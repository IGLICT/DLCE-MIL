import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class ScaledDepthFusion(nn.Module):
    def __init__(self, dim=512, max_depth=1000, init_scale=0.0):
        super().__init__()
        self.depth_emb = nn.Embedding(num_embeddings=max_depth, embedding_dim=dim)
        self.alpha = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x, depth_indices):
        depth_indices = depth_indices.clamp(0, self.depth_emb.num_embeddings - 1).long()
        d_vec = self.depth_emb(depth_indices)
        return x + self.alpha * d_vec


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class PPEG1DDepthwise(nn.Module):
    def __init__(self, dim=512, k_list=(7, 5, 3)):
        super().__init__()
        k7, k5, k3 = k_list

        self.proj  = nn.Conv1d(dim, dim, kernel_size=k7, padding=k7//2, groups=dim, bias=False)
        # self.proj1 = nn.Conv1d(dim, dim, kernel_size=k5, padding=k5//2, groups=dim, bias=False)
        # self.proj2 = nn.Conv1d(dim, dim, kernel_size=k3, padding=k3//2, groups=dim, bias=False)

    def forward(self, x, H=None, W=None):
        # x: [B, N, C], N includes cls token
        B, N, C = x.shape
        cls_token = x[:, 0:1, :]    # [B,1,C]
        feat_token = x[:, 1:, :]    # [B,N-1,C]

        t = feat_token.transpose(1, 2).contiguous()  # [B,C,N-1]
        # out = self.proj(t) + t + self.proj1(t) + self.proj2(t)  # [B,C,N-1]
        out = self.proj(t) + t  # [B,C,N-1]
        out = out.transpose(1, 2).contiguous()  # [B,N-1,C]

        out = torch.cat([cls_token, out], dim=1)  # [B,N,C]
        return out



class TransMILadd2DICE(nn.Module):
    """
    - 输入 features: [B, N, in_dim]，默认 in_dim=384
    - depth 可选：
        * use_depth=True 且提供 depths -> 做 additive depth fusion
        * use_depth=False 或未提供 depths -> 走最原始 TransMILori 主干（不做 depth=0 兜底）
    """
    def __init__(
        self,
        n_classes: int,
        in_dim: int = 384,
        use_depth: bool = True,
        max_depth: int = 1000,
        init_scale: float = 0.0,
        use_ppeg: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.use_depth = bool(use_depth)
        self.use_ppeg = bool(use_ppeg)

        # ✅ ori 风格投影：Linear + ReLU
        self._fc1 = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(inplace=True)
        )

        # depth fusion（只在 use_depth=True 且 depths!=None 时启用）
        self.depth_fusion = ScaledDepthFusion(dim=512, max_depth=max_depth, init_scale=init_scale)

        # TransMIL 组件（与 ori 同结构）
        self.pos_layer = PPEG(dim=512)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):
        input_data = kwargs.get("data")

        # ---- unpack ----
        if isinstance(input_data, dict):
            x = input_data["features"]
            d = input_data.get("depths", None)
        else:
            x = input_data
            d = kwargs.get("depths", None)

        # ---- stem (ori) ----
        x = x.to(dtype=self._fc1[0].weight.dtype)
        h = self._fc1(x)  # [B, N, 512]

        # ---- optional depth fusion (NO depth=0 fallback) ----
        if self.use_depth and (d is not None):
            h = self.depth_fusion(h, d)

        # ---- pad to square ----
        H = h.shape[1]
        _H = _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        # ---- cls token ----
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---- transformer + PPEG ----
        h = self.layer1(h)
        if self.use_ppeg:
            h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        # ---- head ----
        h_cls = self.norm(h)[:, 0]
        logits = self._fc2(h_cls)

        return {
            "logits": logits,
            "Y_prob": F.softmax(logits, dim=1),
            "Y_hat": torch.argmax(logits, dim=1),
            "cls_emb": h_cls,
        }

# 兼容别名
TransMILAdd2DICE = TransMILadd2DICE
