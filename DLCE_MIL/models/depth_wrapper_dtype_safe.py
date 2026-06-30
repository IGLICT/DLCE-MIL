import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthAwareWrapper(nn.Module):
    def __init__(self, transmil_model, input_dim=384, norm="gn", dropout=0.4):
        """
        基于你原本效果最好的 GN 版本：
        1. 保留 norm="gn" (这是你跑出高分的关键)
        2. Dropout 建议设为 0.5 (抗过拟合)  之前0.4 /0.1
        """
        super().__init__()
        self.backbone = transmil_model
        
        # 你的原本逻辑：GroupNorm (num_groups=1 等价于对整个样本做 LN)
        if norm == "gn":
            self.norm_layer = nn.GroupNorm(1, input_dim)
        elif norm == "ln":
            self.norm_layer = nn.LayerNorm(input_dim)
        else:
            self.norm_layer = nn.Identity()

        self.norm_type = norm

        # 你的卷积层
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) # 稍微加大力度

        # 你的归一化逻辑
        if self.norm_type == "gn":
            self.norm = self.norm_layer
        else:
            self.norm = nn.LayerNorm(input_dim)

    def forward(self, **kwargs):
        input_data = kwargs.get("data")

        # 兼容两种输入：
        # 1) Tensor: [B, N, C]
        # 2) dict: {'features': Tensor[B,N,C], 'depths': Tensor[B,N], ...}
        if isinstance(input_data, dict):
            if "features" not in input_data:
                raise RuntimeError(
                    "DepthAwareWrapper got dict input for `data` but missing key 'features'. "
                    f"Available keys: {list(input_data.keys())}"
                )
            x = input_data["features"]
        else:
            x = input_data

        if not torch.is_tensor(x):
            raise RuntimeError(
                "DepthAwareWrapper expects `data` to be a Tensor or a dict containing a Tensor at data['features'], "
                f"but got type={type(x)}"
            )

        if x.dim() != 3:
            raise RuntimeError(f"DepthAwareWrapper expects [B, N, C], got {tuple(x.shape)}")

        feat_dim = int(x.shape[-1])
        expected_dim = int(self.conv.in_channels)
        if feat_dim != expected_dim:
            raise RuntimeError(
                "Feature dim mismatch for 1DConv wrapper: "
                f"got C={feat_dim}, but wrapper is configured with input_dim={expected_dim}. "
                "If you changed feature extraction, update Model.in_dim in your YAML."
            )

        # ==========================================
        # 🚨 唯一的新增：入口处的 L2 归一化 🚨
        # 不管后面是用 GN 还是 LN，这一步是解决
        # "Fold 3 高分 vs Fold 4 低分" 的关键钥匙
        # ==========================================
        x = x.to(dtype=self.conv.weight.dtype)
        x = F.normalize(x, p=2, dim=-1)

        identity = x

        # -> [B, C, N] (为了适应 Conv1d 和 GroupNorm)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)

        # 这里的 GN 很好，保持不动！
        if self.norm_type == "gn":
            x = self.norm(x)

        # -> [B, N, C]
        x = x.transpose(1, 2)

        # 这里的 LN 逻辑保持不动
        if self.norm_type == "ln":
            x = self.norm(x)

        # 残差连接
        x = x + identity

        if isinstance(input_data, dict):
            # 不破坏 depth 等其它字段；仅替换 features
            new_data = dict(input_data)
            new_data["features"] = x
            kwargs["data"] = new_data
        else:
            kwargs["data"] = x

        return self.backbone(**kwargs)