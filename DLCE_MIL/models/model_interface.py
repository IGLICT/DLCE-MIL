import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
from typing import Optional

#---->
from MyOptimizer import create_optimizer
# from MyLoss import create_loss
from MyLoss.loss_factory import create_loss
from utils.utils import cross_entropy_torch
# 假设您的文件夹结构是 MyLoss/focal_loss.py
from MyLoss.focal_loss import FocalLoss
#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pathlib import Path
from MyLoss.ldam_loss import LDAMLoss

#---->
import pytorch_lightning as pl
# 在 model_interface.py 开头
from .TransMILadd2 import TransMILadd2
# from .TransMILcat2 import TransMILcat2
# try:
#     from .TransMIL import ScaledDepthFusion, SimilarityRefinement
# except ImportError:
#     # 备选：如果直接在根目录运行，可能是这种写法
#     from TransMIL import ScaledDepthFusion, SimilarityRefinement

# ... (前面的 import)

# =================================================================
# 1. 定义多头相似度精炼模块 (Standard Multi-Head Self-Attention)
# =================================================================
class SimilarityRefinement(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # 这里的 dim 对应投影后的 512 维
        self.norm1 = nn.LayerNorm(dim)
        # Multihead Attention 能够捕捉更丰富的语义关联
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # FFN (前馈网络) - 增强非线性表达能力
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [B, N, 512]
        
        # Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x) # Q, K, V 都是 x
        x = residual + self.dropout(x)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


# =================================================================
# 2. DINOv2 Adapter (dict-safe)
#    - Projects raw token features (e.g., 384) -> hidden_dim (e.g., 512)
#    - Optionally applies SimilarityRefinement
#    - Keeps depth information untouched when input is a dict
# =================================================================
class DINOv2Adapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int = 384,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        use_sim_refiner: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.act = nn.GELU()
        self.use_sim_refiner = bool(use_sim_refiner)
        self.sim_refiner = (
            SimilarityRefinement(dim=int(hidden_dim), num_heads=8)
            if self.use_sim_refiner
            else nn.Identity()
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, **kwargs):
        input_data = kwargs.get('data')

        if isinstance(input_data, dict):
            if 'features' not in input_data:
                raise RuntimeError(
                    "DINOv2Adapter got dict input for `data` but missing key 'features'. "
                    f"Available keys: {list(input_data.keys())}"
                )
            x = input_data['features']
        else:
            x = input_data

        if x is None:
            raise RuntimeError("DINOv2Adapter expects `data` (Tensor or dict with 'features'). Got None.")

        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.act(x)
        x = self.sim_refiner(x)
        x = self.dropout(x)
        x = self.norm(x)

        if isinstance(input_data, dict):
            new_data = dict(input_data)
            new_data['features'] = x
            kwargs['data'] = new_data
        else:
            kwargs['data'] = x

        return self.backbone(**kwargs)

# =================================================================
# 2. 定义单流适配器 (DINOv2 Adapter)
# =================================================================
# class DINOv2Adapter(nn.Module):
#     def __init__(self, model, input_dim=384, hidden_dim=512, dropout=0.5, use_sim_refiner=True):
#         super().__init__()
#         # 1. 保存原来的模型 (TransMIL)
#         self.backbone = model 
        
#         # 2. 定义适配层: 384 -> 512
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.act = nn.GELU()
        
#         # 3. 🌟 插入相似度精炼（可选） 🌟
#         self.use_sim_refiner = bool(use_sim_refiner)
#         self.sim_refiner = (
#             SimilarityRefinement(dim=hidden_dim, num_heads=8)
#             if self.use_sim_refiner
#             else nn.Identity()
#         )
        
#         # 4. 规范化
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(hidden_dim)
        
#     def forward(self, **kwargs):
#         # 1. 获取输入数据 [B, N, 384]
#         x = kwargs['data'] 
        
#         # 2. 投影 + 激活
#         x = x.to(dtype=self.fc1.weight.dtype) # 确保类型一致
#         x = self.fc1(x)  # -> [B, N, 512]
#         x = self.act(x)
        
#         # 3. 🌟 相似度增强（可选） 🌟
#         x = self.sim_refiner(x)
        
#         x = self.dropout(x)
#         x = self.norm(x)
        
#         # 4. 把增强后的特征传给原来的 TransMIL
#         # 注意：TransMIL 里的 input_dim 必须设为 512，因为它接收的是这里输出的 512 维
#         kwargs['data'] = x 
#         return self.backbone(**kwargs)

# depth
# class DINOv2Adapter(nn.Module):
#     def __init__(self, model, input_dim=384, hidden_dim=512, dropout=0.5, use_sim_refiner=True):
#         super().__init__()
#         # 1. 保存原来的模型 (TransMIL)
#         self.backbone = model 
        
#         # 2. 定义适配层: 384 -> 512
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.act = nn.GELU()
        
#         # 3. 🌟 插入相似度精炼（可选） 🌟
#         self.use_sim_refiner = bool(use_sim_refiner)
#         self.sim_refiner = (
#             SimilarityRefinement(dim=hidden_dim, num_heads=8)
#             if self.use_sim_refiner
#             else nn.Identity()
#         )
        
#         # 4. 规范化
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(hidden_dim)
        
#         # 5. 🌟 新增：深度融合模块 🌟
#         # 确保 dim 和 hidden_dim 一致 (512)
#         self.depth_fusion = ScaledDepthFusion(dim=hidden_dim, max_depth=1000, init_scale=0.0)
        
#     def forward(self, **kwargs):
#         # ============================================
#         # 🌟 修改点 1: 智能拆包 (Unpacking) 🌟
#         # ============================================
#         # 尝试从 kwargs['data'] 中提取 features 和 depths
#         # 你的 DataLoader 返回的可能是 {'features': ..., 'depths': ...} 字典
#         input_data = kwargs.get('data')
        
#         if isinstance(input_data, dict):
#             # 如果 data 是字典，直接拆
#             x = input_data['features']
#             d = input_data['depths']
#         else:
#             # 兼容旧代码或直接传 tensor 的情况
#             x = input_data
#             d = kwargs.get('depths', None) # 尝试找找有没有单独传 depth

#         # 兜底：如果没有深度信息，生成全 0 (为了不报错)
#         if d is None:
#             d = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.long)

#         # ============================================
#         # 🌟 修改点 2: 投影 -> 融合 -> 增强 🌟
#         # ============================================
        
#         # 1. 投影 + 激活 (384 -> 512)
#         x = x.to(dtype=self.fc1.weight.dtype)
#         x = self.fc1(x)
#         x = self.act(x)
        
#         # 2. 🔥 深度融合 (在这里插入!) 🔥
#         x = self.depth_fusion(x, d)
        
#         # 3. 相似度增强 (你原来的逻辑)
#         x = self.sim_refiner(x)
        
#         # 4. 规范化
#         x = self.dropout(x)
#         x = self.norm(x)
        
#         # ============================================
#         # 🌟 修改点 3: 传回 Backbone 🌟
#         # ============================================
#         # 更新 kwargs['data'] 为处理后的特征，传给 TransMIL
#         kwargs['data'] = x 
        
#         # 如果 Backbone 不需要 depths 参数，最好把它清理掉，防止报错
#         # (视你的 TransMIL forward 实现而定，通常 kwargs 会透传)
#         if 'depths' in kwargs: 
#             pass # 保留也行，如果 backbone 会忽略多余参数
            
#         return self.backbone(**kwargs)




class ModelInterface(pl.LightningModule):

    #---->init
    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        
        # 不需要再写 try-except block 来转换 addict 了
        
        self.load_model()
# 进行加权
        # class_weights = torch.tensor([1.0, 2.0]).float()
        # 加权
        weights = torch.tensor([1.0, 1.3]) 
        # self.loss = nn.CrossEntropyLoss(weight=weights)
        self.loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)
        # self.loss = create_loss(loss)
        # self.loss = FocalLoss(gamma=2.0, alpha=None)
        # cls_num_list = [59, 35] # ⚠️请根据您实际训练集的 0类和1类 数量填写
        # self.loss = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=None)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = Path(kargs['log'])
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.patient_records = {stage: [] for stage in ('train', 'val', 'test')}
        self.patient_log_paths = {
            'train': self.log_path / 'train_patient_probabilities.csv',
            'val': self.log_path / 'val_patient_probabilities.csv',
            'test': self.log_path / 'test_patient_probabilities.csv',
        }
        # ... 后面的代码保持不变 ...

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes=self.n_classes, average='macro')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, average='micro'),
                torchmetrics.CohenKappa(task='multiclass', num_classes=self.n_classes),
                # F1 moved in newer torchmetrics: use classification.MulticlassF1Score when F1 is not available
                (torchmetrics.F1(task='multiclass', num_classes=self.n_classes, average='macro')
                 if hasattr(torchmetrics, 'F1') else
                 torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=self.n_classes, average='macro')),
                torchmetrics.Recall(task='multiclass', average='macro', num_classes=self.n_classes),
                torchmetrics.Precision(task='multiclass', average='macro', num_classes=self.n_classes),
                torchmetrics.Specificity(task='multiclass', average='macro', num_classes=self.n_classes)
            ])
        else : 
            self.AUROC = torchmetrics.AUROC(task='binary', num_classes=2, average='macro')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task='binary', num_classes=2, average='micro'),
                torchmetrics.CohenKappa(task='binary', num_classes=2),
                # F1 moved in newer torchmetrics: use classification.BinaryF1Score when F1 is not available
                (torchmetrics.F1(task='binary', num_classes=2, average='macro')
                 if hasattr(torchmetrics, 'F1') else
                 torchmetrics.classification.BinaryF1Score(task='binary', average='macro')),
                torchmetrics.Recall(task='binary', average='macro', num_classes=2),
                torchmetrics.Precision(task='binary', average='macro', num_classes=2)
            ])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label, slide_id = self._unpack_batch(batch)
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(logits, label)

        #---->acc log
        Y_hat_int = int(Y_hat)
        Y = self._tensor_to_int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat_int == Y)
        self._record_patient_result('train', slide_id, Y, Y_hat_int, batch_idx, y_prob=Y_prob)

        return {'loss': loss} 

    def training_epoch_end(self, training_step_outputs):
        total_correct = 0
        total_count = 0
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = 0.0
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            
            # Log class-wise accuracy
            self.log(f'train_acc_class{c}', acc, prog_bar=True, on_epoch=True, logger=True)
            
            total_correct += correct
            total_count += count
            
        # Log overall accuracy
        if total_count > 0:
            train_acc = float(total_correct) / total_count
            self.log('train_acc', train_acc, prog_bar=True, on_epoch=True, logger=True)

        self._flush_patient_records('train')
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ----> log lambda_sim (TransMIL_Bias)
        self._log_lambda_sim(stage='train')

    def validation_step(self, batch, batch_idx):
        data, label, slide_id = self._unpack_batch(batch)
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        Y = self._tensor_to_int(label)
        Y_hat_int = int(Y_hat)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat_int == Y)
        self._record_patient_result('val', slide_id, Y, Y_hat_int, batch_idx, y_prob=Y_prob)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        # self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs[:, 1], target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)

        # ----> Save ROC curve (binary only)
        try:
            self._save_roc_curve(
                stage='val',
                pos_probs=probs[:, 1],
                target=target.squeeze(),
                epoch=int(self.current_epoch),
            )
        except Exception as e:
            print(f"[WARN] Failed to save val ROC curve: {e}")

        #---->acc log
        per_class_accs = []
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = 0.0
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            
            # Log class-wise accuracy for validation
            self.log(f'val_acc_class{c}', acc, prog_bar=True, on_epoch=True, logger=True)

            # For balanced accuracy, only include classes that appear in this epoch
            if count > 0:
                per_class_accs.append(acc)

        # Log balanced accuracy for checkpoint monitor compatibility
        if len(per_class_accs) == 0:
            val_balanced_acc = 0.0
        else:
            val_balanced_acc = float(sum(per_class_accs)) / float(len(per_class_accs))
        self.log('val_balanced_acc', val_balanced_acc, prog_bar=True, on_epoch=True, logger=True)
            
        self._flush_patient_records('val')
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ----> log lambda_sim (TransMIL_Bias)
        self._log_lambda_sim(stage='val')
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)


    def _log_lambda_sim(self, stage: str):
        """Log TransMIL_Bias learnable similarity bias scale(s).

        If the current model does not include any `lambda_sim` parameters,
        this is a no-op.
        """

        if not hasattr(self, 'model') or self.model is None:
            return

        lambda_params = []
        try:
            for name, param in self.model.named_parameters():
                if name.endswith('lambda_sim'):
                    lambda_params.append((name, param))
        except Exception:
            return

        if len(lambda_params) == 0:
            return

        with torch.no_grad():
            values = []
            parts = []
            for name, param in lambda_params:
                # Most implementations keep it as a scalar
                v = param.detach().float().reshape(-1)[0]
                values.append(v)
                parts.append(f"{name}={v.item():.6f}")

            values_t = torch.stack(values)
            mean_v = values_t.mean().item()
            max_abs_v = values_t.abs().max().item()

        print(f"[lambda_sim] {stage}: " + ", ".join(parts))
        # Use distinct metric names to avoid collisions with user metrics
        self.log(f"{stage}_lambda_sim_mean", mean_v, prog_bar=False, on_epoch=True, logger=True)
        self.log(f"{stage}_lambda_sim_max_abs", max_abs_v, prog_bar=False, on_epoch=True, logger=True)
    


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label, slide_id = self._unpack_batch(batch)
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = self._tensor_to_int(label)
        Y_hat_int = int(Y_hat)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat_int == Y)
        self._record_patient_result('test', slide_id, Y, Y_hat_int, batch_idx, y_prob=Y_prob)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    # def test_epoch_end(self, output_results):
    #     probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
    #     max_probs = torch.stack([x['Y_hat'] for x in output_results])
    #     target = torch.stack([x['label'] for x in output_results], dim = 0)
        
    #     #---->
    #     # auc = self.AUROC(probs, target.squeeze())

    #     # auc = self.AUROC(probs[:, 1], target.squeeze())
    #     auc = self.AUROC(probs[:, 0], target.squeeze())
    #     metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
    #     metrics['auc'] = auc
    #     for keys, values in metrics.items():
    #         print(f'{keys} = {values}')
    #         metrics[keys] = values.cpu().numpy()
    #     print()
    #     #---->acc log
    #     for c in range(self.n_classes):
    #         count = self.data[c]["count"]
    #         correct = self.data[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    #     #---->
    #     result = pd.DataFrame([metrics])
    #     result.to_csv(self.log_path / 'result.csv')
    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        target = torch.stack([x['label'] for x in output_results], dim = 0)
        
        # 1. 计算 AUC (使用第 1列)
        pos_probs = probs[:, 1]
        auc = self.AUROC(pos_probs, target.squeeze())

        # 2. 自动寻找最佳阈值
        fpr, tpr, thresholds = torchmetrics.functional.roc(pos_probs, target.squeeze(), task='binary')
        optimal_idx = torch.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]
        print(f"\n🌟 Best Threshold found: {best_threshold:.4f}")

        # ----> Save ROC curve (binary only)
        try:
            self._save_roc_curve(
                stage='test',
                pos_probs=pos_probs,
                target=target.squeeze(),
                epoch=None,
            )
        except Exception as e:
            print(f"[WARN] Failed to save test ROC curve: {e}")

        # 3. 基于最佳阈值生成新的预测
        new_preds = (pos_probs >= best_threshold).int()

        # 4. 计算基础指标 (Accuracy, F1, Recall, etc.)
        metrics = self.test_metrics(new_preds.squeeze(), target.squeeze())
        metrics['auc'] = auc
        # 将 Tensor 类型的指标转为数值，方便打印和保存
        for keys, values in metrics.items():
            # print(f'{keys} = {values}') # 可以先不打，后面统一打
            if hasattr(values, 'cpu'):
                metrics[keys] = values.cpu().numpy()

        # -------------------------------------------------------
        # 🌟🌟🌟 核心修改：手动计算分列准确率并保存 🌟🌟🌟
        print("\n📊 修正后的真实分类结果 (基于最佳阈值):")
        
        # Class 0 (镰刀菌)
        mask0 = (target.squeeze() == 0)
        total0 = mask0.sum().item()
        if total0 > 0:
            correct0 = (new_preds[mask0] == 0).sum().item()
            acc0 = correct0 / total0
        else:
            acc0 = 0.0
            correct0 = 0
        print(f"Class 0: Acc {acc0:.4f}, Correct {correct0}/{total0}")
        
        # Class 1 (曲霉菌)
        mask1 = (target.squeeze() == 1)
        total1 = mask1.sum().item()
        if total1 > 0:
            correct1 = (new_preds[mask1] == 1).sum().item()
            acc1 = correct1 / total1
        else:
            acc1 = 0.0
            correct1 = 0
        print(f"Class 1: Acc {acc1:.4f}, Correct {correct1}/{total1}")

        # ✅ 把算出来的结果塞进 metrics 字典，这样就会保存到 CSV 了
        metrics['test_Class0_Acc'] = acc0
        metrics['test_Class1_Acc'] = acc1
        metrics['best_threshold'] = best_threshold.item() # 把最佳阈值也存下来备忘
        # -------------------------------------------------------

        # 6. 保存 CSV
        # 此时 metrics 里已经包含了 auc, f1, class0_acc, class1_acc 等所有信息
        result = pd.DataFrame([metrics])
        save_path = self.log_path / 'result1.csv'
        result.to_csv(save_path)
        print(f"✅ Metrics saved to {save_path}\n")

        # 统计 Class 1
        mask1 = (target.squeeze() == 1)
        correct1 = (new_preds[mask1] == 1).sum().item() # 这里的 1 对应正类
        total1 = mask1.sum().item()
        print(f"Class 1: Acc {correct1/total1:.4f}, Correct {correct1}/{total1}")
        self._flush_patient_records('test')


    def _save_roc_curve(self, stage: str, pos_probs: torch.Tensor, target: torch.Tensor, epoch: Optional[int] = None):
        """Save ROC curve plot (.png) and raw curve data (.csv).

        Notes:
        - Only supports binary classification (positive class probability).
        - Designed to be safe: failures should not crash training.
        """

        if int(self.n_classes) != 2:
            return

        pos_probs = pos_probs.detach().float().reshape(-1)
        target = target.detach().int().reshape(-1)

        # If only one class is present in target, ROC is undefined
        unique = torch.unique(target)
        if unique.numel() < 2:
            return

        fpr, tpr, thresholds = torchmetrics.functional.roc(pos_probs, target, task='binary')

        # AUC from ROC points
        auc_from_curve = float(torch.trapz(tpr, fpr).item())

        # Build filenames
        if epoch is None:
            stem = self.log_path / f'roc_{stage}'
        else:
            stem = self.log_path / f'roc_{stage}_epoch{epoch:03d}'

        # Save raw curve data
        try:
            import pandas as pd

            df = pd.DataFrame({
                'fpr': fpr.detach().cpu().numpy(),
                'tpr': tpr.detach().cpu().numpy(),
                'threshold': thresholds.detach().cpu().numpy(),
            })
            df.to_csv(str(stem) + '.csv', index=False)
        except Exception:
            pass

        # Save plot
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fpr_np = fpr.detach().cpu().numpy()
            tpr_np = tpr.detach().cpu().numpy()

            plt.figure(figsize=(5, 5))
            plt.plot(fpr_np, tpr_np, label=f'AUC={auc_from_curve:.4f}', linewidth=2)
            plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC ({stage})')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(str(stem) + '.png', dpi=200)
            plt.close()
        except Exception:
            # Plotting is optional
            pass


    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, label, slide_id = batch
        else:
            data, label = batch
            slide_id = None
        return data, label, self._normalize_slide_id(slide_id)

    @staticmethod
    def _normalize_slide_id(slide_id):
        if isinstance(slide_id, (list, tuple)):
            if len(slide_id) == 1:
                slide_id = slide_id[0]
            else:
                slide_id = ','.join(map(str, slide_id))
        if isinstance(slide_id, torch.Tensor):
            slide_id = slide_id.item() if slide_id.numel() == 1 else str(slide_id.tolist())
        if slide_id is None:
            return 'unknown'
        return str(slide_id)

    @staticmethod
    def _tensor_to_int(value):
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    @staticmethod
    def _extract_binary_probs(y_prob):
        """Return (class0_prob, class1_prob) from model Y_prob tensor."""
        if not isinstance(y_prob, torch.Tensor):
            return None, None
        if y_prob.numel() == 0:
            return None, None

        probs = y_prob.detach().float()
        if probs.dim() == 1:
            # shape [C]
            p0 = float(probs[0].item()) if probs.shape[0] > 0 else None
            p1 = float(probs[1].item()) if probs.shape[0] > 1 else None
            return p0, p1

        # shape [B, C], keep first sample for patient-level record
        row = probs[0]
        p0 = float(row[0].item()) if row.shape[0] > 0 else None
        p1 = float(row[1].item()) if row.shape[0] > 1 else None
        return p0, p1

    def _record_patient_result(self, stage, slide_id, label, pred, batch_idx, y_prob=None):
        prob0, prob1 = self._extract_binary_probs(y_prob)
        record = {
            'epoch': int(self.current_epoch),
            'step': int(self.global_step),
            'batch_idx': int(batch_idx),
            'stage': stage,
            'patient_id': slide_id,
            'target': label,
            'prediction': pred,
            'correct': int(pred == label),
            'prob_class0': prob0,
            'prob_class1': prob1,
            '镰刀菌(0)概率': prob0,
            '曲霉菌(1)概率': prob1,
        }
        self.patient_records[stage].append(record)

    def _flush_patient_records(self, stage):
        records = self.patient_records.get(stage, [])
        if not records:
            return
        df = pd.DataFrame(records)
        path = self.patient_log_paths[stage]
        header = not path.exists()
        df.to_csv(path, mode='a', header=header, index=False)
        self.patient_records[stage] = []


    # def load_model(self):
    #     # support both mapping (dict) and attribute-style (addict.Dict/omegaconf)
    #     if isinstance(self.hparams.model, dict):
    #         name = self.hparams.model.get('name')
    #     else:
    #         name = getattr(self.hparams.model, 'name')
    #     # Change the `trans_unet.py` file name to `TransUnet` class name.
    #     # Please always name your model file name as `trans_unet.py` and
    #     # class name or funciton name corresponding `TransUnet`.
    #     if '_' in name:
    #         camel_name = ''.join([i.capitalize() for i in name.split('_')])
    #     else:
    #         camel_name = name
    #     try:
    #         Model = getattr(importlib.import_module(
    #             f'models.{name}'), camel_name)
    #     except:
    #         raise ValueError('Invalid Module File Name or Invalid Class Name!')
    #     self.model = self.instancialize(Model)

    # # def load_model(self):
    #     # ... (前面的代码保持不变) ...
    #     # # try:
    #     # Model = getattr(importlib.import_module(
    #     #     f'models.{name}'), camel_name)
    #     # except:
    #     #     raise ValueError('Invalid Module File Name or Invalid Class Name!')
        
    #     # 1. 实例化原始模型 (TransMIL)
    #     # self.model = self.instancialize(Model)

    #     # # ====================================================
    #     # # 🌟 核心修改：如果是 TransMIL，自动套上 Adapter 包装器
    #     # # ====================================================
    #     # if camel_name == 'TransMIL':
    #     #     print(f"🔥 [Model Interface] 检测到 TransMIL，正在应用 HandcraftedGuidedAdapter (Input 385 -> 384)...")
    #     #     # 用 GuidedModelWrapper 替换 self.model
    #     #     self.model = GuidedModelWrapper(self.model, dim=384)
    #     # if camel_name == 'TransMIL':
    #     #     feat_dim = 384  
    #     #     print(f"🔥 [Model Interface] Wrapping TransMIL with DepthAwareWrapper (dim={feat_dim})...")
            
    #     #     # 这里代码不用变，只要上面 import 对了，这里就能跑
    #     #     self.model = DepthAwareWrapper(self.model, input_dim=feat_dim, dropout=0.1)
    #     pass

    def load_model(self):
        # support both mapping (dict) and attribute-style (addict.Dict/omegaconf)
        if isinstance(self.hparams.model, dict):
            name = self.hparams.model.get('name')
        else:
            name = getattr(self.hparams.model, 'name')
            
        # 驼峰命名转换（保留原本大小写，不用 .capitalize()，否则会把 TransMIL 变成 Transmil）
        if '_' in name:
            parts = [p for p in str(name).split('_') if p]
            camel_name = ''.join([p[:1].upper() + p[1:] for p in parts])
        else:
            camel_name = name
            
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        
        # ========================================================
        # Generic optional wrappers (Adapter / SimilarityRefinement / 1DConv)
        # - Previously only enabled for TransMIL/TransmilBias.
        # - Now available for any model as long as its __init__ accepts
        #   the relevant args (instancialize() auto-filters unknown args).
        # ========================================================

        # Keep backward-compatible defaults: TransMIL family defaults to adapter+sim_refiner,
        # other models default to no extra wrappers unless explicitly enabled in YAML.
        default_use_adapter = True if camel_name in ('TransMIL', 'TransmilBias') else False
        default_use_sim_refiner = True if camel_name in ('TransMIL', 'TransmilBias') else False

        if isinstance(self.hparams.model, dict):
            use_adapter = self.hparams.model.get('use_adapter', default_use_adapter)
            use_sim_refiner = self.hparams.model.get('use_sim_refiner', default_use_sim_refiner)
            use_1dconv = self.hparams.model.get('use_1dconv', False)
            raw_in_dim = self.hparams.model.get('in_dim', 384)
            hidden_dim = self.hparams.model.get('project_dim', 512)
            adapter_dropout = self.hparams.model.get('adapter_dropout', 0.5)
            conv_norm = self.hparams.model.get('conv_norm', self.hparams.model.get('norm', 'gn'))
            conv_dropout = self.hparams.model.get('conv_dropout', self.hparams.model.get('dropout', 0.4))
        else:
            use_adapter = getattr(self.hparams.model, 'use_adapter', default_use_adapter)
            use_sim_refiner = getattr(self.hparams.model, 'use_sim_refiner', default_use_sim_refiner)
            use_1dconv = getattr(self.hparams.model, 'use_1dconv', False)
            raw_in_dim = getattr(self.hparams.model, 'in_dim', 384)
            hidden_dim = getattr(self.hparams.model, 'project_dim', 512)
            adapter_dropout = getattr(self.hparams.model, 'adapter_dropout', 0.5)
            conv_norm = getattr(self.hparams.model, 'conv_norm', getattr(self.hparams.model, 'norm', 'gn'))
            conv_dropout = getattr(self.hparams.model, 'conv_dropout', getattr(self.hparams.model, 'dropout', 0.4))

        use_adapter = bool(use_adapter)
        use_sim_refiner = bool(use_sim_refiner)
        use_1dconv = bool(use_1dconv)

        print(f"🔥 [Model Interface] Loaded backbone: {camel_name} (file=models.{name})")
        if use_adapter:
            print(f"   -> Adapter enabled: {raw_in_dim} -> {hidden_dim} (sim_refiner={use_sim_refiner})")
        else:
            print(f"   -> Adapter disabled")
        if use_1dconv:
            print(f"   -> 1DConv enabled: norm={conv_norm}, dropout={conv_dropout}")

        # Build pipeline
        if use_adapter:
            backbone = self.instancialize(Model, in_dim=int(hidden_dim))
            pipeline = DINOv2Adapter(
                backbone,
                input_dim=int(raw_in_dim),
                hidden_dim=int(hidden_dim),
                dropout=float(adapter_dropout),
                use_sim_refiner=use_sim_refiner,
            )
        else:
            pipeline = self.instancialize(Model, in_dim=int(raw_in_dim))

        if use_1dconv:
            from .depth_wrapper_dtype_safe import DepthAwareWrapper
            pipeline = DepthAwareWrapper(
                pipeline,
                input_dim=int(raw_in_dim),
                norm=str(conv_norm),
                dropout=float(conv_dropout),
            )

        self.model = pipeline


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        # NOTE: `inspect.getargspec` is deprecated and fails when the target
        # function contains keyword-only parameters or annotations.
        # Use getfullargspec / signature for broader compatibility.
        accepts_varkw = False
        try:
            spec = inspect.getfullargspec(Model.__init__)
            class_args = list(spec.args[1:]) + list(spec.kwonlyargs)
            accepts_varkw = spec.varkw is not None
        except Exception:
            sig = inspect.signature(Model.__init__)
            params = list(sig.parameters.values())[1:]
            class_args = [
                p.name
                for p in params
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            accepts_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
        # support both dict-like and attribute-like access for hparams.model
        if hasattr(self.hparams.model, 'keys'):
            inkeys = self.hparams.model.keys()
        elif isinstance(self.hparams.model, dict):
            inkeys = self.hparams.model.keys()
        else:
            inkeys = []

        args1 = {}

        if accepts_varkw:
            # Model can accept **kwargs: pass through all config keys.
            for key in inkeys:
                if isinstance(self.hparams.model, dict):
                    args1[key] = self.hparams.model[key]
                else:
                    args1[key] = getattr(self.hparams.model, key)
        else:
            # Only pass keys that appear in the __init__ signature.
            for arg in class_args:
                if arg in inkeys:
                    if isinstance(self.hparams.model, dict):
                        args1[arg] = self.hparams.model[arg]
                    else:
                        args1[arg] = getattr(self.hparams.model, arg)

        # Explicit overrides win
        args1.update(other_args)
        return Model(**args1)