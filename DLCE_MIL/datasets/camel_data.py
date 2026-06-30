import random
import torch
import pandas as pd
from pathlib import Path
import torch.utils.data as data
import os
class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None, data_dir=None):
        # 将传入参数设为属性
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #----> 配置参数
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = data_dir if data_dir else self.dataset_cfg.data_dir
        
        #----> 读取标签 CSV
        self.csv_dir = os.path.join(self.dataset_cfg.label_dir, f'fold{self.fold}.csv')
        self.slide_data = pd.read_csv(self.csv_dir, index_col=None)

        #----> 是否打乱 (Bag 内部打乱)
        self.shuffle = self.dataset_cfg.data_shuffle

        #----> 数据集划分 (Train/Val/Test)
        # 🌟 核心修复: 使用 reset_index(drop=True) 确保索引连续 (0, 1, 2...)
        # 否则 __getitem__ 中的 idx 可能会越界
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'train_label'].dropna().reset_index(drop=True)
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'val_label'].dropna().reset_index(drop=True)
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'test_label'].dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取 Slide ID 和 Label
        slide_id = self.data[idx]
        label = int(self.label[idx])
        
        # 构建特征文件路径
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'

        # ============================================================
        # 1. 加载数据 (含容错处理)
        # ============================================================
        try:
            data = torch.load(full_path)
        except FileNotFoundError:
            print(f"⚠️ Warning: File not found {full_path}, returning zeros.")
            # 🔥 修复: 防止 feature_dim 为 None 导致的报错
            safe_dim = getattr(self.dataset_cfg, 'feature_dim', None) or 384
            return {
                'features': torch.zeros(1, safe_dim), 
                'depths': torch.zeros(1, dtype=torch.long)
            }, label, str(slide_id)

        # ============================================================
        # 2. 解析特征 (Features)
        # ============================================================
        if isinstance(data, torch.Tensor):
            features = data
        elif isinstance(data, dict):
            # 🌟 优先找 'features' (新提取的格式)
            if 'features' in data:
                features = data['features']
            # 兼容旧格式 (根据配置决定)
            else:
                mode = getattr(self.dataset_cfg, "feat_mode", "mean")
                if mode == "cls":
                    features = data["cls"]
                elif mode == "cat":
                    # 拼接 cls 和 mean
                    features = data.get("cat", torch.cat([data["cls"], data["mean"]], dim=-1))
                else:
                    # 默认回退到 'mean'
                    features = data.get("mean", list(data.values())[0])
        else:
             # 未知格式兜底
             safe_dim = getattr(self.dataset_cfg, 'feature_dim', None) or 384
             features = torch.zeros(1, safe_dim)

        # ============================================================
        # 3. 解析深度 (Depths)
        # ============================================================
        if isinstance(data, dict) and 'depths' in data:
            depths = data['depths']
        else:
            # 如果没有深度信息，生成全 0 (类型必须是 Long)
            depths = torch.zeros((features.shape[0]), dtype=torch.long)

        # ============================================================
        # 4. 同步打乱 (Shuffle)
        # ============================================================
        if self.shuffle:
            # 生成随机索引
            indices = torch.randperm(features.shape[0])
            # 必须同时应用到特征和深度，保持对应关系
            features = features[indices]
            depths = depths[indices]

        # ============================================================
        # 5. 打包返回
        # ============================================================
        data_dict = {
            'features': features, # [N, Dim]
            'depths': depths      # [N]
        }

        return data_dict, label, str(slide_id)