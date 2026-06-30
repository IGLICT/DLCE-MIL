# from pathlib import Path
# import time
# from pathlib import Path
# #---->read yaml
# import yaml
# from addict import Dict
# def read_yaml(fpath=None):
#     with open(fpath, mode="r") as file:
#         yml = yaml.load(file, Loader=yaml.Loader)
#         return Dict(yml)

# #---->load Loggers
# from pytorch_lightning import loggers as pl_loggers
# def load_loggers(cfg):

#     log_path = cfg.General.log_path
#     Path(log_path).mkdir(exist_ok=True, parents=True)
#     log_name = Path(cfg.config).parent 
#     version_name = Path(cfg.config).name[:-5]

#     timestamp = time.strftime("%m-%d_%H-%M")
#     cfg.log_path = str(Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}_{timestamp}')
#     # cfg.log_path = str(Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}')
#     print(f'---->Log dir: {cfg.log_path}')
    
#     #---->TensorBoard
#     tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_name),
#                                              name = version_name, version = f'fold{cfg.Data.fold}',
#                                              log_graph = True, default_hp_metric = False)
#     #---->CSV
#     csv_logger = pl_loggers.CSVLogger(log_path+str(log_name),
#                                       name = version_name, version = f'fold{cfg.Data.fold}', )
    
#     return [tb_logger, csv_logger]


# #---->load Callback
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# def load_callbacks(cfg):

#     Mycallbacks = []
#     # Make output path
#     # ensure output_path is a pathlib.Path (cfg.log_path may be a str)
#     output_path = Path(cfg.log_path) if not isinstance(cfg.log_path, Path) else cfg.log_path
#     output_path.mkdir(exist_ok=True, parents=True)

#     early_stop_callback = EarlyStopping(
#         monitor='val_loss',
#         min_delta=0.00,
#         patience=cfg.General.patience,
#         verbose=True,
#         mode='min'
#     )
#     Mycallbacks.append(early_stop_callback)

#     if cfg.General.server == 'train' :
#         Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
#                                          dirpath = str(cfg.log_path),
#                                          filename = '{epoch:02d}-{val_loss:.4f}',
#                                          verbose = True,
#                                          save_last = True,
#                                          save_top_k = 1,
#                                          mode = 'min',
#                                          save_weights_only = True))
#     return Mycallbacks

# #---->val loss
# import torch
# import torch.nn.functional as F
# def cross_entropy_torch(x, y):
#     x_softmax = [F.softmax(x[i]) for i in range(len(x))]
#     x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
#     loss = - torch.sum(x_log) / len(y)
#     return loss



from pathlib import Path
import time
import os
import re
import yaml
from addict import Dict

#---->read yaml
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers

def load_loggers(cfg):
    # 1. 准备基础路径
    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    
    log_name = Path(cfg.config).parent 
    version_name = Path(cfg.config).name[:-5]

    # 2. 生成带时间戳的最终路径
    # 如果用户通过 --log_dir 指定了 manual_log_dir，则在该目录下创建一个带时间戳的子目录，
    # 否则使用默认的 logs/<dataset>/<version>/fold... 结构。
    manual_dir = None
    try:
        manual_dir = getattr(cfg.General, 'manual_log_dir', None)
    except Exception:
        manual_dir = None

    # 检查是否已经生成过时间戳路径 (防止 train.py 被调用两次导致路径不一致)
    # 规则：
    # - 如果 cfg.log_path 形如 .../fold3_01-26_12-33-27_2045950：认为已是最终目录，不再改
    # - 如果 cfg.log_path 形如 .../fold3：自动补时间戳 -> .../fold3_<timestamp_pid>
    # - 否则：把 cfg.log_path 当作“实验根目录”，自动创建子目录 fold{fold}_<timestamp_pid>
    # 仅检查最后一级目录名，避免把 fold5_1Dconv 这种实验名误判为 fold 目录。
    raw = str(getattr(cfg, 'log_path', ''))
    raw = raw.strip() if raw is not None else ''
    # include seconds and pid to avoid collisions when launching multiple runs quickly
    timestamp = time.strftime("%m-%d_%H-%M-%S") + f"_{os.getpid()}"

    if manual_dir is not None and str(manual_dir) != '':
        base = Path(str(manual_dir))
    else:
        if raw == '' or raw in ('logs', 'logs/'):
            base = Path(log_path) / log_name / version_name
        else:
            base = Path(raw)

    last = base.name
    # include seed in directory name when available
    seed_str = ''
    try:
        if hasattr(cfg, 'General') and getattr(cfg.General, 'seed', None) is not None:
            seed_str = f'_seed{int(cfg.General.seed)}'
    except Exception:
        seed_str = ''

    # Accept both old pattern (fold{n}_{timestamp}) and new pattern (fold{n}_seed{m}_{timestamp})
    if re.match(r'^fold\d+(_seed\d+)?_\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+$', last):
        final_dir = base
    elif re.match(r'^fold\d+$', last):
        final_dir = base.parent / f'{last}{seed_str}_{timestamp}'
    else:
        final_dir = base / f'fold{int(cfg.Data.fold)}{seed_str}_{timestamp}'

    cfg.log_path = str(final_dir)
    if hasattr(cfg, 'General'):
        cfg.General.log_path = cfg.log_path
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
    
    print(f'---->Log dir: {cfg.log_path}')
    
    # 3. 初始化 Loggers (关键修改！)
    # 强制让它们保存到 cfg.log_path，并不再创建子目录
    
    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir = cfg.log_path, # 🌟 修改：直接用最终路径
        name = "",               # 🌟 修改：留空，不创建子文件夹
        version = "",            # 🌟 修改：留空
        log_graph = True, 
        default_hp_metric = False
    )
    
    #---->CSV (metrics.csv 生成器)
    csv_logger = pl_loggers.CSVLogger(
        save_dir = cfg.log_path, # 🌟 修改：直接用最终路径
        name = "",               # 🌟 修改：留空
        version = ""             # 🌟 修改：留空
    )
    
    return [tb_logger, csv_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


class BestMetricsToTxtCallback(pl.Callback):
    def __init__(self, log_file: Path, acc_key: str, auc_key: str = 'auc', balanced_key: str = 'val_balanced_acc'):
        super().__init__()
        self.log_file = Path(log_file)
        self.acc_key = str(acc_key)
        self.auc_key = str(auc_key)
        self.balanced_key = str(balanced_key)

        self.best_acc = None
        self.best_auc = None
        self.best_balanced = None

    @staticmethod
    def _to_float(v):
        try:
            if hasattr(v, 'detach'):
                v = v.detach()
            if hasattr(v, 'item'):
                return float(v.item())
            return float(v)
        except Exception:
            return None

    def _append(self, line: str):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line.rstrip('\n') + '\n')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics = trainer.callback_metrics

        acc = self._to_float(metrics.get(self.acc_key, None))
        auc = self._to_float(metrics.get(self.auc_key, None))
        bal = self._to_float(metrics.get(self.balanced_key, None))
        epoch = int(getattr(trainer, 'current_epoch', 0))

        def fmt(v):
            return 'nan' if v is None else f"{v:.4f}"

        # 1) best accuracy
        if acc is not None and (self.best_acc is None or acc > self.best_acc):
            self.best_acc = acc
            self._append(
                f"best_accuracy-epoch={epoch:02d}-{self.acc_key}={fmt(acc)}-auc={fmt(auc)}-val_balanced_acc={fmt(bal)}"
            )

        # 2) best auc
        if auc is not None and (self.best_auc is None or auc > self.best_auc):
            self.best_auc = auc
            self._append(
                f"best_auc-epoch={epoch:02d}-{self.auc_key}={fmt(auc)}-{self.acc_key}={fmt(acc)}-val_balanced_acc={fmt(bal)}"
            )

        # 3) best balanced acc
        if bal is not None and (self.best_balanced is None or bal > self.best_balanced):
            self.best_balanced = bal
            self._append(
                f"best_balanced_acc-epoch={epoch:02d}-{self.balanced_key}={fmt(bal)}-{self.acc_key}={fmt(acc)}-auc={fmt(auc)}"
            )

def load_callbacks(cfg):
    Mycallbacks = []
    
    # 确保路径存在
    output_path = Path(cfg.log_path) if not isinstance(cfg.log_path, Path) else cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train':
        # 1) keep val_loss checkpoint (minimize)
        Mycallbacks.append(ModelCheckpoint(
            monitor='val_loss',
            dirpath=str(output_path),
            filename='best_val_loss-{epoch:02d}-{val_loss:.4f}',
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode='min',
            save_weights_only=True
        ))

        # 2) best AUC (maximize) — the training code logs this metric as 'auc'
        Mycallbacks.append(ModelCheckpoint(
            monitor='auc',
            dirpath=str(output_path),
            filename='best_auc-{epoch:02d}-{auc:.4f}',
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode='max',
            save_weights_only=True
        ))

        # 3) best Accuracy (maximize)
        # Choose appropriate accuracy metric name depending on number of classes
        acc_monitor = 'val_accuracy'
        try:
            n_cls = None
            if hasattr(cfg, 'Model') and hasattr(cfg.Model, 'n_classes'):
                n_cls = int(cfg.Model.n_classes)
            elif hasattr(cfg, 'Data') and hasattr(cfg.Data, 'n_classes'):
                n_cls = int(cfg.Data.n_classes)
            elif hasattr(cfg, 'Data') and hasattr(cfg.Data, 'num_classes'):
                n_cls = int(cfg.Data.num_classes)
            if n_cls == 2:
                acc_monitor = 'val_BinaryAccuracy'
            elif n_cls and n_cls > 2:
                acc_monitor = 'val_MulticlassAccuracy'
        except Exception:
            acc_monitor = 'val_accuracy'

        Mycallbacks.append(ModelCheckpoint(
            monitor=acc_monitor,
            dirpath=str(output_path),
            filename=f'best_accuracy-{{epoch:02d}}-{{{acc_monitor}:.4f}}',
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode='max',
            save_weights_only=True
        ))

        # 4) best Balanced Accuracy (maximize)
        Mycallbacks.append(ModelCheckpoint(
            monitor='val_balanced_acc',
            dirpath=str(output_path),
            filename='best_balanced_acc-{epoch:02d}-{val_balanced_acc:.4f}',
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode='max',
            save_weights_only=True
        ))

        # 5) write best metrics to log.txt (append-only)
        Mycallbacks.append(BestMetricsToTxtCallback(
            log_file=output_path / 'log.txt',
            acc_key=acc_monitor,
            auc_key='auc',
            balanced_key='val_balanced_acc'
        ))

    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F

def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=0) for i in range(len(x))] # 加上 dim=0 消除警告
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss