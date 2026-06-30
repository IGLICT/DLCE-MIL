import argparse
from pathlib import Path
import numpy as np
import glob
import sys
import os
from omegaconf import OmegaConf 
from omegaconf import DictConfig, ListConfig

# 🌟 关键：只需要标准的 DataInterface 和 ModelInterface
# 不需要引入具体的 CrossAttentionModule，因为已经封装在 ModelInterface 里了
from datasets import DataInterface
from models import ModelInterface

from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# 🌟🌟🌟 1. Logger 类 (保留你原本的优秀设计) 🌟🌟🌟
class Logger(object):
    def __init__(self, filename="Default.log", terminal=None):
        self.terminal = terminal if terminal is not None else sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        # 屏幕上正常显示所有内容
        self.terminal.write(message)
        
        # 🌟 文件写入逻辑：智能过滤
        important_keywords = ['val_loss', 'auc', 'acc', 'Epoch', 'class', 'Correct', 'best', 'Best']
        is_important = any(k in message for k in important_keywords)
        
        # 定义进度条特征
        is_progressbar = 'it/s' in message or '█' in message or '\r' in message or '%' in message
        
        # 决策：如果是重要信息，或者它根本不是进度条，就写入文件
        if is_important or not is_progressbar:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml', type=str)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--fold', default=0)
    parser.add_argument('--start_fold', type=int, default=None, help='Start fold index (inclusive). Default: --fold')
    parser.add_argument('--end_fold', type=int, default=None, help='End fold index (exclusive). Default: start_fold+1, or Data.nfold if --all_folds')
    parser.add_argument('--all_folds', action='store_true', help='Run folds sequentially: [start_fold, Data.nfold)')
    parser.add_argument('--log_dir', default=None, type=str, help='手动指定要测试的模型文件夹路径')
    parser.add_argument('--seed', default=None, type=int, help='Override config General.seed')
    args = parser.parse_args()
    return args

#---->main
def main(cfg, original_stdout=None):
    if original_stdout is None:
        original_stdout = sys.stdout

    # make sure we don't chain Logger(terminal=Logger(...)) across folds
    sys.stdout = original_stdout

    # 转换配置格式（兼容 DictConfig / dict / addict.Dict，且避免 cfg.to_dict 被字段覆盖成 None）
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    else:
        to_dict_attr = getattr(cfg, 'to_dict', None)
        if callable(to_dict_attr):
            cfg = OmegaConf.create(to_dict_attr())
        elif isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        else:
            try:
                cfg = OmegaConf.create(dict(cfg))
            except Exception:
                cfg = OmegaConf.create(cfg)

    # 🌟🌟🌟 2. 启动日志保存逻辑 🌟🌟🌟
    from utils.utils import load_loggers, load_callbacks
    _ = load_loggers(cfg) 
    
    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'terminal_log.txt'
    print(f"----> Clean Log will be saved to: {log_file}")
    
    # 重定向输出
    sys.stdout = Logger(str(log_file), terminal=original_stdout)
    # 🌟🌟🌟 日志设置结束 🌟🌟🌟

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    loggers = load_loggers(cfg) 
    callbacks = load_callbacks(cfg)

    #---->Define Data 
    # Dataset 会自动读取你的新 .pt 文件 (768维)
    DataInterface_dict = {
        'train_batch_size': cfg.Data.train_dataloader.batch_size,
        'train_num_workers': cfg.Data.train_dataloader.num_workers,
        'test_batch_size': cfg.Data.test_dataloader.batch_size,
        'test_num_workers': cfg.Data.test_dataloader.num_workers,
        'dataset_name': cfg.Data.dataset_name,
        'dataset_cfg': cfg.Data,
    }
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    # ModelInterface 内部会调用 GuidedModelWrapper
    # Wrapper 会自动把 768 维拆成 384+384 并做 Cross Attention
    ModelInterface_dict = {
        'model': cfg.Model,
        'loss': cfg.Loss,
        'optimizer': cfg.Optimizer,
        'data': cfg.Data,
        'log': cfg.log_path
    }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.General.epochs,
        min_epochs=50, 
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model=model, datamodule=dm)
    else:
        # 保持你原本完善的测试逻辑
        if cfg.General.manual_log_dir is not None:
            manual = Path(cfg.General.manual_log_dir)
            ckpts_in_manual = list(manual.glob('*.ckpt')) if manual.exists() else []
            if len(ckpts_in_manual) > 0:
                target_folder = manual
            else:
                candidate_dirs = [d for d in manual.iterdir() if d.is_dir()]
                found = None
                latest_mtime = 0
                for d in candidate_dirs:
                    ck = list(d.glob('*.ckpt'))
                    if len(ck) > 0:
                        m = max(p.stat().st_mtime for p in ck)
                        if m > latest_mtime:
                            latest_mtime = m
                            found = d
                if found is not None:
                    target_folder = found
                else:
                    target_folder = manual
        else:
            parent_log_path = Path(cfg.log_path).parent
            all_subdirs = [d for d in parent_log_path.iterdir() if d.is_dir() and 'fold' in d.name]
            all_subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            target_folder = None
            model_paths = []
            
            for folder in all_subdirs:
                ckpts = list(folder.glob('*.ckpt'))
                valid_ckpts = [str(p) for p in ckpts if 'epoch' in str(p)]
                if len(valid_ckpts) > 0:
                    target_folder = folder
                    model_paths = valid_ckpts
                    break
            
            if target_folder is None:
                print("❌ Error: 没找到任何包含模型的文件夹！")
                return

        print(f"\n📂 Testing Model in: {target_folder}")
        cfg.log_path = str(target_folder)
        
        model_paths = list(Path(cfg.log_path).glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]

        for path in model_paths:
            print(f"🚀 Loading model: {path}")
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

    # restore stdout for safety (e.g. multi-fold loop)
    sys.stdout = original_stdout

if __name__ == '__main__':
    args = make_parse()
    raw_cfg = read_yaml(args.config)
    # normalize to plain dict for OmegaConf (avoid raw_cfg.to_dict shadowed by YAML key)
    to_dict_attr = getattr(raw_cfg, 'to_dict', None)
    if callable(to_dict_attr):
        raw_cfg = to_dict_attr()
    elif isinstance(raw_cfg, dict):
        pass
    else:
        raw_cfg = dict(raw_cfg)
    base_cfg = OmegaConf.create(raw_cfg)

    base_cfg.config = args.config
    base_cfg.General.gpus = args.gpus
    base_cfg.General.server = args.stage
    base_cfg.General.manual_log_dir = args.log_dir
    # Allow CLI to override YAML seed
    if args.seed is not None:
        try:
            base_cfg.General.seed = int(args.seed)
        except Exception:
            base_cfg.General.seed = args.seed

    start_fold = int(args.start_fold) if args.start_fold is not None else int(args.fold)
    if args.all_folds:
        end_fold = int(args.end_fold) if args.end_fold is not None else int(base_cfg.Data.nfold)
    else:
        end_fold = int(args.end_fold) if args.end_fold is not None else start_fold + 1

    if end_fold <= start_fold:
        raise ValueError(f"Invalid fold range: start_fold={start_fold}, end_fold={end_fold}")

    original_stdout = sys.stdout
    try:
        for fold in range(start_fold, end_fold):
            cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
            cfg = OmegaConf.create(cfg_dict)
            cfg.Data.fold = int(fold)
            main(cfg, original_stdout=original_stdout)
    finally:
        sys.stdout = original_stdout