import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()

 

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 1. 检查 kwargs 里是否有我们要的路径 (防止 YAML 没改报错)
        # 这里的 train_dir 和 test_dir 对应你 YAML 文件里 Data 部分的新增字段
        train_path = self.kwargs.get('train_dir') 
        val_path = self.kwargs.get('val_dir')

        # 🌟 新增：如果 kwargs 里没有，尝试从 dataset_cfg 里找
        dataset_cfg = self.kwargs.get('dataset_cfg')
        if dataset_cfg:
            # 注意：OmegaConf/addict 支持 .get()
            if not train_path: train_path = dataset_cfg.get('train_dir')
            if not val_path: val_path = dataset_cfg.get('val_dir')
            
            # 如果 YAML 里没写 train_dir/test_dir，尝试回退到旧的 data_dir (为了兼容性)
            if not train_path: train_path = dataset_cfg.get('data_dir')
            if not val_path: val_path = dataset_cfg.get('data_dir')

        # 如果 YAML 里没写 train_dir/test_dir，尝试回退到旧的 data_dir (为了兼容性)
        if not train_path: train_path = self.kwargs.get('data_dir')
        if not val_path: val_path = self.kwargs.get('data_dir')
        
        # 2. 分配训练集和验证集
        if stage == 'fit' or stage is None:
            # 关键点：显式传递 data_dir 参数，这会覆盖 self.kwargs 里的同名参数
            # state='train' 传给 dataset 类，用于加载增强逻辑
            self.train_dataset = self.instancialize(state='train', data_dir=train_path)
            
            # state='val' 传给 dataset 类，用于加载纯净逻辑
            # 注意：验证集使用 val_path (纯净数据)
            self.val_dataset = self.instancialize(state='val', data_dir=val_path)

        # 3. 分配测试集
        if stage == 'test' or stage is None:
            # 测试集也使用 val_path (纯净数据)
            self.test_dataset = self.instancialize(state='test', data_dir=val_path)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)


    def load_data_module(self):
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        argspec = inspect.getargspec(self.data_module.__init__)
        class_args = argspec.args[1:]
        has_kwargs = argspec.keywords is not None
        
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        
        if has_kwargs:
            args1.update(other_args)
        else:
            for k, v in other_args.items():
                if k in class_args:
                    args1[k] = v
                    
        return self.data_module(**args1)