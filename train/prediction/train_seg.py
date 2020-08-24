from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader,random_split
from lr_scheduler import CosineAnnealingWarmUpRestarts
from torchvision import transforms

import PIL.Image as Image

from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

from model import DeepLab_ResNet50,FCN_ResNet50
from data import PCategoryDataset,NPCategoryDataset

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateLogger,ModelCheckpoint
from pytorch_lightning import Trainer,loggers,seed_everything


class Model(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--image_dir', type=str, default='../../data/np/rgbd')
        parser.add_argument('--category_dir', type=str, default='../../data/np/category')
        parser.add_argument('--person', type=str, default='np')

        parser.add_argument('--batch_size', type=int, default=30)

        parser.add_argument('--optimizer', type=str, default='adamw')
        parser.add_argument('--scheduler', type=str, default='plateau')

        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--lr_decay', type=float, default=0.5) # sqrt(2)=0.7 1/2=0.5 1/3=0.33

        parser.add_argument('--num_inputs', type=int, default=4)
        parser.add_argument('--num_classes', type=int, default=60)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size
        self.model = DeepLab_ResNet50(self.hparams.num_inputs,self.hparams.num_classes,pretrained=False,aux_loss=True)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, prefix):
        x,y_true = batch
        output = self(x)
        out_loss = F.cross_entropy(output['out'],y_true,ignore_index=0)
        aux_loss = F.cross_entropy(output['aux'],y_true,ignore_index=0)
        loss = 0.6*out_loss+0.4*aux_loss

        return {'loss': loss, 'out_loss':out_loss.item(),'aux_loss':aux_loss.item()}
    
    def _shared_step_end(self,outputs,prefix):
        avg_loss = torch.stack([o[f'loss'] for o in outputs]).mean()
        out_loss = torch.tensor([o[f'out_loss'] for o in outputs]).mean()
        aux_loss = torch.tensor([o[f'aux_loss'] for o in outputs]).mean()

        tensorboard_logs = {f'{prefix}_loss': avg_loss,f'{prefix}_out_loss':out_loss,f'{prefix}_aux_loss':aux_loss}

        return {f'loss': avg_loss, f'{prefix}_loss':avg_loss, 'log': tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,'train')
    
    def training_epoch_end(self, outputs):
        return self._shared_step_end(outputs,'train')    

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,'val')

    def validation_epoch_end(self, outputs):
        return self._shared_step_end(outputs,'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch,batch_idx,'test')

    def test_epoch_end(self, outputs):
        return self._shared_step_end(outputs,'test')

    def configure_optimizers(self):
        opt = self.hparams.optimizer
        sch = self.hparams.scheduler
        lr = self.hparams.learning_rate if sch!='cosine2' else 1e-6

        if opt == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),lr=lr)
        elif opt == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        
        if sch == 'cosine1':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2, eta_min=1e-6)
        elif sch == 'cosine2':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, T_up=5, eta_max=self.hparams.learning_rate, gamma=self.hparams.lr_decay)
        elif sch == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1)
        elif sch == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=self.hparams.lr_decay)
        else:
            scheduler = None

        if scheduler is None:
            return optimizer
        else:
            return [optimizer],[scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def prepare_data(self):
        dst_cls = PCategoryDataset if self.hparams.person == 'p' else NPCategoryDataset
        self.train_dataset = dst_cls(self.hparams.image_dir,self.hparams.category_dir,'../../data/ids.train.csv')
        self.valid_dataset = dst_cls(self.hparams.image_dir,self.hparams.category_dir,'../../data/ids.valid.csv')
        self.test_dataset = dst_cls(self.hparams.image_dir,self.hparams.category_dir,'../../data/ids.test.csv')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=False, drop_last=True)

def main(hparams):
    seed_everything(hparams.seed)
    model = Model(hparams)
    
    if hparams.version is None:
        hparams.version = f'{hparams.optimizer}_{hparams.scheduler}'
    tb_logger = loggers.TensorBoardLogger(hparams.save_dir,name=hparams.name,version=hparams.version)
    lr_logger = LearningRateLogger()

    if hparams.saver=='best':
        saver = ModelCheckpoint(prefix='best_')
    else:
        saver = ModelCheckpoint(save_top_k=-1,period=1,prefix='period_')
    trainer = Trainer.from_argparse_args(hparams,checkpoint_callback=saver)

    trainer.logger = tb_logger
    trainer.callbacks.append(lr_logger)

    trainer.fit(model)
    #trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--saver', type=str, default='best')
    parser.add_argument('--save_dir', type=str, default='../../logs/')
    parser.add_argument('--name', type=str, default='seg')
    parser.add_argument('--version', type=str, default=None)
    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    main(hparams)
