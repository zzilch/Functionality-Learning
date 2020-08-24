from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader,random_split
from lr_scheduler import CosineAnnealingWarmUpRestarts
from torchvision import transforms

from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

from model import ResNet50
from prm import peak_response_mapping
from data import ClsDataset

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateLogger,ModelCheckpoint
from pytorch_lightning import Trainer,loggers,seed_everything


class Model(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--pretrained_backbone', type=str, default=None)
        parser.add_argument('--filter_type', type=str, default='median')

        parser.add_argument('--image_dir', type=str, default='../../data/p/rgbd')

        parser.add_argument('--batch_size', type=int, default=30)

        parser.add_argument('--optimizer', type=str, default='adamw')
        parser.add_argument('--scheduler', type=str, default='plateau')

        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--lr_decay', type=float, default=0.5) # sqrt(2)=0.7 1/2=0.5 1/3=0.33

        parser.add_argument('--num_inputs', type=int, default=4)
        parser.add_argument('--num_classes', type=int, default=34)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size
        self.model = peak_response_mapping(ResNet50(self.hparams.num_inputs,self.hparams.num_classes,False),filter_type='mean' if self.hparams.filter_type is None else self.hparams.filter_type,win_size=3)
        if self.hparams.pretrained_backbone is not None:
            self.model[0].backbone.load_state_dict(torch.load(self.hparams.pretrained_backbone))

    def forward(self, x, class_threshold=0, peak_threshold=6):
        return self.model(x, class_threshold=class_threshold, peak_threshold=peak_threshold)

    def _shared_step(self, batch, batch_idx, prefix):
        x,y_true,_ = batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output,y_true)

        y_pred = (output.detach().sigmoid()>0.5).cpu()
        y_true = y_true.cpu()

        return {'loss': loss,'y_pred':y_pred,'y_true':y_true}
    
    def _shared_step_end(self,outputs,prefix):
        avg_loss = torch.stack([o[f'loss'] for o in outputs]).mean()
        y_true = torch.cat([o['y_true'] for o in outputs])
        y_pred = torch.cat([o['y_pred'] for o in outputs])
        report = classification_report(y_true,y_pred,output_dict=True)
        
        micro_f1 = report['micro avg']['f1-score']
        macro_f1 = report['macro avg']['f1-score']

        tensorboard_logs = {f'{prefix}_loss': avg_loss,f'{prefix}_micro_f1':micro_f1,f'{prefix}_macro_f1':macro_f1}
        return {f'loss': avg_loss
        , f'{prefix}_loss':avg_loss, 'log': tensorboard_logs}

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

        if self.hparams.pretrained_backbone is not None:
            params = [
                {'params': self.model[0].backbone.parameters(),'lr':self.hparams.learning_rate*0.1},
                {'params': self.model[0].classifier.parameters(), 'lr': self.hparams.learning_rate}
            ]
        else:
            params = self.parameters()

        if opt == 'sgd':
            optimizer = torch.optim.SGD(params,lr=lr)
        elif opt == 'adamw':
            optimizer = torch.optim.AdamW(params,lr=lr)
        else:
            optimizer = torch.optim.Adam(params,lr=lr)
        
        if sch == 'cosine1':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 40, T_mult=1, eta_min=1e-6)
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
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor()
        ])

        self.train_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels.json',self.hparams.num_classes,train_transform)
        self.valid_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels_valid.json',self.hparams.num_classes,test_transform)
        self.test_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels_test.json',self.hparams.num_classes,test_transform)

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

    if hparams.saver=='period':
        saver = ModelCheckpoint(save_top_k=-1,period=1,prefix='period_')
    else:
        saver = ModelCheckpoint(prefix='best_')
    trainer = Trainer.from_argparse_args(hparams,checkpoint_callback=saver)

    trainer.logger = tb_logger
    trainer.callbacks.append(lr_logger)

    trainer.fit(model)
    #trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--saver', type=str, default='best')
    parser.add_argument('--save_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='cls')
    parser.add_argument('--version', type=str, default=None)
    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    main(hparams)
