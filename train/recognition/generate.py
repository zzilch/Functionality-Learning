from argparse import ArgumentParser

import os
import skimage.io as io
from skimage import img_as_ubyte

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader,random_split
from lr_scheduler import CosineAnnealingWarmUpRestarts
from torchvision import transforms

from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore") 

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
        self.model = peak_response_mapping(ResNet50(self.hparams.num_inputs,self.hparams.num_classes,False),filter_type='median' if self.hparams.filter_type is None else self.hparams.filter_type,win_size=5)
        if self.hparams.pretrained_backbone is not None:
            self.model[0].backbone.load_state_dict(torch.load(self.hparams.pretrained_backbone))

    def forward(self, x, class_threshold=6, peak_threshold=6):
        return self.model(x, class_threshold=class_threshold, peak_threshold=peak_threshold)
        
    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            self.model.inference()
            x,y_true,id = batch
            output = self(x,class_threshold=6, peak_threshold=6)
            id = id[0]
            save_dir = f'../../output/prm/{id}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if output is not None:
                aggregation, class_response_maps, valid_peak_list, peak_response_maps = output
                y_peak = torch.unique(valid_peak_list[:,1])
                for y in y_peak:
                    peaks = [valid_peak_list[:,1]==y]
                    prm = peak_response_maps[peaks].sum(0)
                    prm = prm/prm.max()
                    save_path = f'{save_dir}/{id}.{y}.png'
                    io.imsave(save_path,prm.cpu().numpy())

            self.model.zero_grad()
        return None

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

        self.train_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels.json',self.hparams.num_classes,test_transform)
        self.valid_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels_valid.json',self.hparams.num_classes,test_transform)
        self.test_dataset = ClsDataset(self.hparams.image_dir,'../../data/labels.json',self.hparams.num_classes,test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=False, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=False, drop_last=False)

def main(hparams):
    seed_everything(hparams.seed)
    model = Model.load_from_checkpoint(hparams.ckpt)
    model.model.inference()
    model.hparams.batch_size = 1
    
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

    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--saver', type=str, default='best')
    parser.add_argument('--save_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='cls')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--ckpt', type=str, required=True)
    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    main(hparams)
