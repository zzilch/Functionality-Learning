from PIL import Image
# from skimage import io
import torchvision.transforms.functional as TF

import json
import torch
import numpy as np
from torch.utils.data import Dataset

class PCategoryDataset(Dataset):

    def __init__(self,image_dir,category_dir,ids_path):
        super(PCategoryDataset, self).__init__()
        self.ids = open(ids_path).read().split('\n')[1:]
        self.image_dir = image_dir
        self.category_dir = category_dir
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,i):
        id = self.ids[i]

        img = Image.open(f'{self.image_dir}/{id}.png')
        img = TF.to_tensor(TF.resize(img,224,Image.BILINEAR))
        cat = np.array(TF.resize(Image.open(f'{self.category_dir}/{id}.category.p.encoded.png'),224,Image.NEAREST))
        cat = torch.tensor(cat).long()
        return  img,cat

class NPCategoryDataset(Dataset):

    def __init__(self,image_dir,category_dir,ids_path):
        super(NPCategoryDataset, self).__init__()
        self.ids = open(ids_path).read().split('\n')[1:]
        self.image_dir = image_dir
        self.category_dir = category_dir
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,i):
        id = self.ids[i]

        img = Image.open(f'{self.image_dir}/{id}.png')
        img = TF.to_tensor(TF.resize(img,224,Image.BILINEAR))
        cat = np.array(TF.resize(Image.open(f'{self.category_dir}/{id}.category.np.encoded.png'),224,Image.NEAREST))
        cat = torch.tensor(cat).long()
        return  img,cat

class ActDataset(Dataset):
    def __init__(self,image_dir,act_dir,acts_path):
        super(ActDataset, self).__init__()
        self.acts = open(acts_path).read().split('\n')
        self.image_dir = image_dir
        self.act_dir = act_dir
    
    def __len__(self):
        return len(self.acts)
    
    def __getitem__(self,i):
        rid,pid,_ = self.acts[i].split(' ')

        img = Image.open(f'{self.image_dir}/{rid}.png')
        img = TF.to_tensor(TF.resize(img,224,Image.BILINEAR))

        act = np.array(Image.open(f'{self.act_dir}/{rid}/{rid}.{pid}.png'))/255.0
        act_mean = 1.0*(act>act[act>0].mean())
        act = torch.tensor(act)
        act_mean = torch.tensor(act_mean)

        pid = torch.tensor(int(pid)).long()

        return  img,pid,act,act_mean