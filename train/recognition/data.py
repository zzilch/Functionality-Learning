from PIL import Image
import torchvision.transforms.functional as TF

import json
import torch
from torch.utils.data import Dataset

class ClsDataset(Dataset):

    def __init__(self,image_dir,label_path,num_classes=34,transform=None):
        super(ClsDataset, self).__init__()
        self.image_dir = image_dir
        self.labels = json.load(open(label_path))
        self.ids = list(self.labels.keys())
        self.num_classes = num_classes
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,i):
        id = self.ids[i]

        img = Image.open(f'{self.image_dir}/{id}.png')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.zeros(self.num_classes)
        label[self.labels[id]] = 1.0

        return  img,label,id