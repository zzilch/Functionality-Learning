import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class ResNet50(nn.Module):

    def __init__(self, num_inputs, num_classes, pretrained):
        super(ResNet50, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_classes = num_classes

        model = models.resnet50(pretrained)
        conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3,bias=False)

        # feature encoding
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1',conv1),
            ('bn1',model.bn1),
            ('relu',model.relu),
            ('maxpool',model.maxpool),
            ('layer1',model.layer1),
            ('layer2',model.layer2),
            ('layer3',model.layer3),
            ('layer4',model.layer4),
        ]))

        # classifier: feature map -> class response map
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x