from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

def DeepLab_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,num_classes=num_classes,aux_loss=aux_loss)
    model.backbone.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

class _ConditionalSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None, num_classes=44, num_cond=256):
        super(_ConditionalSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        self.embd = nn.Embedding(num_classes,num_cond)
        self.fc_out = nn.Linear(num_cond,2048,bias=False)
        #if self.aux_classifier is not None:
        self.fc_aux = nn.Linear(num_cond,1024,bias=False)

    def forward(self, x, c):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        condition = self.embd(c)

        result = OrderedDict()
        x = features["out"]+self.fc_out(condition).view(-1,2048,1,1)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x.squeeze()

        if self.aux_classifier is not None:
            x = features["aux"]+self.fc_aux(condition).view(-1,1024,1,1)
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x.squeeze()

        return result

def Conditional_DeepLab_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = DeepLab_ResNet50(num_inputs, 1, pretrained, aux_loss)
    return _ConditionalSegmentationModel(model.backbone,model.classifier,aux_classifier=model.aux_classifier,num_classes=num_classes)
