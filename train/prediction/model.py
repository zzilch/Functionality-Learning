from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

def DeepLab_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,num_classes=num_classes,aux_loss=aux_loss)
    model.backbone.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model

def Conditional_DeepLab_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = DeepLab_ResNet50(num_inputs, 1, pretrained, aux_loss)
    return _ConditionalSegmentationModel(model.backbone,model.classifier,aux_classifier=model.aux_classifier,num_classes=num_classes)

def FCN_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = models.segmentation.fcn_resnet50(pretrained=pretrained,num_classes=num_classes,aux_loss=aux_loss)
    model.backbone.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3,bias=False)
    return model   

def Conditional_FCN_ResNet50(num_inputs, num_classes, pretrained, aux_loss):
    model = FCN_ResNet50(num_inputs, 1, pretrained, aux_loss)
    return _ConditionalSegmentationModel(model.backbone,model.classifier,aux_classifier=model.aux_classifier,num_classes=num_classes)
