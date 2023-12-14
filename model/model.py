import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import torch
import os

class EfficientnetB4(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', num_classes = num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class Resnet50(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.backbone = timm.create_model('resnet50',num_classes = num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_075', num_classes = num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class Densenet169(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('densenet169', num_classes = num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
