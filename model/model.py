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
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b4', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x


