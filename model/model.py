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

class EfficientnetV2s(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnetv2_rw_s', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientnetV2m(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnetv2_rw_m', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class MultiHeadEfficientnetB4(EfficientnetB4):
    def __init__(self, num_classes=8):
        super().__init__(num_classes)
        self.num_classes = num_classes
        
        backbone = timm.create_model('efficientnet_b4', pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        
        self.age_fc = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
        )
        self.mask_fc = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
        )
        self.gender_fc = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )
        
    def forward(self, x):
        x = self.model(x)
        
        age = self.age_fc(x)
        mask = self.mask_fc(x)
        gender = self.gender_fc(x)
                
        return age, mask, gender