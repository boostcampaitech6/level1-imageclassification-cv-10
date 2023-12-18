import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

import timm

import os

    
class EfficientnetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientnetB2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientnetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientnetB5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class ConvNextB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
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
        self.backbone = timm.create_model('mobilenetv2_100', num_classes = num_classes, pretrained=True)
        
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

class Squeezenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace = False),
            nn.Conv2d(512, num_classes, kernel_size = (1, 1), stride = (1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.model.classifier = self.classifier
    
    def forward(self, x):
        x = self.model(x)
        return x

class ShufflenetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
        self.classifier = nn.Linear(in_features = 1024, out_features = num_classes, bias = True)
        self.model.fc = self.classifier

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientnetV2s(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnetv2_rw_s', num_classes = self.num_classes, pretrained = True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class EfficientNetV2m(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

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