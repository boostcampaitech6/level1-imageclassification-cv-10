import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class EfficientNetV2m(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
class ConvNextB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)