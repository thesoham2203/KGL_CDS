import torch
import torch.nn as nn
import torchvision.models as models

class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super(CSRNet, self).__init__()
        # Use ResNet-18 pretrained on ImageNet
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if load_pretrained else None
        )

        # Extract layers up to layer4
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # CSRNet regressor to produce a density map
        self.regressor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        # Output is a density map (B×1×H×W)
        return self.regressor(x)
