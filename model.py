import torch
import torch.nn as nn
import torchvision.models as models

class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super(CSRNet, self).__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if load_pretrained else None
        )

        # Extract convolutional layers up to layer4 (excluding avgpool and fc)
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

        # CSRNet-style regression head (tuned with dropout & batchnorm)
        self.regressor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 1, kernel_size=1)  # Output = 1-channel density map
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

class CountRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)
