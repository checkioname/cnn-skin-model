import torch.nn as nn
from torchvision import models


class SetupModelResNet152:
    def setup_model(self, device, dropout_prob=0.5):
        resnet152 = models.resnet152(pretrained=True)

        for param in resnet152.parameters():
            param.requires_grad = False

        num_features = resnet152.fc.in_features
        resnet152.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1)
        )

        model = resnet152.to(device)

        return model
