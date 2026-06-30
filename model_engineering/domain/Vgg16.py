import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


class SetupModelVgg:
    def setup_model(self, device, dropout_prob=0.5):
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)

        for param in vgg.features.parameters():
            param.requires_grad = False

        num_features = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1)
        )

        model = vgg.to(device)

        return model
