import torch.nn as nn
from torchvision import models


class SetupModelViT:
    def setup_model(self, device, dropout_prob=0.5):
        vit = models.vit_b_16(pretrained=True)

        for param in vit.parameters():
            param.requires_grad = False

        num_features = vit.heads.in_features
        vit.heads = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1)
        )

        model = vit.to(device)

        return model
