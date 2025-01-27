import torch.nn as nn
from torchvision import models
from torch import optim

class SetupModelVgg:
    def setup_model(self, device, dropout_prob=0.5):
        # Carregar o modelo VGG16 pré-treinado
        vgg = models.vgg16(pretrained=True)

        # Congelar todas as camadas convolucionais
        for param in vgg.features.parameters():
            param.requires_grad = False

        # Substituir a última camada fully connected (FC) para se adequar ao seu problema
        num_features = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 128),  # Camada intermediária
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1),  # Saída para classificação binária
            nn.Sigmoid()
        )

        # Mover o modelo para o device (GPU/CPU)
        model = vgg.to(device)

        # Definir loss, optimizer e scheduler
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return model, loss_fn, optimizer, scheduler