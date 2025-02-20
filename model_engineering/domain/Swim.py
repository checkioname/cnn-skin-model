import torch.nn as nn
from torchvision import models
from torch import optim

class SetupModelSwin:
    def setup_model(self, device, dropout_prob=0.5):
        # Carregar o modelo SWIN Transformer pré-treinado
        swin = models.swin_transformer_tiny(pretrained=True)  # 'swin_transformer_tiny' pode ser alterado para outro tamanho

        # Congelar todas as camadas do transformer
        for param in swin.parameters():
            param.requires_grad = False

        # Substituir a última camada fully connected (FC) para se adequar ao seu problema
        num_features = swin.head.in_features
        swin.head = nn.Sequential(
            nn.Linear(num_features, 128),  # Camada intermediária
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1),  # Saída para classificação binária
            nn.Sigmoid()
        )

        # Mover o modelo para o device (GPU/CPU)
        model = swin.to(device)

        # Definir loss, optimizer e scheduler
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.head.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return model, loss_fn, optimizer, scheduler
