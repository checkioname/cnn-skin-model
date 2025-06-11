import torch.nn as nn
from torchvision import models
from torch import optim
import torch

class SetupModelSwin:
    def setup_model(self, device, dropout_prob=0.5, pretrained_path=None):
        swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)  # 'swin_transformer_tiny' pode ser alterado para outro tamanho

        # Congelar todas as camadas do transformer
        for param in swin.parameters():
            param.requires_grad = False

        num_features = swin.head.in_features
        swin.head = nn.Sequential(
            nn.Linear(num_features, 128),  
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1),  
            nn.Sigmoid()
        )

        model = swin.to(device)

        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict)
                print(f"Pesos pré-treinados carregados de: {pretrained_path}")
            except FileNotFoundError:
                print(f"Erro: Arquivo não encontrado em: {pretrained_path}")
            except RuntimeError as e:
                print(f"Erro ao carregar os pesos: {e}")
                print("Certifique-se de que a estrutura do modelo salvo corresponde à estrutura atual.")

        # Definir loss, optimizer e scheduler
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.head.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return model, loss_fn, optimizer, scheduler
