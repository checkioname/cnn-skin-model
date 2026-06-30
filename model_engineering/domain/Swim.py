import torch.nn as nn
from torchvision import models
import torch


class SetupModelSwin:
    def setup_model(self, device, dropout_prob=0.5, pretrained_path=None):
        swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)

        for param in swin.parameters():
            param.requires_grad = False

        num_features = swin.head.in_features
        swin.head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1)
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

        return model
