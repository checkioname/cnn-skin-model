import torch.nn as nn
import torch.optim as optim

from Vgg16 import SetupModelVgg
from ResNet152 import SetupModelResNet152
from Vit import SetupModelViT
from Swim import SetupModelSwin

class SetupModel:
    def __init__(self, model_name, num_classes=1, dropout_prob=0.5):
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def setup_model(self, device):
        """Configura o modelo, a função de perda, o otimizador e o scheduler."""
        model = self._initialize_model(device)
        loss_fn = nn.BCELoss()  # Função de perda para classificação binária
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Otimizador SGD
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  # Scheduler

        return model, loss_fn, optimizer, scheduler

    def _initialize_model(self, device):
        """Delegar a criação do modelo para classes específicas."""
        if self.model_name == "vgg16":
            model_setup = SetupModelVgg(self.num_classes, self.dropout_prob)
            model = model_setup.setup_model(device)
        elif self.model_name == "resnet152":
            model_setup = SetupModelResNet152(self.num_classes, self.dropout_prob)
            model = model_setup.setup_model(device)
        elif self.model_name == "vit":
            model_setup = SetupModelViT(self.num_classes, self.dropout_prob)
            model = model_setup.setup_model(device)
        elif self.model_name == "swin":
            model_setup = SetupModelSwin(self.num_classes, self.dropout_prob)
            model = model_setup.setup_model(device)

        else:
            raise ValueError(f"Modelo '{self.model_name}' não suportado.")

        return model
