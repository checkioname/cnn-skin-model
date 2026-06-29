import torch.nn as nn
import torch.optim as optim

from domain.Vgg16 import SetupModelVgg
from domain.ResNet152 import SetupModelResNet152
from domain.Vit import SetupModelViT
from domain.Swim import SetupModelSwin


SCHEDULERS = {
    'plateau': lambda opt, **kw: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, **kw),
    'step':    lambda opt, **kw: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1, **kw),
    'cosine':  lambda opt, **kw: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, **kw),
}

OPTIMIZERS = {
    'sgd':  lambda params, **kw: optim.SGD(params, **kw),
    'adam': lambda params, **kw: optim.Adam(params, **kw),
}


class SetupModel:
    def __init__(self, model_name, scheduler='plateau', num_classes=1, dropout_prob=0.5):
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.scheduler_name = scheduler

    def setup_model(self, device, lr=0.01, momentum=0.9, weight_decay=0.0, optimizer_name='sgd', scheduler_name=None, epochs=50):
        scheduler_name = scheduler_name or self.scheduler_name

        model = self._initialize_model(device)
        loss_fn = nn.BCELoss()

        if optimizer_name in OPTIMIZERS:
            opt_fn = OPTIMIZERS[optimizer_name]
            if optimizer_name == 'sgd':
                optimizer = opt_fn(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            else:
                optimizer = opt_fn(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        if scheduler_name in SCHEDULERS:
            sched_fn = SCHEDULERS[scheduler_name]
            if scheduler_name == 'plateau':
                scheduler = sched_fn(optimizer)
            elif scheduler_name == 'cosine':
                scheduler = sched_fn(optimizer, T_max=epochs)
            else:
                scheduler = sched_fn(optimizer)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        return model, loss_fn, optimizer, scheduler

    def _initialize_model(self, device):
        if self.model_name == "vgg16":
            model_setup = SetupModelVgg()
            model = model_setup.setup_model(device, self.dropout_prob)
        elif self.model_name == "resnet152":
            model_setup = SetupModelResNet152()
            model = model_setup.setup_model(device, self.dropout_prob)
        elif self.model_name == "vit":
            model_setup = SetupModelViT()
            model = model_setup.setup_model(device, self.dropout_prob)
        elif self.model_name == "swin":
            model_setup = SetupModelSwin()
            model = model_setup.setup_model(device, self.dropout_prob)
        elif self.model_name == "custom":
            raise NotImplementedError("Modelo 'custom' requer implementação manual")
        else:
            raise ValueError(f"Modelo '{self.model_name}' não suportado.")

        return model
