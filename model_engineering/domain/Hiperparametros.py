import torch
import os
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from domain.NeuralNetwork import NeuralNetwork


class Hiperparameters():
    def __init__(self):
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.001
        
    def setup_model(self, layers_config, device):
        model = NeuralNetwork(layers_config, device, dropout_prob=0.1).to(device)
        
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)        
        # Registre detalhes no TensorBoard
        
        return model, loss_fn, optimizer, scheduler
    

    def register_config(self, model, scheduler, optimizer, batch_size, loss_fn, writer):
        writer.add_graph(model, torch.randn(batch_size, 3, 224, 224))
        writer.add_text("Model Configuration", str(model))
        writer.add_text("Scheduler", str(scheduler))
        writer.add_text("Optimizer Configuration", str(optimizer))
        writer.add_text("Loss Function", str(loss_fn))
        