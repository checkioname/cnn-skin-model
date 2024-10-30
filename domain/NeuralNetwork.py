import torch
import torch.nn as nn

# Defina uma semente para a inicialização
seed = 422
torch.manual_seed(seed)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# O resize da nossa imagem foi para 224x224 e ela é colorida ( 3 channels)
class NeuralNetwork(nn.Module):
    def __init__(self, layers_config, device,  dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        self.device = device
        
        # Tamanho da imagem de entrada
        self.input_size = (3, 224, 224)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, layers_config[0], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers_config[0], layers_config[1], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers_config[1], layers_config[2], kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers_config[2], layers_config[3], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers_config[3], layers_config[4], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.dropout = nn.Dropout(p=dropout_prob)

        # Calcule o tamanho da camada linear de acordo com o tamanho de entrada
        self.flatten = Flatten()
        flattened_size = self._get_flattened_size(self.input_size)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, int(flattened_size/2)),
            nn.SELU(),
            nn.Linear(int(flattened_size/2), 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.conv_layers(x)
        x = self.dropout(x)
        x = self.fc_layers(x)
        return x

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            output = self.conv_layers(dummy_input)
            flattened_output = self.flatten(output)
            return flattened_output.size(1)
        
    