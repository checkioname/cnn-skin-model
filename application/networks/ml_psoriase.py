#!/usr/bin/env python
# coding: utf-8



import numpy as np
from random import randint
import torch
import torchvision
import numpy as np
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from PIL import Image
import os
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

#Pytorch possibilida o usa facil de gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(f"Using {device} device")




#transformando as imagens (image augmentation)
#Normalize = normalifor image_file in os.listdir(dir):ze a tensor image with mean and standard deviation.
#Pode transformar em grayscale? (acho que nao kkkkk)
## Pre processamento

transforms = transforms.Compose([
    #transforms.RandomRotation(50,fill=1),
    #transforms.RandomCrop(size=(224, 224)),
    #transforms.RandomResizedCrop((224,224)),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),  # Converte para tensor
])




#Creating a custom dataset (rotulando, etc...)
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.loc[idx, 'img_name']
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path)
        label = str(self.data.loc[idx, 'labels'])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)
        
        return image, label

# Criando KFold cross-validator
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Criando DataLoader para o conjunto de treinamento
custom_dataset = CustomDataset(csv_file='data_labels.csv', img_dir='/content/sample_data/psoriasis/data', transform=transforms, target_transform=None)

labels = custom_dataset.labels
for fold, (train_index, val_index) in enumerate(kf.split(range(len(labels)), labels)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Salvando os índices de treinamento e validação
    torch.save(train_index, os.path.join(f'/content/drive/MyDrive/psoriasis/train_index_fold{fold}.pt'))
    torch.save(val_index, os.path.join(f'/content/drive/MyDrive/psoriasis/val_index_fold{fold}.pt'))

    print(train_index, val_index)



from torchvision.transforms import ToTensor

train_dataset = CustomDataset(csv_file='train_set.csv', img_dir='data/', transform=transforms,target_transform=None )
test_dataset = CustomDataset(csv_file='test_set.csv',img_dir='data/', transform=transforms)
val_dataset = CustomDataset(csv_file='val_set.csv',img_dir='data/', transform=transforms)

batch_size = 32

train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)    




def get_data_dimensions(dataloader):
    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f'Length of X: {len(X)}')
        print(f"Length/type of y: {len(y)} {type(y)}")
        print(f"Size of dataloader: {len(dataloader)}")
        break


def plot_data(dataloader,img_num):
    train_features, train_labels = next(iter(dataloader))
    for i in range(img_num):
        i = randint(0,len(train_features)-1)
        img = train_features[i].squeeze().permute(1, 2, 0)
        label = train_labels[i]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")




# definindo o modelo
test_1 = [224,128,64,32,16]
test_2 = [128,64,32,16,8]
test_3 = [64,32,16,8,4]
test_4 = [16,32,64,128,224]
test_5 = [64,32,16,32,8]
test_6 = [32, 64, 128, 128, 256]   
test_7 = [16, 32, 64, 64, 128]     
test_8 = [64, 64, 128, 128, 256]   
test_9 = [32, 64, 64, 128, 128]    
test_10 = [32, 64, 64, 64, 128]    
test_11 = [128, 128, 256, 256, 512] 

tests = [test_6,
         test_7,
         test_8,
         test_9,
         test_10,
         test_11]         





# Defina uma semente para a inicialização
seed = 422
torch.manual_seed(seed)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# O resize da nossa imagem foi para 224x224 e ela é colorida ( 3 channels)
class NeuralNetwork(nn.Module):
    def __init__(self, test, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        
        # Tamanho da imagem de entrada
        self.input_size = (3, 224, 224)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, test[0], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(test[0], test[1], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(test[1], test[2], kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(test[2], test[3], kernel_size=3, padding=0),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(test[3], test[4], kernel_size=3, padding=0),
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


#batch_labels_numeric = [class_to_idx[label] for label in batch_labels]
class_to_idx = {"psoriasis": 0, "melanome": 1}


def testing_entries(model,dataloader):
    size = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_labels_numeric = [class_to_idx[label] for label in y]
        batch_labels_tensor = torch.tensor(batch_labels_numeric).float()
        #print(batch)
        print('shape tensor imagem:',X.shape)
        print('shape tensor y antes tranformacao:',len(y))
        print('shape tensor label:',batch_labels_tensor.shape)
        print(batch_labels_tensor)
        break



# Função de treinamento
def train(dataloader, model, loss_fn, optimizer, class_to_idx, writer, device,epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_labels_numeric = [class_to_idx[label] for label in y]
        y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        pred = pred.squeeze(1)

        loss = loss_fn(pred, y)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")

# Função de teste
def test(data, model, loss_fn, class_to_idx, epoch, writer, device):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            batch_labels_numeric = [class_to_idx[label] for label in y]
            y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum().item()
    
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)
    print(f"Test Error: Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")

# Exemplo de uso
class_to_idx = {"psoriasis": 1, "melanome": 0}

# Definir as funções de perda e otimizadores
epochs = int(input("Digite o número de épocas: "))


# Crie um objeto StepLR para ajustar a taxa de aprendizado
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

for i, obj in enumerate(tests):
    print('rodando camadas: ',obj)

    lst = len(os.listdir('runs/'))
    writer = SummaryWriter(f"runs/ml-model-test-{lst}")

    model = NeuralNetwork(obj, dropout_prob=0.1).to(device)
    print(obj[0])
    momentum = 0.9
    weight_decay = 0.001
    learning_rate = 0.01
    
    
    loss_fn = nn.BCELoss()
    
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)



    writer.add_graph(model, torch.randn(batch_size, 3, 224, 224))
    # Registre os detalhes no TensorBoard
    writer.add_text("Model Configuration", str(model))
    writer.add_text("Scheduler", str(scheduler))
    writer.add_text("Optimizer Configuration", str(optimizer))
    writer.add_text("Loss Function", str(loss_fn))

    print(f"Teste - {i+1}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, class_to_idx, writer, device, t)
        test(test_loader, model, loss_fn, class_to_idx, t, writer, device)
        scheduler.step()

    

        # Registre outros detalhes relevantes, como hiperparâmetros
        writer.add_hparams({"learning_rate": learning_rate, "batch_size": batch_size, "momentum": momentum,"weight decay": weight_decay}, {})

    print("Done!")
    writer.flush()
    writer.close()
    torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'runs/ml-model-test-{lst}/model.pt')
