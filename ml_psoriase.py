#!/usr/bin/env python
# coding: utf-8

# In[3]:


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



# In[4]:


#Pytorch possibilida o usa facil de gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(f"Using {device} device")


# In[5]:


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


# In[6]:


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


# # data = pd.read_csv('data_labels.csv')
# image_name = data.loc[10, 'img_name']
# label = data.loc[10,'labels']
# print(image_name)
# print('classe: ',label)
# Image.open(os.path.join('data/',image_name))

# In[7]:


#train_dataset = datasets.ImageFolder(root='data/train/',transform=transform)
from torchvision.transforms import ToTensor

train_dataset = CustomDataset(csv_file='train_set.csv', img_dir='data/', transform=transforms,target_transform=None )
test_dataset = CustomDataset(csv_file='test_set.csv',img_dir='data/', transform=transforms)
val_dataset = CustomDataset(csv_file='val_set.csv',img_dir='data/', transform=transforms)


train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)    


# In[8]:


def get_data_dimensions(dataloader):
    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f'Length of X: {len(X)}')
        print(f"Length/type of y: {len(y)} {type(y)}")
        print(f"Size of dataloader: {len(dataloader)}")
        break


# In[10]:


# Display image and label.


def plot_data(dataloader,img_num):
    train_features, train_labels = next(iter(dataloader))
    for i in range(img_num):
        i = randint(0,len(train_features)-1)
        img = train_features[i].squeeze().permute(1, 2, 0)
        label = train_labels[i]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")


# In[13]:


# O resize da nossa imagem foi para 224x224 e ela é colorida ( 3 channels)
input_size = (3, 224, 224)

# tamanho correto do tensor de entrada após as 


# In[22]:


# definindo o modelo
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

test_1 = [224,128,64,32,16]
test_2 = [128,64,32,16,8]
test_3 = [64,32,16,8,4]
test_4 = [16,32,64,128,224]
test_5 = [64,32,16,32,8]

tests = [test_1,test_2,test_3,test_4,test_5]

class NeuralNetwork(nn.Module):
    def __init__(self,test):
        super().__init__()
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, test[0], kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(test[0], test[1], kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(test[1], test[2], kernel_size=3, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(test[2], test[3], kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(test[3], test[4], kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.flatten = Flatten()
        flattened_size = self._get_flattened_size(input_size)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            output = self.conv_layers(dummy_input)
            flattened_output = self.flatten(output)
            return flattened_output.size(1)

# In[16]:


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


# In[17]:




def train(dataloader, model, loss_fn, optimizer, class_to_idx):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_labels_numeric = [class_to_idx[label] for label in y]
        y = torch.tensor(batch_labels_numeric, dtype=torch.float32)
        X, y = X.to(device), y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Compute prediction
        pred = model(X)
        pred = pred.squeeze(1)

        # Compute loss and backpropagation
        loss = loss_fn(pred, y)
        writer.add_scalar("Loss/train",loss, batch)
        loss.backward()
        #Performs a single optimization step
        optimizer.step()

        if batch % 100 == 0:
            current = batch * 64
            print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")


# In[12]:


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# In[18]:


def test(data, model, loss_fn, class_to_idx):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            batch_labels_numeric = [class_to_idx[label] for label in y]
            y = torch.tensor(batch_labels_numeric, dtype=torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")

# Exemplo de uso
class_to_idx = {"psoriasis": 1, "melanome": 0}


# In[ ]:


# Definir as funções de perda e otimizadores
##input (Tensor) – Tensor of arbitrary shape as probabilities.
##target (Tensor) – Tensor of the same shape as input with values between 0 and 1.
# testar esse learning rate lr=0.01

for i,obj in enumerate(tests):
    lst = len(os.listdir('runs/'))
    writer = SummaryWriter(f"runs/ml-model-test-{lst}")
    model = NeuralNetwork(tests[i]).to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

    print(f"Teste - {i+1}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, class_to_idx)
        test(test_loader, model, loss_fn, class_to_idx)
    print("Done!")
    writer.flush()
    writer.close()