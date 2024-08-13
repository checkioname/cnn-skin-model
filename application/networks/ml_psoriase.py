#!/usr/bin/env python
# coding: utf-8


from random import randint
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
#from torch.optim.lr_scheduler import ReduceLROnPlateau

from application.networks.custom_network import NeuralNetwork
from application.utils.training import Training

#Pytorch possibilida o usa facil de gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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


training = Training()

training.test()

training.train()


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
