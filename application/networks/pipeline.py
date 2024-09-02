#!/usr/bin/env python
# coding: utf-8

import sys
import os
from torch.utils.data import DataLoader, Subset

from domain.custom_network import NeuralNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain import Hiperparametros
from application.cmd import training
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from application.preprocessing.custom_dataset import CustomDataset


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

batch_size = 32

########################
# type of grid search  #
########################

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


# Using to convert dataset labels names do numbers
class_to_idx = {"psoriasis": 0, "melanome": 1}


def testing_entries(model, dataloader):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_labels_numeric = [class_to_idx[label] for label in y]
        batch_labels_tensor = torch.tensor(batch_labels_numeric).float()
        # print(batch)
        print('shape tensor imagem:',X.shape)
        print('shape tensor y antes tranformacao:',len(y))
        print('shape tensor label:',batch_labels_tensor.shape)
        print(batch_labels_tensor)
        break


training = training.Training()

# Definir as funções de perda e otimizadores
epochs = int(input("Digite o número de épocas: "))


train_transforms = transforms.Compose([
     transforms.RandomRotation(50,fill=1),
     transforms.RandomResizedCrop((224,224)),
     transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),  # Converte para tensor
 ])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Crie um objeto StepLR para ajustar a taxa de aprendizado
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
num_folds = 3
for i, layer_config in enumerate(tests):
    print('rodando camadas: ',layer_config)

    lst = len(os.listdir('application/rag/content/runs'))
    writer = SummaryWriter(f"runs/ml-model-test-{lst}")



    # K-fold training
    for fold in range(num_folds):    
        train_index = torch.load(f'/home/king/Documents/PsoriasisEngineering/application/rag/content/index/train_index_fold{fold}.pt')
        val_index = torch.load(f'/home/king/Documents/PsoriasisEngineering/application/rag/content/index/val_index_fold{fold}.pt')

        custom_dataset = CustomDataset(csv_file='/home/king/Documents/PsoriasisEngineering/image_labels.csv', transform=train_transforms, target_transform=None)

        #conjunto de treino e teste
        train_loader = DataLoader(Subset(custom_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(custom_dataset, val_index), batch_size=batch_size, shuffle=True)

        
        # Setup model
        modelSetup = Hiperparametros.SetupModel()
        model, loss_fn, optimizer, scheduler = modelSetup.setup_model(layer_config,device)

        writer.add_graph(model, torch.randn(batch_size, 3, 224, 224))
        # Registre os detalhes no TensorBoard
        writer.add_text("Model Configuration", str(model))
        writer.add_text("Scheduler", str(scheduler))
        writer.add_text("Optimizer Configuration", str(optimizer))
        writer.add_text("Loss Function", str(loss_fn))


        print(f"Teste - {i+1}")
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            training.train(model, train_loader, writer, loss_fn, optimizer, device, t)
            training.test(model, writer, test_loader, loss_fn, t, device)
            scheduler.step()

        print(f"Done fold - {fold}!")
        writer.flush()
        writer.close()
        torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'runs/ml-model-test-{lst}/model.pt')
