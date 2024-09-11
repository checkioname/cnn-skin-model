#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse

from application.preprocessing.pre_processing import ImageProcessing
from domain.custom_network import NeuralNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain import Hiperparametros
from application.cmd import training
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from application.preprocessing.custom_dataset import CustomDataset


# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", required=True, help="number of epochs on training", type=int)
args = parser.parse_args()


#Pytorch possibilida o usa facil de gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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

tests = [test_6]



def testing_entries(model, dataloader):
    class_to_idx = {"psoriasis": 0, "melanome": 1}
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

dataset = ImageProcessing()

train_set, test_set = dataset.pre_processing()

# Definir as funções de perda e otimizadores
epochs = args.epoch
batch_size = 32
class_to_idx = {"psoriasis": 0, "dermatite": 1}


# Crie um objeto StepLR para ajustar a taxa de aprendizado
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

for i, layer_config in enumerate(tests):
    print('rodando camadas: ',layer_config)

    lst = len(os.listdir('application/rag/content/runs'))
    writer = SummaryWriter(f"runs/ml-model-test-{lst}")

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
        training.train(model, train_loader, writer, loss_fn, optimizer, class_to_idx, device, t)
        training.test(model, writer, test_loader, loss_fn, class_to_idx, t, device)
        scheduler.step()

    print("Done!")
    writer.flush()
    writer.close()
    torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'runs/ml-model-test-{lst}/model.pt')
