#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time

from application.preprocessing.pre_processing import ImageProcessing
from application.utils.utils import generate_csv_from_dir
from domain.custom_network import NeuralNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain import Hiperparametros
from application.cmd import training
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from application.preprocessing.custom_dataset import CustomDataset

#Pytorch possibilida o usa facil de gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("--"*50)
print(f"UTILIZANDO O DEVICE - {device}")

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", required=True, help="number of epochs on training", type=int)
parser.add_argument("-f", "--func", required=False, help="Which function to run: \n 1- generate csv from data \n 2 - generate stratified dataset", type=int)


args = parser.parse_args()

if (args.func == 1):
    path = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
    print(path)
    root_path = os.path.join('infraestructure/db')
    generate_csv_from_dir(root_path, output_csv='image_labels.csv')
else:
    print('Not generating csv dataset')




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


batch_size = 32
epochs = args.epochs
class_to_idx = {"psoriasis": 0, "dermatite": 1}


dataset = ImageProcessing()
train_loader, test_loader = dataset.pre_processing(4, batch_size)


# Crie um objeto StepLR para ajustar a taxa de aprendizado
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
for i, layer_config in enumerate(tests):
    print('rodando camadas: ',layer_config)
    
    time = time.time()
    lst = len(os.listdir('application/rag/content/runs'))
    save_path = f"runs/ml-model-test-{time}"
    writer = SummaryWriter(save_path)
    modelSetup = Hiperparametros.SetupModel()
    model, loss_fn, optimizer, scheduler = modelSetup.setup_model(layer_config,device)
    training = training.Training(train_loader,model, writer)

    writer.add_graph(model, torch.randn(batch_size, 3, 224, 224))
    # Registre os detalhes no TensorBoard
    writer.add_text("Model Configuration", str(model))
    writer.add_text("Scheduler", str(scheduler))
    writer.add_text("Optimizer Configuration", str(optimizer))
    writer.add_text("Loss Function", str(loss_fn))

    print(f"Teste - {i+1}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training.train(loss_fn, optimizer, class_to_idx, device, t)
        training.test(loss_fn, class_to_idx, t, device)
        scheduler.step()

    print("Done!")
    writer.flush()
    writer.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path + '/' + 'model.pt')
