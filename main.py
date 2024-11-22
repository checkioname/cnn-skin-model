#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time

from application.callbacks import EarlyStopping
from application.preprocessing.PreProcessing import ImageProcessing
from application.utils.utils import generate_csv_from_dir
from domain.Hiperparametros import Hiperparameters
from application.cmd.Training import Training 

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from application.dataset.CustomDataset import CustomDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{"--"*50} \n UTILIZANDO O DEVICE {device}')
    return device

# Argumentos
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", required=True, help="number of epochs on training", type=int)
    parser.add_argument("-f", "--func", required=False, help="Which function to run: \n 1- generate csv from data \n 2 - generate stratified dataset", type=int)
    return parser.parse_args()

def handle_arguments(args):
    if (args.func == 1):
        path = os.path.dirname(os.path.abspath(__file__))
        root_path = path +'/infrastructure/db'
        print(root_path)
        generate_csv_from_dir(root_path, output_csv='image_labels.csv')
    else:
        print('Not generating csv dataset')

def save_model_stats(model, writer, scheduler, optimizer, loss_fn):
    writer.add_graph(model, torch.randn(32, 3, 224, 224))
    writer.add_text("Model Configuration", str(model))
    writer.add_text("Scheduler", str(scheduler))
    writer.add_text("Optimizer Configuration", str(optimizer))
    writer.add_text("Loss Function", str(loss_fn))

def save_model_checkpoint(model, optimizer, epoch, save_path):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_path, 'model.pt'))
    print(f"Modelo salvo em: {save_path}")


def train_model(epochs, device):
    dataset = ImageProcessing()
    num_folds = 4

    # Definindo as configurações de camadas do modelo
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

    for i, layer_config in enumerate(tests):
        print(f'Iniciando teste {i+1} com camadas: {layer_config}')
        run_training(layer_config, dataset, epochs, device, num_folds)


def run_training(layer_config, dataset, epochs, device, kfolds):
    timestamp = time.time()
    
    for fold in range(kfolds):
        save_path = f"runs/ml-model-test-{timestamp}-fold{fold}"
        train_loader, test_loader = dataset.pre_processing(fold=fold, batch_size=32)
        model, loss_fn, optimizer, scheduler = Hiperparameters().setup_model(layer_config, device)
        writer = SummaryWriter(save_path)
        scheduler = ReduceLROnPlateau(optimizer, 'min') 
        training = Training(train_loader, test_loader, model, writer, callbacks=[])
        save_model_stats(model, writer, scheduler, optimizer, loss_fn)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Fold {fold}\n{'-'*30}")
            training.train(loss_fn, optimizer, device, epoch)
            training.test(loss_fn, epoch, device)
            scheduler.step()

    save_model_checkpoint(model, optimizer, epoch, save_path)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parse_arguments()
    device = get_device()
    handle_arguments(args)
    train_model(args.epochs, device)
