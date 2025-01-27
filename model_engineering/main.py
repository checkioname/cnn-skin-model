#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time

import psutil

from application.preprocessing.PreProcessing import ImageProcessing

from application.utils.utils import generate_stratified_dataset
from application.cmd.Training import Training

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


from domain.Vgg16 import SetupModelVgg

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

def monitor_resources():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
    cpu_usage = process.cpu_percent(interval=1)
    print(f"Memory Usage: {mem_usage:.2f} GB, CPU Usage: {cpu_usage}%")
    return mem_usage, cpu_usage

def train_model(epochs, device):
    class_to_idx = {"psoriasis": 0, "dermatite": 1}
    dataset = ImageProcessing()
    train_loader, test_loader = dataset.pre_processing(fold=1, batch_size=32)

    # Definindo as configurações de camadas do modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = SetupModelVgg()
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)

    run_training(model, train_loader, test_loader, epochs, device, optimizer, loss_fn, scheduler)

def run_training(model, train_loader, test_loader, epochs, device, optimizer, loss_fn, scheduler):
    timestamp = time.time()
    save_path = f"runs/ml-model-test-{timestamp}"
    writer = SummaryWriter(save_path)
    # model, loss_fn, optimizer, scheduler = model_setup.setup_model(layer_config, device)
    
    training = Training(train_loader, test_loader, model, writer, None)

    # save_model_stats(model, writer, scheduler, optimizer, loss_fn)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n{'-' * 30}")
        training.train(loss_fn, optimizer, device, epoch)
        training.test(loss_fn, epoch, device)
        monitor_resources()
        scheduler.step()
    save_model_checkpoint(model, optimizer, epoch, save_path)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parse_arguments()
    device = get_device()
    train_model(args.epochs, device)