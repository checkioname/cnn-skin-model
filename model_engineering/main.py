#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time
import psutil
import numpy as np
import random
import cv2
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from application.preprocessing.PreProcessing import ImageProcessing
from application.cmd.Training import Training
from domain.SetupModel import SetupModel

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch.backends.cudnn.benchmark = True

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{"--"*50} \n UTILIZANDO O DEVICE {device}')
    return device

# Argumentos
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", required=True, help="number of epochs on training", type=int)
    parser.add_argument("-m", "--model", required=True, help="Model architecture: vgg16, resnet152, vit, swin", type=str)
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

def _prepare_cam_sample(test_loader):
    from application.preprocessing.PreProcessing import OpenCVPreprocessing
    from torchvision import transforms

    subset = test_loader.dataset
    idx = subset.indices[0]
    custom_dataset = subset.dataset
    img_path = custom_dataset.data.iloc[idx]['img_name']

    cv_img = cv2.imread(img_path)
    if cv_img is None:
        return None
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    crop = min(h, w)
    y, x = (h - crop) // 2, (w - crop) // 2
    cv_img = cv_img[y:y+crop, x:x+crop]
    cv_img = cv2.resize(cv_img, (512, 512))
    raw_np = cv_img.astype(np.float32) / 255.0

    fixed_transform = transforms.Compose([
        OpenCVPreprocessing(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(cv_img)
    input_tensor = fixed_transform(pil_img).unsqueeze(0)

    return raw_np, input_tensor


def train_model(epochs, model_name):
    class_to_idx = {"psoriasis": 0, "dermatite": 1}
    dataset = ImageProcessing()
    train_loader, test_loader = dataset.pre_processing(fold=1, batch_size=32)

    cam_sample = _prepare_cam_sample(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = SetupModel(model_name)
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)

    run_training(model, train_loader, test_loader, epochs, device, optimizer, loss_fn, scheduler,
                 model_name=model_name, cam_sample=cam_sample)

def run_training(model, train_loader, test_loader, epochs, device, optimizer, loss_fn, scheduler,
                 model_name=None, cam_sample=None):
    timestamp = time.time()
    save_path = f"runs/ml-model-test-{timestamp}"
    writer = SummaryWriter(save_path)

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    writer.add_graph(model, dummy_input)
    
    training = Training(train_loader, test_loader, model, writer, None,
                        model_name=model_name, cam_sample=cam_sample)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n{'-' * 30}")
        training.train(loss_fn, optimizer, device, epoch)
        test_loss = training.test(loss_fn, epoch, device)
        training.visualize_gradcam(epoch, device)
        monitor_resources()
        scheduler.step(test_loss)
    save_model_checkpoint(model, optimizer, epoch, save_path)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    seed = 42

    torch.manual_seed(seed)

        # Se você estiver usando a GPU, você também deve definir a semente para CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Para múltiplas GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


    args = parse_arguments()
    train_model(args.epochs, args.model)