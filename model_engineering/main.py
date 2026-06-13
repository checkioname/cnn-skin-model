#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time
import json
import psutil
import numpy as np
import random
import cv2
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from application.preprocessing.PreProcessing import ImageProcessing, OpenCVPreprocessing
from application.cmd.Training import Training
from domain.SetupModel import SetupModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", required=True, type=int)
    parser.add_argument("-m", "--model", required=True, choices=['vgg16', 'resnet152', 'vit', 'swin'])
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--dp", action='store_true', help="Enable DataParallel")
    return parser.parse_args()


def setup_ddp():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Rank {local_rank} of {dist.get_world_size()}")
        return local_rank, dist.get_world_size()
    return 0, 1


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def save_model_checkpoint(model, optimizer, epoch, save_path):
    raw_model = model.module if hasattr(model, 'module') else model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_path, 'model.pt'))
    print(f"Modelo salvo em: {save_path}")


def save_training_metrics(metrics, save_path):
    path = os.path.join(save_path, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {path}")


def monitor_resources():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)
    cpu_usage = process.cpu_percent(interval=1)
    print(f"Memory: {mem_usage:.2f} GB, CPU: {cpu_usage}%")
    return mem_usage, cpu_usage


def _prepare_cam_sample(test_loader):
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


def train_model(epochs, model_name, batch_size, num_workers, use_dp):
    class_to_idx = {"psoriasis": 0, "dermatite": 1}
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = ImageProcessing()
    train_loader, test_loader = dataset.pre_processing(
        fold=1, batch_size=batch_size, num_workers=num_workers,
        rank=local_rank, world_size=world_size
    )

    cam_sample = _prepare_cam_sample(test_loader) if is_main_process() else None

    setup = SetupModel(model_name)
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)

    if is_main_process():
        print(f"\nGPUs disponiveis: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    parallelism = "DDP" if world_size > 1 else ("DataParallel" if use_dp else "none")
    if use_dp and world_size == 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        parallelism = "DataParallel"
        print(f"[DataParallel] usando {torch.cuda.device_count()} GPUs")
    elif world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        print(f"[DDP] Rank {local_rank} pronto")

    run_training(model, train_loader, test_loader, epochs, device, optimizer,
                 loss_fn, scheduler, model_name=model_name, cam_sample=cam_sample,
                 rank=local_rank, world_size=world_size, parallelism=parallelism)


def run_training(model, train_loader, test_loader, epochs, device, optimizer,
                 loss_fn, scheduler, model_name=None, cam_sample=None,
                 rank=0, world_size=1, parallelism="none"):
    cam_interval = max(1, epochs // 5)
    timestamp = time.time()
    save_path = f"runs/ml-model-test-{timestamp}"

    writer = SummaryWriter(save_path) if rank == 0 else None
    start_time = time.time()

    if writer:
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        writer.add_graph(model, dummy_input)
        writer.add_text("Config/parallelism", parallelism)
        writer.add_text("Config/world_size", str(world_size))
        writer.add_text("Config/model", model_name)

    training = Training(train_loader, test_loader, model, writer, None,
                        model_name=model_name, cam_sample=cam_sample,
                        cam_interval=cam_interval, rank=rank, world_size=world_size)

    for epoch in range(epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}\n{'-' * 30}")
        training.train(loss_fn, optimizer, device, epoch)
        test_loss = training.test(loss_fn, epoch, device)
        training.visualize_gradcam(epoch, device)
        scheduler.step(test_loss)

    total_time = time.time() - start_time

    if rank == 0:
        save_model_checkpoint(model, optimizer, epoch, save_path)

        total_images = len(train_loader.dataset) * epochs
        avg_img_per_sec = total_images / total_time if total_time > 0 else 0
        avg_epoch_time = sum(training.epoch_times) / len(training.epoch_times) if training.epoch_times else 0

        metrics = {
            "model": model_name,
            "parallelism": parallelism,
            "epochs": epochs,
            "total_time_seconds": round(total_time, 2),
            "avg_epoch_time_seconds": round(avg_epoch_time, 2),
            "total_images_processed": total_images,
            "avg_throughput_img_per_sec": round(avg_img_per_sec, 2),
            "gpu_count": world_size if world_size > 1 else torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }

        if training.throughputs:
            metrics["avg_epoch_throughput"] = round(np.mean(training.throughputs), 2)
            metrics["peak_throughput"] = round(max(training.throughputs), 2)

        save_training_metrics(metrics, save_path)
        writer.add_hparams({"model": model_name, "parallelism": parallelism},
                           {"final_test_loss": test_loss, "total_time": total_time})

        print(f"\n{'='*40}")
        print(f"Treinamento concluido em {total_time:.1f}s")
        print(f"Throughput medio: {avg_img_per_sec:.1f} imagens/s")
        print(f"{'='*40}")

        writer.flush()
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    args = parse_arguments()
    torch.backends.cudnn.benchmark = not args.dp

    train_model(args.epochs, args.model, args.batch_size, args.num_workers, args.dp)
