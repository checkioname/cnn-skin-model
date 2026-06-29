#!/usr/bin/env python
"""
Busca de hiperparâmetros com Optuna.

Uso:
    python optuna_search.py model=vgg16 --n-trials=50
"""
import sys
import os
import json
import numpy as np
import optuna
import mlflow
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from application.preprocessing.PreProcessing import ImageProcessing
from application.cmd.Training import Training
from domain.SetupModel import SetupModel
from main import set_seed, _prepare_cam_sample, run_training, is_main_process


def objective(trial, cfg, local_rank, world_size):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    cfg.training.lr = lr
    cfg.training.optimizer = optimizer_name
    cfg.training.batch_size = batch_size
    cfg.training.weight_decay = weight_decay

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = ImageProcessing()
    train_loader, test_loader = dataset.pre_processing(
        fold=cfg.data.fold, batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        rank=local_rank, world_size=world_size
    )

    cam_sample = _prepare_cam_sample(test_loader) if is_main_process() else None

    setup = SetupModel(cfg.model)
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)

    parallelism = "none"
    if cfg.dp and world_size == 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        parallelism = "DataParallel"

    metrics, _ = run_training(
        model, train_loader, test_loader, 15, device, optimizer,
        loss_fn, scheduler, model_name=cfg.model, cam_sample=cam_sample,
        rank=local_rank, world_size=world_size, parallelism=parallelism,
        patience=cfg.training.patience, min_delta=cfg.training.min_delta
    )

    return metrics.get("auroc", 0)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"cnn-skin-{cfg.model}-optuna",
        storage=f"sqlite:///optuna_{cfg.model}.db",
        load_if_exists=True,
    )

    local_rank, world_size = 0, 1

    def optuna_objective(trial):
        val = objective(trial, cfg, local_rank, world_size)
        return val

    study.optimize(optuna_objective, n_trials=HydraConfig.get().runtime.choices.get("n_trials", 30))

    print(f"\n{'='*60}")
    print("  MELHORES HIPERPARÂMETROS")
    print(f"{'='*60}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best AUROC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    with open(f"optuna_best_{cfg.model}.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)

    mlflow.set_experiment(f"cnn-skin-{cfg.model}-optuna")
    with mlflow.start_run(run_name="best"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_auroc", study.best_value)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
