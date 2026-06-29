#!/usr/bin/env python
"""
Cross-validation completa: treina em todos os folds e agrega métricas.

Uso:
    python cross_validate.py model=vgg16 training.epochs=50
"""
import sys
import os
import json
import numpy as np
import mlflow
from omegaconf import OmegaConf

import hydra

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from main import set_seed, train_fold, setup_ddp, is_main_process

import torch.distributed as dist


@hydra.main(version_base=None, config_path="config", config_name="config")
def cross_validate(cfg):
    set_seed(cfg.seed)

    mlflow.set_experiment(f"cnn-skin-{cfg.model}-cv")

    fold_metrics = []

    for fold in range(cfg.data.folds):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1}/{cfg.data.folds}")
        print(f"{'='*60}")

        cfg.data.fold = fold + 1

        with mlflow.start_run(run_name=f"fold-{fold + 1}") as run:
            local_rank, world_size = setup_ddp()
            OmegaConf.save(cfg, "hydra_config.yaml")
            mlflow.log_artifact("hydra_config.yaml")
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

            metrics = train_fold(cfg, fold + 1, local_rank, world_size)
            fold_metrics.append(metrics)

            if is_main_process():
                with open(f"metrics_fold_{fold + 1}.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                mlflow.log_artifact(f"metrics_fold_{fold + 1}.json")

        if dist.is_initialized():
            dist.destroy_process_group()

    if is_main_process():
        aggregated = aggregate_metrics(fold_metrics)
        print(f"\n{'='*60}")
        print("  RESULTADOS AGREGADOS (cross-validation)")
        print(f"{'='*60}")
        for metric, values in aggregated.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f} "
                  f"(min={values['min']:.4f}, max={values['max']:.4f})")

        with open("cross_validation_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        mlflow.log_artifact("cross_validation_results.json")


def aggregate_metrics(fold_metrics):
    keys = ["accuracy", "auroc", "mcc", "f1", "precision", "recall",
            "specificity", "kappa"]
    aggregated = {}
    for key in keys:
        values = [m.get(key, 0) for m in fold_metrics if key in m]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": [float(v) for v in values],
            }
    return aggregated


if __name__ == "__main__":
    cross_validate()
