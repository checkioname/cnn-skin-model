#!/usr/bin/env python
"""
Avaliação detalhada de um modelo treinado com métricas acadêmicas e
interpretabilidade (Captum + GradCAM).

Uso:
    python evaluate.py model=vgg16 checkpoint=runs/ml-model-xxx/model.pt
"""
import sys
import os
import json
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from omegaconf import OmegaConf

import hydra
import mlflow

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from application.preprocessing.PreProcessing import ImageProcessing, OpenCVPreprocessing
from application.cmd.Training import Training, TARGET_LAYERS, _unwrap_model
from domain.SetupModel import SetupModel
from main import set_seed


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup = SetupModel(cfg.model)
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)

    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint carregado: época {checkpoint['epoch']}")

    dataset = ImageProcessing()
    _, test_loader = dataset.pre_processing(
        fold=cfg.data.fold, batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    mlflow.set_experiment(f"cnn-skin-{cfg.model}-evaluation")
    with mlflow.start_run(run_name=f"eval-fold-{cfg.data.fold}") as run:
        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze(1)
                prob = torch.sigmoid(pred)

                all_labels.extend(y.cpu().numpy().astype(int))
                all_probs.extend(prob.cpu().numpy())
                all_preds.extend((prob > 0.5).cpu().numpy().astype(int))

        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, matthews_corrcoef,
                                      cohen_kappa_score, confusion_matrix,
                                      classification_report, roc_curve)
        import matplotlib.pyplot as plt

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*60}")
        print("  RELATÓRIO DE AVALIAÇÃO")
        print(f"{'='*60}")
        print(classification_report(all_labels, all_preds,
              target_names=["psoriasis", "dermatite"]))
        print(f"  AUROC:     {auroc:.4f}")
        print(f"  MCC:       {mcc:.4f}")
        print(f"  Kappa:     {kappa:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "mcc": mcc,
            "kappa": kappa,
        })

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {cfg.model}")
        plt.legend()
        plt.savefig("roc_curve.png", dpi=150)
        mlflow.log_artifact("roc_curve.png")

        print("\n  Captum - Integrated Gradients na primeira imagem de cada classe...")
        try:
            from captum.attr import IntegratedGradients

            dataset_full = ImageProcessing()
            train_loader_full, test_loader_full = dataset_full.pre_processing(
                fold=cfg.data.fold, batch_size=1, num_workers=1
            )

            unwrapped = _unwrap_model(model)
            ig = IntegratedGradients(unwrapped)

            for class_name, class_idx in [("psoriasis", 0), ("dermatite", 1)]:
                for X, y in test_loader_full:
                    if y.item() == class_idx:
                        X = X.to(device)
                        X.requires_grad = True
                        attr, delta = ig.attribute(X, target=0, return_convergence_delta=True)
                        attr = attr.cpu().detach().numpy()

                        attr_img = np.transpose(attr[0], (1, 2, 0))
                        attr_img = (attr_img - attr_img.min()) / (attr_img.max() - attr_img.min() + 1e-8)

                        plt.figure()
                        plt.imshow(attr_img)
                        plt.title(f"Integrated Gradients - {class_name}")
                        plt.axis("off")
                        plt.savefig(f"captum_ig_{class_name}.png", dpi=150)
                        mlflow.log_artifact(f"captum_ig_{class_name}.png")
                        print(f"  Captum salvo: captum_ig_{class_name}.png")
                        break
                break
        except Exception as e:
            print(f"  Captum nao disponivel: {e}")


if __name__ == "__main__":
    main()
