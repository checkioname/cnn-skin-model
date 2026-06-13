import os
import torch
import torch.distributed as dist
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from application.dataset.CustomDataset import CustomDataset

import cv2


class OpenCVPreprocessing:
    def __call__(self, image):
        image = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=10, templateWindowSize=5, searchWindowSize=19
        )
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        return to_pil_image(equalized_rgb)


class ImageProcessing():
    def __init__(self):
        self.transforms = transforms.Compose([
            OpenCVPreprocessing(),
            transforms.RandomResizedCrop(512, scale=(0.3, 1.0)),
            transforms.RandomRotation(50, fill=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def pre_processing(self, fold, batch_size, num_workers=4, rank=0, world_size=1):
        self.generate_stratified_dataset(4, self.transforms)
        try:
            train_index, val_index = self._load_idx(fold)
        except FileNotFoundError as e:
            print(f"Erro ao carregar os índices de treino/validação: {e}")
            return None, None

        custom_dataset = CustomDataset(csv_file='dataset.csv', transform=self.transforms)
        print(f"TAMANHO DO DATASET: {len(custom_dataset.data)}")

        train_subset = Subset(custom_dataset, train_index)
        val_subset = Subset(custom_dataset, val_index)

        train_labels = [custom_dataset.labels[i] for i in train_index]
        class_counts = {l: train_labels.count(l) for l in set(train_labels)}
        total_train = len(train_labels)
        weights = [total_train / (len(class_counts) * class_counts[l]) for l in train_labels]

        if world_size > 1 and dist.is_initialized():
            train_sampler = DistributedSampler(
                train_subset, num_replicas=world_size, rank=rank, shuffle=True
            )
            if rank == 0:
                print(f"[DDP] DistributedSampler ativo ({world_size} GPUs)")
        else:
            train_sampler = WeightedRandomSampler(weights, total_train, replacement=True)

        test_sampler = DistributedSampler(
            val_subset, num_replicas=world_size, rank=rank, shuffle=False
        ) if world_size > 1 and dist.is_initialized() else None

        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers,
            pin_memory=True, prefetch_factor=4
        )
        test_loader = DataLoader(
            val_subset, batch_size=batch_size,
            sampler=test_sampler, shuffle=test_sampler is None,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, test_loader

    def _load_idx(self, fold) -> ([], []):
        base_path = 'application/rag/content/index'
        train_index_path = os.path.join(base_path, f'train_index_fold{fold}.pt')
        val_index_path = os.path.join(base_path, f'val_index_fold{fold}.pt')

        train_index = torch.load(train_index_path, weights_only=False)
        val_index = torch.load(val_index_path, weights_only=False)

        return train_index, val_index

    def generate_stratified_dataset(self, num_folds, transforms) -> None:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        custom_dataset = CustomDataset(csv_file='dataset.csv', transform=transforms)

        df = custom_dataset.data.copy()
        df['patient_id'] = custom_dataset.patient_ids

        patient_df = df.groupby('patient_id')['labels'].first().reset_index()
        labels = patient_df['labels'].tolist()

        base_path = 'application/rag/content/index'
        os.makedirs(base_path, exist_ok=True)

        for fold, (train_patient_idx, val_patient_idx) in enumerate(
            kf.split(patient_df['patient_id'], labels)
        ):
            train_patient_ids = set(patient_df.iloc[train_patient_idx]['patient_id'])
            val_patient_ids = set(patient_df.iloc[val_patient_idx]['patient_id'])

            train_index = df[df['patient_id'].isin(train_patient_ids)].index.tolist()
            val_index = df[df['patient_id'].isin(val_patient_ids)].index.tolist()

            print(f"Fold {fold + 1}/{num_folds} - "
                  f"Train: {len(train_index)} amostras, "
                  f"Val: {len(val_index)} amostras")

            torch.save(train_index, os.path.join(base_path, f'train_index_fold{fold}.pt'))
            torch.save(val_index, os.path.join(base_path, f'val_index_fold{fold}.pt'))
