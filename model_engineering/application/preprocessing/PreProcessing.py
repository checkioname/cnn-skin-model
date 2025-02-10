
import sys
import os

import cv2
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from application.dataset.CustomDataset import CustomDataset


class ImageProcessing():
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(50,fill=1),
            transforms.RandomResizedCrop((224,224)),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # Converte para tensor
        ])

    def _opencv_preprocessing(self, image):
        image = np.array(image)

        # Aplicar Denoise (Non-Local Means Denoising)
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 
                                                   h=10,  # Força do filtro
                                                   templateWindowSize=5, 
                                                   searchWindowSize=19)
        
        
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        equalized = cv2.equalizeHist(gray)
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        return to_pil_image(equalized_rgb)

    def pre_processing(self, fold, batch_size):
        ## Gerar o dataset estratificado
        try:
            train_index, val_index = self._load_idx(fold)
        except FileNotFoundError as e:
            print(f"Erro ao carregar os índices de treino/validação: {e}")
            return None, None

        custom_dataset = CustomDataset(csv_file='dataset.csv', transform=self.transforms, target_transform=None)
        print(f"TAMANHO DO DATASET: {len(custom_dataset.data)}")
        print(custom_dataset.data.head())


        #conjunto de treino e teste
        train_loader = DataLoader(Subset(custom_dataset, train_index), batch_size=batch_size, shuffle=True, num_workers=1)
        test_loader = DataLoader(Subset(custom_dataset, val_index), batch_size=batch_size, shuffle=True, num_workers=1)

        return train_loader, test_loader

    def _load_idx(self,fold):
        base_path = 'application/rag/content/index'
        train_index_path = os.path.join(base_path, f'train_index_fold{fold}.pt')
        val_index_path = os.path.join(base_path, f'val_index_fold{fold}.pt')

        train_index = torch.load(train_index_path)
        val_index = torch.load(val_index_path)

        return train_index, val_index