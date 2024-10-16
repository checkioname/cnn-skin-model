import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
from application.dataset.CustomDataset import CustomDataset
from application.utils.background_remover.BackgroundRemoverPixelwise import PixelWiseRemover
from torchvision import transforms

from application.utils.utils import generate_stratified_dataset


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

        

    def pre_processing(self, fold, batch_size):
        ## Gerar o dataset estratificado

        # processar pre processar cada uma das imagens com threads
        # generate_stratified_dataset(fold, self.transforms, "/home/king/Documents/PsoriasisEngineering/image_labels.csv")

        try:
            train_index, val_index = self._load_idx(fold)
        except FileNotFoundError as e:
            print(f"Erro ao carregar os índices de treino/validação: {e}")
            return None, None

        custom_dataset = CustomDataset(csv_file='image_labels.csv', transform=self.transforms, target_transform=None)
        print(custom_dataset.data.head())


        #conjunto de treino e teste
        train_loader = DataLoader(Subset(custom_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(custom_dataset, val_index), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def _load_idx(self,fold):
        base_path = 'application/rag/content/index'
        train_index_path = os.path.join(base_path, f'train_index_fold{fold}.pt')
        val_index_path = os.path.join(base_path, f'val_index_fold{fold}.pt')

        train_index = torch.load(train_index_path)
        val_index = torch.load(val_index_path)

        return train_index, val_index