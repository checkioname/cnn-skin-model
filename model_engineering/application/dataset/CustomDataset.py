#Creating a custom dataset (rotulando, etc...)
import os
import torch
from PIL import Image
from pandas import read_csv
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


#transformando as imagens (image augmentation)
#Normalize = normalifor image_file in os.listdir(dir):ze a tensor image with mean and standard deviation.
#Pode transformar em grayscale? (acho que nao kkkkk)
## Pre processamento


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.data = read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.labels = [str(label) for label in self.data['labels']]
        self.class_to_idx = {"psoriasis": 0, "dermatite": 1}




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[int(idx), 'img_name']
        image = Image.open(image_path)
        label = str(self.data.loc[int(idx), 'labels'])
        
        if self.transform:
            image = self.transform(image)

        label_numeric = torch.tensor(self.class_to_idx[label], dtype=torch.float32)

        
        return image, label_numeric
    
    transforms = transforms.Compose([
        transforms.RandomRotation(50,fill=1),
        transforms.RandomResizedCrop((224,224)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),  # Converte para tensor
    ])
