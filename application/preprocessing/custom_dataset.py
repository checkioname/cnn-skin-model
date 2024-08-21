#Creating a custom dataset (rotulando, etc...)
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


#transformando as imagens (image augmentation)
#Normalize = normalifor image_file in os.listdir(dir):ze a tensor image with mean and standard deviation.
#Pode transformar em grayscale? (acho que nao kkkkk)
## Pre processamento


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.loc[idx, 'img_name']
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path)
        label = str(self.data.loc[idx, 'labels'])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)
        
        return image, label
    
    transforms = torch.transforms.Compose([
        #transforms.RandomRotation(50,fill=1),
        #transforms.RandomCrop(size=(224, 224)),
        #transforms.RandomResizedCrop((224,224)),
        torch.transforms.Resize((224,224)),
        torch.transforms.RandomHorizontalFlip(p=0.5),
        torch.transforms.RandomVerticalFlip(p=0.5),
        torch.transforms.ToTensor(),  # Converte para tensor
    ])


    # Criando KFold cross-validator
    def generate_stratified_dataset(self, num_folds, transforms) -> None:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Criando DataLoader para o conjunto de treinamento
        custom_dataset = CustomDataset(csv_file='data_labels.csv', img_dir='/content/sample_data/psoriasis/data', transform=transforms, target_transform=None)

        labels = custom_dataset.labels
        for fold, (train_index, val_index) in enumerate(kf.split(range(len(labels)), labels)):
            print(f"Fold {fold + 1}/{num_folds}")

            # Salvando os índices de treinamento e validação
            torch.save(train_index, os.path.join(f'/content/drive/MyDrive/psoriasis/train_index_fold{fold}.pt'))
            torch.save(val_index, os.path.join(f'/content/drive/MyDrive/psoriasis/val_index_fold{fold}.pt'))

            print(train_index, val_index)