import os
import re
import torch
from PIL import Image
from pandas import read_csv
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None, data_dir=None):
        print(f"[DEBUG] Lendo CSV de: {csv_file}")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Arquivo {csv_file} não encontrado.")

        self.data = read_csv(csv_file)
        if self.data is None or "labels" not in self.data.columns:
            raise ValueError("CSV inválido ou coluna 'labels' ausente.")

        self.data_dir = data_dir or os.getenv('DATA_DIR', '')
        if self.data_dir:
            self.data['img_name'] = self.data['img_name'].apply(
                lambda p: os.path.join(self.data_dir, p) if not os.path.isabs(p) else p
            )

        self.transform = transform
        self.target_transform = target_transform
        self.labels = [str(label) for label in self.data['labels']]
        self.class_to_idx = {"psoriasis": 0, "dermatite": 1}

        self.patient_ids = []
        for path in self.data['img_name']:
            match = re.search(r'\((\d{15,})\)', str(path))
            patient_id = match.group(1) if match else str(path)
            self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[int(idx)]['img_name']
        image = Image.open(image_path)
        label = self.data.iloc[idx]['labels']
            
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
