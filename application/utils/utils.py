import os
import csv
import torch
from pandas import read_csv

from sklearn.model_selection import StratifiedKFold
from application.dataset.CustomDataset import CustomDataset
from torchvision import transforms


# Exemplo de uso:
# generate_csv_from_dir('/home/king/Documents/PsoriasisEngineering/infrastructure/db')
# Run on the terminal like
# python -m application.utils.utils


def generate_csv_from_dir(root_path, output_csv='image_labels.csv'):
    # Lista todas as subpastas dentro do diretório raiz (db)
    subfolders = [f.name for f in os.scandir(root_path) if f.is_dir()]
    print(subfolders)
    data = []
    # Itera por cada subpasta e seus arquivos de imagem
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_path, subfolder)
        # Lista todos os arquivos dentro da subpasta
        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtra por tipos de arquivo de imagem
                image_path = os.path.join(subfolder_path, image_file)
                label = subfolder  # O rótulo é o nome da subpasta
                data.append([image_path, label])

    # Escreve os dados no arquivo CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_name", "labels"])
        print(f"ESCREVENDO OS DADOS {data}")
        writer.writerows(data)

    df = read_csv("image_labels.csv")
    print(df.head())

    print(f"CSV salvo como {output_csv}")


# Criando KFold cross-validator
def generate_stratified_dataset(num_folds, csv_path) -> None:
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    transform = transforms.Compose([
        transforms.RandomRotation(50,fill=1),
        transforms.RandomResizedCrop((224,224)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),  # Converte para tensor
    ])

    # Criando DataLoader para o conjunto de treinamento
    custom_dataset = CustomDataset(csv_file=csv_path, transform=transform, target_transform=None)

    labels = custom_dataset.labels
    for fold, (train_index, val_index) in enumerate(kf.split(range(len(labels)), labels)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Salvando os índices de treinamento e validação
        torch.save(train_index, os.path.join(f'application/rag/content/index/train_index_fold{fold}.pt'))
        torch.save(val_index, os.path.join(f'application/rag/content/index/val_index_fold{fold}.pt'))

        print(train_index, val_index)



def testing_entries(model, dataloader):
    class_to_idx = {"psoriasis": 0, "melanome": 1}
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_labels_numeric = [class_to_idx[label] for label in y]
        batch_labels_tensor = torch.tensor(batch_labels_numeric).float()
        # print(batch)
        print('shape tensor imagem:',X.shape)
        print('shape tensor y antes tranformacao:',len(y))
        print('shape tensor label:',batch_labels_tensor.shape)
        print(batch_labels_tensor)
        break
