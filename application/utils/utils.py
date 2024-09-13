import os
import csv
import torch
import argparse

from sklearn.model_selection import StratifiedKFold
from application.preprocessing.custom_dataset import CustomDataset

# Exemplo de uso:
# generate_csv_from_dir('/home/king/Documents/PsoriasisEngineering/infrastructure/db')
# Run on the terminal like
# python -m application.utils.utils






def generate_csv_from_dir(root_path, output_csv='image_labels.csv'):
    # Lista todas as subpastas dentro do diretório raiz (db)
    subfolders = [f.name for f in os.scandir(root_path) if f.is_dir()]

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
        writer.writerows(data)

    print(f"CSV salvo como {output_csv}")


# Criando KFold cross-validator
def generate_stratified_dataset(num_folds, transforms, csv_path) -> None:
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Criando DataLoader para o conjunto de treinamento
    custom_dataset = CustomDataset(csv_file=csv_path, img_dir='image_labels.csv', transform=transforms, target_transform=None)

    labels = custom_dataset.labels
    for fold, (train_index, val_index) in enumerate(kf.split(range(len(labels)), labels)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Salvando os índices de treinamento e validação
        torch.save(train_index, os.path.join(f'application/rag/content/index/train_index_fold{fold}.pt'))
        torch.save(val_index, os.path.join(f'application/rag/content/index/val_index_fold{fold}.pt'))

        print(train_index, val_index)