import sys
import torch
from torch.utils.data import DataLoader, Subset
from application.preprocessing.custom_dataset import CustomDataset
from application.utils.background_remover.background_remover_pixelwise import PixelWiseRemover


sys.path.append('/home/king/Documents/PsoriasisEngineering/application')

path = "infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093312541.jpg"

## IMAGENS DE PERTO
p1 ="infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093333119.jpg"
p2 = "infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093334632.jpg"
p3 = "infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093346411.jpg"
p4 = "infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093355724.jpg"
p5= "infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093352604.jpg"

## Preprocessar um conjunto de dados de modo paralelo

class ImageProcessing():
    def __init__(self, path, transforms):
        self.path = path 
        self.transforms = transforms
        

    def pre_processing(self, fold, batch_size):
        ## Gerar o dataset estratificado

        # processar pre processar cada uma das imagens com threads
        self.customDataset.generate_stratified_dataset()
        train_index = torch.load(f'/content/drive/MyDrive/psoriasis/train_index_fold{fold}.pt')
        val_index = torch.load(f'/content/drive/MyDrive/psoriasis/val_index_fold{fold}.pt')

        custom_dataset = CustomDataset(csv_file='/content/drive/MyDrive/psoriasis/augmented_data.csv', img_dir='/content/sample_data/', transform=self.transforms, target_transform=None)

        #conjunto de treino e teste
        train_loader = DataLoader(Subset(custom_dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(custom_dataset, val_index), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader



    
