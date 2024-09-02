from PIL import Image
from pandas import read_csv
from torch.utils.data import Dataset


#transformando as imagens (image augmentation)
#Normalize = normalifor image_file in os.listdir(dir):ze a tensor image with mean and standard deviation.
#Pode transformar em grayscale? (acho que nao kkkkk)
## Pre processamento


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform, target_transform=None):
        self.data = read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.labels = [str(label) for label in self.data['labels']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, 'img_name']
        image = Image.open(image_path)
        label = str(self.data.loc[idx, 'labels'])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)
        
        return image, label
    
    # transforms = transforms.Compose([
    #     transforms.RandomRotation(50,fill=1),
    #     transforms.RandomResizedCrop((224,224)),
    #     transforms.Resize((224,224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.ToTensor(),  # Converte para tensor
    # ])