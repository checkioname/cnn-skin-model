import os
import torch 
from domain.Swim import SetupModelSwin
from torchvision import transforms
from PIL import Image

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50



import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import numpy as np 

# Load an image and preprocess it
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = cv2.crop(img(512, 512))  
    img = cv2.resize(img, (512, 512))  
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img



if __name__ == "__main__":
    img_path = "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/dermatite/COSTA SOARES, RHAYAN MIGUEL  (20240430090521504) 20240430091026337.jpg"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_file_path = '/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/runs/ml-model-test-1742432834.2114196/model.pt' 

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (512, 512))
    rgb_img = np.float32(rgb_img) / 255


    input_tensor = preprocess_image(img_path)

    setup = SetupModelSwin()
    swin, _, _, _ = setup.setup_model(device, pretrained_path=pretrained_file_path)
    target_layers_swin = [swin.features[-1][-1].norm1]

    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(0)]

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=swin, target_layers=target_layers_swin) as cam:
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs


        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.axis('off')  # Remove os eixos para uma visualização mais limpa
        plt.title("Grad-CAM Visualizatin")
        plt.show()