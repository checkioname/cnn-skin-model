import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2

class BackgroundRemoverNN():

    model_architecture = 'deeplabv3_resnet101'
    repo = 'pytorch/vision:v0.10.0'

    #Pre processing for the selected model
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.4556,0.406], std=[0.229,0.224,0.225]),
        ])

    def load_model(self):
        model = torch.hub.load(self.repo, self.model_architecture, pretrained=True)
        model.eval()
        return model

    def make_transparent_foreground(self, pic, mask):
        # split the image into channels
        b, g, r = cv2.split(np.array(pic).astype('uint8'))
        # add an alpha channel with and fill all with transparent pixels (max 255)
        a = np.ones(mask.shape, dtype='uint8') * 255
        # merge the alpha channel back
        alpha_im = cv2.merge([b, g, r, a], 4)
        # create a transparent background
        bg = np.zeros(alpha_im.shape)
        # setup the new mask
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        # copy only the foreground color pixels from the original image where mask is set
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground

    def remove_background(self, model, input_file):
        input_image = Image.open(input_file)
        input_tensor = self.preprocess(input_image).unsqueeze(0)

        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)

        # Create a black and white mask of the profile foreground
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask,255, background).astype(np.uint8)

        foreground = self.make_transparent_foreground(input_image, bin_mask)
        return foreground, bin_mask 

    def transform_image(self, input_image):
        
        deeplab_model = self.load_model()
        foreground, _ = self.remove_background(deeplab_model, input_image)

        plt.imshow(foreground)
        print("-------- Saving image with no background --------")
        plt.imsave(f'{input_image.split("/")[-1]}_nobackground.jpg', foreground)
