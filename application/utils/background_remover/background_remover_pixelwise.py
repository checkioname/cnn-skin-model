from PIL import Image
import cv2
import numpy as np

class pixelwise_remover():
    # def __init__(self, path):
    #     self.path = path

    def rgb_to_hsv(self, image_path):
        img = cv2.imread(image_path)
        bgr_img = np.array(img)

        # this function convert rgb to hsv
        image_hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        print(image_hsv)

        # Perform color-segmentation to get the binary mask
        lwr = np.array([100, 150, 0]) #lower range
        upr = np.array([140, 255, 255]) #upper range
        msk = cv2.inRange(image_hsv, lwr, upr)

        # Set all pixels within the blue range to black in the original image
        image_hsv[msk > 0] = [0, 0, 0]

        # Save or display the resulting image
        # cv2.imshow('Transformed Image', image_hsv)
        # cv2.waitKey(0)

        # Utilizando um kernel para fazer operacao de diilatacao na imagem
        # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # dlt = cv2.dilate(msk, krn, iterations=10)0
        # res = 255 - cv2.bitwise_and(dlt, msk)

        # cv2.imshow("res", res)
        # cv2.waitKey(0)

        # print(Image.fromarray(image_hsv))



bg_remover = pixelwise_remover()

bg_remover.rgb_to_hsv("infraestructure/db/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093355069.jpg")