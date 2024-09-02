import cv2
import numpy as np

class PixelWiseRemover():

    def rgb_to_hsv(self, image_path):
        bgr_img = cv2.imread(image_path)

        # this function convert rgb to hsv
        image_ycbcr = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2YCrCb)

        # Y=image_ycbcr( :,:,1)
        # Cb=image_ycbcr( :,:,2)
        # Cr=image_ycbcr( :,:,3)


        print(image_ycbcr)

        # Perform color-segmentation to get the binary mask
        lwr = np.array([100, 150, 0]) #lower range
        upr = np.array([140, 255, 255]) #upper range
        msk = cv2.inRange(image_ycbcr, lwr, upr)

        # Set all pixels within the blue range to black in the original image
        image_ycbcr[msk > 0] = [0, 0, 0] 

        # Save or display the resulting image
        # cv2.imshow('Transformed Image', image_hsv)
        cv2.waitKey(0)

        # Utilizando um kernel para fazer operacao de diilatacao na imagem
        # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # dlt = cv2.dilate(msk, krn, iterations=10)0
        # res = 255 - cv2.bitwise_and(dlt, msk)
        # cv2.imshow("res", res)
        # cv2.waitKey(0)

        # print(Image.fromarray(image_hsv))


# Y: [10, 255]• Cr: [135, 180]• Cb: [85, 135]

bg_remover = PixelWiseRemover()

bg_remover.rgb_to_hsv("/home/king/Documents/PsoriasisEngineering/infrastructure/db/MARIA ROSA DE JESUS SOUSA - 607682/MARIA ROSA DE JESUS SOUSA - 607682.jpg")