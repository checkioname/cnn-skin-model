import sys
sys.path.append('/home/king/Documents/PsoriasisEngineering/application')

from utils.remove_background import BackgroundRemover

backgroundRemover = BackgroundRemover()
backgroundRemover.transform_image("application/dataset/data/acral_melanoma2.jpg")