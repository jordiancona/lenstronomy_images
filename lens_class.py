
import numpy as np
from random import uniform
from create_lens import Lenses as lss
import os

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

class lens:
    def __init__(self, total_images = 1000, model = 'SIE') -> None:
        self.total_images = total_images
        self.model = model
    
    def Generate_Images(self, theta_E = 1, images_path = './images/', fits_path = './fits/', **kwargs):
        self.__dict__.update(kwargs)

        for _ in range(self.total_images):
            gamma1, gamma2 = uniform(-0.5,0.5), uniform(-0.5,0.5)
            e1, e2 = uniform(-0.5,0.5), uniform(-0.5,0.5)
            center_x, center_y = uniform(-100,100), uniform(-100,100)

            lss.Create_PNG(model = self.model,
                           theta_E = theta_E,
                           e1 = e1,
                           e2 = e2,
                           gamma1 = gamma1,
                           gamma2 = gamma2,
                           center_x = center_x,
                           center_y = center_y)
        
    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for file in os.listdir(self.path):
            if file.endswith('.fits'):
                self.files.append(file)
