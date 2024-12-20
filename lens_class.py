
import os
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from create_lens import Lenses as lss
from dataclasses import dataclass

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

@dataclass
class lens:
    total_images: int
    
    def Generate_Images(self, images_path = './images/', fits_path = './fits/', **kwargs):
        self.__dict__.update(kwargs)

        for _ in range(self.total_images):

            lss.makelens(f = 0.6,
                sigmav = 200,
                zl = 0.2,
                zs = 1.5,
                gamma1 = -0.01,
                gamma2 = 0.03,
                center_x = 0.1,
                center_y = 0.1,)

    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for _ in range(self.total_images):
            for file in os.listdir(self.path):
                if file.endswith('.fits'):
                    self.files.append(file)
