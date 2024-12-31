
import os
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from create_lens import Lenses as lss
from dataclasses import dataclass
import random as rd

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

@dataclass
class lens:
    total_images: int

    def Generate_Train_Images(self, images_path = './lenses/train', fits_path = './fits/', **kwargs):
        self.__dict__.update(kwargs)
        for i in range(self.total_images):
            lss.makelens(n = i+1,
                         path = images_path,
                         f = rd.random(),
                         sigmav = 200,
                         zl = rd.uniform(0.,1.),
                         zs = rd.uniform(1.,2.),
                         gamma1 = rd.uniform(-0.5,0.5),
                         gamma2 = rd.uniform(-0.5,0.5),
                         center_x = rd.uniform(0.,0.5),
                         center_y = rd.uniform(0.,0.5))

    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for _ in range(self.total_images):
            for file in os.listdir(self.path):
                if file.endswith('.fits'):
                    self.files.append(file)

Lens = lens(total_images = 10)
Lens.Generate_Train_Images()