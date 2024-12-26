
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

    def LoadParameters(self, filename = 'input.dat'):
        d = {'f':'f','sigmav':'sigmav','zl':'zl', "zs":'zs', 'gamma1': 'gamma2', 'center_x': 'center_x', 'center_y': 'center_y'}
        FILE = open(filename)
        for line in FILE:
            name, value = line.split("=")
            value = value.strip()
            if " " in value:
                value = map(float, value.split())
            else:
                value = float(value)
            setattr(self, d[name], value)

    def Generate_Images(self, images_path = './images/', fits_path = './fits/', **kwargs):
        self.__dict__.update(kwargs)

        for _ in range(self.total_images):
            class F(object):pass

            file = F()
            self.LoadParameters(file)
            lss.makelens(f = file['f'],
                         sigmav = file['sigmav'],
                         zl = file['zl'],
                         zs = file['zs'],
                         gamma1 =file['gamma1'],
                         gamma2 = file['gamma2'],
                         center_x = file['center_x'],
                         center_y = file['center_y'])

    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for _ in range(self.total_images):
            for file in os.listdir(self.path):
                if file.endswith('.fits'):
                    self.files.append(file)
