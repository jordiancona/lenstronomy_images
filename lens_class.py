
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
        FILE = open(filename)
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.f = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.sigmav = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.zl = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.zs = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.gamma1 = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.gamma2 = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.center_x = float(s[1])
        s = FILE.readline().split('=')
        if len(s) == 2:
            self.center_y = float(s[1])

    def Generate_Images(self, images_path = './images/', fits_path = './fits/', **kwargs):
        self.__dict__.update(kwargs)
        for _ in range(self.total_images):
            class F(object): pass

            self.LoadParameters()
            lss.makelens(f = self.f,
                         sigmav = self.sigmav,
                         zl = self.zl,
                         zs = self.zs,
                         gamma1 = self.gamma1,
                         gamma2 = self.gamma2,
                         center_x = self.center_x,
                         center_y = self.center_y)

    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for _ in range(self.total_images):
            for file in os.listdir(self.path):
                if file.endswith('.fits'):
                    self.files.append(file)

Lens = lens(total_images = 1)
Lens.Generate_Images()