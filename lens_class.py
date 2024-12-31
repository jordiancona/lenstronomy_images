
import os
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from create_lens import Lenses as lss
from dataclasses import dataclass
import random as rd
import astropy.io.fits as fits
from time import gmtime, strftime

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

@dataclass
class lens:
    total_images: int
    train_path: str = './lenses/train/'
    fits_path: str = './fits/'

    def Read_FITS(self, path):
        self.files = []
        
        self.path = path
        for _ in range(self.total_images):
            for file in os.listdir(self.path):
                if file.endswith('.fits'):
                    self.files.append(file)
    

    def Generate_Train_Images(self, **kwargs):
        self.__dict__.update(kwargs)
        for i in range(self.total_images):
            lss.makelens(n = i+1,
                         path = self.train_path,
                         f = rd.uniform(0.5,1.),
                         sigmav = 200,
                         zl = rd.uniform(0.,1.),
                         zs = rd.uniform(1.,2.),
                         gamma1 = rd.uniform(-0.2,0.1),
                         gamma2 = rd.uniform(-0.2,0.1),
                         center_x = rd.uniform(0.,0.4),
                         center_y = rd.uniform(0.,0.4))
            
            lss.Create_FITS(path = self.fits_path)
    
    def Save_FITS(self):
        files = []
        path = self.fits_path
        for file in os.listdir(path):
            if file.endswith('.fits'):
                files.append(file)
        
        images_hdus = []

        for file in files:
            with fits.open(path + file) as hdul:
                if not isinstance(hdul[0], fits.ImageHDU):
                    image_hdu = fits.ImageHDU(header = hdul[0].header , data = hdul[0].data)
                    images_hdus.append(image_hdu)
                else:
                    images_hdus.append(hdul[0].copy())

        first_file_time = strftime('%Y-%m-%d %H:%M:%S', gmtime())
        hdr = fits.Header()
        hdr['DATE'] = first_file_time
        hdr['COMMENT'] = 'General file of fits.'
        primary_hdu = fits.PrimaryHDU(header = hdr)

        hdu = fits.HDUList([primary_hdu] + images_hdus)
        hdu.writeto('lens_fits.fits', overwrite = True)

Lens = lens(total_images = 10)
Lens.Generate_Train_Images()
Lens.Save_FITS()