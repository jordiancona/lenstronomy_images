
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
    fits_name: str = 'lens_fits.fits'

    # Genera una matriz de las imégenes de lentes gravitacionales para entrenamiento
    def Examples(self, n):
        try:
            with fits.open(self.fits_name) as hdul:
                plt.figure(figsize = (5,5))
                for i in range(15):
                    data = hdul[i+1].data
                    plt.subplot(5, 3, i+1)
                    plt.grid(False)
                    plt.imshow(data, cmap = 'gray', aspect = 'auto')
                    plt.axis('off')
                plt.suptitle('Example of lenses')
                plt.tight_layout()
                plt.savefig('lenses_images.png')
                plt.close()

        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    # Guarda las imágenes y etiquetas para entrenamiento y validación
    def Trian_and_Val_Images(self, ntrain):
        try:
            with fits.open(self.fits_name) as hdul:
                self.labels = ['theta_E','e1','e2','gamma1','gamma2','center_x','center_y']
                self.train_images = []
                self.train_labels = []
                for i in range(ntrain):
                    file = hdul[i+1]
                    hdr = file.header
                    self.train_images.append(file.data)
                    self.train_labels.append([hdr['theta_E'],
                                            hdr['e1'],
                                            hdr['e2'],
                                            hdr['gamma1'],
                                            hdr['gamma2'],
                                            hdr['center_x'],
                                            hdr['center_y']])
                print(self.train_labels)

        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    # Da un resumen del archivo general FITS
    def Generate_Summary(self):
        try:
            with fits.open(self.fits_name) as hdul:
                summary = {"total_images": len(hdul) - 1,
                           "file_info": hdul.info()}
                
                print("Dataset Summary:", summary)
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    # Se entrena el modelo
    def Train_and_Val(self):
        pass
        #train_data = 
    
    # Se evalua el modelo
    def Evaluate(self):
        pass

    # Se generan las imágenes y archivos FITS
    def Generate_Images(self, **kwargs):
        self.__dict__.update(kwargs)
        try:
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
        except:
            print('Class Lenses not downloaded!')

    # Se guardan los archivos FITS en un general
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
        hdu.writeto(self.fits_name, overwrite = True)

Lens = lens(total_images = 100)
#Lens.Generate_Images()
#Lens.Save_FIT# Genera una matriz de las imégenes de lentes gravitacionales para entrenamientoS()
#Lens.Examples(2)
#Lens.Trian_and_Val_Images(60)
Lens.Generate_Summary()