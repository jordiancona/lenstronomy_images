
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime
from create_lens import Lenses as lss
from models import alexnet
import tensorflow as tf
import keras
from keras.optimizers import Adam # type: ignore
import astropy.io.fits as fits
from dataclasses import dataclass

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train', action = 'store_true', help = 'Train the DL model.')
parser.add_argument('-db', '--database', action = 'store_true', help = 'Generate the images for the training.')
parser.add_argument('-sh', '--show', action = 'store_true', help = 'Return an example of the images for the training.')
parser.add_argument('-sm', '--summary', action = 'store_true', help = 'Gives a summary of the dataset.')

args = parser.parse_args()

@dataclass
class lens:
    total_images: int
    train_path: str = './lenses/train/'
    fits_path: str = './fits/'
    fits_name: str = 'lens_fits.fits'
    batch_size: int = 64
    input_shape = (400, 400)

    # Genera una matriz de las imégenes de lentes gravitacionales para entrenamiento
    def Examples(self, n):
        try:
            with fits.open(self.fits_name) as hdul:
                plt.figure(figsize = (6,6))
                for i in range(9):
                    data = hdul[i+1].data
                    plt.subplot(3, 3, i+1)
                    plt.grid(False)
                    plt.imshow(data, cmap = 'gist_heat', aspect = 'auto')
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
                self.val_images = []
                self.val_labels = []

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
                #print(self.train_labels)
                for i in range(ntrain, self.total_images):
                    file = hdul[i]
                    hdr = file.header
                    self.val_images.append(file.data)
                    self.val_labels.append([hdr['theta_E'],
                                            hdr['e1'],
                                            hdr['e2'],
                                            hdr['gamma1'],
                                            hdr['gamma2'],
                                            hdr['center_x'],
                                            hdr['center_y']])

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
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-4,
                                                                  decay_steps = 10000,
                                                                  decay_rate = 0.9)
        optimizer = Adam(learning_rate = lr_schedule) # 'adam', 'sgd'
        self.model = alexnet.AlexNet(input_shape = self.input_shape)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['mae'])
        history = self.model.fit(self.train_images, epochs = 100, validation_data = self.val_images)
    
    # Se evalua el modelo
    def Evaluate(self):
        test_loss, test_mae = self.model.evaluate(self.test_images)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        predictions = self.model.predict(self.test_images)

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
        path = self.fits_path
        files = [file for file in os.listdir(path) if file.endswith('.fits')]
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

if args.train:
    Lens.Trian_and_Val_Images(60)
    Lens.Train_and_Val()

if args.database:
    Lens.Generate_Images()
    Lens.Save_FITS()

if args.show:
    Lens.Examples(2)

if args.summary:
    Lens.Generate_Summary()
