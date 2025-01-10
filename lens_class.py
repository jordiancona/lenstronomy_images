#!/Users/juananconaflores/.pyenv/shims/python

import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime
from create_lens import Lenses as lss
from models import alexnet
from keras.optimizers import Adam # type: ignore
import astropy.io.fits as fits
from dataclasses import dataclass
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import lenstronomy as ln
except:
    print("Lenstronomy not installed!")

parser = argparse.ArgumentParser()
parser.add_argument('-db', '--database', action = 'store_true', help = 'Generate the images for training.')
parser.add_argument('-sh', '--show', action = 'store_true', help = 'Return an example of the images for the training.')
parser.add_argument('-sm', '--summary', action = 'store_true', help = 'Gives a summary of the dataset.')
parser.add_argument('-tr', '--train', help = 'Train the DL model.')
parser.add_argument('-ev', '--evaluate', action = 'store_true', help = 'Evaluate the model.')

args = parser.parse_args()

class lens:
    def __init__(self, total_images):
        self.total_images = total_images
        self.train_path = './dataset/train/lenses/'
        self.val_path = './dataset/val/lenses/'
        self.test_path = './dataset/test/'
        self.fits_path = './fits/'
        self.fits_name = './lens_fits.fits'
        self.batch_size = 64
        self.input_shape = (389, 389, 4)

    # Genera una matriz de las imégenes de lentes gravitacionales para entrenamiento
    def Examples(self):
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
                plt.savefig('lenses_images.png', bbox_inches = "tight")
                plt.close()

        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    # Da un resumen del archivo general FITS
    def Generate_Summary(self):
        try:
            with fits.open(self.fits_name) as hdul:
                summary = {'total_images': len(hdul) - 1, 'file_info': hdul.info()}
                
                print("Dataset Summary:", summary)
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    def Train_and_Val_Images(self):
        try:
            with fits.open(self.fits_name) as hdul:
                self.labels = ['theta_E','e1','e2','gamma1','gamma2','center_x','center_y']

                train_lbs = []
                train_images = []
                for idx in range(self.total_images):
                    file = hdul[idx+1]
                    hdr = file.header
                    plt.imshow(file.data, cmap = 'gist_heat')
                    plt.axis('off')
                    plt.margins(0,0)
                    plt.savefig(f'{self.train_path}lens_{idx+1}.png', bbox_inches = "tight")
                    plt.close()
                    train_lbs.append([hdr[label] for label in self.labels])

                for file in os.listdir(self.train_path):
                    if file.endswith('.png'):
                        img = Image.open(os.path.join(self.train_path, file))
                        train_images.append(np.asarray(img))

                train_images, train_lbs = np.array(train_images), np.array(train_lbs)
                self.train_df, self.val_df, self.train_labels, self.val_labels = train_test_split(train_images, train_lbs, test_size = 0.33, random_state = 42)
                self.train_df, self.val_df = self.train_df / 255., self.val_df / 255.
                self.train_labels, self.val_labels = np.array(self.train_labels), np.array(self.val_labels)
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")
    
    # Se entrena el modelo
    def Train_and_Val(self, epochs):
        optimizer = Adam(learning_rate = 1e-4) # 'adam', 'sgd'
        self.model = alexnet.AlexNet(input_shape = self.input_shape)
        self.model.compile(optimizer = optimizer,
                        loss = 'mean_squared_error',
                        metrics = ['mae'])

        self.history = self.model.fit(self.train_df, self.train_labels, epochs = epochs, validation_data = (self.val_df, self.val_labels))
        self.Plot_Results()
    
    # Se evalua el modelo
    def Evaluate(self):
        test_loss, test_mae = self.model.evaluate(self.val_df, self.val_labels, batch_size = 128)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        #predictions = self.model.predict(self.test_images)

    def Plot_Results(self):
        plt.figure()
        plt.plot(self.history.history['mae'], label = f'Trainning mae', c = 'k', lw = 0.8)
        plt.plot(self.history.history['val_mae'], label = f'Validation mae', c = 'r', lw = 0.8)
        plt.title('MAE')
        plt.xlabel('epoch')
        plt.ylabel('mae')
        plt.legend()
        plt.show()
    
    # Se generan las imágenes y archivos FITS
    def Generate_Images(self, **kwargs):
        self.__dict__.update(kwargs)
        for i in range(self.total_images):
            lss.makelens(n = i,
                        path = './dataset/images/',
                        f = rd.uniform(0.5,1.),
                        sigmav = 200,
                        zl = rd.uniform(0.,1.),
                        zs = rd.uniform(1.,2.),
                        gamma1 = rd.uniform(-0.2,0.1),
                        gamma2 = rd.uniform(-0.2,0.1),
                        center_x = rd.uniform(0.,0.4),
                        center_y = rd.uniform(0.,0.4))
            
            lss.Create_FITS(path = self.fits_path)

    # Se guardan los archivos FITS en un general
    def Save_FITS(self):
        path = self.fits_path
        files = [file for file in os.listdir(path) if file.endswith('.fits')]
        images_hdus = []

        for file in files:
            with fits.open(path + file, lazy_load = True) as hdul:
                for hdu in hdul:
                    if isinstance(hdu, fits.ImageHDU):
                        images_hdus.append(hdu.copy())

        first_file_time = strftime('%Y-%m-%d %H:%M:%S', gmtime())
        hdr = fits.Header()
        hdr['DATE'] = first_file_time
        hdr['COMMENT'] = 'General file of fits.'
        primary_hdu = fits.PrimaryHDU(header = hdr)

        hdu = fits.HDUList([primary_hdu] + images_hdus)
        hdu.writeto(self.fits_name, overwrite = True)

Lens_instance = lens(total_images = 300)

if args.database:
    Lens_instance.Generate_Images()
    Lens_instance.Save_FITS()

if args.show:
    Lens_instance.Examples()

if args.summary:
    Lens_instance.Generate_Summary()

if args.train:
    Lens_instance.Train_and_Val_Images()
    Lens_instance.Train_and_Val(int(args.train))

if args.evaluate:
    Lens_instance.Evaluate()
