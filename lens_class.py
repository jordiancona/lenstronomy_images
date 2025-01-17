#!/Users/juananconaflores/.pyenv/shims/python

import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from time import gmtime, strftime
from create_lens import Lenses as lss
from models import alexnet
from keras.optimizers import Adam # type: ignore
import astropy.io.fits as fits
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-db', '--database', action = 'store_true', help = 'Generate the images for training.')
parser.add_argument('-sh', '--show', action = 'store_true', help = 'Return an example of the images for the training.')
parser.add_argument('-sm', '--summary', action = 'store_true', help = 'Gives a summary of the dataset.')
parser.add_argument('-tr', '--train', help = 'Train the DL model.')
parser.add_argument('-ev', '--evaluate', action = 'store_true', help = 'Evaluate the model.')
parser.add_argument('-sv', '--save', action = 'store_true', help = 'Save the model.')
args = parser.parse_args()

class Lens:
    def __init__(self, total_images):
        self.total_images = total_images
        self.train_path = './dataset/train/lenses/'
        self.test_path = './dataset/test/'
        self.fits_path = './fits/'
        self.fits_name = './lens_fits.fits'
        self.labels = ['theta_E','e1','e2','gamma1','gamma2','center_x','center_y']
        self.batch_size = 64
        self.input_shape = (390, 390, 4)

    # Genera una matriz de las imégenes de lentes gravitacionales para entrenamiento
    def Examples(self):
        try:
            with fits.open(self.fits_name) as hdul:
                plt.figure(figsize = (10,8))
                for i in range(9):
                    file = hdul[i+1]
                    hdr = file.header
                    data = file.data
                    plt.subplot(3, 3, i+1)
                    plt.grid(False)
                    plt.imshow(data, cmap = 'gist_heat', aspect = 'auto')
                    text_values = [f'{label}: {hdr[label]:.2}' for label in self.labels]
                    y_start = 195
                    y_step = 25
                    for j, text in enumerate(text_values):
                        plt.text(400, y_start - j * y_step, text, fontsize = 8, ha = 'left')
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

    # Genera la base de datos para entrenamiento y validación
    def Train_and_Val_Images(self):
        try:
            with fits.open(self.fits_name) as hdul:
                train_lbs = []
                train_images = []

                for idx in range(self.total_images):
                    file = hdul[idx+1]
                    hdr = file.header
                    file_name = hdr['NAME']
                    train_lbs.append([hdr[label] for label in self.labels])
                    img = Image.open(os.path.join(self.train_path, file_name))
                    train_images.append(np.asarray(img))

                train_images, train_lbs = np.array(train_images), np.array(train_lbs)
                self.train_df, self.test_df, self.train_labels, self.test_labels = train_test_split(train_images, train_lbs, test_size = 0.33, random_state = 42)
                self.train_df, self.test_df = self.train_df / 255., self.test_df / 255.
                self.val_df, self.val_labels = self.train_df[-100:], self.train_labels[-100:]
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")
    
    # Se entrena el modelo
    def Train_and_Val(self, epochs):
        optimizer = Adam(learning_rate = 1e-4) # 'adam', 'sgd'
        self.model = alexnet.AlexNet(input_shape = self.input_shape, classes = 7)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['mae'])

        self.history = self.model.fit(self.train_df, self.train_labels, epochs = epochs, validation_data = (self.val_df, self.val_labels))
        self.Plot_Results('mae')
        self.Plot_Results('loss')

    # Se guarda el modelo
    def Save_model(self):
        self.model.save('./cnn_model/my_model.h5')
    
    # Se evalua el modelo
    def Evaluate(self):
        test_loss, test_mae = self.model.evaluate(self.test_df, self.test_labels, batch_size = 128)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        predictions = self.model.predict(self.test_df[:10])
        print(f'Len predictions {len(predictions)} \n predictions: \n{predictions}')

    def Plot_Results(self, metric):
        plt.figure()
        plt.plot(self.history.history[f'{metric}'], label = f'Trainning {metric}', c = 'k', lw = 0.8)
        plt.plot(self.history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
        plt.title(metric.upper())
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'{metric.upper()}.png')
        plt.close()
    
    # Se generan las imágenes y archivos FITS
    def Generate_Images(self):
        #self.__dict__.update(kwargs)
        for i in range(self.total_images):
            lss.makelens(n = i,
                        path = self.train_path,
                        f = 1.,
                        sigmav = 200,
                        zl = rd.uniform(0.,1.),
                        zs = rd.uniform(1.,2.),
                        gamma1 = rd.uniform(-0.2,0.1),
                        gamma2 = rd.uniform(-0.2,0.1),
                        center_x = 0.,
                        center_y = 0.)
            
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

Lens_instance = Lens(total_images = 500)

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
