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
from tqdm import tqdm
from scipy.ndimage import rotate

parser = argparse.ArgumentParser()
parser.add_argument('-db', '--database', action = 'store_true', help = 'Generate the images for training.')
parser.add_argument('-sh', '--show', action = 'store_true', help = 'Return an example of the images for the training.')
parser.add_argument('-sm', '--summary', action = 'store_true', help = 'Gives a summary of the dataset.')
parser.add_argument('-tr', '--train', help = 'Train the DL model.')
parser.add_argument('-sv', '--save', action = 'store_true', help = 'Save the model.')
parser.add_argument('-ag', '--augment', help = 'Generate a Data Augmentation.')
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
        self.input_shape = (100, 100, 1)

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
                    plt.imshow(np.log10(data), cmap = 'gist_heat', aspect = 'auto')
                    text_values = [f'{label}: {hdr[label]:.2f}' for label in self.labels]
                    y_start = 50
                    y_step = 8
                    for j, text in enumerate(text_values):
                        plt.text(100, y_start - j * y_step, text, fontsize = 8, ha = 'left')
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
    def Train_and_Val_Images(self, augment = False):
        try:
            with fits.open(self.fits_name) as hdul:
                self.train_lbs = []
                self.train_images = []

                for idx in range(self.total_images):
                    file = hdul[idx+1]
                    hdr = file.header
                    file_name = hdr['NAME']
                    img = file.data
                    #img_resized = img.resize((224, 224), Image.BILINEAR)
                    self.train_lbs.append([hdr[label] for label in self.labels])
                    self.train_images.append(np.asarray(np.log10(img)))

                self.train_images, self.train_lbs = np.array(self.train_images), np.array(self.train_lbs)
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")
    
    # Se entrena el modelo
    def Train_and_Val(self, epochs):
        train_df, test_df, train_labels, test_labels = train_test_split(self.train_images, self.train_lbs, test_size = 0.33, random_state = 42, shuffle = True)
        #train_df, test_df = train_df / 255., test_df / 255.
        val_df, val_labels = train_df[-100:], train_labels[-100:]

        optimizer = Adam(learning_rate = 1e-3) # 'adam', 'sgd'
        self.model = alexnet.AlexNet(input_shape = self.input_shape, classes = 7)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['mae'])

        self.history = self.model.fit(train_df, train_labels, epochs = epochs, validation_data = (val_df, val_labels))
        self.Plot_Results('mae')
        self.Plot_Results('loss')

        test_loss, test_mae = self.model.evaluate(test_df, test_labels, batch_size = 128)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        predictions = self.model.predict(test_df[:10])
        print(f'Len predictions {len(predictions)} \n predictions: \n{predictions}')

        correlation = np.corrcoef(predictions, test_labels)[0,1]
        print(f'Coeficiente de correlación - R: {correlation:.2f}')
        print(f'Coeficiente de determinación - R^2: {correlation**2:.2f}')

    def Save_model(self):
        self.model.save('./cnn_model/my_model.h5')

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
    def Generate_Images(self, augment = False):
        #self.__dict__.update(kwargs)
        for i in tqdm(range(self.total_images)):
            f = rd.uniform(0,1.)
            deg = 60
            pa = deg/180*np.pi
            self.e1, self.e2 = (1 - f)/(1 + f)*np.cos(2*pa), (1 - f)/(1 + f)*np.sin(2*pa)
            lss.makelens(n = i,
                         e1 = self.e1,
                         e2 = self.e2,
                         sigmav = 200,
                         zl = rd.uniform(0.5,1.0),
                         zs = rd.uniform(1.,3.),
                         gamma1 = rd.uniform(0,.1),
                         gamma2 = rd.uniform(0,.1),
                         center_x = 0.,
                         center_y = 0.)
            
            lss.Create_FITS(path = self.fits_path)

    # Se guardan los archivos FITS en un general
    def Save_FITS(self, augment = False):
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

        if augment == True:
            self.Augment_Data()
    
    def Rotate_Parameters(self, e1, e2, gamma1, gamma2, angle=45):
        theta = np.radians(angle)
        cos_theta = np.cos(2 * theta)
        sin_theta = np.sin(2 * theta)
        e1_new = e1 * cos_theta - e2 * sin_theta
        e2_new = e1 * sin_theta + e2 * cos_theta
        gamma1_new = gamma1 * cos_theta - gamma2 * sin_theta
        gamma2_new = gamma1 * sin_theta + gamma2 * cos_theta
        return e1_new, e2_new, gamma1_new, gamma2_new
    
    def Augment_Data(self):
        try:
            with fits.open(self.fits_name, mode = 'update') as hdul:
                for i in range(1, len(hdul)):
                    file = hdul[i]
                    hdr = file.header
                    img = file.data
                
                    rotated_data = rotate(img, 45, reshape = False)
                    hdr['e1'], hdr['e2'], hdr['gamma1'], hdr['gamma2'] = self.Rotate_Parameters(
                        hdr['e1'], hdr['e2'], hdr['gamma1'], hdr['gamma2'])
                    
                    new_hdu = fits.ImageHDU(rotated_data, header = hdr)
                    hdul.append(new_hdu)
                hdul.flush()
        except FileNotFoundError:
            print(f'File {self.fits_name} not found.')

Lens_instance = Lens(total_images = 400)

if args.database:
    Lens_instance.Generate_Images()
    Lens_instance.Save_FITS(bool(args.augment))

if args.show:
    Lens_instance.Examples()

if args.summary:
    Lens_instance.Generate_Summary()

if args.train:
    Lens_instance.Train_and_Val_Images()
    Lens_instance.Train_and_Val(int(args.train))
