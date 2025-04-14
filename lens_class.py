#!/Users/juananconaflores/.pyenv/shims/python

import os
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime
from create_lens import Lenses as lss
from create_lens import sie_lens
from make_lens import MakeLens
from models import alexnet_informed
from models import alexnet
from models import physics_informed_loss
import tensorflow as tf
from keras.optimizers import Adam, Nadam # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-db', '--database', action = 'store_true', help = 'Generate the images for training.')
parser.add_argument('-sh', '--show', action = 'store_true', help = 'Return an example of the images for the training.')
parser.add_argument('-sm', '--summary', action = 'store_true', help = 'Gives a summary of the dataset.')
parser.add_argument('-tr', '--train', help = 'Train the DL model.')
parser.add_argument('-sv', '--save', action = 'store_true', help = 'Save the model.')
args = parser.parse_args()

class Lens:
    def __init__(self, total_images):
        self.total_images = total_images
        self.fits_path = './fits/'
        self.fits_name = './lens_fits.fits'
        self.labels = ['theta_E','f_axis','e1','e2','gamma1','gamma2']
        self.classes = 6
        self.batch_size = 64
        self.input_shape = (100, 100, 1)

    # Genera una matriz de las imágenes de lentes gravitacionales para entrenamiento
    def Examples(self):
        try:
            with fits.open(self.fits_name) as hdul:
                plt.figure(figsize = (10,8))
                for i in range(9):
                    idx = np.random.randint(1, self.total_images)
                    file = hdul[idx]
                    hdr = file.header
                    data = file.data
                    plt.subplot(3, 3, i+1)
                    plt.title(f'image: {idx}')
                    plt.grid(False)
                    plt.imshow(np.log10(data), cmap = 'gist_heat', aspect = 'auto')
                    plt.axis('equal')
                    text_values = [f'{label}: {hdr[label]:.4f}' for label in self.labels]
                    y_start = 50
                    y_step = 8
                    for j, text in enumerate(text_values):
                        plt.text(100, y_start - j * y_step, text, fontsize = 8, ha = 'left')
                    plt.axis('off')
                plt.suptitle('Example of lenses')
                plt.tight_layout()
                plt.savefig('lenses_images.png', bbox_inches = 'tight')
                plt.close()

        except FileNotFoundError:
            print(f'File {self.fits_name} not found.')

    # Da un resumen del archivo general FITS
    def Generate_Summary(self):
        try:
            with fits.open(self.fits_name) as hdul:
                summary = {'total_images': len(hdul) - 1}
                print('Dataset Summary:', summary)
                
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")

    def Rotate_Parameters(self, e1, e2, gamma1, gamma2, angle = 45):
        pa = angle/180*np.pi
        e1_new = e1 * np.cos(2*pa) - e2 * np.sin(2*pa)
        e2_new = e1 * np.sin(2*pa) + e2 * np.cos(2*pa)
        gamma1_new = gamma1 * np.cos(2*pa) - gamma2 * np.sin(2*pa)
        gamma2_new = gamma1 * np.sin(2*pa) + gamma2 * np.cos(2*pa)
        return e1_new, e2_new, gamma1_new, gamma2_new
    
    def Augment_Data_Special(self):
        try:
            with fits.open(self.fits_name, mode = 'update') as hdul:
                for i in range(1, len(hdul)):
                    file = hdul[i]
                    hdr = file.header
                    img = file.data
                    hdr['e1'] = rd.uniform(0.5, 0.9)
                    hdr['e2'] = rd.uniform(0.5, 0.9)
                    rotated_data = MakeLens(thetaE = hdr['theta_E'],
                                            e1 = hdr['e1'],
                                            e2 = hdr['e2'],
                                            gamma1 = hdr['gamma1'],
                                            gamma2 = hdr['gamma2'],
                                            center_x = hdr['center_x'],
                                            center_y = hdr['center_y'])
                    
                    new_hdu = fits.ImageHDU(rotated_data, header = hdr)
                    hdul.append(new_hdu)
                hdul.flush()
        except FileNotFoundError:
            print(f'File {self.fits_name} not found.')

    def Augment_Data(self, angle):
        try:
            with fits.open(self.fits_name, mode = 'update') as hdul:
                for i in range(1, len(hdul)):
                    file = hdul[i]
                    hdr = file.header
                    img = file.data
                    hdr['e1'], hdr['e2'], hdr['gamma1'], hdr['gamma2'] = self.Rotate_Parameters(hdr['e1'],
                                                                                                hdr['e2'],
                                                                                                hdr['gamma1'],
                                                                                                hdr['gamma2'],
                                                                                                angle = angle)
                    rotated_data = MakeLens(thetaE = hdr['theta_E'],
                                            e1 = hdr['e1'],
                                            e2 = hdr['e2'],
                                            gamma1 = hdr['gamma1'],
                                            gamma2 = hdr['gamma2'],
                                            center_x = hdr['center_x'],
                                            center_y = hdr['center_y'])
                    
                    new_hdu = fits.ImageHDU(rotated_data, header = hdr)
                    hdul.append(new_hdu)
                hdul.flush()
        except FileNotFoundError:
            print(f'File {self.fits_name} not found.')

    # Genera la base de datos para entrenamiento y validación
    def Train_and_Val_Images(self):
        try:
            with fits.open(self.fits_name) as hdul:
                self.train_lbs = []
                self.train_images = []

                for _ in range(self.total_images):
                    idx = np.random.randint(0,self.total_images)
                    file = hdul[idx+1]
                    hdr = file.header
                    file_name = hdr['NAME']
                    img = file.data
                    # img_resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
                    self.train_lbs.append([hdr[label] for label in self.labels])
                    self.train_images.append(np.asarray(np.log10(img)))

                self.train_images, self.train_lbs = np.array(self.train_images), np.array(self.train_lbs)
        
        except FileNotFoundError:
            print(f"File {self.fits_name} not found.")
    
    # Se entrena el modelo
    def Train_and_Val(self, epochs, device, percentage):
        
        if device == 'yes':
            tf.config.set_visible_devices([],'GPU')
        
        train_df, test_df, train_labels, test_labels = train_test_split(self.train_images, self.train_lbs, test_size = 0.2, random_state = 42, shuffle = True)
        pcg = percentage*len(train_df)//100
        val_df, val_labels = train_df[-pcg:], train_labels[-pcg:]
        train_df, train_labels = train_df[:-pcg], train_labels[:-pcg]
        
        print(f'Imágenes de entrenamiento: {len(train_df)}')
        print(f'Imágenes de validación: {len(val_df)}')
        print(f'Imágenes de prueba: {len(test_df)}')

        early_stopping = EarlyStopping(monitor = 'val_loss', start_from_epoch = 4, patience = 3)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, min_lr = 1e-5)
        optimizer = Nadam(learning_rate = 1e-4) # 'adam', 'sgd', 'test ema momentum'

        def weighted_mse_loss(weights):

            weights = tf.constant(weights, dtype = tf.float32)

            def loss(y_true, y_pred):
                squared_diff = tf.square(y_true - y_pred)
                weighted_squared_diff = squared_diff * weights
                return tf.reduce_mean(weighted_squared_diff)
            return loss

        weights = [2.9, 1.0, 1.5, 1.5, 0.5, 0.5]
        loss_fn = weighted_mse_loss(weights)
        #self.model = hybrid_model.Hybird_Model(input_shape = self.input_shape, classes = self.classes)
        self.model = alexnet_informed.AlexNet(input_shape = self.input_shape, classes = self.classes)
        
        self.model.compile(optimizer = optimizer, 
                           loss = loss_fn,#{'Decoder':'mse', 'Regressor':'mse'},
                           metrics = ['mae']) 

        self.history = self.model.fit(train_df,
                                      train_labels, #[train_df, train_labels]
                                      epochs = epochs,
                                      validation_data = (val_df, val_labels), # [val_df, val_labels]
                                      callbacks = [early_stopping, reduce_lr], 
                                      batch_size = 32)
        self.Plot_Metrics('mae')
        self.Plot_Metrics('loss')

        test_n = 5000
        test_loss, test_mae = self.model.evaluate(test_df[:test_n], test_labels[:test_n], batch_size = 128)
        print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
        
    def Save_model(self):
        self.model.save('./cnn_model/my_model_informed.keras')

    def Plot_Metrics(self, metric):
        plt.figure()
        plt.plot(self.history.history[f'{metric}'], label = f'Training {metric}', c = 'k', lw = 0.8)
        plt.plot(self.history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
        plt.title(metric.upper())
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'{metric.lower()}.png')
        plt.close()
    
    # Se generan las imágenes y archivos FITS
    def Generate_Images(self):
        #self.__dict__.update(kwargs)
        for i in tqdm(range(self.total_images)):
            f = rd.uniform(0,1.)
            deg = 30
            pa = deg/180*np.pi
            self.sigmav = 200
            self.zl = rd.uniform(0.5,1.0)
            self.zs = rd.uniform(1.0,2.)
            self.co = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
            dl = self.co.angular_diameter_distance(self.zl)
            ds = self.co.angular_diameter_distance(self.zs)
            dls = self.co.angular_diameter_distance_z1z2(self.zl, self.zs)
            y1, y2 = 0, 0
            SIE = sie_lens(self.co, zl = self.zl, zs = self.zs, sigmav = self.sigmav, f = f, pa = pa)
            x, phi = SIE.phi_ima(y1,y2)
            gamma1, gamma2 = SIE.gamma(x, phi)
            e1, e2 = (1 - f)/(1 + f)*np.cos(2*pa), (1 - f)/(1 + f)*np.sin(2*pa)
            thetaE = 1e6*(4.0*np.pi*self.sigmav**2/c**2*dls/ds*180.0/np.pi*3600.0).value

            lss.makelens(n = i,
                         f = f,
                         thetaE = thetaE,
                         e1 = e1,
                         e2 = e2,
                         gamma1 = gamma1[1],
                         gamma2 = gamma2[1],
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
        #if augment == True:
            #self.Augment_Data_Special()
        #    self.Augment_Data(30)
        #    self.Augment_Data(270)

Lens_instance = Lens(total_images = 80000)

if args.database:
    Lens_instance.Generate_Images()
    Lens_instance.Save_FITS(True)

if args.show:
    Lens_instance.Examples()

if args.summary:
    Lens_instance.Generate_Summary()

if args.train:
    Lens_instance.Train_and_Val_Images()
    Lens_instance.Train_and_Val(int(args.train), device = 'no', percentage = 25)

if args.save:
    Lens_instance.Save_model()
