
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import R2Score
from models import alexnet, resnet
from tensorflow.keras.optimizers import Adam, Nadam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from astropy.io import fits
from tqdm import tqdm
import os
import random

plt.rc('axes', labelsize = 15)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)

# --- PARAMETROS ---
CLASSES = 4
TOTAL_IMAGES = 50000
FITS_NAME = './csst_catalog/lens_fits_100.fits' # Imágenes de 100 x 100
LEARNING_RATE = 1e-4
labels = ['theta_E','f_axis','e2','e2']
input_dimensions = (100, 100, 1)
dropuots = [(0.3, 0.2), (0.2, 0.2), (0.0, 0.0)]
losses = []
val_losses = []
maes = []
val_maes = []

# --- SEMILLA ---
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def weighted_mse_loss(weights):

    weights = tf.constant(weights, dtype = tf.float32)

    def loss(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = squared_diff * weights
        return tf.reduce_mean(weighted_squared_diff)
    return loss

def Plot_Metrics(history, metric, path, n):
        plt.figure()
        plt.plot(history.history[f'{metric}'], label = f'Training {metric}', c = 'k', lw = 0.8)
        plt.plot(history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
        plt.title(metric.upper())
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(path + f'{metric.lower()}_.png')
        plt.close()

def main():
    try:
        raw_images = []
        train_lbs = []
        train_images = []
        for fits_file in tqdm(os.listdir('./csst_catalog/fits/'), desc = 'Loading FITS files'):

            hdul = fits.open('./csst_catalog/fits/' + fits_file)
            idx = np.random.randint(0,TOTAL_IMAGES)
            file = hdul[0]
            hdr = file.header
            img = file.data.astype(np.float32)
        
            raw_images.append(img)
            train_lbs.append([hdr[label] for label in labels])

        raw_images = np.array(raw_images)

        epsilon = 1e-6  # valor pequeño para evitar log(0)
        log_images = np.log10(np.maximum(raw_images, 0) + epsilon)

        # Calcular estadísticas globales
        per_image_min = np.min(log_images, axis = (1, 2), keepdims = True)
        per_image_max = np.max(log_images, axis = (1, 2), keepdims = True)

        # Normalizar cada imagen individualmente 
        train_images = (log_images - per_image_min) / (per_image_max - per_image_min + 1e-8)
        
        # Agrega la dimensión del canal al final (100, 100) -> (100, 100, 1)
        train_images = train_images[..., np.newaxis]

        train_images, train_lbs = np.array(train_images), np.array(train_lbs)

        print(f'Train Data size :{train_images.shape} \n Train Labels size :{train_lbs.shape}')

    except FileNotFoundError:
        print(f"File {FITS_NAME} not found.")

    train_df, test_df, train_labels, test_labels = train_test_split(train_images, train_lbs, test_size = 0.2, random_state = 42, shuffle = True)
    train_df, val_df, train_labels, val_labels = train_test_split(train_df, train_labels, test_size = 0.2, random_state = 42)

    early_stopping = EarlyStopping(monitor = 'val_loss', start_from_epoch = 4, patience = 3)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, min_lr = 1e-5)

    for n, dpts in enumerate(dropuots):
        dp1, dp2 = dpts
        path = f'./csst_catalog/test1/alexnet_{n+1}/'

        print(f'--------PRUEBA {n+1}--------')

        weights = [1.5, 1.5, 1.5, 1.5]
        loss_fn = weighted_mse_loss(weights)
        optimizer = Nadam(learning_rate = LEARNING_RATE) # Optimizador y LR

        model = alexnet.AlexNet(input_shape = input_dimensions, classes = CLASSES, dp1 = dp1, dp2 = dp2)
        model.compile(optimizer = optimizer, loss = loss_fn, metrics = ['mae'])

        print(f'Imágenes de entrenamiento: {len(train_df)}')
        print(f'Imágenes de validación: {len(val_df)}')
        print(f'Imágenes de prueba: {len(test_df)}')

        start = time.time()
        history = model.fit(train_df,
                            train_labels,
                            epochs = 16,
                            validation_data = (val_df, val_labels),
                            callbacks = [reduce_lr],
                            batch_size = 32)
        end = time.time()
        train_time = end - start
        Plot_Metrics(history, 'mae', path, n)
        Plot_Metrics(history, 'loss', path, n)
        losses.append(history.history['loss'])
        maes.append(history.history['mae'])
        val_losses.append(history.history['val_loss'])
        val_maes.append(history.history['val_mae'])

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(path + f'training_history_{n+1}.csv', index = False)

        start = time.time()
        test_n = 5000
        test_loss, test_mae = model.evaluate(test_df[:test_n], test_labels[:test_n], batch_size = 64)
        end = time.time()
        test_time = end - start
        print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

        train_time = int(train_time // 60) if train_time > 60 > 0 else train_time
        test_time = int(test_time // 60) if test_time > 60 > 0 else test_time
        print(f'training time: {train_time} min - test_time: {test_time:.4f} seconds')
        model.save(path + f'alexnet_paper_{n+1}.keras')
    
    for n, dpts in enumerate(dropuots):
        dp1, dp2 = dpts
        plt.plot(maes[n], lw = 0.8, label = f'd1: {dp1} d2: {dp2}')
    plt.title('MAE through training')
    plt.gca().set(xlabel = 'epoch', ylabel = 'mae')
    plt.legend()
    plt.savefig(path+'maes.png')
    plt.close()

    for n, dpts in enumerate(dropuots):
        dp1, dp2 = dpts
        plt.plot(val_maes[n], lw = 0.8, label = f'd1: {dp1} d2: {dp2}')
    plt.title('MAE through validation')
    plt.gca().set(xlabel = 'epoch', ylabel = 'mae')
    plt.legend()
    plt.savefig(path+'val_maes.png')
    plt.close()

    for n, dpts in enumerate(dropuots):
        dp1, dp2 = dpts
        plt.plot(losses[n], lw = 0.8, label = f'd1: {dp1} d2: {dp2}')
    plt.title('Loss through training')
    plt.gca().set(xlabel = 'epoch', ylabel = 'loss')
    plt.legend()
    plt.savefig(path+'losses.png')
    plt.close()

    for n, dpts in enumerate(dropuots):
        dp1, dp2 = dpts
        plt.plot(val_losses[n], lw = 0.8, label = f'd1: {dp1} d2: {dp2}')
    plt.title('Loss through validation')
    plt.gca().set(xlabel = 'epoch', ylabel = 'loss')
    plt.legend()
    plt.savefig(path+'val_losses.png')
    plt.close()

if __name__=='__main__':
    main()