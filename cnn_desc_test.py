
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
import configparser

plt.rc('axes', labelsize = 15)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)

# PARAMETROS
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
CLASSES = int(main_config['MODEL']['classes'])
TOTAL_IMAGES = int(main_config['MODEL']['total_images'])
MAIN_PATH = main_config['PATHS']['main_path']
MODEL_PATH = main_config['PATHS']['model_path']
FITS_PATH = main_config['PATHS']['fits_path'] # Imágenes de 200 x 200
FITS_FILE = main_config['PATHS']['fits_file'] # Si se guarda un archivo FITS con todas las imágenes
LEARNING_RATE = float(main_config['MODEL']['learning_rate'])
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNLES = int(main_config['MODEL']['channels'])
DROPOUTS = [float(item.strip()) for item in main_config['MODEL']['dropouts'].split(',')]
BATCH_SIZE = int(main_config['MODEL']['batch_size'])
EPOCHS = int(main_config['MODEL']['epochs'])
TRAIN_SPLIT = float(main_config['MODEL']['train_split'])
VAL_SPLIT = float(main_config['MODEL']['val_split'])
PRUEBA = int(main_config['MODEL']['prueba'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNLES)

losses = []
val_losses = []
maes = []
val_maes = []

# SEMILLA
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def weighted_mse_loss(weights):
    '''
    Función de pérdida MSE ponderada.
    '''
    weights = tf.constant(weights, dtype = tf.float32)

    def loss(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = squared_diff * weights
        return tf.reduce_mean(weighted_squared_diff)
    return loss

def adaptative_weighted_mse():
    '''
    Función de pérdida MSE ponderada adaptativa basada en la varianza de las etiquetas verdaderas.
    '''
    def loss(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        var = tf.math.reduce_mean(tf.square(y_true -tf.reduce_mean(y_true, axis = 0)), axis = 0)
        weights = 1.0/(var + 1e-8)
        weighted_squared_diff = weights*squared_diff
        return tf.reduce_mean(weighted_squared_diff)
    return loss

def Plot_Metrics(history, metric, path):
    '''
    Plots and saves a graph of training and validation metrics over epochs.
    '''
    plt.figure()
    plt.plot(history.history[f'{metric}'], label = f'Training {metric}', c = 'k', lw = 0.8)
    plt.plot(history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
    plt.title(metric.upper())
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(path + f'{metric.lower()}_{PRUEBA}.png')
    plt.close()

def plot_training_results(metrics, metric_names, path):
    '''
    Plots and saves graphs for training and validation metrics.
    '''
    for metric_name in metric_names:
        dp1, dp2 = DROPOUTS
        plt.plot(metrics['train'][metric_name][n], lw=0.8, label=f'd1: {dp1} d2: {dp2}')
        plt.title(f'{metric_name.upper()} through training')
        plt.gca().set(xlabel='epoch', ylabel=metric_name)
        plt.legend()
        plt.savefig(path + f'{metric_name.lower()}_training_{PRUEBA}.png')
        plt.close()

        dp1, dp2 = DROPOUTS
        plt.plot(metrics['val'][metric_name][n], lw=0.8, label=f'd1: {dp1} d2: {dp2}')
        plt.title(f'{metric_name.upper()} through validation')
        plt.gca().set(xlabel='epoch', ylabel=metric_name)
        plt.legend()
        plt.savefig(path + f'{metric_name.lower()}_validation_{PRUEBA}.png')
        plt.close()

def main():
    try:
        raw_images = []
        train_lbs = []
        train_images = []
        for fits_file in tqdm(os.listdir(FITS_PATH), desc = 'Loading FITS files'):

            hdul = fits.open(FITS_PATH + fits_file)
            idx = np.random.randint(0,TOTAL_IMAGES)
            file = hdul[1]
            hdr = file.header
            img = file.data.astype(np.float32)
        
            raw_images.append(img)
            train_lbs.append([hdr[label] for label in LABELS])

        raw_images = np.array(raw_images)

        epsilon = 1e-6  # valor pequeño para evitar log(0)
        log_images = np.maximum(raw_images, 0) + epsilon # np.log10

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
        print(f"File {FITS_PATH} not found.")

    X_train, X_test, y_train, y_test = train_test_split(train_images, train_lbs, test_size = TRAIN_SPLIT, random_state = 42, shuffle = True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = TEST_SPLIT, random_state = 42)

    early_stopping = EarlyStopping(monitor = 'val_loss', start_from_epoch = 4, patience = 3)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, min_lr = 1e-5)

    # Se definen los parámetros y entrena el modelo
    dp1, dp2 = DROPOUTS
    print(f'--------Inicia prueba--------\n')
    print(f'Dropout 1: {dp1}, Dropout 2: {dp2}\n')
    print(f'Input Shape: {INPUT_SHAPE}\n')
    print(f'Epochs: {EPOCHS}\n')
    print(f'Model path: {path}\n')
    print(f'Learning Rate: {LEARNING_RATE}\n')

    weights = [1.0, 1.0, 2.5, 2.5]
    loss_fn = weighted_mse_loss(weights)
    optimizer = Nadam(learning_rate = LEARNING_RATE) # Optimizador y LR

    model = alexnet.AlexNet(input_shape = INPUT_SHAPE, classes = CLASSES, dp1 = dp1, dp2 = dp2)
    model.compile(optimizer = optimizer, loss = loss_fn, metrics = ['mae'])

    print(f'Imágenes de entrenamiento: {len(X_train)}')
    print(f'Imágenes de validación: {len(X_val)}')
    print(f'Imágenes de prueba: {len(X_test)}')

    start = time.time()
    history = model.fit(X_train,
                        y_train,
                        epochs = EPOCHS,
                        validation_data = (X_val, y_val),
                        callbacks = [reduce_lr],
                        batch_size = BATCH_SIZE)
    
    end = time.time()
    train_time = end - start

    Plot_Metrics(history, 'mae', MODEL_PATH)
    Plot_Metrics(history, 'loss', MODEL_PATH)

    losses.append(history.history['loss'])
    maes.append(history.history['mae'])
    val_losses.append(history.history['val_loss'])
    val_maes.append(history.history['val_mae'])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL_PATH + f'training_history_{n+1}.csv', index = False)

    start = time.time()
    test_n = 5000
    test_loss, test_mae = model.evaluate(X_test[:test_n], y_test[:test_n], batch_size = 64)
    end = time.time()
    test_time = end - start
    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    train_time = int(train_time // 60) if train_time > 60 > 0 else train_time
    test_time = int(test_time // 60) if test_time > 60 > 0 else test_time
    print(f'training time: {train_time} min - test_time: {test_time:.4f} seconds')
    model.save(MODEL_PATH + f'alexnet_paper_{PRUEBA}.keras')

# Call the function to plot results
metrics = {'train': {'mae': maes, 'loss': losses},
           'val': {'mae': val_maes, 'loss': val_losses}
           }

plot_training_results(metrics, ['mae', 'loss'], MODEL_PATH)

if __name__=='__main__':
    main()
