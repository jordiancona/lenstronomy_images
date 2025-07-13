
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from models import alexnet
from keras.optimizers import Adam, Nadam # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from astropy.io import fits
from tqdm import tqdm

fits_name = './lens_fits_100.fits'
total_images = 50000
labels = ['theta_E','f_axis','e1','e2','gamma1','gamma2']
input_dimensions = (100, 100, 1)
percentage = 20
dropuots = [(0.2, 0.3), (0.1, 0.5), (.05, 0.1), (0.0, 0.0)]
dp1 = sys.argv[1]
dp2 = sys.argv[2]

try:
    with fits.open(fits_name) as hdul:
        raw_images = []
        train_lbs = []
        train_images = []

        for _ in tqdm(range(total_images), desc = 'Cargando imágenes'):
            idx = np.random.randint(0,total_images)
            file = hdul[idx+1]
            hdr = file.header
            img = file.data.astype(np.float32) # Asegurar que sea float
        
            raw_images.append(img)
            train_lbs.append([hdr[label] for label in labels])

        raw_images = np.array(raw_images)

        # --- PASO 2: Aplicar una escala logarítmica segura ---
        # Se evita tomar el log de cero o valores negativos
        epsilon = 1e-6  # Un valor pequeño para evitar log(0)
        log_images = np.log10(np.maximum(raw_images, 0) + epsilon)

        # --- PASO 3: Calcular estadísticas GLOBALES ---
        per_image_min = np.min(log_images, axis=(1, 2), keepdims=True)
        per_image_max = np.max(log_images, axis=(1, 2), keepdims=True)

        # --- PASO 4: Normalizar CADA IMAGEN individualmente ---
        train_images = (log_images - per_image_min) / (per_image_max - per_image_min + 1e-8)
        
        # Agrega la dimensión del canal al final (100, 100) -> (100, 100, 1)
        train_images = train_images[..., np.newaxis]

        train_images, train_lbs = np.array(train_images), np.array(train_lbs)

except FileNotFoundError:
    print(f"File {fits_name} not found.")

def weighted_mse_loss(weights):

            weights = tf.constant(weights, dtype = tf.float32)

            def loss(y_true, y_pred):
                squared_diff = tf.square(y_true - y_pred)
                weighted_squared_diff = squared_diff * weights
                return tf.reduce_mean(weighted_squared_diff)
            return loss

def Plot_Metrics(history, metric):
        plt.figure()
        plt.plot(history.history[f'{metric}'], label = f'Training {metric}', c = 'k', lw = 0.8)
        plt.plot(history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
        plt.title(metric.upper())
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'{metric.lower()}.png')
        plt.close()

train_df, test_df, train_labels, test_labels = train_test_split(train_images, train_lbs, test_size = 0.2, random_state = 42, shuffle = True)
pcg = percentage*len(train_df)//100
train_df, train_labels = train_df[:-pcg], train_labels[:-pcg]
val_df, val_labels = train_df[-pcg:], train_labels[-pcg:]


model = alexnet.AlexNet(input_shape = input_dimensions, classes = 6, dp1 = dp1, dp2 = dp2)
model.summary()

weights = [2.5, 1.0, 1.0, 1.0, 1.5, 1.5]
loss_fn = weighted_mse_loss(weights)
optimizer = Adam(learning_rate = 1e-4)
model.compile(optimizer = optimizer, loss = loss_fn, metrics = ['mae'])

print(f'Imágenes de entrenamiento: {len(train_df)}')
print(f'Imágenes de validación: {len(val_df)}')
print(f'Imágenes de prueba: {len(test_df)}')

early_stopping = EarlyStopping(monitor = 'val_loss', start_from_epoch = 4, patience = 3)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, min_lr = 1e-5)

history = model.fit(train_df,
                    train_labels, #[train_df, train_labels]
                    epochs = 50,
                    validation_data = (val_df, val_labels), # [val_df, val_labels]
                    callbacks = [reduce_lr, early_stopping],
                    batch_size = 32)

Plot_Metrics(history, 'mae')
Plot_Metrics(history, 'loss')

test_n = 5000
test_loss, test_mae = model.evaluate(test_df[:test_n], test_labels[:test_n], batch_size = 64)
print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

model.save('./cnn_model/alexnet_paper.keras')
