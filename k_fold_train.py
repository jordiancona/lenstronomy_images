

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from models import alexnet, alexnet_original
from tensorflow.keras.optimizers import Adam, Nadam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from astropy.io import fits
import os
import random
import configparser
import sys
import argparse
plt.rc('axes', labelsize = 15)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# PARAMETROS
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
PRUEBA = int(main_config['CONFIG']['prueba'])
CLASSES = int(main_config['MODEL']['classes'])
MAIN_PATH = main_config['PATHS']['main_path']
MODEL_PATH = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
TFRECORD_PATH_TRAIN = main_config['PATHS']['tfrecords_path_train']
TFRECORD_PATH_TEST = main_config['PATHS']['tfrecords_path_test']
LEARNING_RATE = float(main_config['MODEL']['learning_rate'])
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNELS = int(main_config['MODEL']['channels'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
DROPOUTS = [float(item.strip()) for item in main_config['MODEL']['dropouts'].split(',')]
BATCH_SIZE = int(main_config['MODEL']['batch_size'])
EPOCHS = int(main_config['MODEL']['epochs'])
TRAIN_SPLIT = float(main_config['MODEL']['train_split']) # p.ej. 0.8 (para 80% train+val)
VAL_SPLIT = float(main_config['MODEL']['val_split'])   # p.ej. 0.2 (para 20% de validación del set de train+val)
N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])

parser = argparse.ArgumentParser()
parser.add_argument('--model','-m', type = str, default = 'alexnet', help = 'Model to use (alexnet or alexnet_original)')
args = parser.parse_args()
ARCHITECTURE = args.model.lower()
if ARCHITECTURE not in ['alexnet', 'alexnet_original']:
    print(f"{RED}Invalid model architecture specified. Use 'alexnet' or 'alexnet_original'.{ENDC}")
    sys.exit(1)

# SEED
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def get_weighted_loss(loss_weights=None, num_outputs=4):
    '''
    Función de pérdida ponderada para múltiples salidas.
    '''
    if loss_weights is None:
        loss_weights = [1.0] * num_outputs
        if num_outputs >= 2:
            loss_weights[-2] = 3.0
            loss_weights[-1] = 3.0
    
    def weighted_mse(y_true, y_pred):
        total_loss = 0

        for i in range(num_outputs):
            true_val = y_true[:, i]  # Shape: (batch,)
            pred_val = y_pred[:, i]  # Shape: (batch,)
            
            mse = tf.reduce_mean(tf.square(true_val - pred_val))
            total_loss += loss_weights[i] * mse
        
        return total_loss
    return weighted_mse

def Plot_Metrics(history, metric, path, fold_idx):
    ''' Plots metrics guardando el indice del fold '''
    plt.figure()
    try:
        plt.plot(history.history[f'{metric}'], label = f'Training {metric}', c = 'k', lw = 0.8)
        plt.plot(history.history[f'val_{metric}'], label = f'Validation {metric}', c = 'r', lw = 0.8)
        plt.title(f'{metric.upper()} - FOLD {fold_idx}')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(path + f'{metric.lower()}_fold_{fold_idx}.png')
        plt.close()
    except KeyError:
        print(f"{RED}Metric {metric} not found in history{ENDC}")

def parse_tfrecord(example_proto):
    '''
    Function to parse the TFRecord file.
    '''
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        'pa_l': tf.io.FixedLenFeature([], tf.float32),
        'center_x': tf.io.FixedLenFeature([], tf.float32),
        'center_y': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
    }
    
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed_example['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)

    # Normalization to [0, 1]
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) + 1e-6)

    label = tf.stack([parsed_example[label] for label in LABELS], axis = 0)
    return image, label

def count_tfrecord_examples(tfrecord_files):
    '''
    Counts the number of examples in a TFRecord file.
    '''
    total = 0
    for file in tfrecord_files:
        total += sum(1 for _ in tf.data.TFRecordDataset(file))
    return total

def load_tfrecord_dataset(tfrecord_files, batch_size, shuffle=False, repeat=False):
    '''
    Load and process the TFRecord dataset.
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 1000, seed = 42, reshuffle_each_iteration = True)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    return dataset

def main():
    try:
        print(f'{YELLOW}Loading datasets from {TFRECORD_PATH_TRAIN} and {TFRECORD_PATH_TEST}{ENDC}\n')
        train_tfrecord_files = sorted([os.path.join(TFRECORD_PATH_TRAIN, file) for file in os.listdir(TFRECORD_PATH_TRAIN)if file.endswith('.tfrecord')])
        test_tfrecord_files = sorted([os.path.join(TFRECORD_PATH_TEST, file)for file in os.listdir(TFRECORD_PATH_TEST)if file.endswith('.tfrecord')])

        # Counts the number of examples
        num_train_total = count_tfrecord_examples(train_tfrecord_files)
        num_test_total = count_tfrecord_examples(test_tfrecord_files)
        print(f"\nTraining examples: {num_train_total * TRAIN_SPLIT}")
        print(f"Validation examples: {num_train_total * VAL_SPLIT}")
        print(f"Test examples: {num_test_total}\n")

        # Divide train into train and validation sets
        steps_per_epoch = num_train_total // BATCH_SIZE
        validation_steps = int(num_train_total * VAL_SPLIT // BATCH_SIZE)
        
        num_train = int(num_train_total * TRAIN_SPLIT)
        num_val = num_train_total - num_train

        print(f"\nsteps_per_epoch = {steps_per_epoch}")
        print(f"validation_steps = {validation_steps}\n")
        
        # Datasets
        full_ds = tf.data.TFRecordDataset(train_tfrecord_files).map(parse_tfrecord)

        test_dataset = load_tfrecord_dataset(test_tfrecord_files, BATCH_SIZE).take(10)

        for fold in range(N_FOLDS):
            if N_FOLDS == 1:
                print(f"{YELLOW}Training without K-Fold (N_FOLDS=1){ENDC}")
                val_dataset = load_tfrecord_dataset(train_tfrecord_files, BATCH_SIZE, shuffle=False, repeat=False)
                train_dataset = load_tfrecord_dataset(train_tfrecord_files, BATCH_SIZE, shuffle=True, repeat=True)
                steps_per_epoch = num_train_total // BATCH_SIZE
                validation_steps = num_train_total // BATCH_SIZE
                fold = 0 # Para mantener consistencia en nombres de archivos
            else:
                print(f"\n{CYAN}========================================{ENDC}")
                print(f"{CYAN}       TRAINING FOLD {fold + 1}/{N_FOLDS}       {ENDC}")
                print(f"{CYAN}========================================{ENDC}")

                #Limpiar sesión de Keras
                tf.keras.backend.clear_session()

                # FOLDS para validar
                # Split usando .shard()
                val_dataset_raw = full_ds.shard(num_shards=N_FOLDS, index=fold)

                # Entrenamiento
                train_dataset_raw = None
                for i in range(N_FOLDS):
                    if i != fold:
                        shard = full_ds.shard(num_shards=N_FOLDS, index=i)
                        if train_dataset_raw is None:
                            train_dataset_raw = shard
                        else:
                            train_dataset_raw = train_dataset_raw.concatenate(shard)

                # 3. Calcular pasos (Steps)
                # Aproximación de tamaños
                num_val_fold = num_train_total // N_FOLDS
                num_train_fold = num_train_total - num_val_fold

                steps_per_epoch = num_train_fold // BATCH_SIZE
                validation_steps = num_val_fold // BATCH_SIZE

                print(f"Train samples (approx): {num_train_fold} | Steps: {steps_per_epoch}")
                print(f"Val samples (approx):   {num_val_fold} | Steps: {validation_steps}")

                # 4. Procesamiento final del pipeline (Shuffle, Batch, Prefetch)
                # IMPORTANTE: Shuffle solo en train
                train_dataset = (train_dataset_raw
                    .shuffle(buffer_size=1000, seed=seed_value)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                    .repeat() # Necesario porque pasamos steps_per_epoch
                )

                val_dataset = (val_dataset_raw
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                )

                # 5. Definir Callbacks (Nuevos para cada fold)
                #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)

                # Modelo cargado por cada FOLD
                match ARCHITECTURE:
                    case 'alexnet':
                        dp1, dp2 = DROPOUTS
                        model = alexnet.AlexNet(input_shape=INPUT_SHAPE, classes=CLASSES, dp1=dp1, dp2=dp2)
                        RESULTS_PATH = os.path.join(MAIN_PATH, f"alexnet/")
                        os.makedirs(RESULTS_PATH, exist_ok=True)
                    case 'alexnet_original':
                        model = alexnet_original.AlexNet_original(input_shape=INPUT_SHAPE, classes=CLASSES, dp1=0.5, dp2=0.5)
                        RESULTS_PATH = os.path.join(MAIN_PATH, f"alexnet_original/")
                        os.makedirs(RESULTS_PATH, exist_ok=True)

                weights = [1.0, 1.0, 3.0, 3.0]
                loss_fn = get_weighted_loss(weights)
                optimizer = Nadam(learning_rate = LEARNING_RATE)

                model.compile(optimizer=optimizer,
                            loss=loss_fn,
                            metrics=['mae', tf.metrics.MeanAbsolutePercentageError()])

                # 7. Entrenamiento
                start = time.time()
                history = model.fit(
                    train_dataset,
                    validation_data = val_dataset,
                    epochs = EPOCHS,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps,
                    callbacks=[reduce_lr], # Agregué early_stopping aquí
                    verbose=1
                )
                end = time.time()

                # 8. Guardado de Resultados Específicos del Fold

                # Guardar Historial
                history_df = pd.DataFrame(history.history)
                history_df.to_csv(os.path.join(RESULTS_PATH, f'history_fold_{fold+1}.csv'), index=False)

                # Guardar Gráficas
                Plot_Metrics(history, 'mae', RESULTS_PATH, fold+1)
                Plot_Metrics(history, 'loss', RESULTS_PATH, fold+1)

                # Guardar Modelo para el Ensamble
                model_name = f'alexnet_fold_{fold+1}.keras'
                model.save(os.path.join(RESULTS_PATH, model_name))
                print(f"{GREEN}Model saved: {model_name}{ENDC}")
                print(f"Training time fold {fold+1}: {(end - start)/60:.2f} min")

            print(f"\n{GREEN}All {N_FOLDS} folds completed successfully!{ENDC}")

    except Exception as e:
        # Imprime la línea del error para facilitar debugging
        print(f'{RED}Error: in line {sys.exc_info()[2].tb_lineno} - {e}{ENDC}')
        raise e # Re-lanzar para ver el traceback completo si es necesario

if __name__ == '__main__':
    main()
