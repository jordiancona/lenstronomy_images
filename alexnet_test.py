
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
PRUEBA = main_config['CONFIG']['prueba']
CLASSES = int(main_config['MODEL']['classes'])
MAIN_PATH = main_config['PATHS']['main_path']
MODEL_PATH = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
TFRECORD_PATH_TRAIN = main_config['PATHS']['tfrecords_path_train']
TFRECORD_PATH_TEST = main_config['PATHS']['tfrecords_path_test']
LEARNING_RATE = float(main_config['MODEL']['learning_rate'])
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNLES = int(main_config['MODEL']['channels'])
DROPOUTS = [float(item.strip()) for item in main_config['MODEL']['dropouts'].split(',')]
BATCH_SIZE = int(main_config['MODEL']['batch_size'])
EPOCHS = int(main_config['MODEL']['epochs'])
TRAIN_SPLIT = float(main_config['MODEL']['train_split']) # p.ej. 0.8 (para 80% train+val)
VAL_SPLIT = float(main_config['MODEL']['val_split'])   # p.ej. 0.2 (para 20% de validación del set de train+val)
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNLES)

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
        full_train_ds = tf.data.TFRecordDataset(train_tfrecord_files).map(parse_tfrecord)
        train_dataset = (full_train_ds
            .take(int(num_train_total * TRAIN_SPLIT))
            .shuffle(1000, seed=42)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )

        val_dataset = (
            full_train_ds
            .skip(int(num_train_total * TRAIN_SPLIT))
            .take(int(num_train_total * VAL_SPLIT))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = load_tfrecord_dataset(test_tfrecord_files, BATCH_SIZE).take(10)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)

        # Model
        dp1, dp2 = DROPOUTS
        model = alexnet_original.AlexNet_original(input_shape=INPUT_SHAPE, classes=CLASSES)
        
        with open(MODEL_PATH + f'alexnet_summary_{PRUEBA}.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"{YELLOW}Model Summary saved to {MODEL_PATH}alexnet_summary_{PRUEBA}.txt{ENDC}")

        weights = [1.0, 1.0, 3.0, 3.0]
        loss_fn = get_weighted_loss(weights)
        optimizer = Nadam(learning_rate = LEARNING_RATE)

        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['mae', tf.metrics.MeanAbsolutePercentageError()]
                      )

        print(f"\n{YELLOW}Training with {steps_per_epoch} steps and {validation_steps} validation steps{ENDC}\n")

        # Training
        start = time.time()
        history = model.fit(train_dataset,
                            validation_data = val_dataset,
                            epochs = EPOCHS,
                            steps_per_epoch = steps_per_epoch,
                            validation_steps = validation_steps,
                            callbacks=[reduce_lr])
        end = time.time()

        train_time = end - start

        # Evaluation
        test_loss, test_mae, test_mape = model.evaluate(test_dataset)
        print(f'\n{YELLOW}Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}{ENDC}')

        # plotting metrics
        Plot_Metrics(history, 'mae', MODEL_PATH)
        Plot_Metrics(history, 'loss', MODEL_PATH)
        Plot_Metrics(history, 'mean_absolute_percentage_error', MODEL_PATH)

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(MODEL_PATH + f'training_history_{PRUEBA}.csv', index=False)

        model.save(MODEL_PATH + f'ViT_{PRUEBA}.keras')

        print(f"{YELLOW}Training time: {train_time/60:.2f} min{ENDC}")

    except Exception as e:
        print(f'{RED}Error: in line {sys.exc_info()[2].tb_lineno} - {e}{ENDC}')

if __name__ == '__main__':
    main()
