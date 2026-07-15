
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import random
import corner
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import configparser
import os
import time

# --- COLOR CODES FOR TERMINAL OUTPUT ---
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# --- 1. CONFIGURATION AND DATA LOADING ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
    MAIN_PATH = main_config['PATHS']['main_path']
    TEST_PATH = main_config['PATHS']['tfrecords_path_test']
    PRUEBA = int(main_config['CONFIG']['prueba'])
    NUM_PIX = int(main_config['MODEL']['num_pix'])
    CHANNELS = int(main_config['MODEL']['channels'])
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    BATCH_SIZE = int(main_config['MODEL']['batch_size'])
    TEST_IMAGES = int(main_config['MODEL']['test_images'])
    N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])
except Exception as e:
    print(f"{RED}Error cargando configuración: {e}{ENDC}")
    MAIN_PATH = './'
    TEST_PATH = './tfrecords/test'
    PRUEBA = 1
    NUM_PIX = 100
    CHANNELS = 1
    LABELS = ['theta_E', 'f_axis', 'e1', 'e2']
    N_FOLDS = 5

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
OUTPUT_DIR = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. AUXILIARY FUNCTIONS FOR TFRECORD ---
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_weighted_loss(loss_weights=None, num_outputs=4):
    if loss_weights is None:
        loss_weights = [1.0] * num_outputs
        if num_outputs >= 2:
            loss_weights[-2] = 3.0
            loss_weights[-1] = 3.0
    
    def weighted_mse(y_true, y_pred):
        total_loss = 0
        n_out = min(num_outputs, y_pred.shape[1])
        for i in range(n_out):
            mse = tf.reduce_mean(tf.square(y_true[:, i] - y_pred[:, i]))
            total_loss += loss_weights[i] * mse
        return total_loss
    return weighted_mse

def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'f_s': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        'center_x': tf.io.FixedLenFeature([], tf.float32),
        'center_y': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        're_l': tf.io.FixedLenFeature([], tf.float32),
        'pa_l': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
        'e1_s': tf.io.FixedLenFeature([], tf.float32),
        'e2_s': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) + 1e-6)
    return image, parsed

def load_tfrecord_dataset(tfrecord_dir, batch_size):
    if not os.path.exists(tfrecord_dir):
        print(f"{RED}Path no encontrado: {tfrecord_dir}{ENDC}")
        return None
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 4. ENSEMBLE AND PREDICTION ---
def mc_ensamble_predictions(models, images_array, num_samples=50):
    '''
    Returns the mean, uncertainty, and full stack of predictions.
    models: list of loaded Keras models
    images_array: numpy array of shape (n_samples, height, width, channels)
    num_samples: number of MC Dropout samples per model
    Returns:
        predictions_mean: (n_samples, n_labels)
        uncertainty: (n_samples, n_labels)
        predictions_stacked: (n_folds, n_samples, n_labels)
    '''
    preds_folds = []
    
    inference_batch = 64 
    
    for i, model in enumerate(models):
        start_time = time.time()
        print(f"Predicting with model {i+1}/{len(models)}...")
        for _ in range(num_samples):
            p = model(images_array, training=True)
            preds_folds.append(p)
        end_time = time.time()
        print(f"{YELLOW}Time taken for model {i+1}: {end_time - start_time:.2f} seconds{ENDC}")
    
    # Stack: (n_folds, n_samples, n_labels)
    predictions_stacked = np.stack(preds_folds, axis=0)
    
    # Mean and standard deviation across all predictions
    predictions_mean = np.mean(predictions_stacked, axis=0) # (n_samples, n_labels)
    uncertainty = np.std(predictions_stacked, axis=0)       # (n_samples, n_labels)

    return predictions_mean, uncertainty, predictions_stacked

# --- LOAD MODELS ---
def main():
    # --- 3. DATA LOADING ---
    print(f'{YELLOW}Loading dataset from {TEST_PATH}{ENDC}')
    inference_batch_size = 64
    dataset = load_tfrecord_dataset(TEST_PATH, inference_batch_size)

    print(f"{YELLOW}Extracting data into memory (be careful with RAM)...{ENDC}")
    all_images = []
    all_parsed = []

    for img_batch, parsed_batch in tqdm(dataset, desc="Loading Data"):
        all_images.append(img_batch.numpy())
        batch_len = img_batch.shape[0]
        keys = parsed_batch.keys()
        for i in range(batch_len):
            single_item = {k: parsed_batch[k][i].numpy() for k in keys}
            all_parsed.append(single_item)

    images_array = np.concatenate(all_images, axis=0)
    print(f'{CYAN}Total Dataset:{ENDC} {len(all_parsed)} samples loaded.')
    print(f'{CYAN}Images shape:{ENDC} {images_array.shape}')

    # --- 4. ENSEMBLE AND PREDICTION ---
    print(f"{YELLOW}Loading models from {OUTPUT_DIR}{ENDC}")
    weights = [1.0, 1.0, 3.0, 3.0]
    custom_loss = get_weighted_loss(weights, num_outputs=len(LABELS))
    models = []

    for i in range(N_FOLDS):
        name = f'alexnet_fold_{i+1}.keras'
        model_path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(model_path):
            try:
                model = load_model(model_path, custom_objects={'loss': custom_loss, 'weighted_mse': custom_loss}, compile=False)
                models.append(model)
            except Exception as e:
                print(f"{RED}Error loading {name}: {e}{ENDC}")
        else:
            print(f"{RED}Model {name} not found.{ENDC}")

    if not models:
        raise ValueError("No models loaded. Check paths.")

    print(f"{YELLOW}Making predictions (returning full stack)...{ENDC}")
    start_time = time.time()
    predictions_mean, uncertainty_total, predictions_stacked = mc_ensamble_predictions(models, images_array, num_samples=50)
    end_time = time.time()
    print(f'{GREEN}Predictions completed. Stack shape: {predictions_stacked.shape}{ENDC}')
    print(f"{YELLOW}Time taken for predictions: {end_time - start_time:.2f} seconds{ENDC}")

    # --- 5. SAVE CSV ---
    print(f"{YELLOW}Generating CSV...{ENDC}")
    rows = []
    for i, parsed in enumerate(all_parsed):
        # Here 'id' might be corrupted, but 'i' is the real row index
        row = {
            'row_idx': i, # Save the real row index
            'original_id': int(parsed.get('image_idx', -1)), # Save the original ID just in case
            # Real values
            'theta_E_true': parsed['theta_E'],
            'f_true': parsed['f_axis'],
            'e1_true': parsed['e1'],
            'e2_true': parsed['e2'],
            
            # Predictions
            'theta_E_pred': predictions_mean[i, 0],
            'f_pred': predictions_mean[i, 1],
            'e1_pred': predictions_mean[i, 2],
            'e2_pred': predictions_mean[i, 3],
            
            # Extra parameters
            'f_s': parsed['f_s'],
            're_s': parsed['re_s'],
            're_l': parsed['re_l'],
            'pa_l': parsed['pa_l'],
            'pa_s': parsed['pa_s'],
            'x_s': parsed['center_x'],
            'y_s': parsed['center_y'],
            'e1_s': parsed['e1_s'],
            'e2_s': parsed['e2_s'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'predictions_vs_real.csv')
    df.to_csv(csv_path, index=False)
    print(f'{GREEN}CSV saved in {csv_path}{ENDC}')

    npy_path = os.path.join(OUTPUT_DIR, 'predictions_stacked_mcdeepensemble.npy')
    np.save(npy_path, predictions_stacked)
    print(f'{GREEN}Predictions stack saved in {npy_path}{ENDC}')
    print(f"{YELLOW}DataFrame and predictions stack saved in {OUTPUT_DIR} for further analysis.{ENDC}")
    
if __name__ == "__main__":
    main()