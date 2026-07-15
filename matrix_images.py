
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
from argparse import ArgumentParser
from tqdm import tqdm
import os

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# Function to load configuration from INI file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
NUM_PIX = main_config.getint('MODEL', 'num_pix')
CHANNLES = main_config.getint('MODEL', 'channels')
IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNLES)
TRAIN_OUTPUT_DIR = main_config['PATHS']['tfrecords_path_train']
MAIN_PATH = main_config['PATHS']['main_path']
CMAP = 'gray'

parser = ArgumentParser()
parser.add_argument('--rows', type = int, default = 3, help = 'Number of rows in the image matrix')
parser.add_argument('--cols', type = int, default = 6, help = 'Number of columns in the image matrix')
args = parser.parse_args()

ROWS = args.rows
COLS = args.cols

# Functions to load TFRecord dataset
def load_tfrecord_dataset(path, batch_size=1):
    files = tf.data.Dataset.list_files(os.path.join(path, '*.tfrecord'))
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=10, block_length=1)

    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.map(parse_tfrecord).batch(batch_size)
    return dataset

#Function to parse TFRecord
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
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed_example['image'], tf.float32)
    image = tf.reshape(image, IMGSHAPE)
    label = tf.stack([parsed_example[label] for label in LABELS], axis = 0)

    params = {
        'image_idx': parsed_example['image_idx'],
        'thetaE': parsed_example['theta_E'],
        'e1': parsed_example['e1'], 
        'e2': parsed_example['e2'],
        }
    return image, label, params

def main():

    fig = plt.figure(figsize=(COLS*2, ROWS*2))#, facecolor='black')
    gs = plt.GridSpec(ROWS, COLS, wspace = 0.01, hspace = 0.01)

    dataset = load_tfrecord_dataset(TRAIN_OUTPUT_DIR, batch_size = 1)
    iterator = iter(dataset)

    count = 0
    max_attempts = 10000 # Evitar loop infinito si no hay suficientes datos
    attempts = 0

    # UMBRAL DE FILTRADO: Ajusta este valor según tus datos.
    # Valores más altos = arcos más grandes y visibles.
    THETA_E_THRESHOLD = 1.8
    MIN_ELLIPTICITY = 0.2

    with tqdm(total=ROWS*COLS, desc=f"{CYAN}Searching for lenses and drawing vectors{ENDC}") as pbar:
        while count < ROWS * COLS and attempts < max_attempts:
            try:
                # Randomly select an image
                image_tensor, label, params = next(iterator)
                
                # Convertimos a valor numérico
                theta_val = params['thetaE'].numpy()[0] 
                
                # fillter condition
                ellipticity = np.sqrt(np.square(params['e1'].numpy()[0]) + np.square(params['e2'].numpy()[0]))
                if theta_val > THETA_E_THRESHOLD and ellipticity > MIN_ELLIPTICITY:
                    
                    image = image_tensor[0].numpy()
                    ax = plt.subplot(gs[count])
                    
                    if image.shape[-1] == 1:
                        ax.imshow(image[:,:,0], vmin=image.min(), vmax=image.max(), cmap=CMAP)
                    else:
                        ax.imshow(image)
                    
                    ax.axis('off')
                    count += 1
                    pbar.update(1)
                
                attempts += 1
                
            except StopIteration:
                print(f'{RED}The dataset has been exhausted.{ENDC}')
                break

    plt.savefig(MAIN_PATH + f'images_matrix_{CMAP}.pdf', bbox_inches = 'tight', pad_inches = 0.015, facecolor = 'black')

if __name__ == '__main__':
    main()
