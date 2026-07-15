
import numpy as np
import random
import tensorflow as tf
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.table import Table
import multiprocessing
import configparser
from argparse import ArgumentParser
from tqdm import tqdm
import subprocess
import os

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
NUM_PIX = main_config.getint('MODEL', 'num_pix')
CHANNLES = main_config.getint('MODEL', 'channels')
DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
TRAIN_OUTPUT_DIR = main_config['PATHS']['tfrecords_path_train']
TEST_OUTPUT_DIR = main_config['PATHS']['tfrecords_path_test']
IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNLES)
TOTAL_IMAGES = main_config.getint('MODEL', 'total_images')
TEST_IMAGES = main_config.getint('MODEL', 'test_images')

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_tfrecord_batch(lens_data_batch, output_path, start_idx = 0):
    with tf.io.TFRecordWriter(output_path) as writer:
        for lens_data in lens_data_batch:
            try:
                image_data, params = generate_lens_image(lens_data)
                image_data = image_data.astype(np.float32)
                image_bytes = image_data.tobytes()

                image_idx = start_idx + 1
                feature = {
                    'image_idx': _int_feature(image_idx),
                    'image': _bytes_feature(image_bytes),
                    'theta_E': _float_feature(params['thetaE']),
                    'f_axis': _float_feature(params['f_l']),
                    'f_s': _float_feature(params['f_s']),
                    'e1': _float_feature(params['e1']),
                    'e2': _float_feature(params['e2']),
                    'center_x': _float_feature(params['center_x']),
                    'center_y': _float_feature(params['center_y']),
                    're_s': _float_feature(params['re_s']),
                    'pa_s': _float_feature(params['pa_s']),
                    're_l': _float_feature(params['re_l']),
                    'pa_l': _float_feature(params['pa_l']),
                    'e1_s': _float_feature(params['e1_s']),
                    'e2_s': _float_feature(params['e2_s']),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except Exception as e:
                print(f"Error processing lens data: {e}")

    print(f'Wrote {len(lens_data_batch)} lenses to {output_path}')

def generate_lens_image(lens_data):
    '''
    Genera la imagen de una lente gravitacional y devuelve los datos de la imagen y los parámetros.
    '''

    band_names = ['g', 'r', 'i', 'z']

    # zero point for each band of the source galaxy
    zp_source = {
        'g': lens_data['mag_g_s0'],
        'r': lens_data['mag_r_s0'],
        'i': lens_data['mag_i_s0'],
        'z': lens_data['mag_z_s0'],
    }

    # zero point for each band of the lens galaxy
    zp_lens = {
        'g': lens_data['mag_g_l'],
        'r': lens_data['mag_r_l'],
        'i': lens_data['mag_i_l'],
        'z': lens_data['mag_z_l'],
    }

    # Exposure time for Wide-Field (150s*2)
    EXPOSURE_TIME = 300.0  # seconds

    pa_l = np.deg2rad(10)
    pa_s = np.deg2rad(lens_data['pa_s0'])
    f_l = lens_data['q_l']
    f_s = lens_data['q_s0']
    thetaE = lens_data['thetaE_s0']
    x_s, y_s = lens_data['x_s0'], lens_data['y_s0']
    re_s = lens_data['re_s0']
    re_l = lens_data['re_l']

    # Ellipticities
    e_l = (1 - f_l) / (1 + f_l)
    e_s = (1 - f_s) / (1 + f_s)
    e1_l, e2_l = e_l * np.cos(2 * pa_l), e_l * np.sin(2 * pa_l)
    e1_s, e2_s = e_s * np.cos(2 * pa_s), e_s * np.sin(2 * pa_s)

    # Crear grid de coordenadas
    x, y = np.meshgrid(np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX),
                       np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX))

    lens_model_list = ['SIE']
    lens_model = LensModel(lens_model_list)

    lens_light_model_list = ['SERSIC_ELLIPSE']
    lens_light_model = LightModel(lens_light_model_list)

    source_light_model_list = ['SERSIC_ELLIPSE']
    source_light_model = LightModel(source_light_model_list)

    lens_kwargs = [{
        'theta_E': thetaE,
        'e1': e1_l,
        'e2': e2_l,
        'center_x': 0.0,
        'center_y': 0.0
    }]

    stacked_image = np.zeros((NUM_PIX, NUM_PIX), dtype=np.float32)

    for band in band_names:
        col_name = f'mag_{band}_l'
        mag_l = lens_data[col_name]
        mag_s = lens_data[f'mag_{band}_s0']

        flux_l = EXPOSURE_TIME * 10 ** (-0.4 * (mag_l - zp_lens[band]))
        flux_s = EXPOSURE_TIME * 10 ** (-0.4 * (mag_s - zp_source[band]))
        
        lens_light_kwargs = [{
            'amp': flux_l,
            'R_sersic': re_l,
            'n_sersic': 4.0,
            'e1': e1_l,
            'e2': e2_l,
            'center_x': 0.0,
            'center_y': 0.0
        }]

        source_light_kwargs = [{
            'amp': flux_s,
            'R_sersic': re_s,
            'n_sersic': 2.0,
            'e1': e1_s,
            'e2': e2_s,
            'center_x': x_s,
            'center_y': y_s
        }]

        flux_image_lens = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
        flux_image_lens_reshaped = flux_image_lens.reshape(NUM_PIX, NUM_PIX)

        x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)

        flux_image_source = source_light_model.surface_brightness(x_lensed, y_lensed, source_light_kwargs)        
        flux_image_source_reshaped = flux_image_source.reshape(NUM_PIX, NUM_PIX)

        stacked_image += flux_image_lens_reshaped + flux_image_source_reshaped
    
    stacked_image /= len(band_names)

    total_image = np.expand_dims(stacked_image, axis=-1)  # Add a channel dimension to stacked_image

    params = {'f_l': f_l,
              'f_s': f_s,
              'thetaE': thetaE,
              'e1': e1_l,
              'e2': e2_l,
              'center_x': x_s,
              'center_y': y_s,
              're_s': re_s,
              'pa_s': pa_s,
              're_l': re_l,
              'pa_l': pa_l,
              'e1_s': e1_s,
              'e2_s': e2_s,
              }

    return total_image, params

def process_lens_batch(args):
    '''
    Procesa un batch de lentes y guarda las imágenes en un archivo TFRecord.
    '''
    batch_idx, lens_data_batch, output_dir, global_start = args
    output_path = os.path.join(output_dir, f'lens_data_batch_{batch_idx + 1}.tfrecord')
    write_tfrecord_batch(lens_data_batch, output_path, start_idx = global_start)

def main():
    '''
    Main function. Parses arguments and manages multiprocessing for lens image generation.
    '''
    parser = ArgumentParser()
    parser.add_argument('-np', '--num_processes', type=int, default=8, help='Number of processes for multiprocessing.')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size for TFRecords.')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    NUM_PROCESSES = main_config.getint('PARALLEL', 'cores')
    FILE = main_config['PATHS']['csv_file']

    try:
        lens_samples = Table.read(FILE)
        print(f"\033[32mFile {FILE} loaded.\033[0m")
    except FileNotFoundError:
        print(f"\033[31mError: The file {FILE} was not found.\033[0m")
        return

    # Dividir en conjuntos de entrenamiento y prueba
    train_samples = lens_samples[:TOTAL_IMAGES]
    test_samples = lens_samples[TOTAL_IMAGES:TOTAL_IMAGES+TEST_IMAGES]

    train_tasks = [(batch_idx, train_samples[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE], TRAIN_OUTPUT_DIR, batch_idx * BATCH_SIZE)
                   for batch_idx in range((len(train_samples) + BATCH_SIZE - 1) // BATCH_SIZE)]

    test_tasks = [(batch_idx, test_samples[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE], TEST_OUTPUT_DIR, 70000 + batch_idx * BATCH_SIZE)
                  for batch_idx in range((len(test_samples) + BATCH_SIZE - 1) // BATCH_SIZE)]

    print(f"\033[33mStarting a Pool with {NUM_PROCESSES} processes.\033[0m")

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_lens_batch, train_tasks)
        pool.map(process_lens_batch, test_tasks)

    print('\033[32m¡Process completed!\033[0m')

if __name__ == "__main__":
    main()
