
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import configparser
from argparse import ArgumentParser
from tqdm import tqdm
import os

CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

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
CMAP = "gray_r"

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
    dataset = dataset.map(parse_tfrecord).batch(batch_size)
    return dataset

# --- Función parse_tfrecord MODIFICADA para incluir f_axis ---
def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32), # Aseguramos que esté aquí
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
        # ... [los demás campos se quedan igual] ...
    }
    # (Agrega aquí el resto de los campos de tu función original si son necesarios)
    
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed_example['image'], tf.float32)
    image = tf.reshape(image, IMGSHAPE)
    
    # Extraemos los parámetros necesarios para el texto del gráfico
    params = {
        'image_idx': parsed_example['image_idx'],
        'thetaE': parsed_example['theta_E'],
        'e1': parsed_example['e1'], 
        'e2': parsed_example['e2'],
        'f_axis': parsed_example['f_axis'],
        'center_x': parsed_example['center_x'],
        'center_y': parsed_example['center_y'],
        'pa_l': parsed_example['pa_l'],
    }
    return image, None, params # El label no se usa en este script según tu código

def main():
    # Aumentamos un poco el tamaño para acomodar visualizaciones
    fig = plt.figure(figsize=(COLS*3, ROWS*3)) 
    gs = plt.GridSpec(ROWS, COLS, wspace=0.1, hspace=0.3) 

    dataset = load_tfrecord_dataset(TRAIN_OUTPUT_DIR, batch_size=1)
    iterator = iter(dataset)

    count = 0
    max_attempts = 10000
    attempts = 0

    THETA_E_THRESHOLD = 1.2
    MIN_ELLIPTICITY = 0.2

    # Parámetro para la longitud visual del vector de elipticidad
    # Ajusta esto para que se vea bien (0.5 = mitad del radio de la imagen)
    VECTOR_SCALE = 0.4 * (NUM_PIX / 2) 

    with tqdm(total=ROWS*COLS, desc=f"{CYAN}Searching for lenses and drawing vectors{ENDC}") as pbar:
        while count < ROWS * COLS and attempts < max_attempts:
            try:
                image_tensor, _, params = next(iterator)
                
                # Extracción de valores numéricos
                theta_val = params['thetaE'].numpy()[0] 
                e1_val = params['e1'].numpy()[0]
                e2_val = params['e2'].numpy()[0]
                f_axis_val = params['f_axis'].numpy()[0]
                
                # Centro de la lente (asumiendo que está en arcsec relativos al centro)
                # Convertimos a coordenadas de píxel
                cx_pix = (params['center_x'].numpy()[0] / DELTA_PIX) + (NUM_PIX / 2)
                cy_pix = (params['center_y'].numpy()[0] / DELTA_PIX) + (NUM_PIX / 2)
                
                # --- CÁLCULO DE COMPONENTES VISUALES ---
                # 1. Magnitud de la elipticidad total
                e_total = np.sqrt(e1_val**2 + e2_val**2)
                phi_rad = params['pa_l'].numpy()[0] #0.5 * np.arctan2(e2_val, e1_val)
                phi_deg = 2 * np.degrees(phi_rad)

                # 2. Ángulo de posición (PA) en radianes
                # La fórmula estándar es: 2*phi = arctan2(e2, e1)

                if theta_val > THETA_E_THRESHOLD and e_total > MIN_ELLIPTICITY:
                    image = image_tensor[0].numpy()
                    ax = plt.subplot(gs[count])
                    
                    if image.shape[-1] == 1:
                        ax.imshow(image[:,:,0], vmin=image.min(), vmax=image.max(), cmap=CMAP, origin='lower')
                    else:
                        ax.imshow(image, origin='lower') # origin='lower' suele ser importante para astronomía
                    
                    ax.set_xticks([])
                    ax.set_yticks([])

                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)
                    
                    # B. Dibujar el Radio de Einstein (theta_E) como un círculo
                    # Convertimos theta_E de arcsec a píxeles
                    centerx_0 = NUM_PIX / 2
                    centery_0 = NUM_PIX / 2
                    theta_E_pix = theta_val / DELTA_PIX
                    circle = patches.Circle((centerx_0, centery_0), theta_E_pix, 
                                            linewidth=1., edgecolor='red', facecolor='none', 
                                            linestyle='--', alpha=0.7)
                    ax.add_patch(circle)

                    ellipse = patches.Ellipse((centerx_0, centery_0), width=theta_E_pix*2, height=theta_E_pix*2*(1-e_total),
                                              angle=phi_deg, edgecolor='blue', facecolor='none', linewidth=1.,
                                              linestyle='-', alpha=0.7)
                    ax.add_patch(ellipse)
                    ax.legend([r'$\theta_e$', r'$\epsilon$'], loc='upper right', fontsize=8, frameon=False, handlelength=0.8, handletextpad=0.5)

                    # --- AÑADIR TEXTO (Mantenemos la actualización anterior) ---
                    textstr = '\n'.join((
                        r'$\theta_E=%.2f^{\prime\prime}$' % (theta_val, ),
                        r'$\epsilon_x, \epsilon_y$ = %.2f, %.2f' % (e1_val, e2_val),
                        r'$f_{axis}=%.2f$' % (f_axis_val, )))
                    
                    ax.text(0.5, -0.10, textstr, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    #ax.axis('off')
                    ax.set_xlim(0, NUM_PIX)
                    ax.set_ylim(0, NUM_PIX)

                    count += 1
                    pbar.update(1)
                
                attempts += 1
            except StopIteration:
                print(f'{RED}The dataset has been exhausted.{ENDC}')
                break

    plt.savefig(MAIN_PATH + f'images_matrix_{CMAP}.pdf', bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print(f"{GREEN}Plot saved in {MAIN_PATH}{ENDC}")

if __name__ == '__main__':
    main()
