import os
import configparser
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuración de estilos
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

# --- CONFIGURACIÓN DE COLORES ---
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

# --- CARGA DE CONFIGURACIÓN ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    NUM_PIX = main_config.getint('MODEL', 'num_pix')
    CHANNELS = main_config.getint('MODEL', 'channels')
    IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
    MAIN_PATH = main_config['PATHS']['main_path']
    KEY = str(main_config['CONFIG']['prueba'])
    MODEL_DIR_CFG = main_config['CONFIG'].get('model_dir', '').strip()
    N_FOLDS = main_config.getint('DEEPENSAMBLE', 'n_folds', fallback=5)
except Exception as e:
    print(f"{RED}Error cargando configuración: {e}{ENDC}")
    LABELS = ['theta_E', 'f_axis', 'e1', 'e2']
    NUM_PIX = 100
    CHANNELS = 1
    IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
    MAIN_PATH = './'
    KEY = 'alexnet_original'
    MODEL_DIR_CFG = ''
    N_FOLDS = 5

if MODEL_DIR_CFG and os.path.exists(MODEL_DIR_CFG):
    OUT_DIR = MODEL_DIR_CFG
else:
    possible_paths = [
        os.path.join(MAIN_PATH, f'{KEY}/'),
        os.path.join(MAIN_PATH, f'alexnet_{KEY}/'),
        MAIN_PATH
    ]
    OUT_DIR = possible_paths[0]
    for p in possible_paths:
        if os.path.exists(p):
            OUT_DIR = p
            break

# --- FUNCIONES DE CARGA DE DATOS ---
def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
    }
    try:
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_raw(parsed_example['image'], tf.float32)
        image = tf.reshape(image, IMGSHAPE)
        return image
    except Exception:
        return tf.zeros(IMGSHAPE)

def load_tfrecord_dataset(path, batch_size=1):
    if not os.path.exists(path):
        return None
    tfrecord_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tfrecord")])
    if not tfrecord_files:
        return None
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- FUNCIONES MATEMÁTICAS ---
def namask(*args):
    m = np.ones(args[0].shape, dtype=bool)
    for a in args:
        m &= np.isfinite(a)
    return m

def normalize_minmax(img, mask=None, low=0.0, high=1.0, pmin=1, pmax=99):
    if mask is None: mask = np.isfinite(img)
    vals = img[mask]
    if len(vals) == 0: return img
    vmin, vmax = np.percentile(vals, [pmin, pmax])
    if vmin == vmax: vmax = vmin + 1e-9
    img_clipped = np.clip(img, vmin, vmax)
    x = (img_clipped - vmin) / (vmax - vmin)
    return x * (high - low) + low

def mse(a, b, mask=None):
    if mask is None: mask = namask(a, b)
    d = (a - b)[mask]
    if len(d) == 0: return 0.0
    return np.mean(d**2)

def psnr(a, b, mask=None):
    if mask is None: mask = namask(a, b)
    m = mse(a, b, mask)
    if m <= 0: return 100.0
    peak = np.nanmax(a[mask]) if np.any(mask) else 1.0
    return 10 * np.log10((peak**2) / m)

def plot_comparison_histogram(data_dict, xlabel, filename):
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet']
    
    for i, (key, values) in enumerate(data_dict.items()):
        if key == 'Ensemble': continue
        valid_data = [d for d in values if np.isfinite(d)]
        if not valid_data: continue
        plt.hist(valid_data, bins=40, alpha=0.3, density=True, 
                 label=f'{key} (Mean: {np.mean(valid_data):.2f})', color=colors[i % len(colors)])

    if 'Ensemble' in data_dict:
        ens_data = [d for d in data_dict['Ensemble'] if np.isfinite(d)]
        if ens_data:
            plt.hist(ens_data, bins=40, alpha=0.6, density=True, color='blue', 
                     histtype='step', linewidth=2, label=f'Ensemble (Mean: {np.mean(ens_data):.2f})')
            plt.axvline(np.median(ens_data), color='blue', linestyle='--', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    plt.title(f'{xlabel} Comparison: Models vs Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

# --- MAIN ---
def main():
    results_psnr = {'Ensemble': []}
    results_mse = {'Ensemble': []}
    
    path_orig = os.path.join(OUT_DIR, 'original/')
    path_pred = os.path.join(OUT_DIR, 'predictions/')
    
    # Also check if individual fold subdirectories exist
    fold_dirs = [os.path.join(MAIN_PATH, f'alexnet_{i+1}', 'predictions/') for i in range(N_FOLDS)]
    valid_fold_dirs = [fd for fd in fold_dirs if os.path.exists(fd)]

    ds_original = load_tfrecord_dataset(path_orig)
    if ds_original is None:
        print(f"{RED}Original TFRecord dataset not found in {path_orig}{ENDC}")
        return

    if valid_fold_dirs:
        for i in range(len(valid_fold_dirs)):
            results_psnr[f'Model {i+1}'] = []
            results_mse[f'Model {i+1}'] = []
        ds_preds = [load_tfrecord_dataset(p) for p in valid_fold_dirs]
        ds_preds = [dp for dp in ds_preds if dp is not None]
        combined_dataset = tf.data.Dataset.zip((ds_original, *ds_preds))
    else:
        ds_ens = load_tfrecord_dataset(path_pred)
        if ds_ens is None:
            print(f"{RED}Prediction TFRecord dataset not found in {path_pred}{ENDC}")
            return
        combined_dataset = tf.data.Dataset.zip((ds_original, ds_ens))

    print(f"\033[33mProcessing Deep Ensemble PSNR comparison...\033[0m")
    count = 0

    for data in combined_dataset:
        orig_tensor = data[0].numpy()[0]
        if valid_fold_dirs and len(data) > 2:
            pred_tensors = [d.numpy()[0] for d in data[1:]]
            if orig_tensor.shape[-1] == 1:
                orig_tensor = orig_tensor.squeeze(-1)
                pred_tensors = [p.squeeze(-1) for p in pred_tensors]
            ensemble_img = np.mean(np.array(pred_tensors), axis=0)
        else:
            ens_tensor = data[1].numpy()[0]
            if orig_tensor.shape[-1] == 1:
                orig_tensor = orig_tensor.squeeze(-1)
                ens_tensor = ens_tensor.squeeze(-1)
            ensemble_img = ens_tensor
            pred_tensors = []

        mask_common = namask(orig_tensor, ensemble_img)
        orig_norm = normalize_minmax(orig_tensor, mask=mask_common)
        ens_norm = normalize_minmax(ensemble_img, mask=mask_common)
        
        mse_ens = mse(orig_norm, ens_norm, mask=mask_common)
        psnr_ens = psnr(orig_norm, ens_norm, mask=mask_common)
        
        results_mse['Ensemble'].append(mse_ens)
        results_psnr['Ensemble'].append(psnr_ens)
        
        for i, pred in enumerate(pred_tensors):
            pred_norm = normalize_minmax(pred, mask=mask_common)
            mse_ind = mse(orig_norm, pred_norm, mask=mask_common)
            psnr_ind = psnr(orig_norm, pred_norm, mask=mask_common)
            results_mse[f'Model {i+1}'].append(mse_ind)
            results_psnr[f'Model {i+1}'].append(psnr_ind)

        count += 1

    print(f'\n\033[32mAnalysis completed. Total: {count} images.\033[0m')

    if len(results_psnr['Ensemble']) > 0:
        plot_comparison_histogram(results_psnr, 'PSNR', 'ensemble_psnr_comparison.pdf')
        plot_comparison_histogram(results_mse, 'MSE', 'ensemble_mse_comparison.pdf')
        
        print("\n--- Summary Stats ---")
        ens_mean_psnr = np.mean(results_psnr['Ensemble'])
        print(f"\033[1mEnsemble Mean PSNR: {ens_mean_psnr:.4f}\033[0m")
        for k in results_psnr:
            if k != 'Ensemble' and len(results_psnr[k]) > 0:
                ind_mean = np.mean(results_psnr[k])
                print(f"{k} Mean PSNR: {ind_mean:.4f}")
    else:
        print("No data gathered.")

if __name__ == "__main__":
    main()
