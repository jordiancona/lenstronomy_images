
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
except FileNotFoundError as e:
    print(f"{RED}Error cargando configuración: {e}{ENDC}")
    LABELS = ['theta_E', 'f_axis', 'f_s', 'e1', 'e2', 'x_s', 'y_s']
    DELTA_PIX = 0.08
    NUM_PIX = 100 # Ajustar acompañando con tus datos reales
    CHANNELS = 1
    IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
    MAIN_PATH = './'

# --- FUNCIONES DE CARGA DE DATOS ---
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
    try:
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_raw(parsed_example['image'], tf.float32)
        image = tf.reshape(image, IMGSHAPE)
        return image
    except:
        return tf.zeros(IMGSHAPE)

def load_tfrecord_dataset(path, batch_size=1):
    tfrecord_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tfrecord")])
    if not tfrecord_files:
        raise FileNotFoundError(f"No se encontraron archivos .tfrecord en {path}")
    
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

def robust_stats(x, mask=None):
    if mask is None: mask = np.isfinite(x)
    xm = x[mask]
    if len(xm) == 0: return 0.0, 1.0
    med = np.median(xm)
    mad = np.median(np.abs(xm - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(xm) if len(xm) > 1 else 1.0
    return med, sigma

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
    '''
    Plotea un histograma comparativo: Individuales vs Ensemble
    data_dict: {'Model 1': [vals], 'Model 2': [vals], ..., 'Ensemble': [vals]}
    '''
    plt.figure(figsize=(10, 6))
    
    # Estilos para diferenciar
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    # Plotear modelos individuales (transparente)
    for i, (key, values) in enumerate(data_dict.items()):
        if key == 'Ensemble': continue
        valid_data = [d for d in values if np.isfinite(d)]
        plt.hist(valid_data, bins=40, alpha=0.3, density=True, 
                 label=f'{key} (Mean: {np.mean(valid_data):.2f})', color=colors[i%3])

    # Plotear Ensemble (más fuerte)
    if 'Ensemble' in data_dict:
        ens_data = [d for d in data_dict['Ensemble'] if np.isfinite(d)]
        plt.hist(ens_data, bins=40, alpha=0.6, density=True, color='blue', 
                 histtype='step', linewidth=2, label=f'Ensemble (Mean: {np.mean(ens_data):.2f})')
        
        # Linea de la mediana del ensamble
        plt.axvline(np.median(ens_data), color='blue', linestyle='--', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    plt.title(f'{xlabel} Comparison: Individual Models vs Deep Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MAIN_PATH, filename))
    plt.close()

# --- MAIN ---
def main():
    # Estructuras para guardar resultados
    results_psnr = {'Model 1': [], 'Model 2': [], 'Model 3': [], 'Ensemble': []}
    results_mse = {'Model 1': [], 'Model 2': [], 'Model 3': [], 'Ensemble': []}
    
    # 1. Preparar Datasets
    # Asumimos que la carpeta 'original' es idéntica en todos, cargamos la del 1
    path_orig = os.path.join(MAIN_PATH, 'alexnet_1', 'original/')
    
    # Rutas de predicciones para cada miembro del ensamble
    paths_pred = [os.path.join(MAIN_PATH, f'alexnet_{i+1}', 'predictions/') for i in range(3)]
    
    try:
        ds_original = load_tfrecord_dataset(path_orig)
        ds_preds = [load_tfrecord_dataset(p) for p in paths_pred]
        
        # 2. ZIPEAR DATASETS: Esto alinea (Orig, Pred1, Pred2, Pred3)
        # Nota: ds_preds se desempaqueta con *
        combined_dataset = tf.data.Dataset.zip((ds_original, *ds_preds))
        
    except Exception as e:
        print(f"\033[31mError cargando datasets: {e}\033[0m")
        return

    print(f"\033[33mProcesando Deep Ensemble...\033[0m")
    
    count = 0
    # Iteramos sobre la tupla (batch_orig, batch_p1, batch_p2, batch_p3)
    for data in combined_dataset:
        # Extraer tensores
        # data[0] es original, data[1] es pred1, data[2] es pred2...
        orig_tensor = data[0].numpy()[0]
        pred_tensors = [d.numpy()[0] for d in data[1:]] # Lista de arrays numpy
        
        # Ajuste de dimensiones si es canal 1
        if orig_tensor.shape[-1] == 1:
            orig_tensor = orig_tensor.squeeze(-1)
            pred_tensors = [p.squeeze(-1) for p in pred_tensors]

        # --- LÓGICA ENSAMBLE ---
        # 1. Calcular promedio de predicciones (Ensemble Mean)
        # Stackeamos para tener (3, H, W) y hacemos media en axis 0
        ensemble_img = np.mean(np.array(pred_tensors), axis=0)
        
        # --- NORMALIZACIÓN Y MÉTRICAS ---
        
        # Preparamos máscara común (basada en NaNs del original y ensemble)
        mask_common = namask(orig_tensor, ensemble_img)
        
        # Normalizamos Original y Ensemble
        orig_norm = normalize_minmax(orig_tensor, mask=mask_common)
        ens_norm = normalize_minmax(ensemble_img, mask=mask_common)
        
        # Calculamos métricas del ENSAMBLE
        mse_ens = mse(orig_norm, ens_norm, mask=mask_common)
        psnr_ens = psnr(orig_norm, ens_norm, mask=mask_common)
        
        results_mse['Ensemble'].append(mse_ens)
        results_psnr['Ensemble'].append(psnr_ens)
        
        # (Opcional) Calculamos métricas individuales para comparar
        for i, pred in enumerate(pred_tensors):
            pred_norm = normalize_minmax(pred, mask=mask_common)
            mse_ind = mse(orig_norm, pred_norm, mask=mask_common)
            psnr_ind = psnr(orig_norm, pred_norm, mask=mask_common)
            
            results_mse[f'Model {i+1}'].append(mse_ind)
            results_psnr[f'Model {i+1}'].append(psnr_ind)

        count += 1
        if count % 50 == 0:
            print(f"  Procesadas {count} imágenes...", end='\r')

    print(f'\n\033[32mAnálisis completado. Total: {count} imágenes.\033[0m')

    # --- PLOTTING ---
    if len(results_psnr['Ensemble']) > 0:
        # Plot PSNR Comparison
        plot_comparison_histogram(results_psnr, 'PSNR', 'ensemble_psnr_comparison.pdf')
        # Plot MSE Comparison
        plot_comparison_histogram(results_mse, 'MSE', 'ensemble_mse_comparison.pdf')
        
        # Imprimir resumen numérico
        print("\n--- Summary Stats ---")
        ens_mean_psnr = np.mean(results_psnr['Ensemble'])
        print(f"\033[1mEnsemble Mean PSNR: {ens_mean_psnr:.4f}\033[0m")
        for i in range(3):
            ind_mean = np.mean(results_psnr[f'Model {i+1}'])
            print(f"Model {i+1} Mean PSNR: {ind_mean:.4f}")
            
    else:
        print("No data gathered.")

if __name__ == "__main__":
    main()
