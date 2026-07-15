
import os
import configparser
import numpy as np
import tensorflow as tf
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm

try:
    from skimage.metrics import structural_similarity as ssim_sk
except ImportError:
    print("skimage is required for SSIM calculation. Please install it via 'pip install scikit-image'")
    ssim_sk = None

# --- Configuración y Colores ---
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

if os.path.exists('main_config.ini'):
    main_config = load_config('main_config.ini')
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
    NUM_PIX = main_config.getint('MODEL', 'num_pix')
    CHANNELS = main_config.getint('MODEL', 'channels')
    IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
    MAIN_PATH = main_config['PATHS']['main_path']
    KEY = main_config['CONFIG']['prueba']
    NUM_SAMPLES = 3
else:
    LABELS = ['theta_E', 'f_axis', 'f_s', 'e1', 'e2', 'x_s', 'y_s']
    DELTA_PIX = 0.08
    NUM_PIX = 100 
    CHANNELS = 1
    IMGSHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
    MAIN_PATH = './'

# --- Funciones de Procesamiento ---

def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        #'f_axis': tf.io.FixedLenFeature([], tf.float32),
        #'f_s': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        'x_s': tf.io.FixedLenFeature([], tf.float32),
        'y_s': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        're_l': tf.io.FixedLenFeature([], tf.float32),
        #'pa_l': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
        #'e1_s': tf.io.FixedLenFeature([], tf.float32),
        #'e2_s': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed_example['image'], tf.float32)
    image = tf.reshape(image, IMGSHAPE)
    return image, parsed_example['theta_E']

def load_tfrecord_dataset(path, batch_size=1):
    tfrecord_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files).map(parse_tfrecord)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_einstein_ring_mask(shape, theta_e, delta_pix, width=0.2):
    '''
    Creates a boolean mask that is True for pixels within a ring of width `width` around the Einstein radius `theta_e`.
     - `shape`: tuple (height, width) of the image.
     - `theta_e`: Einstein radius in the same units as `delta_pix`.
     - `delta_pix`: pixel scale (size of one pixel in the same units as `theta_e`).
     - `width`: half-width of the ring in the same units as `theta_e` (default 0.2).
     Returns a boolean array of the same shape as the image, where True indicates pixels within the ring.
     The ring includes pixels where the distance from the center is between (theta_e - width) and (theta_e + width).
     This mask can be used to focus SSIM calculations on the region around the Einstein radius, which is often the most relevant area for lensing analyses.
    '''
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (shape[0] // 2, shape[1] // 2)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2) * delta_pix
    mask = (dist_from_center >= (theta_e - width)) & (dist_from_center <= (theta_e + width))
    return mask

def namask(*args):
    m = np.ones(args[0].shape, dtype=bool)
    for a in args:
        m &= np.isfinite(a)
    return m

def mse_masked(a, b, mask=None):
    if mask is None:
        mask = namask(a, b)
    d = (a - b)[mask]
    return np.mean(d**2)

def plot_text(ax, x, y, val, text, color='white', fontsize=12):
    if val == 0:
        val_text = '0'
    elif val < 1e-3:
        val_text = '{:.2e}'.format(val)
        mantissa, exponent = val_text.split('e')
        val_text = fr'${mantissa}×10^{{{int(exponent)}}}$'
    else:
        val_text = f'{val:.4f}'
    ax.text(x, y, text + ': ' + val_text, color=color, fontsize=fontsize)

def normalize_minmax(img, pmin=1, pmax=99):
    vmin, vmax = np.percentile(img, [pmin, pmax])
    img_clipped = np.clip(img, vmin, vmax)
    return (img_clipped - vmin) / (vmax - vmin + 1e-9)

def calculate_ssim(img1, img2):
    # SSIM requiere rango dinámico. Usamos max de img1 como referencia.
    # Se expanden dimensiones para cumplir con (Batch, H, W, C) que pide TF
    t1 = tf.convert_to_tensor(img1[..., np.newaxis], dtype=tf.float32)
    t2 = tf.convert_to_tensor(img2[..., np.newaxis], dtype=tf.float32)
    return tf.image.ssim(t1, t2, max_val=1.0).numpy()

def plot_histogram(n, ax, data, bins_count, color, xlabel):
    valid_data = [d for d in data if np.isfinite(d)]
    if not valid_data: return
    
    counts, bins, _ = ax[n].hist(valid_data, bins=bins_count, color='khaki', alpha=0.8, density=True)
    median_val = np.median(valid_data)
    
    ax[n].vlines(median_val, 0, counts.max(), colors='r', linestyles='--', label=f'Med: {median_val:.3f}')
    ax[n].plot([], [], label = f'max: {np.max(valid_data):.3f}')
    ax[n].plot([], [], label = f'min: {np.min(valid_data):.3f}')
    #ax[n].set_title(f'Model {n+1}')
    ax[n].set_xlabel(xlabel)
    ax[n].legend(fontsize=12)

def arcs_detection(image, theta_e, delta_pix, umbral=0.1):
    """
    Verifica si hay suficiente flujo en la zona del anillo de Einstein.
    Umbral es un valor relativo al brillo máximo de la imagen.
    """
    mask = get_einstein_ring_mask(image.shape, theta_e, delta_pix, width=0.15)
    # Si el promedio de brillo en el anillo es significativo
    flujo_en_anillo = np.mean(image[mask])
    return flujo_en_anillo > umbral

def visualizar_ejemplo_ssim(ds_orig, ds_pred, delta_pix, num_samples, path_saved):
    '''
    Plots an example of the SSIM map for a given example:
    - Column 1: Original image with two circles: the true Einstein radius and the inferred one.
    - Column 2: Predicted image with the same circles.
    - Column 3: SSIM similarity map between both images.
     Each row corresponds to a different example.
     The resulting figure will be saved in path_guardado.
    '''
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)

    samples_collected = 0
    # Taken randomly from the datasets (assuming they are in the same order and have the same number of samples)
    #dataset_zip = zip(ds_orig, ds_pred)
    dataset_zip = zip(ds_orig.take(num_samples), ds_pred.take(num_samples))

    for i, ((img_o, te_o), (img_p, te_p)) in enumerate(zip(ds_orig, ds_pred)):
        if samples_collected >= num_samples:
            break
        # Procesamiento de imágenes
        im_real = img_o.numpy()[0].squeeze() # normalize_minmax(img_o.numpy()[0].squeeze())
        im_pred = img_p.numpy()[0].squeeze() # normalize_minmax(img_p.numpy()[0].squeeze())
        
        n_temp = normalize_minmax(im_real)
        if not arcs_detection(n_temp, te_o.numpy()[0], delta_pix):
            continue
        
        # Radios de Einstein
        i = samples_collected
        
        val_te_real = te_o.numpy()[0]
        val_te_pred = te_p.numpy()[0]
        
        # Procesamiento para visualización
        im_real_norm = normalize_minmax(im_real)
        im_pred_norm = normalize_minmax(im_pred)
        
        # Cálculo de SSIM y métricas estructurales
        score, ssim_map = ssim_sk(im_real, im_pred, full=True, data_range=1.0, win_size=7)
        d_ssim = 1 - score
        mse_val = mse_masked(im_real_norm, im_pred_norm)
        mse_einstein = np.mean((val_te_real - val_te_pred)**2) # Aquí podrías aplicar una máscara de anillo si quieres

        center = (im_real.shape[1] // 2, im_real.shape[0] // 2)
        rad_real_px = val_te_real / delta_pix
        rad_pred_px = val_te_pred / delta_pix

        # Visualización (igual a tu código original)
        axes[i, 0].imshow(im_real, cmap='gray', origin='lower')
        plot_text(axes[i, 0], 5, 10, val_te_real, text=r"$\theta_E$", color='white', fontsize=16)
        axes[i, 0].add_patch(patches.Circle(center, rad_real_px, linewidth=1.5, edgecolor='lime', facecolor='none'))
        axes[i, 0].set_xlabel('píxeles')
        axes[i, 0].set_ylabel('píxeles')
        
        axes[i, 1].imshow(im_pred, cmap='gray', origin='lower')
        plot_text(axes[i, 1], 5, 10, val_te_pred, text=r"$\theta_E$", color='white', fontsize=16)
        axes[i, 1].add_patch(patches.Circle(center, rad_pred_px, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'))
        axes[i, 1].set_xlabel('píxeles')
        axes[i, 1].set_ylabel('píxeles')

        im_ssim = axes[i, 2].imshow(ssim_map, cmap='Spectral', vmin=0, vmax=1, origin='lower')
        plot_text(axes[i, 2], 5, 5, d_ssim, text=r"$\text{D}_{\mathrm{SSIM}}$", color='white', fontsize=14)
        plot_text(axes[i, 2], 5,12, mse_val, text=r"$\text{MSE}_{\mathrm{Img}}$", color='white', fontsize=14)
        axes[i, 2].set_title(f"SSIM: {score:.4f}", fontsize=16)
        axes[i, 2].set_xlabel('píxeles')
        axes[i, 2].set_ylabel('píxeles')
        
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_ssim, cax=cax)

        for j in range(3):
            axes[i, j].set_xticks(np.arange(0, im_real.shape[1]+1, 25))
            axes[i, j].set_yticks(np.arange(0, im_real.shape[0]+1, 25))
        
        samples_collected += 1

    plt.tight_layout()
    plt.savefig(path_saved, bbox_inches='tight', dpi=150)
    print(f"{GREEN}Matriz de arcos guardada en: {path_saved}{ENDC}")

def main():
    metrics = {KEY: {'psnr': [], 'mse': [], 'ssim_global': [], 'ssim_ring': []}}

    PATH = os.path.join(MAIN_PATH, f'{KEY}/')
    try:
        ds_orig = load_tfrecord_dataset(os.path.join(PATH, 'original/'))
        ds_pred = load_tfrecord_dataset(os.path.join(PATH, 'predictions/'))
    except Exception as e:
        print(f"{RED}Error loading {KEY}: {e}{ENDC}")

    print(f"{YELLOW}Processing model {KEY}...{ENDC}")
    
    for (img_o, tE_o), (img_p, tE_p) in zip(ds_orig, ds_pred):
        # Preparar imágenes
        im1 = img_o.numpy()[0].squeeze()
        im2 = img_p.numpy()[0].squeeze()
        tE = tE_o.numpy()[0] # Radio de Einstein del original

        # Normalización
        n1 = normalize_minmax(im1)
        n2 = normalize_minmax(im2)
        
        # 1. SSIM Global
        s_global = calculate_ssim(n1, n2)
        
        # 2. SSIM Anillo de Einstein
        mask_ring = get_einstein_ring_mask(n1.shape, tE, DELTA_PIX)
        # Aplicamos máscara: fuera del anillo ponemos 0 para concentrar el SSIM
        s_ring = calculate_ssim(n1 * mask_ring, n2 * mask_ring)

        # Guardar resultados
        metrics[KEY]['ssim_global'].append(s_global)
        metrics[KEY]['ssim_ring'].append(s_ring)
        metrics[KEY]['psnr'].append(10 * np.log10(1.0 / mse_masked(n1, n2)))  # PSNR basado en MSE global
        
        # Métricas previas (opcional)
        mse_val = np.mean((n1 - n2)**2)
        metrics[KEY]['mse'].append(mse_val)

    # Reporte por modelo
    print(f"{GREEN}Resultados Modelo {KEY}:{ENDC}")
    print(f"  > Median SSIM Global: {np.median(metrics[KEY]['ssim_global']):.4f}")
    print(f"  > Median SSIM Ring: {np.median(metrics[KEY]['ssim_ring']):.4f}")

    # Plot SSIM Histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_histogram(0, ax, metrics[KEY]['ssim_global'], bins_count=20, color='blue', xlabel='SSIM Global')
    plot_histogram(1, ax, metrics[KEY]['ssim_ring'], bins_count=20, color='orange', xlabel='SSIM Ring')
    plt.xlabel('SSIM')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, f'ssim_histogram_model_{KEY}.png'), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_histogram(0, [ax], metrics[KEY]['psnr'], bins_count=20, color='blue', xlabel='PSNR')
    plt.xlabel('PSNR')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, f'psnr_histogram_model_{KEY}.png'), bbox_inches='tight')
    plt.close()

    output_viz = os.path.join(PATH, f'ssim_map_analysis_{KEY}.png')
    visualizar_ejemplo_ssim(ds_orig, ds_pred, DELTA_PIX, NUM_SAMPLES, output_viz)

if __name__ == "__main__":
    main()