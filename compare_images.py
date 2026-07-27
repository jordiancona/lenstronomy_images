
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from lenstronomy.LensModel.lens_model import LensModel # pyrefly: ignore [missing-import]
from lenstronomy.Plots import lens_plot # pyrefly: ignore [missing-import]
import configparser
import sys
import os
plt.rc('axes', labelsize=20)
plt.rc('axes', titlesize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

# Terminal colors
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

IDX = int(main_config['PSNR']['idx'])
N = main_config['CONFIG']['prueba']
MAIN_PATH = main_config['PATHS']['main_path']
PATH = os.path.join(MAIN_PATH, f'{N}/')
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNELS = int(main_config['MODEL']['channels'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
DELTA_PIX = main_config.getfloat('MODEL','delta_pix')
labels = ['theta_E', 'f_axis', 'e1', 'e2']
plot_labels = [r'$\theta_E$', r'$f$', r'$e_x$', r'$e_y$']

def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        're_l': tf.io.FixedLenFeature([], tf.float32),
        'x_s': tf.io.FixedLenFeature([], tf.float32),
        'y_s': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    return image, parsed

def load_tfrecord(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(parse_tfrecord)
    return list(dataset)

def commoon_region(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]

def namask(*args):
    m = np.ones(args[0].shape, dtype=bool)
    for a in args:
        m &= np.isfinite(a)
    return m

def robust_stats(x, mask=None):
    if mask is None:
        mask = np.isfinite(x)
    xm = x[mask]
    med = np.median(xm)
    mad = np.median(np.abs(xm - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(xm) if len(xm) > 1 else 1.0
    return med, sigma

def normalize_zscore(img, mask=None):
    med, sigma = robust_stats(img, mask=mask)
    sigma = sigma if sigma > 0 else 1.0
    return (img - med) / sigma

def normalize_minmax(img, mask=None, low=0.0, high=1.0, pmin=1, pmax=99):
    if mask is None:
        mask = np.isfinite(img)
    vals = img[mask]
    vmin, vmax = np.percentile(vals, [pmin, pmax])
    if vmin == vmax:
        vmax = vmin + 1
    img_min = vmin
    img_max = vmax
    x = (img - img_min) / (img_max - img_min)
    return x * (high - low) + low

def normalize_affine_match(ref, mov, mask=None):
    if mask is None:
        mask = namask(ref, mov)
    r = ref[mask]
    m = mov[mask]
    a = np.vstack([m, np.ones_like(m)]).T
    alpha, beta = np.linalg.lstsq(a, r, rcond=None)[0]
    return mov * alpha + beta, alpha, beta

def mse(a, b, mask=None):
    if mask is None:
        mask = namask(a, b)
    d = (a - b)[mask]
    return np.mean(d**2)

def psnr(a, b, mask=None):
    if mask is None:
        mask = namask(a, b)
    m = mse(a, b, mask)
    if m == 0:
        return np.inf
    peak = np.nanmax(a[mask])
    return 10 * np.log10((peak**2) / m)

def plot_radius(ax, hdr, ref, radius, edgecolor='green', lw=1.5, linestyle = '-'):
    scale = 1.0 / DELTA_PIX
    x_s = hdr['x_s']*scale + ref.shape[1]/2
    y_s = hdr['y_s']*scale + ref.shape[1]/2
    radius = hdr['theta_E'] * scale
    circle = patches.Circle((x_s, y_s),
                            radius=radius,
                            facecolor='none',
                            edgecolor=edgecolor,
                            lw=lw,
                            linestyle = linestyle,
                            fill=False)
    ax.add_patch(circle)

def compare_and_plot(ref, mov, original_hdr, predicted_hdr, title, plot_title, cmap_img='hot', cmap_diff='magma'):
    m = namask(ref, mov)
    d = ref - mov
    s = 0.025
    diff = np.arcsinh(abs(d) / s)
    val_mse = mse(ref, mov, m)
    val_psnr = psnr(ref, mov, m)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.5))
    #plt.suptitle(f'Model {N}', fontsize=20)

    im1 = ax[0].imshow(ref, origin='lower', cmap = 'gray_r')
    ax[0].set_title(f'Original', fontsize=18)
    ax[0].set_xlabel('píxeles')
    ax[0].set_ylabel('píxeles')
    ax[0].set_xticks(np.arange(0, ref.shape[1]+1, 25))
    ax[0].set_yticks(np.arange(0, ref.shape[1]+1, 25))
    ax[0].text(0.05, 0.05,
                 f"$\\theta_E$={original_hdr['theta_E']:.4f}\n"
                 f"f={original_hdr['f_axis']:.4f}\n"
                 f"e1={original_hdr['e1']:.4f}\n"
                 f"e2={original_hdr['e2']:.4f}",
                 color='white', fontsize=12, transform=ax[0].transAxes,
                 bbox=dict(facecolor='black', alpha=0.4, pad=2))
    plot_radius(ax[0], original_hdr, ref, radius=1.0, lw=1.5)
    plot_radius(ax[0], predicted_hdr, ref, radius=1.0, edgecolor='red', lw=1.5, linestyle = '--')
    divider = make_axes_locatable(ax[0])
    cax1 = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im1, cax=cax1)

    im2 = ax[1].imshow(mov[:,:,0], origin='lower', cmap = 'gray_r')
    ax[1].set_title(f'Reconstrucción', fontsize=18)
    ax[1].set_xlabel('píxeles')
    ax[1].set_ylabel('píxeles')
    ax[1].set_xticks(np.arange(0, ref.shape[1]+1, 25))
    ax[1].set_yticks(np.arange(0, ref.shape[1]+1, 25))
    ax[1].text(0.05, 0.05,
                 f"$\\theta_E$={predicted_hdr['theta_E']:.4f}\n"
                 f"f={predicted_hdr['f_axis']:.4f}\n"
                 f"e1={predicted_hdr['e1']:.4f}\n"
                 f"e2={predicted_hdr['e2']:.4f}",
                 color='white', fontsize=12, transform=ax[1].transAxes,
                 bbox=dict(facecolor='black', alpha=0.4, pad=2))
    divider = make_axes_locatable(ax[1])
    cax2 = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im2, cax=cax2)

    im3 = ax[2].imshow(diff[:, :, 0], origin='lower', cmap=cmap_diff, vmin=0, vmax=2)
    
    if val_mse == 0:
        val_mse_tex = '0'
    elif val_mse < 1e-3:
        val_mse_tex = '{:.2e}'.format(val_mse)
        mantissa, exponent = val_mse_tex.split('e')
        val_mse_tex = fr'${mantissa}×10^{{{int(exponent)}}}$'
    else:
        val_mse_tex = '{:.2g}'.format(val_mse)
        
    ax[2].set_title(f'Residuo \nMSE={val_mse_tex}, PSNR={val_psnr:.2f} dB', fontsize=18, pad=20)
    ax[2].set_xlabel('píxeles')
    ax[2].set_ylabel('píxeles')
    ax[2].set_xticks(np.arange(0, ref.shape[1]+1, 25))
    ax[2].set_yticks(np.arange(0, ref.shape[1]+1, 25))
    divider = make_axes_locatable(ax[2])
    cax3 = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im3, cax=cax3, label=r'asinh(|Original - Pred|$\alpha$)')

    plt.tight_layout()
    plt.savefig(PATH + f'{plot_title.lower()}_model_{N}_{IDX}.png', bbox_inches='tight')
    plt.close()
    return val_mse

def main():
    original_dir = os.path.join(PATH, 'original/')
    predictions_dir = os.path.join(PATH, 'predictions/')

    original_file = os.path.join(original_dir, 'lenses_original.tfrecord')
    prediction_file = os.path.join(predictions_dir, 'lenses_predicted.tfrecord')

    original_dataset = load_tfrecord(original_file)
    predicted_dataset = load_tfrecord(prediction_file)

    original_img, original_hdr = original_dataset[IDX]
    predicted_img, predicted_hdr = predicted_dataset[IDX]

    img1, img2 = commoon_region(original_img, predicted_img)
    mask0 = namask(img1, img2)

    n1_a = normalize_zscore(img1, mask=mask0)
    n2_a = normalize_zscore(img2, mask=mask0)

    n1_b = normalize_minmax(img1, mask=mask0)
    n2_b = normalize_minmax(img2, mask=mask0)

    n2_C, alpha_C, beta_C = normalize_affine_match(img1, img2, mask=mask0)

    print(f'{YELLOW}Comparing images...{ENDC}')
    print(f'Image: {IDX} from {original_file}\n')
    print(f'{YELLOW}========== Results from image {IDX} =========={ENDC}')
    #print('--- Robust Z-score ---')
    #mse_A = compare_and_plot(n1_a, n2_a, original_hdr, predicted_hdr, 'Z-score', f'robust_Z-score')

    print('--- Robust Min-Max (p1–p99) ---')
    mse_B = compare_and_plot(n1_b, n2_b, original_hdr, predicted_hdr, 'Robust Min-Max', f'robust_Min-Max')

    #print('--- Robust Affine (img2→img1) ---')
    #mse_C = compare_and_plot(img1, n2_C, original_hdr, predicted_hdr, f'Affine α={alpha_C:.3g}, β={beta_C:.3g}', f'robust_Affine')

    #print(f'{CYAN}MSE Z-score:{ENDC} {mse_A:.6g}')
    print(f'{CYAN}MSE Min-Max: {ENDC}{mse_B:.6g}')
    #print(f'{CYAN}MSE Affine: {ENDC} {mse_C:.6g} \n')

    print(f'{GREEN}Image saved in {PATH}{ENDC}\n')

if __name__ == '__main__':
    main()
