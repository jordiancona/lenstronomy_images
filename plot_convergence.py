
"""
plot_convergence_SIE.py
=======================
Grafica el mapa de convergencia κ(x1, x2) del modelo SIE (Singular Isothermal
Ellipsoid) para los parámetros **reales** y los **predichos por la red neuronal**,
a partir de los parámetros:
  - theta_E  : radio de Einstein (arcsec)
  - e1, e2   : componentes de elipticidad (convención lenstronomy)
  - (f_axis) : relación axial, derivada de e1,e2 si no se predice directamente

La convergencia del SIE según Meneghetti (2021) Ec. 5.79 / 5.80 es:

    κ(x1, x2) = (sqrt(f) / 2) / sqrt(x1^2 + f^2 * x2^2)

donde f es la relación axial (axis ratio), 0 < f ≤ 1.

La conversión entre (e1, e2) y f viene de Ec. 6.88 del mismo libro:
    e1 = (1-f)/(1+f) * cos(2*phi)
    e2 = (1-f)/(1+f) * sin(2*phi)
  → |e| = sqrt(e1^2+e2^2)  = (1-f)/(1+f)
  → f   = (1 - |e|) / (1 + |e|)
  → phi = 0.5 * arctan2(e2, e1)   (ángulo de posición del eje mayor)

El mapa se rota con phi para reproducir la orientación correcta de la elipse.

Uso rápido:
-----------
  python plot_convergence_SIE.py

Para integrar con tu pipeline de TFRecords, reemplaza los diccionarios
`params_true` y `params_pred` con los valores leídos de `ds_orig` y `ds_pred`.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import configparser
import tensorflow as tf
import os

# ─────────────────────────────────────────────
#  Configuración de tamaño de letra
# ─────────────────────────────────────────────
plt.rc('axes',  labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

# ─────────────────────────────────────────────
#  Configuración de colores
# ─────────────────────────────────────────────
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# ─────────────────────────────────────────────
#  Funciones auxiliares
# ─────────────────────────────────────────────
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

def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        #'center_x': tf.io.FixedLenFeature([], tf.float32),
        #'center_y': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        're_l': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, IMGSHAPE)
    return image, parsed

# Function to load TFRecord dataset
def load_tfrecord_dataset(tfrecord_dir, batch_size):
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
# ══════════════════════════════════════════════════════════════════════════════
#  1.  Funciones físicas (basadas en Meneghetti 2021)
# ══════════════════════════════════════════════════════════════════════════════
def e1e2_to_f_phi(e1, e2):
    """
    Convierte las componentes de elipticidad (e1, e2) al par (f, phi).

    Referencias: Meneghetti 2021, Ec. 6.88
      e1 = (1-f)/(1+f) * cos(2*phi)
      e2 = (1-f)/(1+f) * sin(2*phi)

    Retorna
    -------
    f   : relación axial (eje menor / eje mayor), 0 < f ≤ 1
    phi : ángulo de posición del eje mayor (radianes)
    """
    e_mod = np.sqrt(e1**2 + e2**2)          # magnitud de la elipticidad
    e_mod = np.clip(e_mod, 0.0, 0.9999)     # evitar f=0 (singularidad)
    f = (1.0 - e_mod) / (1.0 + e_mod)       # Ec. 6.88 invertida
    phi = 0.5 * np.arctan2(e2, e1)          # ángulo de posición
    return f, phi


def kappa_SIE(x1, x2, theta_E, f, phi, delta_pix=0.08):
    """
    Mapa de convergencia del SIE (Meneghetti 2021, Ec. 5.79).

        κ(x1,x2) = sqrt(f) / (2 * sqrt(x1'^2 + f^2 * x2'^2))

    donde (x1', x2') son las coordenadas rotadas por el ángulo de posición phi.
    Las coordenadas angulares se miden en unidades del radio de Einstein (θ_E).

    Parámetros
    ----------
    x1, x2   : grids 2D de coordenadas angulares en arcsec
    theta_E  : radio de Einstein en arcsec  (normalización)
    f        : relación axial
    phi      : ángulo de posición del eje mayor (rad)
    delta_pix: escala de pixel en arcsec/px (no se usa en el cálculo,
               sólo para referencia de los ejes)

    Retorna
    -------
    kappa : array 2D con los valores de convergencia (recortados en ±10σ)
    """
    # Rotar ejes al sistema del elipsoide
    c, s = np.cos(phi), np.sin(phi)
    x1r =  c * x1 + s * x2
    x2r = -s * x1 + c * x2

    # Normalizar por theta_E
    u1 = x1r / theta_E
    u2 = x2r / theta_E

    # Convergencia SIE (Ec. 5.79) — denominador en unidades de θ_E
    denom = np.sqrt(u1**2 + f**2 * u2**2)
    denom = np.where(denom < 1e-6, 1e-6, denom)   # evitar divergencia central

    kappa = np.sqrt(f) / (2.0 * denom)
    return kappa

# ══════════════════════════════════════════════════════════════════════════════
#  3.  Funciones auxiliares de anotación
# ══════════════════════════════════════════════════════════════════════════════

def _add_info(ax, params, f, phi, color='white'):
    """Imprime θ_E, e1, e2 y f derivado en la esquina del panel."""
    txt = (
        fr"$\theta_E = {params['theta_E']:.3f}''$"  + "\n"
        fr"$e_1 = {params['e1']:.3f}$"              + "\n"
        fr"$e_2 = {params['e2']:.3f}$"              + "\n"
        fr"$f = {f:.3f}$"                            + "\n"
        fr"$\phi = {np.degrees(phi):.1f}°$"
    )
    ax.text(0.03, 0.97, txt,
            transform=ax.transAxes,
            va='top', ha='left',
            color=color, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='black', alpha=0.45))


def _add_einstein_circle(ax, theta_E, color='lime', ls='-', lw=1.5):
    """Dibuja el círculo de Einstein."""
    import matplotlib.patches as patches
    circ = patches.Circle((0, 0), theta_E,
                           linewidth=lw, edgecolor=color,
                           facecolor='none', linestyle=ls)
    ax.add_patch(circ)


def _add_residual_stats(ax, diff):
    """Imprime estadísticas de la diferencia."""
    finite = diff[np.isfinite(diff)]
    txt = (
        fr"med $|\Delta\kappa|$ = {np.median(finite):.4f}" + "\n"
        fr"max $|\Delta\kappa|$ = {np.max(finite):.4f}"
    )
    ax.text(0.03, 0.97, txt,
            transform=ax.transAxes,
            va='top', ha='left',
            color='white', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='black', alpha=0.45))


def _colorbar(fig, ax, im, label=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label=label)

# ══════════════════════════════════════════════════════════════════════════════
#  4.  Versión para múltiples muestras (integración con TFRecords)
# ══════════════════════════════════════════════════════════════════════════════

def plot_from_datasets(ds_orig, ds_pred,
                       num_samples=3,
                       num_pix=100,
                       delta_pix=0.08,
                       save_path='convergence_comparison.png'):
    """
    Wrapper para usar directamente con los datasets TFRecord del pipeline.

    Parámetros
    ----------
    ds_orig     : tf.data.Dataset  (imagen, theta_E)  — ground truth
    ds_pred     : tf.data.Dataset  (imagen, theta_E)  — predicciones
    num_samples : cuántos ejemplos graficar (uno por fila)
    num_pix     : tamaño de la imagen en píxeles
    delta_pix   : escala en arcsec/px
    save_path   : ruta de guardado

    NOTA: el dataset DEBE contener también e1, e2.
    Si el parse_tfrecord actual solo retorna theta_E, extiéndelo así:

        return image, {
            'theta_E': parsed_example['theta_E'],
            'e1':      parsed_example['e1'],
            'e2':      parsed_example['e2'],
        }
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    half = num_pix * delta_pix / 2.0
    lin  = np.linspace(-half, half, num_pix)
    x1, x2 = np.meshgrid(lin, lin)
    extent  = [-half, half, -half, half]

    norm_k = mcolors.Normalize(vmin=0, vmax=2.0)

    for i, ((_, ds_o), (_, ds_p)) in enumerate(zip(ds_orig.take(num_samples), ds_pred.take(num_samples))):
        # Extrae parámetros (adapta si tu dataset devuelve tensores sueltos)

        p_t = {label: ds_o[label].numpy()[0] for label in enumerate(LABELS)}
        p_p = {label: ds_p[label].numpy()[0] for label in enumerate(LABELS)}

        f_t, phi_t = e1e2_to_f_phi(p_t['e1'], p_t['e2'])
        f_p, phi_p = e1e2_to_f_phi(p_p['e1'], p_p['e2'])

        kt = kappa_SIE(x1, x2, p_t['theta_E'], f_t, phi_t, delta_pix)
        kp = kappa_SIE(x1, x2, p_p['theta_E'], f_p, phi_p, delta_pix)
        diff = np.abs(kt - kp)

        for j, (kmap, params, f, phi, ttl, col, ls) in enumerate([
            (kt,   p_t, f_t, phi_t, r'$\kappa_\mathrm{true}$',  'lime', '-'),
            (kp,   p_p, f_p, phi_p, r'$\kappa_\mathrm{pred}$',  'cyan', '--'),
            (diff, p_t, f_t, phi_t, r'$|\Delta\kappa|$',         'lime', '-'),
        ]):
            ax = axes[i, j]
            if j < 2:
                im = ax.imshow(kmap, origin='lower', extent=extent,
                               norm=norm_k, cmap='inferno')
                _add_info(ax, params, f, phi, color=col)
                _add_einstein_circle(ax, params['theta_E'], color=col, ls=ls)
            else:
                vd = diff[np.isfinite(diff)].max() * 0.6
                im = ax.imshow(kmap, origin='lower', extent=extent,
                               vmin=0, vmax=vd, cmap='RdBu_r')
                _add_einstein_circle(ax, p_t['theta_E'],  color='lime', ls='-')
                _add_einstein_circle(ax, p_p['theta_E'], color='cyan', ls='--')
                _add_residual_stats(ax, diff)

            if i == 0:
                ax.set_title(ttl, fontsize=15)
            _colorbar(fig, ax, im, label=r'$\kappa$' if j < 2 else r'$|\Delta\kappa|$')
            ax.set_xlabel("$x_1$ [arcsec]")
            ax.set_ylabel("$x_2$ [arcsec]")

    plt.suptitle("Convergencia SIE — real vs. predicha", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, f'convergence_comparison_{KEY}.png'), dpi=150, bbox_inches='tight')
    print(f"Figura guardada en: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Demo ejecutable (parámetros de ejemplo)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Parámetros reales de ejemplo ──────────────────────────────────────────
    # f = 0.7, phi = 30° → e1 = (1-0.7)/(1+0.7)*cos(60°) ≈ 0.0882
    #                       e2 = (1-0.7)/(1+0.7)*sin(60°) ≈ 0.1529
    PATH = os.path.join(MAIN_PATH, f'{KEY}/')
    try:
        ds_orig = load_tfrecord_dataset(os.path.join(PATH, 'original/'), batch_size=1)
        ds_pred = load_tfrecord_dataset(os.path.join(PATH, 'predictions/'), batch_size=1)
    except Exception as e:
        print(f"{RED}Error loading {KEY}: {e}{ENDC}")

    print(f"{YELLOW}Processing model {KEY}...{ENDC}")

    plot_from_datasets(
        ds_orig, ds_pred,
        num_pix=NUM_PIX,
        delta_pix=DELTA_PIX,
    )
    print(f"Figure saved in: {PATH}")
