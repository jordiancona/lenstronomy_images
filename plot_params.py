
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns # Añadimos seaborn para mejores paletas y estilos
import configparser
import pandas as pd
import numpy as np
import os

# Temrinal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

#plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpio y profesional
plt.rcParams.update({"axes.titlesize": 24,
                     "axes.labelsize": 24,
                     "xtick.labelsize": 22,
                     "ytick.labelsize": 22,
                     "legend.fontsize": 16,
                     "figure.dpi": 150            # Alta resolución
                     })

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

# Load configuration file
main_config = load_config('main_config.ini')
MAIN_PATH = main_config['PATHS']['main_path']
TRAIN_PATH = main_config['PATHS']['tfrecords_path_train']
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNELS = int(main_config['MODEL']['channels'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)

# Function to parse TFRecord
def parse_tfrecord(example_proto):
    feature_description = {
        'image_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'theta_E': tf.io.FixedLenFeature([], tf.float32),
        'f_axis': tf.io.FixedLenFeature([], tf.float32),
        'e1': tf.io.FixedLenFeature([], tf.float32),
        'e2': tf.io.FixedLenFeature([], tf.float32),
        'center_x': tf.io.FixedLenFeature([], tf.float32),
        'center_y': tf.io.FixedLenFeature([], tf.float32),
        're_s': tf.io.FixedLenFeature([], tf.float32),
        're_l': tf.io.FixedLenFeature([], tf.float32),
        'pa_s': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    return image, parsed

# Function to load TFRecord dataset
def load_tfrecord_dataset(tfrecord_dir, batch_size):
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

features_dataset = load_tfrecord_dataset(TRAIN_PATH, batch_size=1)

# Save labels values in lists for plotting
theta_E_vals, e1_vals, e2_vals, f_axis_vals = [], [], [], []

for i, (image, label) in enumerate(features_dataset):
    theta_E_vals.append(float(label['theta_E'].numpy()))
    e1_vals.append(float(label['e1'].numpy()))
    e2_vals.append(float(label['e2'].numpy()))
    f_axis_vals.append(float(label['f_axis'].numpy()))

data_to_plot = [(theta_E_vals, r"$\theta_E$", "red"),
                (f_axis_vals, r"$f$", "red"),
                (e1_vals, r"$\epsilon_x$", "red"),
                (e2_vals, r"$\epsilon_y$", "red")]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
for i, (data, label, color) in enumerate(data_to_plot):
    ax = axes[i]
    
    # 1. Cálculos optimizados con NumPy (mucho más rápido que sum/len, min, max en bucles)
    data_np = np.asarray(data)
    mean_val = data_np.mean()
    min_val = data_np.min()
    max_val = data_np.max()
    
    print(f"{GREEN}{label}{ENDC} - {YELLOW}Min:{ENDC} {min_val:.2f}, {YELLOW}Max:{ENDC} {max_val:.2f}, {YELLOW}Mean:{ENDC} {mean_val:.2f}")
    
    ax.hist(data_np, bins=25, density=True, alpha=0.3, color=color, histtype='stepfilled')
    ax.hist(data_np, bins=25, density=True, color='black', histtype='step', linewidth=1.5, alpha=0.8)

    # ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'Media: {mean_val:.2f}')

    ax.set_xlabel(label, fontsize=24, fontweight='bold', labelpad=10)
    ax.set_ylabel('Densidad' if i == 0 else '', fontsize=24, labelpad=10)
    ax.tick_params(axis='both', labelsize=16) # Agrandamos los números de los ejes para que hagan juego con el texto grande

    data_range = max_val - min_val
    ax.set_xlim(min_val - 0.05 * data_range, max_val + 0.05 * data_range)
    
    ax.grid(True, linestyle=':', alpha=0.6) # Una cuadrícula
    sns.despine(ax=ax, left=False, bottom=False) # sin cuadrícula

plt.tight_layout()

output_path = os.path.join(MAIN_PATH, 'parameter_distributions.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)

# plot theta_E vs f_axis
theta_E_vals, f_axis_vals = [], []
for i, (image, label) in enumerate(features_dataset):
    theta_E_vals.append(float(label['theta_E'].numpy()))
    f_axis_vals.append(float(label['f_axis'].numpy()))

df = {r'$\theta_E$': theta_E_vals, r'$f$': f_axis_vals}
data_to_plot = pd.DataFrame(df)

fig = plt.figure(figsize=(8, 6))
pplot = sns.pairplot(data=data_to_plot, kind='scatter', corner = True, plot_kws={'s': 10, 'alpha': 0.5, 'color': 'darkblue'}, diag_kws={'color': 'darkblue'}, hue=None)

plt.tight_layout()
output_path = os.path.join(MAIN_PATH, 'theta_E_vs_f.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
