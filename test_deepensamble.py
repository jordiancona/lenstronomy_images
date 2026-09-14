import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import configparser
import argparse

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# --- LENSTRONOMY IMPORTS ---
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
except ImportError:
    print(f'{RED}ERROR: Lenstronomy is not installed. Please install it to proceed.{ENDC}')
    sys.exit(1)

# --- 1. CONFIGURACIÓN ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
MAIN_PATH = main_config['PATHS']['main_path']
TEST_PATH = main_config['PATHS']['tfrecords_path_test']
PRUEBA = str(main_config['CONFIG']['prueba'])
MODEL_DIR_CFG = main_config['CONFIG'].get('model_dir', '').strip()
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNELS = int(main_config['MODEL']['channels'])
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
BATCH_SIZE = int(main_config['MODEL']['batch_size'])
TEST_IMAGES = int(main_config['MODEL']['test_images'])
N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)

# Parse optional arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default=None, help='Model name or path to models directory')
args, _ = parser.parse_known_args()

target_model = args.model if args.model else (MODEL_DIR_CFG if MODEL_DIR_CFG else PRUEBA)

if os.path.exists(target_model) and os.path.isdir(target_model):
    OUTPUT_DIR = target_model
    model_name = os.path.basename(os.path.normpath(target_model))
else:
    model_name = target_model
    OUTPUT_DIR = os.path.join(MAIN_PATH, f"{model_name}/")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. FUNCIONES AUXILIARES TFRECORD ---
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
        print(f"{RED}Path not found: {tfrecord_dir}{ENDC}")
        return None
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 3. CARGA DE DATOS ---
print(f'{YELLOW}Loading dataset from {TEST_PATH}{ENDC}')
inference_batch_size = BATCH_SIZE
dataset = load_tfrecord_dataset(TEST_PATH, inference_batch_size)
if dataset is None:
    print(f"{RED}Error: Unable to load dataset from {TEST_PATH}{ENDC}")
    sys.exit(1)

print(f"{YELLOW}Extracting data into memory...{ENDC}")
all_images = []
all_parsed = []
targets_list = []

for img_batch, parsed_batch in tqdm(dataset, desc="Loading Data"):
    all_images.append(img_batch.numpy())
    batch_len = img_batch.shape[0]
    keys = parsed_batch.keys()
    
    batch_targets = np.stack([
        parsed_batch['theta_E'].numpy(),
        parsed_batch['f_axis'].numpy(),
        parsed_batch['e1'].numpy(),
        parsed_batch['e2'].numpy()
    ], axis=1)
    targets_list.append(batch_targets)
    for i in range(batch_len):
        single_item = {k: parsed_batch[k][i].numpy() for k in keys}
        all_parsed.append(single_item)

images_array = np.concatenate(all_images, axis=0)
targets_array = np.concatenate(targets_list, axis=0)
print(f'{CYAN}Total Dataset:{ENDC} {len(all_parsed)} samples loaded.')
print(f'{CYAN}Images shape:{ENDC} {images_array.shape}')

# --- 4. ENSAMBLE Y PREDICCIÓN ---
def ensamble_predictions(models, images_array):
    preds_folds = []
    inference_batch = BATCH_SIZE
    
    for i, model in enumerate(models):
        print(f"Predicting with model {i+1}/{len(models)}...")
        p = model.predict(images_array, batch_size=inference_batch, verbose=1)
        if hasattr(p, 'numpy'):
            p = p.numpy()
        p = np.array(p, dtype=np.float64)
        preds_folds.append(p)
            
    predictions_stacked = np.stack(preds_folds, axis=0).astype(np.float64)
    predictions_mean = np.mean(predictions_stacked, axis=0)
    uncertainty = np.std(predictions_stacked, axis=0)

    return predictions_mean, uncertainty, predictions_stacked

# Cargar modelos fold
models = []
print(f"{YELLOW}Searching for fold models in {OUTPUT_DIR}{ENDC}")

# Candidate naming schemes
possible_fold_files = []
for i in range(N_FOLDS):
    possible_fold_files.append([
        f'{model_name}_fold_{i+1}.keras',
        f'fold_{i+1}.keras',
        f'alexnet_new_fold_{i+1}.keras',
        f'model_fold_{i+1}.keras'
    ])

for i in range(N_FOLDS):
    loaded = False
    for fname in possible_fold_files[i]:
        mpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(mpath):
            try:
                model = load_model(mpath, compile=False)
                models.append(model)
                print(f"{GREEN}Loaded model fold {i+1}: {fname}{ENDC}")
                loaded = True
                break
            except Exception as e:
                print(f"{RED}Error loading {fname}: {e}{ENDC}")
    if not loaded:
        print(f"{YELLOW}Warning: Fold {i+1} model file not found in {OUTPUT_DIR}{ENDC}")

if not models:
    # Try searching any .keras files in OUTPUT_DIR
    keras_files = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.keras')])
    for kf in keras_files[:N_FOLDS]:
        try:
            m = load_model(kf, compile=False)
            models.append(m)
            print(f"{GREEN}Loaded fallback model: {os.path.basename(kf)}{ENDC}")
        except Exception as e:
            print(f"{RED}Error loading fallback {kf}: {e}{ENDC}")

if not models:
    print(f"{RED}Error: No valid models loaded for Deep Ensemble.{ENDC}")
    sys.exit(1)

print(f"{YELLOW}Making ensemble predictions across {len(models)} models...{ENDC}")
predictions_mean, uncertainty_total, predictions_stacked = ensamble_predictions(models, images_array)
print(f'{GREEN}Predictions completed. Stack shape: {predictions_stacked.shape}{ENDC}')

# --- 5. GUARDAR CSV Y STACK ---
print(f"{YELLOW}Generating CSV...{ENDC}")
rows = []
for i, parsed in enumerate(all_parsed):
    row = {
        'row_idx': i,
        'original_id': int(parsed.get('image_idx', -1)),
        'theta_E_true': float(parsed['theta_E']),
        'f_axis_true': float(parsed['f_axis']),
        'e1_true': float(parsed['e1']),
        'e2_true': float(parsed['e2']),
        'theta_E_pred': float(predictions_mean[i, 0]),
        'f_axis_pred': float(predictions_mean[i, 1]),
        'e1_pred': float(predictions_mean[i, 2]),
        'e2_pred': float(predictions_mean[i, 3]),
        'f_s': float(parsed['f_s']),
        're_s': float(parsed['re_s']),
        're_l': float(parsed['re_l']),
        'pa_l': float(parsed['pa_l']),
        'pa_s': float(parsed['pa_s']),
        'x_s': float(parsed['center_x']),
        'y_s': float(parsed['center_y']),
        'e1_s': float(parsed['e1_s']),
        'e2_s': float(parsed['e2_s']),
    }
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, 'predictions_vs_real.csv')
csv_path_orig = os.path.join(OUTPUT_DIR, 'predictions_vs_original.csv')
df.to_csv(csv_path, index=False)
df.to_csv(csv_path_orig, index=False)
print(f'{GREEN}CSVs saved in {csv_path} and {csv_path_orig}{ENDC}')

npy_path = os.path.join(OUTPUT_DIR, 'predictions_stacked_deepensemble.npy')
np.save(npy_path, predictions_stacked)
print(f'{GREEN}Predictions stack saved in {npy_path}{ENDC}')

# --- 6. GENERACIÓN DE IMÁGENES (LENSTRONOMY) ---
def generate_lens_image(theta_E, f_s, e1_l, e2_l, re_s, re_l, pa_s, x_s, y_s):
    e_s = (1.0 - f_s) / (1.0 + f_s)
    e1_s = e_s * np.cos(2 * pa_s)
    e2_s = e_s * np.sin(2 * pa_s)

    x, y = np.meshgrid(
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX),
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX))

    lens_model = LensModel(['SIE'])
    lens_kwargs = [{'theta_E': theta_E, 'e1': e1_l, 'e2': e2_l, 'center_x': 0.0, 'center_y': 0.0}]

    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    lens_light_kwargs = [{'amp': 8., 'R_sersic': re_l, 'n_sersic': 4.0, 'e1': e1_l, 'e2': e2_l, 'center_x': 0.0, 'center_y': 0.0}]

    source_light_model = LightModel(['SERSIC_ELLIPSE'])
    source_kwargs = [{'amp': 50.0, 'R_sersic': re_s, 'n_sersic': 2.0, 'e1': e1_s, 'e2': e2_s, 'center_x': x_s, 'center_y': y_s}]

    try:
        lens_light = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
        x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)
        source_light = source_light_model.surface_brightness(x_lensed, y_lensed, source_kwargs)
        return (lens_light + source_light).reshape(NUM_PIX, NUM_PIX)
    except Exception as e:
        print(f"Error generating lenstronomy image: {e}")
        return np.zeros((NUM_PIX, NUM_PIX))

# --- 7. GUARDAR TFRECORDS ---
TFRECORD_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "original")
TFRECORD_PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(TFRECORD_ORIGINAL_DIR, exist_ok=True)
os.makedirs(TFRECORD_PRED_DIR, exist_ok=True)

tfrecord_original_path = os.path.join(TFRECORD_ORIGINAL_DIR, "lenses_original.tfrecord")
tfrecord_pred_path = os.path.join(TFRECORD_PRED_DIR, "lenses_predicted.tfrecord")

print(f"{YELLOW}Generating images and saving TFRecords...{ENDC}")
with tf.io.TFRecordWriter(tfrecord_original_path) as writer_true, \
     tf.io.TFRecordWriter(tfrecord_pred_path) as writer_pred:
    
    for i in tqdm(range(len(df)), desc="Saving TFRecords"):
        img_true = generate_lens_image(
            theta_E = df.loc[i, 'theta_E_true'],
            f_s     = df.loc[i, 'f_s'],
            e1_l    = df.loc[i, 'e1_true'],
            e2_l    = df.loc[i, 'e2_true'],
            re_s    = df.loc[i, 're_s'],
            re_l    = df.loc[i, 're_l'],
            pa_s    = df.loc[i, 'pa_s'],
            x_s     = df.loc[i, 'x_s'],
            y_s     = df.loc[i, 'y_s']
        )
        
        img_pred = generate_lens_image(
            theta_E = df.loc[i, 'theta_E_pred'],
            f_s     = df.loc[i, 'f_s'],
            e1_l    = df.loc[i, 'e1_pred'],
            e2_l    = df.loc[i, 'e2_pred'],
            re_s    = df.loc[i, 're_s'],
            re_l    = df.loc[i, 're_l'],
            pa_s    = df.loc[i, 'pa_s'],
            x_s     = df.loc[i, 'x_s'],
            y_s     = df.loc[i, 'y_s']
        )
        
        features_true = {
            'image_idx': _int_feature(int(df.loc[i, 'row_idx'])),
            'image': _bytes_feature(img_true.astype(np.float32).tobytes()),
            'theta_E': _float_feature(df.loc[i, 'theta_E_true']),
            'f_axis': _float_feature(df.loc[i, 'f_axis_true']),
            'e1': _float_feature(df.loc[i, 'e1_true']),
            'e2': _float_feature(df.loc[i, 'e2_true']),
            're_s': _float_feature(df.loc[i, 're_s']),
            're_l': _float_feature(df.loc[i, 're_l']),
            'pa_s': _float_feature(df.loc[i, 'pa_s']),
            'x_s': _float_feature(df.loc[i, 'x_s']),
            'y_s': _float_feature(df.loc[i, 'y_s']),
        }
        writer_true.write(tf.train.Example(features=tf.train.Features(feature=features_true)).SerializeToString())
        
        features_pred = {
            'image_idx': _int_feature(int(df.loc[i, 'row_idx'])),
            'image': _bytes_feature(img_pred.astype(np.float32).tobytes()),
            'theta_E': _float_feature(df.loc[i, 'theta_E_pred']),
            'f_axis': _float_feature(df.loc[i, 'f_axis_pred']),
            'e1': _float_feature(df.loc[i, 'e1_pred']),
            'e2': _float_feature(df.loc[i, 'e2_pred']),
            're_s': _float_feature(df.loc[i, 're_s']),
            're_l': _float_feature(df.loc[i, 're_l']),
            'pa_s': _float_feature(df.loc[i, 'pa_s']),
            'x_s': _float_feature(df.loc[i, 'x_s']),
            'y_s': _float_feature(df.loc[i, 'y_s']),
        }
        writer_pred.write(tf.train.Example(features=tf.train.Features(feature=features_pred)).SerializeToString())

print(f"{GREEN}TFRecords successfully saved in {OUTPUT_DIR}{ENDC}")

# --- 8. VISUALIZACIÓN DE RESULTADOS CON SCATTER ---

def plot_results_with_scatter(df_meta, idx, preds_stacked, output_dir, labels_list):
    row_match = df_meta.loc[df_meta['row_idx'] == idx]
    if len(row_match) == 0:
        return
    row_data = row_match.iloc[0]
    display_id = int(row_data.get('row_idx', idx))
    
    img_true = generate_lens_image(
        row_data['theta_E_true'], row_data['f_s'], row_data['e1_true'], row_data['e2_true'],
        row_data['re_s'], row_data['re_l'], row_data['pa_s'], row_data['x_s'], row_data['y_s']
    )

    img_pred = generate_lens_image(
        row_data['theta_E_pred'], row_data['f_s'], row_data['e1_pred'], row_data['e2_pred'],
        row_data['re_s'], row_data['re_l'], row_data['pa_s'], row_data['x_s'], row_data['y_s']
    )

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    labels_for_plot = [r'$\theta_E$', r'$f$', r'$e_1$', r'$e_2$']
    ax0.imshow(img_true, cmap='hot', origin='lower')
    ax0.set_title(f"Original (Idx: {display_id})", fontsize=12)
    ax0.axis('off')
    txt_true = "\n".join([f"{labels_for_plot[l]}: {row_data[label+'_true']:.3f}" for l, label in enumerate(labels_list)])
    ax0.text(0.05, 0.05, txt_true, color='white', fontsize=9, transform=ax0.transAxes, bbox=dict(facecolor='black', alpha=0.6))

    ax1.imshow(img_pred, cmap='hot', origin='lower')
    ax1.set_title("Prediction", fontsize=12)
    ax1.axis('off')
    txt_pred = "\n".join([f"{labels_for_plot[l]}: {row_data[label+'_pred']:.3f}" for l, label in enumerate(labels_list)])
    ax1.text(0.05, 0.05, txt_pred, color='white', fontsize=9, transform=ax1.transAxes, bbox=dict(facecolor='black', alpha=0.6))

    if idx < preds_stacked.shape[1]:
        folds_data_for_img = preds_stacked[:, idx, :]
        x_positions = np.arange(len(labels_list))

        for i in range(folds_data_for_img.shape[0]):
            ax2.scatter(x_positions, folds_data_for_img[i, :], color='black', alpha=0.4, s=20, zorder=2, label='Models' if i == 0 else "")

        means = [row_data[l+'_pred'] for l in labels_list]
        ax2.scatter(x_positions, means, color='crimson', marker='*', s=80, zorder=2, label='Average')
        
        trues = [row_data[l+'_true'] for l in labels_list]
        ax2.scatter(x_positions, trues, color='green', marker='x', s=80, zorder=2, label='Real')

        ax2.set_title("Ensemble Scatter", fontsize=12)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(labels_for_plot, fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y', zorder=0)
        
        handles, labels_leg = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analysis_idx_{display_id:04d}.png"), dpi=150)
    plt.close(fig)

def plot_histograms_comparison(df, current_labels, output_dir, prueba_id):
    labels_map = {'theta_E': r'$\theta_E$', 'f_axis': r'$f$', 'e1': r'$e_1$', 'e2': r'$e_2$'}
    fig, axs = plt.subplots(1, len(current_labels), figsize=(6*len(current_labels), 5))
    if len(current_labels) == 1:
        axs = [axs]
    for i, label in enumerate(current_labels):
        true_vals = df[f'{label}_true']
        pred_mean_vals = df[f'{label}_pred']
        min_val = min(true_vals.min(), pred_mean_vals.min())
        max_val = max(true_vals.max(), pred_mean_vals.max())
        bins = np.linspace(min_val, max_val, 50)
        mean_true = np.mean(true_vals)
        mean_pred = np.mean(pred_mean_vals)

        axs[i].hist(true_vals, bins=bins, density=True, alpha=0.8, color='skyblue', label='True')
        axs[i].hist(pred_mean_vals, bins=bins, density=True, alpha=0.8, color='pink', label='Predicted')
        axs[i].axvline(mean_true, color='blue', linestyle='dashed', linewidth=1, label=f'True Mean: {mean_true:.3f}')
        axs[i].axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f'Pred Mean: {mean_pred:.3f}')
        axs[i].set_title(f'Distribution: {labels_map.get(label, label)}')
        axs[i].legend()
        axs[i].set_xlabel(labels_map.get(label, label))
        axs[i].set_ylabel('Density')
        axs[i].grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f'hist_{prueba_id}.pdf'), bbox_inches='tight')
    plt.close()

# Visualization
N_VISUALIZATION = 9
total_samples = len(df)
candidate_indices = [849, 919, 10, 204, 305, 412, 512, 612, 712, 1275, 1157, 939, 2185]
selected_row_indices = [idx for idx in candidate_indices if idx < total_samples][:N_VISUALIZATION]
if not selected_row_indices:
    selected_row_indices = list(range(min(total_samples, N_VISUALIZATION)))

labels_for_plot = ['theta_E', 'f_axis', 'e1', 'e2'] 

for idx in tqdm(selected_row_indices, desc="Analysing Results"):
    plot_results_with_scatter(
        df_meta=df,
        idx=idx,
        preds_stacked=predictions_stacked,
        output_dir=OUTPUT_DIR,
        labels_list=labels_for_plot
    )

idx_theta_E_match = df.loc[np.isclose(df['theta_E_true'], 1.2830, atol=1e-3)].index
if len(idx_theta_E_match) > 0:
    plot_results_with_scatter(
        df_meta=df,
        idx=idx_theta_E_match[0],
        preds_stacked=predictions_stacked,
        output_dir=OUTPUT_DIR,
        labels_list=labels_for_plot
    )

print(f"{YELLOW}Generating comparative histograms...{ENDC}")
plot_histograms_comparison(df, labels_for_plot, OUTPUT_DIR, "comparative_histograms")

def get_ensemble_metrics(models, X_data, y_real):
    all_preds = []
    for model in models:
        preds = model.predict(X_data)
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    uncertainties = np.std(all_preds, axis=0)
    ensemble_mean = np.mean(all_preds, axis=0)
    errors = np.abs(ensemble_mean - y_real)
    return uncertainties, errors

split_idx = min(len(images_array) // 2, TEST_IMAGES)
if split_idx > 0 and len(images_array) > split_idx:
    X_train_full = images_array[:split_idx]
    y_train_full = targets_array[:split_idx]
    X_test = images_array[split_idx:]
    y_test = targets_array[split_idx:]
    unc_train, err_train = get_ensemble_metrics(models, X_train_full, y_train_full)
    unc_test, err_test = get_ensemble_metrics(models, X_test, y_test)

    def plot_uncertainty_behavior(uncertainties_train, errors_train, uncertainties_test, errors_test, n_bins=15):
        def get_binned_stats(unc, err, bins):
            bin_indices = np.digitize(unc, bins)
            bin_means_err = []
            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.any(mask):
                    bin_means_err.append(np.mean(err[mask]))
                else:
                    bin_means_err.append(np.nan)
            return np.array(bin_means_err)

        all_unc = np.concatenate([uncertainties_train, uncertainties_test])
        bins = np.linspace(all_unc.min(), all_unc.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        train_err_bins = get_binned_stats(uncertainties_train, errors_train, bins)
        test_err_bins = get_binned_stats(uncertainties_test, errors_test, bins)
        gap = test_err_bins - train_err_bins

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(bin_centers, train_err_bins, 'o-', label='Train Error', color='blue', alpha=0.7)
        ax1.plot(bin_centers, test_err_bins, 's-', label='Test Error', color='red', alpha=0.7)
        ax1.plot([0, bins.max()], [0, bins.max()], '--k', alpha=0.3, label='Perfect Calibration')
        ax1.set_xlabel('Ensemble Uncertainty (STD)')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Uncertainty Calibration')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.bar(bin_centers, gap, width=(bin_centers[1]-bin_centers[0])*0.8, 
                color='salmon', alpha=0.8, label='Gap (Test - Train)')
        ax2.axhline(0, color='black', lw=1)
        ax2.set_xlabel('Ensemble Uncertainty (STD)')
        ax2.set_ylabel('Error Difference (Test - Train)')
        ax2.set_title('Generalization Gap vs. Uncertainty')
        ax2.legend()
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_behavior.pdf'), dpi=300)
        plt.close()

    print(f"{YELLOW}Plotting uncertainty behavior...{ENDC}")
    plot_uncertainty_behavior(unc_train, err_train, unc_test, err_test)

print(f'{GREEN}Deep ensemble testing completed successfully. Outputs saved to {OUTPUT_DIR}.{ENDC}')
