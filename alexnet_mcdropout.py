
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import os
import configparser
import sys
import random
from tqdm import tqdm

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

# Seed
seed_value = 62
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- CONFIGURATION ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    PRUEBA = int(main_config['CONFIG']['prueba'])
    CLASSES = int(main_config['MODEL']['classes'])
    MAIN_PATH = main_config['PATHS']['main_path']
    MODEL_PATH = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
    TFRECORD_PATH_TEST = main_config['PATHS']['tfrecords_path_test']
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    NUM_PIX = int(main_config['MODEL']['num_pix'])
    CHANNLES = int(main_config['MODEL']['channels'])
    BATCH_SIZE = int(main_config['MODEL']['batch_size'])
except Exception as e:
    print(f"{RED}Error loading config: {e}{ENDC}")
    sys.exit(1)

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNLES)
MODEL_FILENAME = f'alexnet_paper_{PRUEBA}.keras'
MODEL_FULL_PATH = os.path.join(MODEL_PATH, MODEL_FILENAME)
DELTA_PIX = 0.08 # Asumido, ajustar si es necesario
NUMSAMPLES = 10 # Número de muestras MC Dropout

# --- FUNCIONES DE CARGA DE DATOS ---
def parse_tfrecord(example_proto):
    '''
    Parsea y devuelve: Imagen, Label (Target), y TODOS los Metadatos (Features)
    necesarios para la reconstrucción posterior.
    '''
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
    
    # Decodificar imagen
    image = tf.io.decode_raw(parsed_example['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    
    # Crear vector de etiquetas objetivo (para calcular error)
    label = tf.stack([parsed_example[label] for label in LABELS], axis = 0)
    
    # Devolvemos también el diccionario completo para reconstruir después
    return image, label, parsed_example

def load_tfrecord_dataset(tfrecord_files, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

# --- FUNCIÓN CORE: MC DROPOUT ---
def predict_mc_dropout(model, dataset, num_samples=50):
    '''
    Realiza inferencia MC Dropout y guarda metadatos para reconstrucción.
    '''
    print(f'{GREEN}Initializing MC Dropout with {num_samples} iterations per image...{ENDC}')
    
    mc_predictions = [] # List of tensors (samples, batch, output)
    y_true_list = []
    metadata_list = [] # Acumulador de diccionarios de metadatos
    
    batch_count = 0
    for x_batch, y_batch, meta_batch in dataset:
        y_true_list.append(y_batch.numpy())
        
        # Guardar metadatos del batch actual
        # meta_batch es un diccionario de tensores, lo convertimos a lista de dicts
        batch_size_current = x_batch.shape[0]
        for i in range(batch_size_current):
            single_meta = {k: v[i].numpy() for k, v in meta_batch.items()}
            metadata_list.append(single_meta)

        # MC Dropout Sampling
        batch_preds = []
        for i in range(num_samples):
            pred = model(x_batch, training=True) # Dropout ON
            batch_preds.append(pred)
        
        # Stack: (num_samples, batch_size, output_dim)
        batch_preds = tf.stack(batch_preds)
        mc_predictions.append(batch_preds)
        
        batch_count += 1
        print(f"{CYAN}Processing batch {batch_count}...{ENDC}", end='\r')

    print(f"\n{YELLOW}Calculating statistics...{ENDC}")
    
    # Concatenar resultados
    y_true = np.concatenate(y_true_list, axis=0)
    
    # mc_predictions es lista de (samples, batch, output). 
    # Concatenamos eje 1 (batch) -> (samples, total_imgs, output)
    all_preds_stack = np.concatenate(mc_predictions, axis=1)
    
    mean_preds = np.mean(all_preds_stack, axis=0)
    std_preds = np.std(all_preds_stack, axis=0)
    
    return mean_preds, std_preds, all_preds_stack, y_true, metadata_list

# --- VISUALIZACIÓN: HISTOGRAMAS Y PARITY ---
def plot_histograms_comparison(df, current_labels, output_dir, prueba_id):
    print(f"\n{GREEN}Generating comparative histograms...{ENDC}")
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

        axs[i].hist(true_vals, bins=bins, density=True, alpha=0.8, color='skyblue', stacked=True, label='True')
        axs[i].hist(pred_mean_vals, bins=bins, density=True, alpha=0.8, color='pink', stacked=True, label='Predicted')
        axs[i].axvline(mean_true, color='blue', linestyle='dashed', linewidth=1, label=f'True Mean: {mean_true:.3f}')
        axs[i].axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f'Pred Mean: {mean_pred:.3f}')
        axs[i].set_title(f'Distribution: {labels_map.get(label, label)}')
        axs[i].legend()
        axs[i].set_xlabel(labels_map.get(label, label))
        axs[i].set_ylabel('Density')
        axs[i].grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f'hist_{prueba_id}.pdf'), bbox_inches='tight')
    plt.close()

def plot_parity_with_confidence(df, current_labels, output_dir, prueba_id, max_points=300):
    print(f"\n{GREEN}Generating parity plots...{ENDC}")
    if len(df) > max_points:
        plot_df = df.sample(n=max_points, random_state=42)
    else:
        plot_df = df

    labels_map = {'theta_E': r'$\theta_E$', 'f': r'$f$', 'e1': r'$e_1$', 'e2': r'$e_2$'}
    for label in current_labels:
        plt.figure(figsize=(10, 10))
        plt.scatter(plot_df[f'{label}_true'], plot_df[f'{label}_pred'], alpha=0.5, color='red', label='Predictions')

        min_ax = min(plot_df[f'{label}_true'].min(), plot_df[f'{label}_pred'].min())
        max_ax = max(plot_df[f'{label}_true'].max(), plot_df[f'{label}_pred'].max())
        plt.plot([min_ax, max_ax], [min_ax, max_ax], 'k--')
        plt.title(f'Parity: {labels_map.get(label, label)}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(output_dir, f'parity_{label}_{prueba_id}.pdf'), bbox_inches='tight')
        plt.close()

# --- LENSTRONOMY GENERATOR ---
def generate_lens_image(theta_E, f_s, e1_l, e2_l, re_s, re_l, pa_s, x_s, y_s):
    # Convertir inputs a formato Lenstronomy
    # f_s es axis ratio q -> e = (1-q)/(1+q)
    e_s_val = (1.0 - f_s) / (1.0 + f_s)
    e1_s = e_s_val * np.cos(2 * pa_s)
    e2_s = e_s_val * np.sin(2 * pa_s)

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
        img = (lens_light + source_light).reshape(NUM_PIX, NUM_PIX)
        return img
    except Exception as e:
        print(f"Lenstronomy Error: {e}")
        return np.zeros((NUM_PIX, NUM_PIX))

# --- PLOT SCATTER VISUALIZATION ---
def plot_results_with_scatter(df_meta, idx, preds_stacked, output_dir, labels_list):
    row_data = df_meta.loc[idx]
    display_id = int(row_data.get('row_idx', idx))
    
    # 1. Imagen Real
    img_true = generate_lens_image(
        theta_E=row_data['theta_E_true'],
        f_s=row_data['f_s'],
        e1_l=row_data['e1_true'], 
        e2_l=row_data['e2_true'],
        re_s=row_data['re_s'],
        re_l=row_data['re_l'],
        pa_s=row_data['pa_s'],
        x_s=row_data['x_s'], 
        y_s=row_data['y_s']
    )

    # 2. Imagen Predicha (Media)
    img_pred = generate_lens_image(
        theta_E=row_data['theta_E_pred'], 
        f_s=row_data['f_s'], # Usamos f_s real pq no se predijo
        e1_l=row_data['e1_pred'],
        e2_l=row_data['e2_pred'],
        re_s=row_data['re_s'],
        re_l=row_data['re_l'],
        pa_s=row_data['pa_s'], 
        x_s=row_data['x_s'], 
        y_s=row_data['y_s']
    )

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    labels_map = {'theta_E': r'$\theta_E$', 'f': r'$f$', 'e1': r'$e_1$', 'e2': r'$e_2$'}
    
    # Plot Original
    ax0.imshow(img_true, cmap='hot', origin='lower')
    txt_true = "\n".join([f"{labels_map[label]}: {row_data[label+'_true']:.3f}" for label in labels_list])
    ax0.text(0.05, 0.05, txt_true, color='white', fontsize=9, transform=ax0.transAxes, bbox=dict(facecolor='black', alpha=0.6))
    ax0.set_title(f"Original (Idx: {display_id})")
    ax0.axis('off')

    # Plot Prediction
    ax1.imshow(img_pred, cmap='hot', origin='lower')
    txt_pred = "\n".join([f"{labels_map[label]}: {row_data[label+'_pred']:.3f}" for label in labels_list])
    ax1.text(0.05, 0.05, txt_pred, color='white', fontsize=9, transform=ax1.transAxes, bbox=dict(facecolor='black', alpha=0.6))
    ax1.set_title("Prediction (Mean)")
    ax1.axis('off')

    # Plot Scatter
    # preds_stacked shape: (n_samples, n_imgs, n_params)
    # Extraemos slice para esta imagen: (n_samples, n_params)
    #img_preds_samples = preds_stacked[:, idx, :]
    x_pos = np.arange(len(labels_list))

    # B) Media y Real
    means = [row_data[l+'_pred'] for l in labels_list]
    trues = [row_data[l+'_true'] for l in labels_list]
    
    ax2.errorbar(x_pos, means, yerr=[row_data[l+'_uncertainty'] for l in labels_list],
                 fmt='o', color='k', ecolor='red', elinewidth=2, capsize=5, label='MCDropout', zorder=4)
    ax2.scatter(x_pos, trues, color='green', marker='x', s=60, label='True', zorder=5)
    
    y_min_plot, y_max_plot = ax2.get_ylim()
    y_range = y_max_plot - y_min_plot

    folds = preds_stacked[:, idx, :]
    for i, label in enumerate(labels_list):
        val_std = row_data[f'{label}_uncertainty']
        max_point = means[i] + val_std

        ax2.text(x_pos[i], max_point, f'$\sigma$: {val_std:.3f}',
                 ha='center', va='bottom', fontsize=8, color='black', zorder=6)
    
    ax2.set_title("Uncertainty Scatter")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([labels_map.get(l, l) for l in labels_list])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"visual_idx_{display_id:04d}.png"), dpi=150)
    plt.close(fig)

# --- TFRECORD HELPERS ---
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# --- MAIN ---
def main():
    try:
        print(f'{GREEN}Searching TFRecords in: {TFRECORD_PATH_TEST}{ENDC}')
        tf_files = sorted([os.path.join(TFRECORD_PATH_TEST, f) for f in os.listdir(TFRECORD_PATH_TEST) if f.endswith('.tfrecord')])
        if not tf_files: raise FileNotFoundError("No files found.")

        # 1. Cargar datos
        test_dataset = load_tfrecord_dataset(tf_files, BATCH_SIZE)

        print(f"{GREEN}Loading model from: {MODEL_FULL_PATH}{ENDC}")
        # Cargar modelo con función de pérdida personalizada de pérdidas ponderadas
        weights = [1.0, 1.0, 3.0, 3.0]
        custom_loss = get_weighted_loss(loss_weights=weights, num_outputs=CLASSES)
        model = load_model(MODEL_FULL_PATH, custom_objects={'loss': custom_loss}, compile=False)

        # 2. Inferencia (Ahora devuelve stack y metadata)
        start = time.time()
        mean_preds, std_preds, stack_preds, y_true, metadata_list = predict_mc_dropout(model, test_dataset, num_samples=NUMSAMPLES)
        print(f"{GREEN}Inference done in {(time.time()-start)/60:.2f} min.{ENDC}")

        # 3. Construir DataFrame Maestro
        print(f"{YELLOW}Building DataFrame...{ENDC}")
        rows = []
        for i, meta in enumerate(metadata_list):
            row = {
                'row_idx': i,
                'image_idx': int(meta.get('image_idx', -1)),
                # True values (from metadata directly to be safe)
                'theta_E_true': meta['theta_E'],
                'f_true': meta['f_axis'],
                'e1_true': meta['e1'],
                'e2_true': meta['e2'],
                # Predictions
                'theta_E_pred': mean_preds[i, 0],
                'f_pred': mean_preds[i, 1],
                'e1_pred': mean_preds[i, 2],
                'e2_pred': mean_preds[i, 3],
                # Uncertainty
                'theta_E_uncertainty': std_preds[i, 0],
                'f_uncertainty': std_preds[i, 1],
                'e1_uncertainty': std_preds[i, 2],
                'e2_uncertainty': std_preds[i, 3],
                # Aux params needed for reconstruction
                'f_s': meta['f_s'],
                're_s': meta['re_s'],
                're_l': meta['re_l'],
                'pa_l': meta['pa_l'],
                'pa_s': meta['pa_s'],
                'x_s': meta['center_x'],
                'y_s': meta['center_y']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # Guardar CSV
        df.to_csv(os.path.join(MODEL_PATH, f'predictions_vs_real.csv'), index=False)

        npy_path = os.path.join(MODEL_PATH, f'predictions_stack_mcdropout.npy')
        np.save(npy_path, stack_preds)
        print(f'{GREEN}Predictions stack saved in {npy_path}{ENDC}')
        print(f"{YELLOW}DataFrame and predictions stack saved in {MODEL_PATH} for further analysis.{ENDC}")

        # 4. Plots Globales
        current_labels = ['theta_E', 'f', 'e1', 'e2']
        plot_histograms_comparison(df, current_labels, MODEL_PATH, PRUEBA)
        plot_parity_with_confidence(df, current_labels, MODEL_PATH, PRUEBA)

        # 5. Visualización detallada (Scatter + Imágenes)
        N_VISUALIZATION = 9
        indices = sorted(random.sample(range(len(df)), min(len(df), N_VISUALIZATION)))
        
        print(f"{YELLOW}Visualizing {len(indices)} samples...{ENDC}")
        for idx in tqdm(indices):
            plot_results_with_scatter(df, idx, stack_preds, MODEL_PATH, current_labels)

        # 6. Guardar TFRecords (Original vs Predicho)
        print(f"{YELLOW}Saving reconstructed TFRecords...{ENDC}")
        out_dir_orig = os.path.join(MODEL_PATH, "original")
        out_dir_pred = os.path.join(MODEL_PATH, "predictions")
        os.makedirs(out_dir_orig, exist_ok=True)
        os.makedirs(out_dir_pred, exist_ok=True)

        writer_orig = tf.io.TFRecordWriter(os.path.join(out_dir_orig, "lenses_original.tfrecord"))
        writer_pred = tf.io.TFRecordWriter(os.path.join(out_dir_pred, "lenses_predicted.tfrecord"))

        for i in tqdm(range(len(df)), desc="Writing Records"):
            # Generar imágenes
            img_true = generate_lens_image(
                df.loc[i, 'theta_E_true'], df.loc[i, 'f_s'], df.loc[i, 'e1_true'], df.loc[i, 'e2_true'],
                df.loc[i, 're_s'], df.loc[i, 're_l'], df.loc[i, 'pa_s'], df.loc[i, 'x_s'], df.loc[i, 'y_s']
            )
            img_pred = generate_lens_image(
                df.loc[i, 'theta_E_pred'], df.loc[i, 'f_s'], df.loc[i, 'e1_pred'], df.loc[i, 'e2_pred'],
                df.loc[i, 're_s'], df.loc[i, 're_l'], df.loc[i, 'pa_s'], df.loc[i, 'x_s'], df.loc[i, 'y_s']
            )

            # Escribir
            ft_true = {'image_idx': _int_feature(int(df.loc[i, 'row_idx'])),
                       'image': _bytes_feature(img_true.astype(np.float32).tobytes())}
            ft_pred = {'image_idx': _int_feature(int(df.loc[i, 'row_idx'])),
                       'image': _bytes_feature(img_pred.astype(np.float32).tobytes())}
            
            writer_orig.write(tf.train.Example(features=tf.train.Features(feature=ft_true)).SerializeToString())
            writer_pred.write(tf.train.Example(features=tf.train.Features(feature=ft_pred)).SerializeToString())
        
        writer_orig.close()
        writer_pred.close()
        print(f"{GREEN}All done.{ENDC}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{RED}Error: {e}{ENDC}")

if __name__ == '__main__':
    main()
