
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from tqdm import tqdm
import configparser

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# --- 1. CONFIGURACIÓN ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
    MAIN_PATH = main_config['PATHS']['main_path']
    TEST_PATH = main_config['PATHS']['tfrecords_path_test']
    PRUEBA = int(main_config['CONFIG']['prueba'])
    NUM_PIX = int(main_config['MODEL']['num_pix'])
    CHANNELS = int(main_config['MODEL']['channels'])
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    BATCH_SIZE = int(main_config['MODEL']['batch_size'])
    N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])
except Exception as e:
    print(f"{RED}Error cargando configuración: {e}{ENDC}")
    # Default fallbacks
    MAIN_PATH = './'
    TEST_PATH = './tfrecords/test'
    PRUEBA = 1
    NUM_PIX = 100
    CHANNELS = 1
    LABELS = ['theta_E', 'f_axis', 'e1', 'e2']
    N_FOLDS = 5

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
OUTPUT_DIR = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/") # Carpeta nueva para CP
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
        print(f"{RED}Path no encontrado: {tfrecord_dir}{ENDC}")
        return None
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 3. DATA LOADING ---
print(f'{YELLOW}Loading dataset from {TEST_PATH}{ENDC}')
inference_batch_size = 64
dataset = load_tfrecord_dataset(TEST_PATH, inference_batch_size)

print(f"{YELLOW}Extracting data into memory...{ENDC}")
all_images = []
all_parsed = []
# Necesitamos los targets reales en un array numpy para CP
targets_list = [] 

for img_batch, parsed_batch in tqdm(dataset, desc="Loading Data"):
    all_images.append(img_batch.numpy())
    batch_len = img_batch.shape[0]
    keys = parsed_batch.keys()
    
    # Extract targets for CP (theta_E, f, e1, e2)
    # Make sure the order matches LABELS
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
targets_array = np.concatenate(targets_list, axis=0) # Shape (N, 4)

print(f'{CYAN}Total Dataset:{ENDC} {len(all_parsed)} samples loaded.')

# ### NUEVO CP ###: División Calibration vs Test
# Usaremos el 50% para calibrar y el 50% para testear (o ajusta N_CAL)
N_TOTAL = len(images_array)
N_CAL = int(N_TOTAL * 0.5) 

# Índices aleatorios (fijamos seed para reproducibilidad)
np.random.seed(42)
indices = np.arange(N_TOTAL)
np.random.shuffle(indices)

cal_idx = indices[:N_CAL]
test_idx = indices[N_CAL:]

print(f"{CYAN}Splitting data:{ENDC} {len(cal_idx)} Calibration samples, {len(test_idx)} Test samples.")

X_cal = images_array[cal_idx]
y_cal = targets_array[cal_idx]
parsed_cal = [all_parsed[i] for i in cal_idx]

X_test = images_array[test_idx]
y_test = targets_array[test_idx]
parsed_test = [all_parsed[i] for i in test_idx]

# --- 4. ENSAMBLE AND PREDICTION ---
def ensamble_predictions(models, images):
    preds_folds = []
    inference_batch = 64 
    for i, model in enumerate(models):
        # Silenciamos el verbose para no llenar la consola
        p = model.predict(images, batch_size=inference_batch, verbose=0)
        preds_folds.append(p)
            
    predictions_stacked = np.stack(preds_folds, axis=0)
    predictions_mean = np.mean(predictions_stacked, axis=0) 
    uncertainty = np.std(predictions_stacked, axis=0)       

    return predictions_mean, uncertainty, predictions_stacked

# Cargar modelos
print(f"{YELLOW}Loading models...{ENDC}")
weights = [1.0, 1.0, 3.0, 3.0]
custom_loss = get_weighted_loss(weights, num_outputs=len(LABELS))
models = []

model_dir_source = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/") 
for i in range(N_FOLDS):
    # Ajusta esta ruta si tus modelos están en otro lado
    name = f'alexnet_fold_{i+1}.keras'
    model_path = os.path.join(model_dir_source, name)
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, custom_objects={'loss': custom_loss, 'weighted_mse': custom_loss}, compile=False)
            models.append(model)
        except Exception as e:
            print(f"{RED}Error loading {name}: {e}{ENDC}")
    else:
        print(f"{RED}Model {name} not found.{ENDC}")

if not models:
    raise ValueError("No models loaded.")

# ### NUEVO CP ###: Predecir en ambos sets
print(f"{YELLOW}Predicting on Calibration Set...{ENDC}")
cal_mean, cal_std, _ = ensamble_predictions(models, X_cal)

print(f"{YELLOW}Predicting on Test Set...{ENDC}")
test_mean, test_std, test_stacked = ensamble_predictions(models, X_test)


# --- 5. LÓGICA CONFORMAL PREDICTION (Locally Adaptive) ---
print(f"{YELLOW}Calibrating Conformal Prediction Intervals...{ENDC}")

ALPHA = 0.05  # 95% Confianza
q_hats = []   # Guardaremos un q_hat por cada label

# Iterar sobre cada parámetro (theta_E, f, e1, e2)
for i in range(len(LABELS)):
    # 1. Calcular scores de no-conformidad en CALIBRATION
    # score = |y_true - y_pred| / (sigma_pred + epsilon)
    # epsilon evita división por cero
    epsilon = 1e-6
    scores = np.abs(y_cal[:, i] - cal_mean[:, i]) / (cal_std[:, i] + epsilon)
    
    # 2. Calcular cuantil (1 - alpha)
    # Usamos np.quantile con método 'higher' para ser conservadores
    n = len(scores)
    q_val = np.quantile(scores, np.ceil((n+1)*(1-ALPHA))/n)
    q_hats.append(q_val)
    
    print(f"  > Label {LABELS[i]}: q_hat = {q_val:.4f}")

q_hats = np.array(q_hats)

# 3. Aplicar a TEST
# Interval width = 2 * (q_hat * sigma)
test_lower = test_mean - (q_hats * (test_std + 1e-6))
test_upper = test_mean + (q_hats * (test_std + 1e-6))

# Calcular cobertura empírica en Test
coverage = np.mean((y_test >= test_lower) & (y_test <= test_upper), axis=0)
print(f"{GREEN}Test Coverage (Target 95%): {coverage}{ENDC}")

# --- 6. GUARDAR CSV ACTUALIZADO ---
print(f"{YELLOW}Generating CSV with CP Intervals...{ENDC}")
rows = []
for i in range(len(test_idx)):
    original_data = parsed_test[i]
    row = {
        'row_idx': test_idx[i], # Índice original global
        'original_id': int(original_data.get('image_idx', -1)),
        
        # Real Values
        'theta_E_true': y_test[i, 0],
        'f_axis_true':       y_test[i, 1],
        'e1_true':      y_test[i, 2],
        'e2_true':      y_test[i, 3],
        
        # Predictions (Mean)
        'theta_E_pred': test_mean[i, 0],
        'f_axis_pred':       test_mean[i, 1],
        'e1_pred':      test_mean[i, 2],
        'e2_pred':      test_mean[i, 3],
        
        # Predictions (Uncertainty Sigma)
        'theta_E_std': test_std[i, 0],
        'f_axis_std':       test_std[i, 1],
        'e1_std':      test_std[i, 2],
        'e2_std':      test_std[i, 3],

        # CP Intervals (Lower & Upper)
        'theta_E_lower': test_lower[i, 0], 'theta_E_upper': test_upper[i, 0],
        'f_axis_lower':  test_lower[i, 1], 'f_axis_upper':  test_upper[i, 1],
        'e1_lower':      test_lower[i, 2], 'e1_upper':      test_upper[i, 2],
        'e2_lower':      test_lower[i, 3], 'e2_upper':      test_upper[i, 3],

        # Extra Params needed for plotting
        'f_s': original_data['f_s'],
        're_s': original_data['re_s'],
        're_l': original_data['re_l'],
        'pa_l': original_data['pa_l'],
        'pa_s': original_data['pa_s'],
        'x_s': original_data['center_x'],
        'y_s': original_data['center_y'],
    }
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, 'predicionts_vs_real.csv')
df.to_csv(csv_path, index=False)
print(f'{GREEN}CSV saved in {csv_path}{ENDC}')

# --- 7. RESULTS PLOTTING FUNCTION ---
def generate_lens_image(theta_E, f_s, e1_l, e2_l, re_s, re_l, pa_s, x_s, y_s):
    # Misma función que tenías
    try:
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
        lens_light = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
        x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)
        source_light = source_light_model.surface_brightness(x_lensed, y_lensed, source_kwargs)
        img = (lens_light + source_light).reshape(NUM_PIX, NUM_PIX)
        return img
    except:
        return np.zeros((NUM_PIX, NUM_PIX))

def plot_cp_results(df_meta, idx_in_df, preds_stacked_test, output_dir, labels_list):
    '''
    Plotea Original | Prediction | CP Intervals
    idx_in_df: índice dentro del DataFrame (0 a len(test_set))
    preds_stacked_test: Stacked predictions correspondientes al Test Set
    '''
    row_data = df_meta.loc[idx_in_df]
    
    img_true = generate_lens_image(
        row_data['theta_E_true'], row_data['f_s'], row_data['e1_true'], row_data['e2_true'],
        row_data['re_s'], row_data['re_l'], row_data['pa_l'], row_data['x_s'], row_data['y_s']
    )
    img_pred = generate_lens_image(
        row_data['theta_E_pred'], row_data['f_s'], row_data['e1_pred'], row_data['e2_pred'],
        row_data['re_s'], row_data['re_l'], row_data['pa_l'], row_data['x_s'], row_data['y_s']
    )

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    labels_plot = [r'$\theta_E$', r'$f$', r'$e_1$', r'$e_2$']
    
    # Imagen Real
    ax0.imshow(img_true, cmap='hot', origin='lower')
    txt_true = "\n".join([f"{labels_plot[l]}: {row_data[label+'_true']:.3f}" for l, label in enumerate(labels_list)])
    ax0.text(0.05, 0.05, txt_true, color='white', fontsize=9, transform=ax0.transAxes, bbox=dict(facecolor='black', alpha=0.6))
    ax0.set_title(f"Original (ID: {idx_in_df})", fontsize=12)
    ax0.axis('off')

    # Imagen Predicha (Media)
    ax1.imshow(img_pred, cmap='hot', origin='lower')
    ax1.set_title("Prediction (Mean)", fontsize=12)
    txt_pred = "\n".join([f"{labels_plot[l]}: {row_data[label+'_pred']:.3f}" for l, label in enumerate(labels_list)])
    ax1.text(0.05, 0.05, txt_pred, color='white', fontsize=9, transform=ax1.transAxes, bbox=dict(facecolor='black', alpha=0.6))
    ax1.axis('off')

    # Scatter + Intervalos CP
    x_pos = np.arange(len(labels_list))
    
    # 1. Intervalos CP (Barras de Error)
    # Calculamos el error asimétrico relativo a la media para errorbar (aunque aquí es simétrico respecto a q*sigma)
    y_err = [
        [row_data[l+'_pred'] - row_data[l+'_lower'] for l in labels_list], # Abajo
        [row_data[l+'_upper'] - row_data[l+'_pred'] for l in labels_list]  # Arriba
    ]
    
    # Dibujar la barra de intervalo CP (95%)
    ax2.errorbar(x_pos, [row_data[l+'_pred'] for l in labels_list], 
                 yerr=y_err, fmt='none', ecolor='k', elinewidth=2, capsize=5,
                 label='95% CP Interval', zorder=1)

    # 2. Scatter de Modelos Individuales (Deep Ensemble)
    # preds_stacked_test shape: (n_folds, n_test_samples, 4)
    # Accedemos usando idx_in_df
    current_folds = preds_stacked_test[:, idx_in_df, :]
    for k in range(current_folds.shape[0]):
        ax2.scatter(x_pos, current_folds[k, :], color='gray', alpha=0.3, s=15, zorder=2)

    # 3. Media
    ax2.scatter(x_pos, [row_data[l+'_pred'] for l in labels_list], 
                color='blue', marker='o', s=60, label='Mean Pred', zorder=3)

    # 4. Realidad
    ax2.scatter(x_pos, [row_data[l+'_true'] for l in labels_list], 
                color='red', marker='x', s=80, label='True Value', zorder=4)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_plot)
    ax2.set_title("Conformal Prediction Intervals", fontsize=12)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CP_analysis_{idx_in_df}.png"), dpi=100)
    plt.close(fig)

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

def barplot_IC(df, labels, output_dir, prueba_id):
    '''
    Generates a barplot of mean values for IC intervals and predicted values.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame containing all metadata
    - labels (list): List of label names in order
    - output_dir (str): Directory to save images
    - prueba_id (int): ID of the current test
    '''
    labels_map = {'theta_E': r'$\theta_E$', 'f_axis': r'$f$', 'e1': r'$e_1$', 'e2': r'$e_2$'}
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    
    for label, ax in zip(labels, ax):
        lower_vals = df[f'{label}_lower']
        upper_vals = df[f'{label}_upper']
        mean_vals = df[f'{label}_pred']
        ax.bar(['Lower', 'Upper', 'Mean'], [lower_vals.mean(), upper_vals.mean(), mean_vals.mean()], color=['blue', 'red', 'green'])
        ax.set_title(f'IC Mean Values: {labels_map.get(label, label)}')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f'barplot_{prueba_id}.pdf'), bbox_inches='tight')
    plt.close()

# Plotting some examples
print(f"{YELLOW}Visualizing results with CP Intervals...{ENDC}")
N_VISUALIZATION = 9
if len(df) > N_VISUALIZATION:
    vis_indices = sorted(random.sample(range(len(df)), N_VISUALIZATION))
else:
    vis_indices = range(len(df))

print(f"{CYAN}Visualizing {len(vis_indices)} examples...{ENDC}")
for idx in tqdm(vis_indices, desc="Plotting"):
    plot_cp_results(df, idx, test_stacked, OUTPUT_DIR, LABELS)

print(f"{YELLOW}Visualizing comparative histograms...{ENDC}")
plot_histograms_comparison(df, LABELS, OUTPUT_DIR, "comparative_histograms")

print(f"{YELLOW}Generating barplots of IC intervals...{ENDC}")
barplot_IC(df, LABELS, OUTPUT_DIR, "barplot_IC")

print(f'{GREEN}CP Process Finished.{ENDC}')
