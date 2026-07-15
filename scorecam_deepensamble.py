
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import glob  # Necesario para buscar múltiples modelos
import cv2

# Colores para la terminal
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

# --- 1. CONFIGURATION AND DATA LOADING ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    DELTA_PIX = 0.05
    MAIN_PATH = main_config['PATHS']['main_path']
    TEST_PATH = main_config['PATHS']['tfrecords_path_test']
    PRUEBA = int(main_config['CONFIG']['prueba'])
    NUM_PIX = int(main_config['MODEL']['num_pix'])
    CHANNELS = int(main_config['MODEL']['channels'])
    LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
    BATCH_SIZE = int(main_config['MODEL']['batch_size'])
    TEST_IMAGES = int(main_config['MODEL']['test_images'])
    N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])
except Exception as e:
    print(f"{RED}Error cargando configuración: {e}{ENDC}")
    MAIN_PATH = './'
    TEST_PATH = './tfrecords/test'
    PRUEBA = 1
    NUM_PIX = 100
    CHANNELS = 1
    LABELS = ['theta_E', 'f_axis', 'e1', 'e2']
    N_FOLDS = 5

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
MODEL_PATH = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
OUTPUT_DIR = os.path.join(MODEL_PATH, 'viz_outputs_ensemble') # Carpeta diferenciada
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_tfrecord(example_proto):
    # ... (Misma función que tenías, sin cambios) ...
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
        parsed = tf.io.parse_single_example(example_proto, feature_description)
    except:
        return tf.zeros(INPUT_SHAPE), {} 
        
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, INPUT_SHAPE)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) + 1e-6)
    return image, parsed

def load_tfrecord_dataset(tfrecord_dir, batch_size):
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_weighted_loss(loss_weights=None, num_outputs=4):
    if loss_weights is None: loss_weights = [1.0] * num_outputs
    def weighted_mse(y_true, y_pred):
        total_loss = 0
        num_to_calc = min(num_outputs, y_pred.shape[1]) 
        for i in range(num_to_calc):
            mse = tf.reduce_mean(tf.square(y_true[:, i] - y_pred[:, i]))
            total_loss += loss_weights[i] * mse
        return total_loss
    return weighted_mse

# --- 2. VISUALIZATION LOGIC (BASE) ---
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('Not found last conv layer.')

def generate_score_cam_single(model, img_tensor, target_index, max_N=None):
    """Calcula Score-CAM para un solo modelo"""
    if len(img_tensor.shape) == 3: img_tensor = tf.expand_dims(img_tensor, 0)
    layer_name = find_last_conv_layer(model)
    activation_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.outputs])
    act_map, _ = activation_model(img_tensor)
    act_map = act_map[0] 
    num_filters = act_map.shape[-1]
    if max_N is not None and max_N < num_filters: num_filters = max_N
    
    masked_images_batch = []
    for i in range(num_filters):
        feature_map = act_map[:, :, i]
        feature_map = tf.expand_dims(tf.expand_dims(feature_map, 0), -1)
        feature_map = tf.image.resize(feature_map, (NUM_PIX, NUM_PIX))
        min_val = tf.reduce_min(feature_map)
        max_val = tf.reduce_max(feature_map)
        norm_feature_map = (feature_map - min_val) / (max_val - min_val + 1e-8)
        masked_image = img_tensor * norm_feature_map
        masked_images_batch.append(masked_image[0]) 
        
    masked_images_batch = tf.stack(masked_images_batch) 
    predictions = model.predict(masked_images_batch, verbose=0)
    scores = predictions[:, target_index]
    scores = tf.nn.softmax(scores).numpy()
    
    cam = np.zeros(act_map.shape[0:2], dtype=np.float32)
    for i in range(num_filters):
        cam += scores[i] * act_map[:, :, i].numpy()
    
    cam = np.maximum(cam, 0)
    if np.max(cam) != 0: cam = cam / np.max(cam)
    cam = tf.expand_dims(tf.expand_dims(cam, 0), -1)
    cam = tf.image.resize(cam, (NUM_PIX, NUM_PIX))[0, :, :, 0].numpy()
    return cam

def get_integrated_gradients_single(model, img_tensor, target_index, m_steps=50, baseline=None):
    """Calcula IG para un solo modelo"""
    img_tensor = tf.cast(img_tensor, tf.float32)
    if len(img_tensor.shape) == 3: img_tensor = tf.expand_dims(img_tensor, 0)
    if baseline is None: baseline = tf.zeros_like(img_tensor)

    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = img_tensor - baseline
    interpolated_images = baseline + alphas_x * delta

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        target_predictions = predictions[:, target_index]

    grads = tape.gradient(target_predictions, interpolated_images)
    avg_grads = tf.reduce_mean(grads, axis=0) 
    integrated_grads = delta[0] * avg_grads
    saliency_map = tf.reduce_mean(tf.abs(integrated_grads), axis=-1)

    min_val = tf.reduce_min(saliency_map)
    max_val = tf.reduce_max(saliency_map)
    saliency_map_norm = (saliency_map - min_val) / (max_val - min_val + 1e-8)
    return saliency_map_norm.numpy()

# --- 3. NEW WRAPPER FUNCTIONS FOR ENSEMBLE ---
def generate_ensemble_score_cam(models, img_tensor, target_index):
    ### ENSEMBLE CHANGE ###
    # Generates the map for each model and returns the average
    cams = []
    for model in models:
        cams.append(generate_score_cam_single(model, img_tensor, target_index))
    
    # Promedio de los mapas
    ensemble_cam = np.mean(cams, axis=0)
    
    # Normalización final del promedio
    if np.max(ensemble_cam) != 0: 
        ensemble_cam = ensemble_cam / np.max(ensemble_cam)
    return ensemble_cam

def get_ensemble_integrated_gradients(models, img_tensor, target_index, m_steps=50):
    ### CAMBIO ENSEMBLE ###
    # Genera el mapa IG para cada modelo y devuelve el promedio
    igs = []
    for model in models:
        igs.append(get_integrated_gradients_single(model, img_tensor, target_index, m_steps))
    
    ensemble_ig = np.mean(igs, axis=0)
    
    # Normalización final del promedio
    min_val = np.min(ensemble_ig)
    max_val = np.max(ensemble_ig)
    ensemble_ig = (ensemble_ig - min_val) / (max_val - min_val + 1e-8)
    return ensemble_ig

def predict_ensemble(models, img_tensor):
    ### CAMBIO ENSEMBLE ###
    # Predice con todos los modelos y saca estadísticas
    preds = []
    for model in models:
        p = model.predict(img_tensor, verbose=0)
        preds.append(p)
    
    preds_np = np.array(preds) # shape: [num_models, batch, num_outputs]
    mean_preds = np.mean(preds_np, axis=0)
    std_preds = np.std(preds_np, axis=0) # Incertidumbre
    return mean_preds, std_preds

# --- 4. PLOTTING FUNCTIONS ---
def create_overlay(img_viz_color, heatmap_map, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = (heatmap_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_viz_color, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def plot_combined_viz(img, score_cam, saliency_map, param_name, true_val, pred_mean, pred_std, output_path):
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img_viz_gray = (img_norm * 255).astype(np.uint8)
    if len(img_viz_gray.shape) == 2 or img_viz_gray.shape[-1] == 1:
        img_viz_color = cv2.cvtColor(img_viz_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_viz_color = img_viz_gray

    # Panel 1: Original + Info Ensemble
    ax[0].imshow(img_viz_color)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    # Mostramos Media y Desviación Estándar (incertidumbre)
    info_text = f"Param: {param_name}\nTrue: {true_val:.4f}\nEns Mean: {pred_mean:.4f}\nEns Std: {pred_std:.4f}"
    ax[0].text(0.03, 0.97, info_text, transform=ax[0].transAxes, 
               color='white', fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    # Panel 2: Score-CAM Heatmap
    heatmap_display = ax[1].imshow(score_cam, cmap='jet', vmin=0, vmax=1)
    ax[1].set_title('Ensemble Score-CAM')
    ax[1].axis('off')
    plt.colorbar(heatmap_display, ax=ax[1], fraction=0.046, pad=0.04)

    # Panel 3: Score-CAM Overlay
    sc_overlay = create_overlay(img_viz_color, score_cam, cv2.COLORMAP_JET)
    ax[2].imshow(sc_overlay)
    ax[2].set_title('Score-CAM Overlay')
    ax[2].axis('off')

    # Panel 4: Integrated Gradients Heatmap
    saliency_display = ax[3].imshow(saliency_map, cmap='hot', vmin=0, vmax=1)
    ax[3].set_title('Ensemble Int. Gradients') 
    ax[3].axis('off')
    plt.colorbar(saliency_display, ax=ax[3], fraction=0.046, pad=0.04)

    # Panel 5: IG Overlay
    sa_overlay = create_overlay(img_viz_color, saliency_map, cv2.COLORMAP_HOT)
    ax[4].imshow(sa_overlay)
    ax[4].set_title('IG Overlay')
    ax[4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- 5. PRINCIPAL EXECUTION ---
weights = [1.0, 1.0, 3.0, 3.0]
custom_loss = get_weighted_loss(weights, num_outputs=len(LABELS))
custom_objects = {'loss': custom_loss}

print(f'{YELLOW}Loading ENSEMBLE models...{ENDC}')

### CAMBIO ENSEMBLE: Cargar todos los modelos .keras en el directorio ###
# Asumimos que todos los .keras en MODEL_PATH son parte del ensamble
model_files = glob.glob(os.path.join(MODEL_PATH, "*.keras"))
if not model_files:
    # Intento alternativo si están en subcarpetas o tienen otro nombre
    print(f"{RED}No .keras files found in {MODEL_PATH}. Checking standard path pattern...{ENDC}")
    # Ajusta esto si tus modelos tienen nombres específicos como 'model_0.keras', 'model_1.keras'
    model_files = [os.path.join(MODEL_PATH, f"alexnet_fold_{PRUEBA}.keras")] 

ensemble_models = []
for m_path in model_files:
    model = load_model(m_path, custom_objects=custom_objects, compile=False)
    ensemble_models.append(model)

print(f"{GREEN}Loaded {len(ensemble_models)} models for the ensemble.{ENDC}")

print(f'{YELLOW}Processing data from: {TEST_PATH}{ENDC}')
dataset = load_tfrecord_dataset(TEST_PATH, 1) 

n = 0
for image_batch, parsed_batch in dataset.take(5):
    n += 1
    print(f"{YELLOW}--- Analyzing image {n} ---{ENDC}")
    img_tensor = image_batch 
    
    ### CAMBIO ENSEMBLE: Predicción usando función wrapper ###
    mean_preds, std_preds = predict_ensemble(ensemble_models, img_tensor)
    
    print(f'Ensemble Predictions:')
    for i, label_name in enumerate(LABELS):
        if i < mean_preds.shape[1]:
            print(f"{CYAN}{label_name}:{ENDC} {mean_preds[0][i]:.4f} (±{std_preds[0][i]:.4f})")
    
    current_labels = LABELS if len(LABELS) == mean_preds.shape[1] else [f"Param_{i}" for i in range(mean_preds.shape[1])]
    
    for i, label_name in enumerate(current_labels):
        print(f"Generating ENSEMBLE viz for parameter {i}: {GREEN}{label_name}{ENDC}")
        
        # Obtener valor verdadero
        if label_name in parsed_batch:
            true_val = parsed_batch[label_name][0].numpy()
        elif label_name == 'e1': 
             true_val = parsed_batch.get('e1', tf.constant([0.0]))[0].numpy()
        else:
            true_val = 0.0
            
        # 1. Generates Ensemble Score-CAM
        ens_score_cam = generate_ensemble_score_cam(ensemble_models, img_tensor, target_index=i)
        
        # 2. Generates Ensemble Integrated Gradients
        ens_ig_map = get_ensemble_integrated_gradients(ensemble_models, img_tensor, target_index=i, m_steps=50)
        
        # 3. Plot combined visualization
        output_filename = f'viz_ensemble_{label_name}_img{n}.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        labels_for_plot = [r'$\theta_E$', r'$f$', r'$e_1$', r'$e_2$']
        plot_combined_viz(img_tensor[0].numpy(), ens_score_cam, ens_ig_map, 
                          labels_for_plot[i], true_val, mean_preds[0][i], std_preds[0][i], output_path)
        print(f'Saved combined visualization: {output_path}')

print(f"{GREEN}Process completed. Images saved in {OUTPUT_DIR}{ENDC}")
