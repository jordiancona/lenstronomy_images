
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import cv2

# Colores para la terminal
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
BLUE = '\033[34m'
ENDC = '\033[0m'

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
PRUEBA = int(main_config['CONFIG']['prueba'])
CLASSES = int(main_config['MODEL']['classes']) 
MAIN_PATH = main_config['PATHS']['main_path']
MODEL_PATH = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
TEST_PATH = main_config['PATHS']['tfrecords_path_test']
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNLES = int(main_config['MODEL']['channels'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNLES)

OUTPUT_DIR = os.path.join(MODEL_PATH, 'viz_outputs') # Carpeta diferenciada
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    # Nota: Si tienes más features en tu TFRecord original, asegúrate de dejarlas aquí
    # Para este ejemplo asumo que la estructura es compatible
    try:
        parsed = tf.io.parse_single_example(example_proto, feature_description)
    except:
        # Fallback genérico si faltan campos en la definición anterior
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

# --- 2. LÓGICA DE VISUALIZACIÓN ---
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('Not found last conv layer.')

def generate_score_cam(model, img_tensor, target_index, max_N=None):
    '''Generates Score-CAM for a given image and model.
    Args:
        img_tensor: Input image tensor of shape (H, W, C) or (1, H, W, C).
        target_index: Index of the target class for which to generate the CAM.
        max_N: Maximum number of feature maps to use (for speed). If None, use all.
    Returns:
        cam: Score-CAM heatmap of shape (H, W).
    '''
    if len(img_tensor.shape) == 3: img_tensor = tf.expand_dims(img_tensor, 0)

    layer_name = find_last_conv_layer(model)
    activation_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.outputs])
    act_map, _ = activation_model(img_tensor)
    act_map = act_map[0] 
    num_filters = act_map.shape[-1]
    if max_N is not None and max_N < num_filters: num_filters = max_N
    print(f'{YELLOW}Using {num_filters} feature maps for Score-CAM.{ENDC}')
    print(f'{CYAN} Activation map shape: {ENDC} {act_map.shape}')

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

# --- INTEGRATED GRADIENTS ---
def get_integrated_gradients(model, img_tensor, target_index, m_steps=50, baseline=None):
    '''
    Computes Integrated Gradients for a given input image and model.
    Args:
        m_steps: Number of interpolation steps (50 is usually sufficient).
        baseline: Base image for computing integrated gradients. If None, a black image is used.
    '''
    img_tensor = tf.cast(img_tensor, tf.float32)
    if len(img_tensor.shape) == 3: 
        img_tensor = tf.expand_dims(img_tensor, 0)

    # 1. Define baseline
    if baseline is None:
        baseline = tf.zeros_like(img_tensor)

    # 2. Generate interpolated images
    # alphas: array from 0.0 to 1.0
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    
    # We need to reshape alphas to multiply by the image tensor
    # alphas shape: [m_steps+1, 1, 1, 1]
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    delta = img_tensor - baseline
    interpolated_images = baseline + alphas_x * delta

    # Calculate gradients for the entire batch of interpolated images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        target_predictions = predictions[:, target_index]

    grads = tape.gradient(target_predictions, interpolated_images)
    
    # 4. Approximate the integral: Average of the gradients
    avg_grads = tf.reduce_mean(grads, axis=0) # shape [H, W, C]

    # 5. Calculate Integrated Gradients: (Input - Baseline) * Avg_Grads
    # delta[0] because delta has shape [1, H, W, C] and avg_grads [H, W, C]
    integrated_grads = delta[0] * avg_grads

    # 6. Processing for visualization
    saliency_map = tf.reduce_mean(tf.abs(integrated_grads), axis=-1)

    # Normalize to [0, 1]
    min_val = tf.reduce_min(saliency_map)
    max_val = tf.reduce_max(saliency_map)
    saliency_map_norm = (saliency_map - min_val) / (max_val - min_val + 1e-8)
    
    return saliency_map_norm.numpy()

import math

import math

def visualize_feature_maps(model, img_tensor, n, layer_name=None, columns=8, max_filters=None, output_dir='.'):
    '''
    Visualiza mapas de características usando subplots (un axis por filtro).
    '''
    # 1. Identificar capas
    layer_outputs = []
    layer_names = []
    
    if layer_name is not None:
        try:
            layer = model.get_layer(layer_name)
            layer_outputs = [layer.output]
            layer_names = [layer.name]
        except ValueError:
            print(f"{RED}Error: La capa {layer_name} no existe.{ENDC}")
            return
    else:
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D)):
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)

    if not layer_names:
        print(f"{RED}No layers found to visualize.{ENDC}")
        return

    # 2. Modelo extractor
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor, verbose=0)
    
    if not isinstance(activations, list):
        activations = [activations]

    # 3. Visualización con Subplots
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        
        print(f"{YELLOW}Layer: {ENDC} {layer_name} | {CYAN}Size: {ENDC} {size}x{size} | {CYAN}Filters: {ENDC} {n_features}")

        # Limitar filtros
        filters_to_show = n_features
        if max_filters is not None and n_features > max_filters:
            print(f"{YELLOW} Limited to {max_filters} filters.{ENDC}")
            filters_to_show = max_filters
        
        # Calcular filas necesarias
        rows = math.ceil(filters_to_show / columns)
        
        # --- AQUÍ ESTÁ EL CAMBIO CLAVE ---
        # Creamos una figura y un array de ejes (axes)
        # figsize: ajustamos el tamaño total de la imagen
        scale = 2.0  # Tamaño en pulgadas por cada subplot
        fig, axes = plt.subplots(rows, columns, figsize=(columns * scale, rows * scale))
        
        # Título general de la figura
        fig.suptitle(f"Layer: {layer_name}", fontsize=24)
        
        # Aplanar los ejes (flatten) para poder iterar sobre ellos en un solo bucle for
        # Si solo hay 1 fila o 1 columna, nos aseguramos de que sea iterable
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

        for i, ax in enumerate(axes_flat):
            if i < filters_to_show:
                # Obtener el filtro individual
                channel_image = layer_activation[0, :, :, i].copy()
                
                # Normalización Min-Max (0.0 a 1.0)
                img_min, img_max = channel_image.min(), channel_image.max()
                if img_max - img_min > 1e-5:
                    channel_image = (channel_image - img_min) / (img_max - img_min)
                else:
                    channel_image = np.zeros_like(channel_image)
                
                # Graficar en el axis correspondiente
                im = ax.imshow(channel_image, cmap='inferno', aspect='auto', vmin=0, vmax=1)
                
                # Estilos del axis
                ax.axis('off') # Ocultar ejes X e Y (números)
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.4)
            else:
                # Si sobran subplots (ej: grid de 8x8 pero solo 60 filtros), los apagamos
                ax.axis('off')
        
        # Ajustar el layout para que no se solapen y dejar espacio blanco
        plt.tight_layout(rect=[0,0,0.9,0.95])
        plt.subplots_adjust(top=0.95) # Dejar espacio para el título superior

        if im is not None:
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.5]) # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, label='Activation Intensity', orientation='vertical', shrink=0.8)
            cbar.ax.tick_params(labelsize=18)
            cbar.ax.yaxis.label.set_size(20)
            
        save_path = os.path.join(output_dir, f'{layer_name}_matrix_img{n}.png')
        fig.savefig(save_path, facecolor='white')
        plt.close(fig) # Importante cerrar la figura para liberar memoria
        
        print(f"{GREEN} Matrix saved in {save_path}{ENDC}")

# --- PLOTTING FUNCTIONS ---
def create_overlay(img_viz_color, heatmap_map, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = (heatmap_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_viz_color, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def plot_combined_viz(img, score_cam, saliency_map, param_name, true_val, pred_value, output_path):
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    # Prepare original image for visualization
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img_viz_gray = (img_norm * 255).astype(np.uint8)
    if len(img_viz_gray.shape) == 2 or img_viz_gray.shape[-1] == 1:
        img_viz_color = cv2.cvtColor(img_viz_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_viz_color = img_viz_gray

    # Panel 1: Original
    ax[0].imshow(img_viz_color)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    info_text = f"Param: {param_name}\nTrue: {true_val:.4f}\nPred: {pred_value:.4f}"
    ax[0].text(0.03, 0.97, info_text, transform=ax[0].transAxes, 
               color='white', fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    # Panel 2: Score-CAM
    heatmap_display = ax[1].imshow(score_cam, cmap='jet', vmin=0, vmax=1)
    ax[1].set_title('Score-CAM Heatmap')
    ax[1].axis('off')
    plt.colorbar(heatmap_display, ax=ax[1], fraction=0.046, pad=0.04)

    # Panel 3: Score-CAM Overlay
    sc_overlay = create_overlay(img_viz_color, score_cam, cv2.COLORMAP_JET)
    ax[2].imshow(sc_overlay)
    ax[2].set_title('Score-CAM Overlay')
    ax[2].axis('off')

    # Panel 4: Integrated Gradients
    # Usamos 'inferno' o 'magma' para IG, se ven muy bien sobre negro
    saliency_display = ax[3].imshow(saliency_map, cmap='hot', vmin=0, vmax=1)
    ax[3].set_title('Integrated Gradients') # Título Actualizado
    ax[3].axis('off')
    plt.colorbar(saliency_display, ax=ax[3], fraction=0.046, pad=0.04)

    # Panel 5: IG Overlay
    sa_overlay = create_overlay(img_viz_color, saliency_map, cv2.COLORMAP_HOT)
    ax[4].imshow(sa_overlay)
    ax[4].set_title('Int. Gradients Overlay') # Título Actualizado
    ax[4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- 3. EJECUCIÓN PRINCIPAL ---
def main():
    model_path = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}", f"alexnet_paper_{PRUEBA}.keras")
    weights = [1.0] * len(LABELS) 
    custom_loss = get_weighted_loss(weights, num_outputs=len(LABELS))
    custom_objects = {'loss': custom_loss}

    print(f'{YELLOW}Loading model...{ENDC}')
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    print(f'{YELLOW}Processing data from: {TEST_PATH}{ENDC}')
    dataset = load_tfrecord_dataset(TEST_PATH, 1) 

    n = 0
    # Analizamos 5 imágenes como ejemplo
    for image_batch, parsed_batch in dataset.take(5):
        n += 1
        print(f"{YELLOW}--- Analyzing image {n} ---{ENDC}")
        img_tensor = image_batch 
        
        preds = model.predict(img_tensor, verbose=0)
        print(f'Predictions:')
        for i, label_name in enumerate(LABELS):
            if i < preds.shape[1]:
                print(f"{CYAN}{label_name}:{ENDC} {preds[0][i]:.4f}")
        
        current_labels = LABELS if len(LABELS) == preds.shape[1] else [f"Param_{i}" for i in range(preds.shape[1])]
        
        for i, label_name in enumerate(current_labels):
            print(f"Generating viz for parameter {i}: {GREEN}{label_name}{ENDC}")
            
            # Obtener valor verdadero
            if label_name in parsed_batch:
                true_val = parsed_batch[label_name][0].numpy()
            elif label_name == 'e1': # Ejemplo de mapeo manual si el nombre difiere
                true_val = parsed_batch.get('e1', tf.constant([0.0]))[0].numpy()
            else:
                true_val = 0.0
                
            # 1. Generar Score-CAM
            score_cam_map = generate_score_cam(model, img_tensor, target_index=i)
            
            # 2. Generar INTEGRATED GRADIENTS (m_steps=50 es estándar)
            # Aquí llamamos a la nueva función
            ig_map = get_integrated_gradients(model, img_tensor, target_index=i, m_steps=50)
            
            # 3. Plotear
            output_filename = f'viz_{label_name}_img{n}.png'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            plot_combined_viz(img_tensor[0].numpy(), score_cam_map, ig_map, label_name, true_val, preds[0][i], output_path)
            print(f'{GREEN}Saved combined visualization: {output_path}{ENDC}')

            print(f"{YELLOW}Visualizing feature maps...{ENDC}")
            visualize_feature_maps(model, img_tensor, n, layer_name=None, columns=16, output_dir=OUTPUT_DIR)

    print(f"{GREEN}Process completed. Images saved in {OUTPUT_DIR}{ENDC}")

if __name__ == "__main__":
    main()