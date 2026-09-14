import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

# Load configuration file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

main_config = load_config('main_config.ini')
DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
MAIN_PATH = main_config['PATHS']['main_path']
TEST_PATH = main_config['PATHS']['tfrecords_path_test']
PRUEBA = main_config['CONFIG']['prueba']
MODEL_DIR_CFG = main_config['CONFIG'].get('model_dir', '').strip()

NUM_PIX = int(main_config['MODEL']['num_pix'])
CHANNELS = int(main_config['MODEL']['channels'])
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
BATCH_SIZE = int(main_config['MODEL']['batch_size'])
TEST_IMAGES = int(main_config['MODEL']['test_images'])
INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)

# Determine output directory and model file location
if MODEL_DIR_CFG and os.path.exists(MODEL_DIR_CFG):
    OUTPUT_DIR = MODEL_DIR_CFG
else:
    possible_dirs = [
        os.path.join(MAIN_PATH, f"{PRUEBA}/"),
        os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/"),
        MAIN_PATH
    ]
    OUTPUT_DIR = possible_dirs[0]
    for d in possible_dirs:
        if os.path.exists(d):
            OUTPUT_DIR = d
            break

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find model file
model_path = None
if MODEL_DIR_CFG and os.path.isfile(MODEL_DIR_CFG):
    model_path = MODEL_DIR_CFG
    OUTPUT_DIR = os.path.dirname(MODEL_DIR_CFG)
else:
    candidates = [
        os.path.join(OUTPUT_DIR, f"{PRUEBA}.keras"),
        os.path.join(OUTPUT_DIR, "alexnet_original.keras"),
        os.path.join(OUTPUT_DIR, "model.keras"),
    ]
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    if not model_path:
        # Search any .keras file in OUTPUT_DIR
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.keras'):
                model_path = os.path.join(OUTPUT_DIR, f)
                break

if not model_path or not os.path.exists(model_path):
    print(f"{RED}Error: No model .keras file found in {OUTPUT_DIR}{ENDC}")
    exit(1)

# Aux functions
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
        for i in range(min(num_outputs, y_pred.shape[1])):
            mse = tf.reduce_mean(tf.square(y_true[:, i] - y_pred[:, i]))
            total_loss += loss_weights[i] * mse
        return total_loss
    return weighted_mse

class Patches(layers.Layer):
    def __init__(self, patch_size=10, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=100, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

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
    tfrecord_files = sorted([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")])
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load dataset
print(f'{YELLOW}Loading dataset from {TEST_PATH}{ENDC}')
dataset = load_tfrecord_dataset(TEST_PATH, 1)
all_data = list(dataset.unbatch().take(TEST_IMAGES))
print(f'Total Dataset: {len(all_data)}')

images = np.array([x[0].numpy() for x in all_data])
parsed_list = [x[1] for x in all_data]

custom_objects = {
    "Patches": Patches,
    "PatchEncoder": PatchEncoder,
}

print(f"{YELLOW}Loading model from {model_path}{ENDC}")
weights = [1.0, 1.0, 3.0, 3.0]
custom_loss = get_weighted_loss(weights, num_outputs=len(LABELS))
model = load_model(model_path, compile=False)

print(f"{YELLOW}Making predictions...{ENDC}")
predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=1)

# Save predictions vs real to CSV
rows = []
for i, parsed in enumerate(parsed_list):
    rows.append({
        'id': int(parsed['image_idx'].numpy()) if 'image_idx' in parsed else i,
        'theta_E_true': float(parsed['theta_E'].numpy()),
        'f_true': float(parsed['f_axis'].numpy()),
        'f_s': float(parsed['f_s'].numpy()),
        'e1_true': float(parsed['e1'].numpy()),
        'e2_true': float(parsed['e2'].numpy()),
        'theta_E_pred': float(predictions[i, 0]),
        'f_pred': float(predictions[i, 1]),
        'e1_pred': float(predictions[i, 2]),
        'e2_pred': float(predictions[i, 3]),
        're_s': float(parsed['re_s'].numpy()),
        're_l': float(parsed['re_l'].numpy()),
        'pa_l': float(parsed['pa_l'].numpy()),
        'pa_s': float(parsed['pa_s'].numpy()),
        'x_s': float(parsed['center_x'].numpy()),
        'y_s': float(parsed['center_y'].numpy()),
        'e1_s': float(parsed['e1_s'].numpy()),
        'e2_s': float(parsed['e2_s'].numpy()),
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, 'predictions_vs_real.csv')
csv_path_orig = os.path.join(OUTPUT_DIR, 'predictions_vs_original.csv')
df.to_csv(csv_path, index=False)
df.to_csv(csv_path_orig, index=False)
print(f'{GREEN}CSVs saved in {csv_path} and {csv_path_orig}{ENDC}')

# Save single fold predictions stack for downstream compatibility
preds_stacked = np.expand_dims(predictions, axis=0) # Shape: (1, n_samples, n_labels)
npy_path = os.path.join(OUTPUT_DIR, 'predictions_stacked_deepensemble.npy')
np.save(npy_path, preds_stacked)
print(f'{GREEN}Single-model predictions stack saved in {npy_path}{ENDC}')

# Generate lens images for comparison
def generate_lens_image(theta_E, f_s, f_l, e1, e2, re_s, re_l, pa_s, pa_l, x_s, y_s):
    e_s = (1.0 - f_s) / (1.0 + f_s)
    e_l = (1.0 - f_l) / (1.0 + f_l)
    e1_s, e2_s = e_s * np.cos(2 * pa_s), e_s * np.sin(2 * pa_s)
    e1, e2 = e_l * np.cos(2 * pa_l), e_l * np.sin(2 * pa_l)

    x, y = np.meshgrid(
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX),
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX))

    lens_model = LensModel(['SIE'])
    lens_kwargs = [{'theta_E': theta_E, 'e1': e1, 'e2': e2}]

    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    lens_light_kwargs = [{'amp': 8., 'R_sersic': re_l, 'n_sersic': 4.0, 'e1': e1, 'e2': e2}]

    source_light_model = LightModel(['SERSIC_ELLIPSE'])
    source_kwargs = [{'amp': 50.0, 'R_sersic': re_s, 'n_sersic': 2.0, 'e1': e1_s, 'e2': e2_s, 'center_x': x_s, 'center_y': y_s}]

    lens_light = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
    x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)
    source_light = source_light_model.surface_brightness(x_lensed, y_lensed, source_kwargs)
    return (lens_light + source_light).reshape(NUM_PIX, NUM_PIX)

# TFRecord paths
TFRECORD_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "original")
TFRECORD_PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(TFRECORD_ORIGINAL_DIR, exist_ok=True)
os.makedirs(TFRECORD_PRED_DIR, exist_ok=True)

tfrecord_original_path = os.path.join(TFRECORD_ORIGINAL_DIR, "lenses_original.tfrecord")
tfrecord_pred_path = os.path.join(TFRECORD_PRED_DIR, "lenses_predicted.tfrecord")

def write_tfrecord(images, df_params, is_pred, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for i, img in enumerate(images):
            prefix = "pred" if is_pred else "true"
            features = {
                'image_idx': _int_feature(int(df_params.loc[i, 'id'])),
                'image': _bytes_feature(img.astype(np.float32).tobytes()),
                'theta_E': _float_feature(float(df_params.loc[i, f'theta_E_{prefix}'])),
                'f_axis': _float_feature(float(df_params.loc[i, f'f_{prefix}'])),
                'f_s': _float_feature(float(df_params.loc[i, f'f_s'])),
                'e1': _float_feature(float(df_params.loc[i, f'e1_{prefix}'])),
                'e2': _float_feature(float(df_params.loc[i, f'e2_{prefix}'])),
                're_s': _float_feature(float(df_params.loc[i, 're_s'])),
                're_l': _float_feature(float(df_params.loc[i, 're_l'])),
                'pa_l': _float_feature(float(df_params.loc[i, 'pa_l'])),
                'x_s': _float_feature(float(df_params.loc[i, 'x_s'])),
                'y_s': _float_feature(float(df_params.loc[i, 'y_s'])),
                'pa_s': _float_feature(float(df_params.loc[i, 'pa_s'])),
                'e1_s': _float_feature(float(df_params.loc[i, 'e1_s'])),
                'e2_s': _float_feature(float(df_params.loc[i, 'e2_s'])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
    print(f"TFRecord saved: {output_path}")

print(f"{YELLOW}Generating images and writing TFRecords...{ENDC}")
imgs_true, imgs_pred = [], []
for i in tqdm(range(len(df)), desc="Generating lens images"):
    imgs_true.append(generate_lens_image(df.loc[i, 'theta_E_true'], df.loc[i, 'f_s'],
                                         df.loc[i, 'f_true'], df.loc[i, 'e1_true'], df.loc[i, 'e2_true'],
                                         df.loc[i, 're_s'], df.loc[i, 're_l'],
                                         df.loc[i, 'pa_s'], df.loc[i, 'pa_l'], 
                                         df.loc[i, 'x_s'], df.loc[i, 'y_s']))
    imgs_pred.append(generate_lens_image(df.loc[i, 'theta_E_pred'], df.loc[i, 'f_s'],
                                         df.loc[i, 'f_true'], df.loc[i, 'e1_pred'], df.loc[i, 'e2_pred'],
                                         df.loc[i, 're_s'], df.loc[i, 're_l'],
                                         df.loc[i, 'pa_s'], df.loc[i, 'pa_l'],
                                         df.loc[i, 'x_s'], df.loc[i, 'y_s']))

imgs_true, imgs_pred = np.array(imgs_true), np.array(imgs_pred)

sample_indices = [849, 919, 10, 204, 305, 412, 512, 612, 712, 1275, 1157, 939, 2185]
valid_sample_indices = [i for i in sample_indices if i < len(df)]

for i in tqdm(valid_sample_indices, desc="Saving comparative sample images"):
    img_true = generate_lens_image(df.loc[i, 'theta_E_true'], df.loc[i, 'f_s'],
                                   df.loc[i, 'f_true'], df.loc[i, 'e1_true'], df.loc[i, 'e2_true'],
                                   df.loc[i, 're_s'], df.loc[i, 're_l'],
                                   df.loc[i, 'pa_l'], df.loc[i, 'pa_s'], df.loc[i, 'x_s'], df.loc[i, 'y_s'])
    img_pred = generate_lens_image(df.loc[i, 'theta_E_pred'], df.loc[i, 'f_s'],
                                   df.loc[i, 'f_true'], df.loc[i, 'e1_pred'], df.loc[i, 'e2_pred'],
                                   df.loc[i, 're_s'], df.loc[i, 're_l'],
                                   df.loc[i, 'pa_l'], df.loc[i, 'pa_s'], 
                                   df.loc[i, 'x_s'], df.loc[i, 'y_s'])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_true, cmap='hot')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[0].text(0.05, 0.05,
                 f"$\\theta_E$={df.loc[i,'theta_E_true']:.4f}\n"
                 f"f={df.loc[i,'f_true']:.4f}\n"
                 f"e1={df.loc[i,'e1_true']:.4f}\n"
                 f"e2={df.loc[i,'e2_true']:.4f}",
                 color='white', fontsize=8, transform=axes[0].transAxes,
                 bbox=dict(facecolor='black', alpha=0.4, pad=2))

    axes[1].imshow(img_pred, cmap='hot')
    axes[1].set_title("Predicción")
    axes[1].axis('off')
    axes[1].text(0.05, 0.05,
                 f"$\\theta_E$={df.loc[i,'theta_E_pred']:.4f}\n"
                 f"f={df.loc[i,'f_pred']:.4f}\n"
                 f"e1={df.loc[i,'e1_pred']:.4f}\n"
                 f"e2={df.loc[i,'e2_pred']:.4f}",
                 color='white', fontsize=8, transform=axes[1].transAxes,
                 bbox=dict(facecolor='black', alpha=0.4, pad=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{i:04d}.png"), dpi=150)
    plt.close(fig)

write_tfrecord(imgs_true, df, False, tfrecord_original_path)
write_tfrecord(imgs_pred, df, True, tfrecord_pred_path)

print(f'{GREEN}Direct testing finished successfully. TFRecords and CSVs saved to {OUTPUT_DIR}.{ENDC}')
