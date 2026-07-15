
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import random
import corner
import seaborn as sns
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import configparser
import os
import time
import sys

# --- COLOR CODES FOR TERMINAL OUTPUT ---
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

# --- 1. CONFIGURATION AND DATA LOADING ---
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
    N_FOLDS = 10

INPUT_SHAPE = (NUM_PIX, NUM_PIX, CHANNELS)
OUTPUT_DIR = os.path.join(MAIN_PATH, f"alexnet_{PRUEBA}/")
CSV_PATH = os.path.join(OUTPUT_DIR, 'predictions_vs_real.csv')
NPY_PATH = os.path.join(OUTPUT_DIR, 'predictions_stacked_mcdeepensemble.npy')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. AUXILIARY FUNCTIONS FOR TFRECORD ---
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# --- 4. GENERATION OF IMAGES (LENSTRONOMY) ---
def generate_lens_image(theta_E, f_s, e1_l, e2_l, re_s, re_l, pa_s, x_s, y_s):
    # CCalculations for the source
    e_s = (1.0 - f_s) / (1.0 + f_s)
    e1_s = e_s * np.cos(2 * pa_s)
    e2_s = e_s * np.sin(2 * pa_s)

    x, y = np.meshgrid(
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX),
        np.linspace(-NUM_PIX / 2 * DELTA_PIX, NUM_PIX / 2 * DELTA_PIX, NUM_PIX))

    lens_model = LensModel(['SIE'])
    lens_kwargs = [{'theta_E': theta_E, 
                    'e1': e1_l, 
                    'e2': e2_l,
                    'center_x': 0.0, 'center_y': 0.0}]

    lens_light_model = LightModel(['SERSIC_ELLIPSE'])
    lens_light_kwargs = [{'amp': 8., 
                          'R_sersic': re_l, 
                          'n_sersic': 4.0,
                          'e1': e1_l, 
                          'e2': e2_l,
                          'center_x': 0.0, 'center_y': 0.0}]

    source_light_model = LightModel(['SERSIC_ELLIPSE'])
    source_kwargs = [{'amp': 50.0, 
                      'R_sersic': re_s, 
                      'n_sersic': 2.0,
                      'e1': e1_s, 
                      'e2': e2_s, 
                      'center_x': x_s, 
                      'center_y': y_s}]

    try:
        lens_light = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
        x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)
        source_light = source_light_model.surface_brightness(x_lensed, y_lensed, source_kwargs)
        img = (lens_light + source_light).reshape(NUM_PIX, NUM_PIX)
        return img
    except Exception as e:
        print(f"{RED}Error while generating image with lenstronomy: {e}{ENDC}")
        return np.zeros((NUM_PIX, NUM_PIX))

# --- 8. VISUALIZATION OF RESULTS WITH SCATTER (MODIFIED) ---
def plot_results_with_scatter(df_meta, idx, preds_stacked, output_dir, labels_list):
    '''
    Generates plt: Original | Predicted | Scatter of models
    idx: MUST be the row index (0 to N), not the image ID.
    df_meta: DataFrame with all metadata
    preds_stacked: (n_folds, n_samples, n_labels)
    output_dir: Directory to save images
    labels_list: List of label names in order
    '''
    # .loc[idx] works if the DF index is RangeIndex (0, 1, 2...)
    row_data = df_meta.loc[idx]
    
    # We use the row index as a visual identifier if the real ID is not reliable
    display_id = int(row_data.get('row_idx', idx))
    
    # Generate images
    img_true = generate_lens_image(
        row_data['theta_E_true'],
        row_data['f_s'],
        row_data['e1_true'],
        row_data['e2_true'],
        row_data['re_s'],
        row_data['re_l'],
        row_data['pa_s'],
        row_data['x_s'],
        row_data['y_s']
    )

    img_pred = generate_lens_image(
        row_data['theta_E_pred'],
        row_data['f_s'],
        row_data['e1_pred'],
        row_data['e2_pred'],
        row_data['re_s'],
        row_data['re_l'],
        row_data['pa_s'],
        row_data['x_s'],
        row_data['y_s']
    )

    # Configurar figura
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    labels_for_plot = [r'$\theta_E$', r'$f$', r'$\epsilon_1$', r'$\epsilon_2$']
    # Panel 1: Original
    ax0.imshow(img_true, cmap='hot', origin='lower')
    ax0.set_title(f"Original (Idx: {display_id})", fontsize=12)
    ax0.axis('off')
    txt_true = "\n".join([f"{labels_for_plot[l]}: {row_data[label+'_true']:.3f}" for l, label in enumerate(labels_list)])
    ax0.text(0.05, 0.05, txt_true, color='white', fontsize=9, transform=ax0.transAxes, bbox=dict(facecolor='black', alpha=0.6))

    # Panel 2: Predicción
    ax1.imshow(img_pred, cmap='hot', origin='lower')
    ax1.set_title("Prediction", fontsize=12)
    ax1.axis('off')
    txt_pred = "\n".join([f"{labels_for_plot[l]}: {row_data[label+'_pred']:.3f}" for l, label in enumerate(labels_list)])
    ax1.text(0.05, 0.05, txt_pred, color='white', fontsize=9, transform=ax1.transAxes, bbox=dict(facecolor='black', alpha=0.6))

    # Panel 3: Scatter
    # Access the stack using the row index 'idx'
    folds_data_for_img = preds_stacked[:, idx, :] # (n_folds, n_labels)
    x_positions = np.arange(len(labels_list))

    # a) Individual points
    for i in range(folds_data_for_img.shape[0]):
        ax2.scatter(x_positions, folds_data_for_img[i, :], color='black', alpha=0.4, s=20, zorder=2, label='Models' if i == 0 else "")

    # b) Average (Star)
    means = [row_data[l+'_pred'] for l in labels_list]
    ax2.scatter(x_positions, means, color='crimson', marker='*', s=80, zorder=2, label='Average')
    
    # c) Real (Cross)
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
    # Guardamos usando el display_id (índice de fila) para evitar colisiones o nombres raros
    plt.savefig(os.path.join(output_dir, f"analysis_idx_{display_id:04d}.png"), dpi=150)
    plt.close(fig)

def plot_histograms_comparison(df, current_labels, output_dir, prueba_id):
    '''
    Generates comparative histograms for true vs predicted values with a Pull plot underneath.
    df: DataFrame with all metadata
    current_labels: List of label names in order
    output_dir: Directory to save images
    prueba_id: Identifier for the plot filename
    '''
    BINS = 30
    labels_map = {'theta_E': r'$\theta_E$', 'f_axis': r'$f$', 'e1': r'$\epsilon_1$', 'e2': r'$\epsilon_2$'}
    n_cols = len(current_labels)
    
    # configuration of the figure with 2 rows (Histogram + Pull) and n_cols columns
    fig, axs = plt.subplots(2, n_cols, 
                            figsize=(6 * n_cols, 8), 
                            sharex='col', 
                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    # Adjustment to handle the case of a single column (axs does not return a 2D array in that case)
    if n_cols == 1:
        axs = np.array(axs).reshape(2, 1)

    for i, label in enumerate(current_labels):
        ax_hist = axs[0, i] # Upper panel (Histograms)
        ax_pull = axs[1, i] # Lower panel (Pull)
        
        true_vals = df[f'{label}_true']
        pred_mean_vals = df[f'{label}_pred']
        
        min_val = min(true_vals.min(), pred_mean_vals.min())
        max_val = max(true_vals.max(), pred_mean_vals.max())
        bins = np.linspace(min_val, max_val, BINS + 1)
        
        # --- UPPER PANEL: HISTOGRAMS ---
        counts_true, _ = np.histogram(true_vals, bins=bins)
        counts_pred, _ = np.histogram(pred_mean_vals, bins=bins)

        # chi-squared calculation
        params = counts_true > 0
        print(fr'Calculating chi^2 for label {label} with {np.sum(params)} active bins.')
        if np.any(params):
            chi2 = np.sum((counts_pred[params] - counts_true[params])**2 / counts_true[params])
            ndf = np.sum(params) # Grados de libertad (número de bins con datos)
            chi2_red = chi2 / ndf # Chi2 reducido
        else:
            chi2 = 0
            ndf = 0
            chi2_red = 0
        
        mean_true = np.mean(true_vals)
        mean_pred = np.mean(pred_mean_vals)

        ax_hist.hist(true_vals, bins=BINS, density=True, alpha=0.2, color='k', stacked=True, label='True', histtype='stepfilled')
        ax_hist.hist(true_vals, bins=BINS, density=True, color='k', histtype='step', linewidth=1.5)
        ax_hist.hist(pred_mean_vals, bins=BINS, density=True, alpha=0.8, color='red', stacked=True, label='MCD+DE', histtype='step')
        
        ax_hist.axvline(mean_true, color='blue', linestyle='dashed', linewidth=1, label=f'True Mean: {mean_true:.3f}')
        ax_hist.axvline(mean_pred, color='red', linestyle='dashed', linewidth=1, label=f'Pred Mean: {mean_pred:.3f}')

        #chi2red_label = rf'$\chi^2_\nu$ = {chi2_red:.2f}'
        #chi2_label = fr'$\chi^2$ = {chi2:.2f}'
        #ax_hist.plot([], [], ' ', label=chi2red_label)
        #ax_hist.plot([], [], ' ', label=chi2_label)
        
        ax_hist.set_title(f'Distribution: {labels_map.get(label, label)}')
        ax_hist.legend()
        ax_hist.set_ylabel('Density')
        ax_hist.grid(True, linestyle='--', alpha=0.5)

        plt.setp(ax_hist.get_xticklabels(), visible=False)
        
        # --- PANEL INFERIOR: PULL ---
        # Pull = (Pred - True) / Error_True. Asumiendo error de Poisson = sqrt(N_true)
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = np.sqrt(counts_true)
            pull = (counts_pred - counts_true) / sigma
        
        # Limpiar NaNs o Infs donde sigma es 0
        pull[np.isnan(pull)] = 0
        pull[np.isinf(pull)] = 0
        
        # Centros de los bins para el plot
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Graficamos el Pull
        #ax_pull.errorbar(bin_centers, pull, yerr=sigma, fmt='o', color='black', ecolor='red', capsize=3)
        ax_pull.scatter(bin_centers, pull, s=10, color='black') # Points for better clarity
        
        # Reference lines (0, +/- 2 sigmas)
        ax_pull.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax_pull.axhline(2, color='red', linestyle=':', linewidth=0.8)
        ax_pull.axhline(-2, color='red', linestyle=':', linewidth=0.8)
        
        ax_pull.set_ylabel('Pull')
        ax_pull.set_xlabel(labels_map.get(label, label))
        ax_pull.set_ylim(-4, 4) # Set limits for visual comparison, adjustable
        ax_pull.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(os.path.join(output_dir, f'hist_pull_{prueba_id}.pdf'), bbox_inches='tight')
    plt.close()

def plot_global_corner(df, output_dir):
    '''
    Generates a global corner plot comparing true vs predicted distributions.
    df: DataFrame with all metadata
    output_dir: Directory to save images
    '''
    
    cols_true = ['theta_E_true', 'f_true', 'e1_true', 'e2_true']
    cols_pred = ['theta_E_pred', 'f_pred', 'e1_pred', 'e2_pred']
    plot_labels = [r'$\theta_E$', r'$f$', r'$\epsilon_1$', r'$\epsilon_2$']
    
    # extract data from df for true and predicted
    data_true = df[cols_true].values
    data_pred = df[cols_pred].values
    
    range_limits = []
    for i in range(4):
        min_val = min(data_true[:, i].min(), data_pred[:, i].min())
        max_val = max(data_true[:, i].max(), data_pred[:, i].max())
        # Add a small margin (8%)
        margin = (max_val - min_val) * 0.08
        range_limits.append((min_val - margin, max_val + margin))

    mean_true = np.mean(data_true, axis=0)
    mean_pred = np.mean(data_pred, axis=0)
    fig = corner.corner(
        data_true,
        labels=plot_labels,
        range=range_limits,
        color='dodgerblue',      # Blue for True
        smooth=0.9,
        smooth1d=1.2,
        truths=mean_true,        # Línea de la media Real
        truth_color='red',       # Rojo para Real
        truth_kwargs={'linewidth': 1.0},
        plot_datapoints=False,   # No pintar puntos individuales (muy pesado para datasets grandes)
        plot_density=True,
        fill_contours=True,
        title_fmt=".3f",
        levels=[0.68, 0.95],
        alpha=0.6,
    )
    
    # 4. OVERLAY the PREDICTIONS plot (Red)
    # Pass fig=fig to draw on top of the existing figure
    corner.corner(
        data_pred,
        labels=plot_labels,
        fig=fig,                 # <--- CLAVE: Dibuja sobre la figura anterior
        range=range_limits,      # Usar los mismos límites
        color='crimson',         # Rojo para Predicción
        smooth=0.9,
        smooth1d=1.2,
        truths=mean_pred,       # Línea de la media Predicha
        truth_color='dodgerblue',# Azul para Predicción
        truth_kwargs={'linewidth': 1.0},
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        quantiles=[0.16, 0.5, 0.84],  # Muestra mediana y 1 sigma en los títulos
        show_titles=True,             # Muestra los valores numéricos arriba
        title_fmt=".3f",
        levels=[0.68, 0.95],
        alpha=0.6,
    )
    
    #blue_line = mlines.Line2D([], [], color='dodgerblue', label='Ground Truth')
    #red_line = mlines.Line2D([], [], color='crimson', label='Prediction')
    
    # Place the legend in the upper right corner of the figure
    #plt.legend(handles=[blue_line, red_line], loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10, frameon=True)
    
    fig.suptitle('Parameter Space: True (blue) vs Predicted (red)', fontsize=16, y=1.02)
    
    save_path = os.path.join(output_dir, 'corner_global_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"{GREEN}Global corner plot saved in {save_path}{ENDC}")

def plot_individual_corner(df_meta, idx, preds_stacked, output_dir, labels_list):
    '''
    Generates corner plot for individual image.
    idx: MUST be the row index (0 to N), not the image ID.
    df_meta: DataFrame with all metadata
    preds_stacked: (n_folds, n_samples, n_labels)
    output_dir: Directory to save images
    labels_list: List of label names in order
    '''

    row_data = df_meta.loc[idx]
    display_id = int(row_data.get('row_idx', idx))

    samples = preds_stacked[:, idx, :]

    truths = [row_data[label+'_true'] for label in labels_list]
    plot_labels = [r'$\theta_E$', r'$f$', r'$\epsilon_1$', r'$\epsilon_2$']
    
    fig = corner.corner(
        samples, 
        labels=plot_labels,
        truths=truths,                # Dibuja las líneas del valor Real
        truth_color='#ff4444',        # Color de la línea Real (Rojo brillante)
        color='black',                # Color de la nube de predicción
        smooth=1.0,                   # Suavizado de contornos
        quantiles=[0.16, 0.5, 0.84],  # Muestra mediana y 1 sigma en los títulos
        show_titles=True,             # Muestra los valores numéricos arriba
        title_fmt=".3f",
        plot_datapoints=True,         # Muestra puntos si la densidad es baja
        plot_density=True,
        fill_contours=True,
        levels=[0.68, 0.95],          # Contornos de confianza (1 y 2 sigma)
    )
    
    fig.suptitle(f"Posterior Distribution for Image ID {display_id}", fontsize=14, y=1.02)
    
    save_path = os.path.join(output_dir, f"corner_indiv_{display_id:04d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"{GREEN}Corner plot saved in {save_path}{ENDC}")

def analyze_and_recalibrate(df_meta, preds_stacked, labels_list, output_dir):
    '''
    Do all the analysis and recalibration steps in one go:
    1. Computes the mean and standard deviation of the predictions.
    2. Applies the factor to recalibrate the uncertainties.
    3. Generates a comparative "Before vs After" plot of the Pull histograms.
    4. Saves the correction factors in a .txt file.
    df_meta: DataFrame with all metadata
    preds_stacked: (n_folds, n_samples, n_labels)
    labels_list: List of label names in order
    output_dir: Directory to save images
    '''
    print(f"{YELLOW}Starting unified calibration and recalibration analysis...{ENDC}")
    
    # Calcular medias y desviaciones estándar del ensamble
    pred_mean = np.mean(preds_stacked, axis=0)
    pred_std = np.std(preds_stacked, axis=0)
    n_labels = len(labels_list)
    
    # Mapeo para etiquetas LaTeX en los plots
    labels_map = {'theta_E': r'$\theta_E$', 'f': r'$f$', 'e1': r'$\epsilon_1$', 'e2': r'$\epsilon_2$'}
    
    # Configurar la figura: 2 filas (Antes/Después) x N columnas (Parámetros)
    fig, axs = plt.subplots(2, n_labels, figsize=(5 * n_labels, 9))
    if n_labels == 1: axs = axs.reshape(2, 1) # Manejo seguro si solo hay 1 parámetro
    
    # Referencia Gaussiana ideal N(0,1) para plotear
    x_ref = np.linspace(-5, 5, 100)
    y_ref = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x_ref**2)
    
    calibration_factors = {}
    report_lines = ["--- Calibration Report ---"]

    for i, label in enumerate(labels_list):
        # Datos
        true_vals = df_meta[f'{label}_true'].values
        mu = pred_mean[:, i]
        sigma_orig = pred_std[:, i]
        # Evitar división por cero
        sigma_orig[sigma_orig < 1e-9] = 1e-9 
        
        label_tex = labels_map.get(label, label)
        
        # =========================================
        # FASE 1: ANÁLISIS ORIGINAL (BEFORE)
        # =========================================
        pulls_orig = (true_vals - mu) / sigma_orig
        std_orig = np.std(pulls_orig)
        mean_orig = np.mean(pulls_orig)
        
        # El factor de escala ES la desviación estándar del pull original
        scale_factor = std_orig
        calibration_factors[label] = scale_factor
        
        # Plot Fila Superior (Original)
        ax_orig = axs[0, i]
        sns.histplot(pulls_orig, ax=ax_orig, kde=True, stat="density", 
                     color='skyblue', bins=30, alpha=0.6, label='Original Data')
        ax_orig.plot(x_ref, y_ref, 'r--', lw=2, label='Ideal N(0,1)')
        
        ax_orig.set_title(f"{label_tex} (Original)\n$\sigma_{{pull}}={std_orig:.2f}$ ($\mu={mean_orig:.2f}$)")
        ax_orig.set_xlim(-5, 5)
        ax_orig.grid(alpha=0.2)
        if i == 0: ax_orig.legend(fontsize=9, loc='upper right')
        
        # =========================================
        # FASE 2: RECALIBRACIÓN (AFTER)
        # =========================================
        # Aplicar el factor: sigma_nueva = sigma_vieja * factor
        sigma_calib = sigma_orig * scale_factor
        
        pulls_calib = (true_vals - mu) / sigma_calib
        std_calib = np.std(pulls_calib) # Esto debería ser muy cercano a 1.0
        
        # Plot Fila Inferior (Recalibrado)
        ax_calib = axs[1, i]
        sns.histplot(pulls_calib, ax=ax_calib, kde=True, stat="density", 
                     color='limegreen', bins=30, alpha=0.6, label='Recalibrated')
        ax_calib.plot(x_ref, y_ref, 'r--', lw=2, label='Ideal N(0,1)')
        
        # En el título mostramos qué factor se usó
        ax_calib.set_title(f"{label_tex} (After Scaling S={scale_factor:.3f})\n$\sigma_{{pull}} \\to {std_calib:.2f}$")
        ax_calib.set_xlim(-5, 5)
        ax_calib.set_xlabel(r'Pull Statistic $(y_{true} - \mu) / \sigma$')
        ax_calib.grid(alpha=0.2)
        if i == 0: ax_calib.legend(fontsize=9, loc='upper right')

        # Reporte de texto
        report_lines.append(f"Param: {label:<8} | Factor S: {scale_factor:.4f} | "
                            f"Orig Pull std: {std_orig:.3f} -> Final Pull std: {std_calib:.3f}")
        print(f"  > {label}: Factor de recalibración calculado S = {scale_factor:.3f}")

    # Finalizar plot
    plt.suptitle("Calibration Analysis: Before & After Temperature Scaling", fontsize=16, y=0.99)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'calibration_comparison_final.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}Comparative plot saved in {plot_path}{ENDC}")
    
    # Save correction factors report
    report_path = os.path.join(output_dir, 'calibration_factors.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"{GREEN}Calibration factors saved in {report_path}{ENDC}")
    
    return calibration_factors

def plot_reliability_loops(df_meta, preds_stacked, calibration_factors, labels_list, output_dir):
    '''
    Generates Reliability Loops (Reliability Plots) comparing Before vs After.
    X-axis: Theoretical Probability (Gaussian CDF).
    Y-axis: Observed Probability (Actual CDF).
    df_meta: DataFrame with all metadata
    preds_stacked: (n_folds, n_samples, n_labels)
    calibration_factors: Dict with recalibration factors per label
    labels_list: List of label names in order
    output_dir: Directory to save images
    '''
    print(f"{CYAN}Generating Reliability Plots (CDF vs CDF)...{ENDC}")
    
    # Calculate original means and stds
    pred_mean = np.mean(preds_stacked, axis=0)
    pred_std_raw = np.std(preds_stacked, axis=0)
    
    # Color and label mapping
    labels_map = {'theta_E': r'$\theta_E$', 'f': r'$f$', 'e1': r'$\epsilon_1$', 'e2': r'$\epsilon_2$'}
    colors = ['blue', 'green', 'orange', 'crimson']
    
    # Create figure with 2 subplots (Before and After)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    # Axis configuration
    for ax in axs:
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal', alpha=0.8) # Perfect diagonal
        ax.set_xlabel('Gaussian CDF', fontsize=12)
        ax.set_ylabel('Actual CDF', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axs[0].set_title('Uncalibrated Reliability Plot', fontsize=16)
    axs[1].set_title('Calibrated Reliability Plot', fontsize=16)

    # --- LOOP THROUGH PARAMETERS ---
    for i, label in enumerate(labels_list):
        label_tex = labels_map.get(label, label)
        color = colors[i % len(colors)]
        
        # Real and Predicted Data
        y_true = df_meta[f'{label}_true'].values
        mu = pred_mean[:, i]
        sigma_raw = pred_std_raw[:, i]
        sigma_raw[sigma_raw < 1e-9] = 1e-9 # Numerical safety
        
        # --- 1. LEFT ---
        # Calculamos el PIT (Probability Integral Transform)
        # Es decir: ¿En qué cuantil de la N(mu, sigma) cae el dato real?
        pit_raw = norm.cdf(y_true, loc=mu, scale=sigma_raw)
        
        # Para graficar CDF vs CDF:
        pit_raw_sorted = np.sort(pit_raw)
        n = len(pit_raw)
        y_vals = np.arange(1, n + 1) / n  # Probabilidad empírica acumulada (1/N, 2/N...)
        
        axs[0].plot(pit_raw_sorted, y_vals, color=color, linewidth=1.5, label=label_tex)
        
        factor = calibration_factors[label]
        sigma_calib = sigma_raw * factor
        
        pit_calib = norm.cdf(y_true, loc=mu, scale=sigma_calib)
        pit_calib_sorted = np.sort(pit_calib)
        
        axs[1].plot(pit_calib_sorted, y_vals, color=color, linewidth=1.5, label=label_tex)

    # Legends
    axs[0].legend(fontsize=10)
    axs[1].legend(fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'reliability_plots_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"{GREEN}Reliability plots saved in {save_path}{ENDC}")

def main():
    # --- 2. LOAD METADATA AND PREDICTIONS ---
    print(f"{YELLOW}Loading metadata and predictions...{ENDC}")
    try:
        df = pd.read_csv(CSV_PATH)
        predictions_stacked = np.load(NPY_PATH)  # Shape: (n_folds, n_samples, n_labels)
        print(f"{GREEN}Metadata and predictions loaded successfully.{ENDC}")
        print(f"{CYAN}DataFrame shape: {ENDC} {df.shape}")
        print(f"{CYAN}Predictions shape: {ENDC} {predictions_stacked.shape}")
    except Exception as e:
        print(f"{RED}Error loading metadata or predictions: {e}{ENDC}")
        sys.exit(1)
    
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
            
            # 1. Generate REAL image
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
            
            # 2. Generar imagen PREDICHA
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
                'image_idx': _int_feature(int(df.loc[i, 'row_idx'])), # Usamos el idx de fila confiable
                'image': _bytes_feature(img_true.astype(np.float32).tobytes()),
            }
            writer_true.write(tf.train.Example(features=tf.train.Features(feature=features_true)).SerializeToString())
            
            features_pred = {
                'image_idx': _int_feature(int(df.loc[i, 'row_idx'])),
                'image': _bytes_feature(img_pred.astype(np.float32).tobytes()),
            }
            writer_pred.write(tf.train.Example(features=tf.train.Features(feature=features_pred)).SerializeToString())

    print(f"{GREEN}TFRecords saved.{ENDC}")

    # Execution of plotting (ROW SAMPLING SELECTION)
    N_VISUALIZATION = 9
    total_samples = len(df)

    print(f"{YELLOW}Selecting {N_VISUALIZATION} random samples for visualization...{ENDC}")

    # Select 9 random row indices from the available rows (0 to total_samples-1)
    if total_samples > N_VISUALIZATION:
        selected_row_indices = sorted(random.sample(range(total_samples), N_VISUALIZATION))
    else:
        # If there are fewer than 9 images, take them all
        selected_row_indices = list(range(total_samples))

    print(f"{CYAN}Visualizing Row Indices: {selected_row_indices}{ENDC}")

    labels_for_plot = ['theta_E', 'f', 'e1', 'e2'] 

    for idx in tqdm(selected_row_indices, desc="Analysing Results"):
        plot_results_with_scatter(
            df_meta=df,
            idx=idx, # We pass the row index, which works for sure
            preds_stacked=predictions_stacked,
            output_dir=OUTPUT_DIR,
            labels_list=labels_for_plot
        )

        plot_individual_corner(
            df_meta=df,
            idx=idx, # We pass the row index, which works for sure
            preds_stacked=predictions_stacked,
            output_dir=OUTPUT_DIR,
            labels_list=labels_for_plot
        )

    print(f"{YELLOW}Generating comparative histograms...{ENDC}")
    plot_histograms_comparison(df, labels_for_plot, OUTPUT_DIR, "comparative_histograms")

    print(f"{YELLOW}Generating pairplot...{ENDC}")
    plot_global_corner(df, OUTPUT_DIR)

    print(f"{YELLOW}Analyzing calibration metrics...{ENDC}")
    factors = analyze_and_recalibrate(df, predictions_stacked, labels_for_plot, OUTPUT_DIR)

    for label, factor in factors.items():
        print(f"{CYAN}Recalibration factor for {label}: {ENDC} {factor:.3f}")
    
    print(f"{YELLOW}Generating reliability plots...{ENDC}")
    plot_reliability_loops(df, predictions_stacked, factors, labels_for_plot, OUTPUT_DIR)
    
    print(f'{GREEN}All done! Process finished successfully.{ENDC}')

if __name__ == "__main__":
    main()
