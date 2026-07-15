

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
import os
import sys
import configparser

# --- 1. CONFIGURATION AND PATHS ---
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
    PLOT_LABELS = [r'$\theta_E$', r'$f$', r'$\epsilon_1$', r'$\epsilon_2$']
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
BINS = 25

def plot_histograms_comparison(results_dict, output_dir):
    '''
    Generates a SINGLE image with 4 subplots (one for each parameter),
    comparing the distributions of predictions vs Ground Truth.
    '''
    
    csv_roots = ['theta_E', 'f', 'e1', 'e2'] 
    
    # Configuración de estilos
    colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']
    linestyles = ['-', '--', '-.', ':']
    

    fig, axs = plt.subplots(1, len(csv_roots), figsize=(6*len(csv_roots), 6))
    axs = axs.flatten() # Asegura que sea iterable fácilmente
    
    # Obtener el primer dataframe para graficar el Ground Truth
    first_key = list(results_dict.keys())[0]
    df_first = results_dict[first_key]
    
    print(f"{YELLOW}Generating combined histogram plot...{ENDC}")

    # Iterar sobre los 4 parámetros
    for i in range(4):
        ax = axs[i]
        root = csv_roots[i]
        label_tex = PLOT_LABELS[i] # e.g. $\theta_E$
        
        col_true = f"{root}_true"
        col_pred = f"{root}_pred"
        
        # 1. Plot Ground Truth (Solo una vez)
        if col_true in df_first.columns:
            data_true = df_first[col_true].values
            # Histograma relleno gris
            median_true = np.median(data_true)
            mean_true = np.mean(data_true)
            ax.hist(data_true, bins=BINS, density=True, alpha=0.2, color='k', 
                    label=f'Truth mean = {mean_true:.3f}', histtype='stepfilled')
            # Borde negro
            ax.hist(data_true, bins=BINS, density=True, color='k', 
                    histtype='step', linewidth=1.5)
            ax.axvline(median_true, color='k', linestyle='--', linewidth=1.5, label='median true')
        else:
            print(f"{RED}Column {col_true} not found via root {root}{ENDC}")

        # 2. Plot Predicciones de cada método
        for j, (method_name, df) in enumerate(results_dict.items()):
            if col_pred in df.columns:
                data_pred = df[col_pred].values
                median_pred = np.median(data_pred)
                mean_pred = np.mean(data_pred)
                color = colors[j % len(colors)]
                style = linestyles[j % len(linestyles)]
                
                ax.hist(data_pred, bins=BINS, density=True, alpha=0.9, 
                        color=color, label=f'{method_name} mean = {mean_pred:.3f}', 
                        histtype='step', linewidth=2, linestyle=style)
                ax.axvline(median_pred, color=color, linestyle='--', linewidth=1.5)
        
        # Estética del subplot
        ax.set_xlabel(f"values", fontsize=14)
        ax.set_title(f'{label_tex}', fontsize=16)
        ax.grid(alpha=0.2, linestyle='--')
        
        # Solo poner label Y y Leyenda en el primer gráfico para no ensuciar
        ax.legend(loc='best', fontsize=10, frameon=True)
        if i == 0:
            ax.set_ylabel('Density', fontsize=14)

    # Título global y guardado
    fig.suptitle('Parameter Distribution Comparison: Real vs Predicted', fontsize=18, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'histograms_comparison_all_params.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"{GREEN}Combined histogram saved successfully at: {save_path}{ENDC}")

def plot_histograms_pull(df, current_labels, output_dir, prueba_id):
    '''
    Generates comparative histograms for true vs predicted values with a Pull plot underneath.
    df: DataFrame with all metadata
    current_labels: List of label names in order
    output_dir: Directory to save images
    prueba_id: Identifier for the plot filename
    '''
    
    labels_map = {'theta_E': r'$\theta_E$',
                  'f': r'$f$',
                  'e1': r'$e_1$',
                  'e2': r'$e_2$'}
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
        #ax_hist.hist(true_vals, bins=BINS, density=True, color='k', histtype='step', linewidth=1.5)
        ax_hist.hist(pred_mean_vals, bins=BINS, density=True, alpha=0.8, color='red', stacked=True, label=f'{prueba_id} mean: {mean_pred:.3f}', histtype='step')
        
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

# --- 3. MAIN EXECUTION ---
def main():
    '''
    Main function for running the comparative analysis.

    Parameters:
    None

    Returns:
    None

    Notes:
    This function will load data from the files specified in files_map,
    plot comparison histograms for each parameter, and plot histograms
    with pull for each method.
    '''
    print(f"{YELLOW}Starting comparative analysis...{ENDC}")
    
    # DEFINITIN OF FILES TO LOAD
    files_map = {'4 folds': 'csst_catalog/test22/alexnet_2/predictions_vs_real.csv',      
                 '10 folds': 'csst_catalog/test21/alexnet_2/predictions_vs_real.csv',
                 }
    
    data_dict = {}
    
    # Load data
    for method, filename in files_map.items():
        full_path = os.path.join(filename)
        if os.path.exists(full_path):
            print(f"Loading {method} from {filename}...")
            try:
                df = pd.read_csv(full_path)
                data_dict[method] = df
            except Exception as e:
                print(f"{RED}Error reading {filename}: {e}{ENDC}")
        else:
            print(f"{RED}Warning: File {filename} not found. Skipping.{ENDC}")

    if not data_dict:
        print(f"{RED}No data loaded. Check the paths.{ENDC}")
        return

    # Plot combined corner plot
    #plot_combined_corner(data_dict, OUTPUT_DIR)

    # Plot comparison histograms for each parameter
    plot_histograms_comparison(data_dict, OUTPUT_DIR)

    # Plot histograms with pull for each method
    #for method, df in data_dict.items():
    #    plot_histograms_pull(df, LABELS, OUTPUT_DIR, method)

    print(f"{GREEN}Analysis completed.{ENDC}")
if __name__ == "__main__":
    main()
