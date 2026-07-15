
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import configparser
import os

# Color codes for terminal output
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

# Load configuration file
main_config = load_config('main_config.ini')
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
plot_labels = [r'$\theta_E$', r'$f$', r'$\epsilon_x$', r'$\epsilon_y$']
PRUEBA = main_config['CONFIG']['prueba']
N_FOLDS = int(main_config['DEEPENSAMBLE']['n_folds'])
MAIN_PATH = main_config['PATHS']['main_path']

def plot_mean_metrics_perepoch(df_history, model_name, output_dir):
    '''
    Plots the mean training and Validación metrics per epoch across folds.
    df_history: DataFrame con el promedio de los 4 folds por cada época.
    model_name: Nombre del modelo para los títulos y nombres de archivo.
    output_dir: Directorio donde se guardarán las gráficas.
    '''
    # Obtenemos solo las métricas base (sin el prefijo 'val_')
    base_metrics = [col for col in df_history.columns if not col.startswith('val_') and col != 'epoch']
    epochs_range = range(1, len(df_history) + 1)

    for metric in base_metrics:
        plt.figure(figsize=(8, 6))
        
        # Graficar Entrenamiento
        plt.plot(epochs_range, df_history[metric], 'b', marker='*', markersize=6, markerfacecolor='k', 
                 markeredgecolor='k', linewidth=0.8, label=f'Mean Training')
        
        # Graficar Validación (si existe)
        val_col = f'val_{metric}'
        if val_col in df_history.columns:
            plt.plot(epochs_range, df_history[val_col], 'r', linestyle='--', marker='+', 
                     markersize=6, markerfacecolor='k', markeredgecolor='k', linewidth=0.8, label=f'Mean Validación')
        
        n = model_name.split('_')[1]
        match metric.lower():
            case 'mean_absolute_percentage_error':
                plt.title(f'Mean MAPE model {n}')
                plt.ylabel('MAPE (%)')
            case 'learning_rate':
                plt.title(f'Mean Learning Rate model {n}')
                plt.ylabel('Learning Rate')
            case _:
                plt.title(f'Mean {metric} model {n}')
                plt.ylabel(metric)
        
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.8)
        
        save_path = os.path.join(output_dir, f"{model_name}_{metric}_mean_perepoch.png")
        plt.savefig(save_path)
        plt.close() # Importante cerrar la figura para liberar memoria

def plot_loss_and_mae(df_history, model_name, output_dir):
    # plot Loss and MAE in just one figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = range(1, len(df_history) + 1)
    plt.subplots_adjust(wspace=0.4)
    #plt.suptitle(f'Model {model_name.split("_")[1]}', fontsize=24)
    # Loss
    ax[0].plot(epochs_range, df_history['loss'], 'b', marker='*', markersize=6, markerfacecolor='k', 
               markeredgecolor='k', linewidth=0.8, label='Entrenamiento')
    ax[0].plot(epochs_range, df_history['val_loss'], 'r', linestyle='--', marker='+', 
               markersize=6, markerfacecolor='k', markeredgecolor='k', linewidth=0.8, label='Validación')
    ax[0].set_title(f'Loss - (WMSE)', fontsize=20)
    ax[0].set_ylabel('Loss', fontsize=20)
    ax[0].set_xlabel('Épocas', fontsize=20)
    ax[0].legend(fontsize=16)
    ax[0].grid(True, linestyle=':', alpha=0.9)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    
    # MAE
    ax[1].plot(epochs_range, df_history['mae'], 'b', marker='*', markersize=6, markerfacecolor='k', 
               markeredgecolor='k', linewidth=0.8, label='Entrenamiento')
    ax[1].plot(epochs_range, df_history['val_mae'], 'r', linestyle='--', marker='+', 
               markersize=6, markerfacecolor='k', markeredgecolor='k', linewidth=0.8, label='Validación')
    ax[1].set_title(f'Mean Absolute Error', fontsize=20)
    ax[1].set_ylabel('MAE', fontsize=20)
    ax[1].set_xlabel('Épocas', fontsize=20)
    ax[1].legend(fontsize=16)
    ax[1].grid(True, linestyle=':', alpha=0.9)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    save_path = os.path.join(output_dir, f"training_history_{model_name.split('_')[1]}.pdf")
    plt.savefig(save_path)
    plt.close()

def plot_enhanced_gap(all_histories, output_dir, model_name):
    '''
    Calcula el gap para cada fold y grafica la media con desviación estándar.
    '''
    gaps = []
    for df in all_histories:
        gaps.append(df['val_loss'] - df['loss'])
    
    gaps_df = pd.concat(gaps, axis=1)
    mean_gap = gaps_df.mean(axis=1)
    std_gap = gaps_df.std(axis=1)
    epochs = range(1, len(mean_gap) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_gap, color='darkorange', label='Mean Generalization Gap')
    plt.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap, 
                     color='orange', alpha=0.5, label='$\sigma$')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Referencia cero
    plt.title(f'Generalization Gap Evolution - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Gap (Val - Train)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f'enhanced_gap_{PRUEBA}.pdf'))
    plt.close()

def plot_advanced_gap(df_history, model_name, output_dir):
    '''
    Plots the loss gap per epoch and the ratio between val_loss and loss
    df_history: DataFrame con el promedio de los 4 folds por cada época.
    model_name: Nombre del modelo para los títulos y nombres de archivo.
    output_dir: Directorio donde se guardarán las gráficas.
    '''
    epochs_range = range(1, len(df_history) + 1)
    gap = df_history['val_loss'] - df_history['loss']
    ratio = df_history['val_loss'] / df_history['loss']
    
    # Identificar el punto de menor val_loss (Punto de inflexión)
    best_epoch = df_history['val_loss'].idxmin() + 1
    gap_at_best = gap.iloc[best_epoch-1]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eje para el GAP Absoluto
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Absolute Loss Gap (Val - Train)', color=color, fontsize=12)
    ax1.plot(epochs_range, gap, color=color, linewidth=2, label='Abs. Gap')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Eje para el RATIO (Normalizado)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Generalization Ratio (Val / Train)', color=color, fontsize=12)
    ax2.plot(epochs_range, ratio, color=color, linestyle='--', alpha=0.7, label='Gen. Ratio')
    ax2.tick_params(axis='y', labelcolor=color)

    # Línea vertical en el mejor modelo
    plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Optimal Epoch: {best_epoch}')
    
    plt.title(f'Generalization Analysis - {model_name}', fontsize=14)
    fig.tight_layout()
    plt.legend(loc='upper left')
    save_path = os.path.join(output_dir, f"generalization_analysis_{PRUEBA}.png")
    plt.savefig(save_path)
    plt.close()
    
    return gap.iloc[-1], ratio.iloc[-1], best_epoch

def main():    
    print(f"{YELLOW}Reading training history and calculating means per epoch...{ENDC}")
    
    model_path = os.path.join(MAIN_PATH, f"{PRUEBA}/")
    all_folds_histories = []

    for m in range(N_FOLDS):
        history_path = os.path.join(model_path, f'{PRUEBA}_history.csv')
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            all_folds_histories.append(df)
    
    if not all_folds_histories:
        print(f"{RED}No history files found in {model_path}{ENDC}")
        return

    concat_df = pd.concat(all_folds_histories)
    mean_history = concat_df.groupby(level=0).mean()
    std_history = concat_df.groupby(level=0).std()

    print(f"{GREEN}Results for test {PRUEBA} (Final Epoch Mean):{ENDC}")
    last_values = mean_history.iloc[-1]
    txt_history_file = os.path.join(model_path, f'mean_history_test_{PRUEBA}.txt')
    with open(txt_history_file, 'w') as f:
        f.write(f"Mean metrics for test {PRUEBA} at final epoch:\n")
        for col in last_values.index:
            f.write(f"{CYAN}{col}:{ENDC} {last_values[col]:.4f} ± {std_history.loc[last_values.name][col]:.4f}\n")
        f.write("\n")
    f.close()
    
    for col in last_values.index:
        print(f"{CYAN}{col}:{ENDC} {last_values[col]:.4f} ± {std_history.loc[last_values.name][col]:.4f}")
        
    print(f"\n{GREEN}Plotting mean metrics for test {PRUEBA}...{ENDC}")
    plot_mean_metrics_perepoch(mean_history, f'alexnet_{PRUEBA}', model_path)

    print(f"{GREEN}Plotting Loss and MAE for test {PRUEBA}...{ENDC}")
    plot_loss_and_mae(mean_history, f'alexnet_{PRUEBA}', model_path)

    all_folds_histories = [pd.read_csv(os.path.join(model_path, f'{PRUEBA}_fold_{m+1}_history.csv')) for m in range(N_FOLDS)]
    print(f"{GREEN}Plotting Enhanced Generalization GAP for test {PRUEBA}...{ENDC}")
    plot_enhanced_gap(all_folds_histories, model_path, f'Model {PRUEBA}')

    print(f"{GREEN}Plotting Advanced Generalization Analysis for test {PRUEBA}...{ENDC}")
    final_gap, final_ratio, best_epoch = plot_advanced_gap(mean_history, f'Model {PRUEBA}', model_path)
    print(f"{CYAN}Final Generalization GAP:{ENDC} {final_gap:.4f}, {CYAN}Final Ratio:{ENDC} {final_ratio:.4f}, {CYAN}Best Epoch:{ENDC} {best_epoch}") 

    print(f"{GREEN}Finished process.{ENDC}")

if __name__ == "__main__":
    main()