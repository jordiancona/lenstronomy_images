
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats
import configparser
from contextlib import redirect_stdout
from scipy.stats import pearsonr
import argparse
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
MAIN_PATH = main_config['PATHS']['main_path']
models = {'alexnet': 'alexnet', 'alexnet_original': 'alexnet_original'}
MODEL_PATH = os.path.join(MAIN_PATH, f"{models[PRUEBA]}/")
CSV_PATH = os.path.join(MODEL_PATH, "predictions_vs_original.csv")

# Auxiliary functions
def relative_error(Vr, Vo):
    return abs((Vr - Vo) / (Vr + 1e-3)) * 100

def smapping_error(Vr, Vo):
    return abs((Vr - Vo) / ((abs(Vr) + abs(Vo))/2 + 1e-2)) * 100

def remove_outliers_iqr(df, labels):
    df_clean = df.copy()
    for col in labels:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

def remove_outliers_sigma(df, labels, sigma=3):
    df_clean = df.copy()
    for col in labels:
        mean = df_clean[col].mean()
        std = df_clean[col].std()

        lower = mean - sigma * std
        upper = mean + sigma * std

        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

def compute_errors(original, predicted):
    errors = {"theta_E": relative_error(original['theta_E'], predicted['theta_E']),
              "f_axis": relative_error(original['f_axis'], predicted['f_axis']),
              "e1": relative_error(original['e1'], predicted['e1']),
              "e2": relative_error(original['e2'], predicted['e2']),}

    return pd.DataFrame(errors)

# Computation of R²
def compute_r2(original, predicted):
    r2_vals = {}
    for key in original:
        r2_vals[key] = r2_score(original[key], predicted[key])
    return r2_vals

# Leer CSV con valores predichos y reales
print(f"{YELLOW}Reading CSV: {CSV_PATH}{ENDC}\n")
df = pd.read_csv(CSV_PATH)

# Separate original and predicted values
original_values = {'theta_E': df['theta_E_true'],
                   'f_axis': df['f_true'],
                   'e1': df['e1_true'],
                   'e2': df['e2_true']
                   }

predicted_values = {'theta_E': df['theta_E_pred'],
                    'f_axis': df['f_pred'],
                    'e1': df['e1_pred'],
                    'e2': df['e2_pred']
                    }

errors_df = compute_errors(original_values, predicted_values)
errors_df = remove_outliers_iqr(errors_df, LABELS)
errors_df.index += 1
errors_df.to_csv(MODEL_PATH + f"errors_tot.csv")

# Stacked predictions
PATH_NPY = os.path.join(MODEL_PATH, 'predictions_stacked_deepensemble.npy')
predictions_stacked = np.load(PATH_NPY)
k_folds = predictions_stacked.shape[0]

# Compute R² values
output_file = os.path.join(MODEL_PATH, f"metrics_model_{PRUEBA}.txt")
with open(output_file, "w") as f:
    with redirect_stdout(f):
        print(f"{YELLOW}======== Stacked std ========{ENDC}")
        for i, label in enumerate(LABELS):
            std_stacked = np.std(predictions_stacked[:, :, i], axis=0, ddof=1)
            print(f"{CYAN}{label} | Stacked STD:{ENDC} {std_stacked.mean():.4f}")
        
        # R^2 Values
        print(f"\n{YELLOW}======== R² Values per Parameter (Ensemble) ========{ENDC}")
        for i, label in enumerate(LABELS):
            y_true = np.array(original_values[label])
            y_pred = np.array(predicted_values[label])
            r2 = r2_score(y_true, y_pred)
            std_r2 = np.std([r2_score(y_true, y_pred) for j in range(k_folds)], ddof=1)
            print(f"{CYAN}{label} | R²:{ENDC} {r2:.4f} ± {std_r2:.4f}")
        
        print(f'\n{YELLOW}======== Error Statistics (Relative Error) ========{ENDC}')
        for i, label in enumerate(LABELS):
            std_rel_error = np.std(errors_df[label].values, ddof=1)
            print(f"{CYAN}{label} | STD:{ENDC} {std_rel_error:.2f}%")

        # Percentils
        print(f"\n{YELLOW}======== Percentils ========={ENDC}")
        percentiles = {}
        for i, label in enumerate(LABELS):
            p50, p65, p90 = np.percentile(errors_df[label].values, [50, 65, 90], axis=0)
            percentiles[label] = {'p50': p50, 'p65': p65, 'p90': p90}
            print(f"{CYAN}{label} | p50:{ENDC} {p50:.2f}% {CYAN}| p65: {ENDC}{p65:.2f}% {CYAN}| p90:{ENDC} {p90:.2f}%")

        # Combined metrics (STD, RMSE, CV)
        print(f"\n{YELLOW}========= Statistical Metrics per Parameter ========={ENDC}")
        for i, label in enumerate(LABELS):
            y_true = np.array(original_values[label])
            y_pred = np.array(predicted_values[label])
            residuals = y_true - y_pred
            
            std = np.std(residuals, ddof=1)
            rmse = np.sqrt(np.mean(residuals**2))
            mean_true = np.mean(y_true)
            cv = (rmse / mean_true) * 100 if mean_true != 0 else 0
            
            print(f"{CYAN}{label} | STD:{ENDC} {std:.4f} {CYAN}| RMSE: {ENDC}{rmse:.4f} {CYAN}| CV: {ENDC}{cv:.2f}%")

# Plot results - Original vs Predicted and Residuals
labels_for_plot = {'theta_E': (r'$\theta_E$', 'Radio de Einstein'),
                   'f': (r'$f$', 'Relación axial'),
                   'e1': (r'$\epsilon_x$', 'Elipticidad x'),
                   'e2': (r'$\epsilon_y$', 'Elipticidad y')}

# Scatter plot enhanced with ensemble uncertainty
def plot_scatter_enhanced():
    print(f'\n{YELLOW}======== Ensemble Analysis (k={k_folds}) ========={ENDC}')
    
    ensemble_mean = np.mean(predictions_stacked, axis=0)
    ensemble_std = np.std(predictions_stacked, axis=0, ddof=1)

    labels = ['theta_E', 'f', 'e1', 'e2']
    fig, axes = plt.subplots(1, len(labels), figsize=(30, 8))
    #plt.suptitle(f"Model {PRUEBA}", fontsize=36)
    for i, label in enumerate(labels):
        residuals = 0
        y_true = df[f'{label}_true'].values
        y_pred_mean = df[f'{label}_pred'].values #ensemble_mean[:, i]
        y_pred_uncertainty = ensemble_std[:, i]

       # Métricas de error (Residuales)
        residuals = y_pred_mean - y_true
        bias = np.median(residuals)
        NMAD = 1.48 * np.median(np.abs(residuals - np.median(residuals)))  # MAD como estimador robusto de la desviación estándar
        r2 = r2_score(y_true, y_pred_mean)
        q16, q84 = np.percentile(residuals, [16, 84])
        err_low = bias - q16
        err_high = q84 - bias

        # --- Gráfica de Dispersión ---
        axes[i].plot([], [], ' ', label=f'$R^2$= {r2:.2f}')
        axes[i].plot([], [], ' ', label=f'$\mu$= {bias:.2f}')
        #axes[i].plot([], [], ' ', label=f'NMAD= {NMAD:.2f}')
        axes[i].plot([], [], ' ', label=f'$\sigma^+$: {err_high:.2f}')
        axes[i].plot([], [], ' ', label=f'$\sigma^-$: {err_low:.2f}')

        axes[i].scatter(y_true, y_pred_mean, alpha=0.2, c='gray', s=10) # dodgerblue
        
        # --- Ajuste de Ejes Iguales (Importante para ver el sesgo) ---
        all_vals = np.concatenate([y_true, y_pred_mean])
        min_val, max_val = all_vals.min(), all_vals.max()
        axes[i].set_xlim(min_val, max_val)
        axes[i].set_ylim(min_val, max_val)
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=4) # Diagonal ideal
        
        # --- Títulos y Leyendas ---
        #axes[i].set_title(f'{labels_for_plot[label][0]}\n$R^2$: {r2:.3f} | $\sigma_{{err}}$: {sigma_error:.3f}', fontsize=22)
        axes[i].set_xlabel('Valor original', fontsize=32)
        if i == 0: axes[i].set_ylabel('Valor predicho', fontsize=32)
        
        axes[i].set_title(f'{labels_for_plot[label][0]} - {labels_for_plot[label][1]}', fontsize=35)
        axes[i].tick_params(axis='both', which='major', labelsize=30)
        axes[i].set_xticks(np.arange(min_val, max_val, (max_val - min_val) / 5))
        #axes[i].set_yticks(np.arange(min_val, max_val, (max_val - min_val) / 5))
        axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axes[i].legend(fontsize=30, loc='upper left', frameon=False, handlelength=0, handletextpad=0)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].set_aspect('equal') # Obliga a que el cuadrado sea perfecto

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, f'original_vs_predicted_{PRUEBA}.pdf'), dpi=150)
    print(f"{GREEN}Scatter plot con incertidumbre guardado.{ENDC}")

# Llamar a la nueva función
plot_scatter_enhanced()

# Residual plots
fig, ax = plt.subplots(1, len(LABELS), figsize=(25, 6))
for n, (label, plot_label) in enumerate(zip(LABELS, labels_for_plot.values())):
    residuals = np.array(original_values[label]) - np.array(predicted_values[label])
    ax[n].scatter(original_values[label], residuals, c='k', s=6)
    ax[n].set_xlabel('Valores originales', fontsize=20)
    ax[n].set_ylabel('Residuales', fontsize=20)
    ax[n].set_title(f'Gráfico de residuos\n{plot_label[0]} - {plot_label[1]}', fontsize=22)
    ax[n].axhline(y=0, c='r', linestyle='--')
    ax[n].grid(True, linestyle=':', alpha=0.8)
    ax[n].tick_params(axis='both', which='major', labelsize=16)
    
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, f'residual_plot_{PRUEBA}.pdf'))
plt.close()

# binned MAE plot
fig, ax = plt.subplots(1, len(LABELS), figsize=(25, 6))
for n, (label, plot_label) in enumerate(zip(LABELS, labels_for_plot.values())):
    residuals = np.abs(np.array(original_values[label]) - np.array(predicted_values[label]))
    bins = np.linspace(min(original_values[label]), max(original_values[label]), 10)
    bin_means, bin_edges, _ = stats.binned_statistic(original_values[label], residuals, statistic='mean', bins=bins)
    
    ax[n].plot(bin_edges[:-1], bin_means, marker='o', c='b')
    ax[n].set_xlabel('Valores originales', fontsize=20)
    if n == 0:
        ax[n].set_ylabel('MAE', fontsize=20)
    ax[n].axhline(y=0, c='r', linestyle='--')
    ax[n].set_title(f'{plot_label[0]} - {plot_label[1]}', fontsize=35)
    ax[n].grid(True, linestyle=':', alpha=0.8)
    ax[n].tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, f'binned_mae_plot_{PRUEBA}.pdf'), dpi=200)
plt.close()

# Residual plots + binned MAE plot
fig, ax = plt.subplots(1, len(LABELS), figsize=(32, 8))
for n, (label, plot_label) in enumerate(zip(LABELS, labels_for_plot.values())):
    residuals = np.array(original_values[label]) - np.array(predicted_values[label])
    original_values_array = np.array(original_values[label][errors_df.index-1])  # Convertir a array para indexación
    median_rel_error = np.median(errors_df[label].values)

    bins = np.linspace(min(original_values_array), max(original_values_array), 10)
    bin_means, bin_edges, _ = stats.binned_statistic(original_values_array, errors_df[label].values, statistic='mean', bins=bins)
    
    ax[n].axhline(y=median_rel_error, c='green', linestyle='--', lw=3.0, label=f'Mediana: {median_rel_error:.2f}%')
    ax[n].plot(bin_edges[:-1], bin_means, marker='o', c='b', lw=3.0, markersize = 10)
    ax[n].scatter(original_values_array, errors_df[label].values, c = 'gray', s = 6, alpha = 0.2)
    ax[n].set_xlabel('Valor original', fontsize=32)
    axtwin = ax[n].twinx()
    axtwin.set_yticks(np.arange(0, max(errors_df[label].values), median_rel_error*1.5, median_rel_error/2))
    axtwin.tick_params(axis='y', labelcolor='b', labelsize=32)

    if n == 0: ax[n].set_ylabel('Error relativo (%)', fontsize=32)
    if n == len(LABELS) - 1:
        axtwin.set_ylabel('$\epsilon_r$ promedio agrupado', fontsize=32, color='b')

    ax[n].set_title(f'{plot_label[0]} - {plot_label[1]}', fontsize=35)
    ax[n].axhline(y=0, c='r', linestyle='--', lw=3.0)
    ax[n].grid(True, linestyle=':', alpha=0.8)
    ax[n].tick_params(axis='both', which='major', labelsize=30)
    ax[n].legend(loc='upper left', fontsize=30, handlelength=0.8, handletextpad=0)
    
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, f'residual_plot_w_binned_mae_plot_{PRUEBA}.pdf'))
plt.close()

print(f'\n{YELLOW}std for relative errors:{ENDC}')
for label in LABELS:
    std_rel_error = np.std(errors_df[label].values, ddof=1)
    print(f"{label}: {std_rel_error:.2f}%")

# Boxplot from errors dataframe
fig, ax = plt.subplots(figsize=(12, 8))
bx1 = ax.boxplot(errors_df,
                 tick_labels=plot_labels,
                 vert=True,
                 patch_artist=True,
                 showfliers=False,
                 manage_ticks=True
                 )

ax.tick_params(axis='x', labelsize=22)

for element in ['boxes', 'whiskers', 'caps', 'medians']:
    plt.setp(bx1[element], color='k')

for patch in bx1['boxes']:
    patch.set_facecolor('papayawhip')
    patch.set_linewidth(1.5)
    patch.set_edgecolor('k')

for median in bx1['medians']:
    median.set(color='k', linewidth=2)

for cap in bx1['caps']:
    cap.set(linewidth=1.5)

# Compute and add mean values
medians = np.median(errors_df, axis=0)
x_positions = np.arange(1, len(LABELS) + 1)
print(f"{GREEN}Medians for test {PRUEBA}:{ENDC} {medians}")

# Agregar texto de la media debajo del bigote inferior
for i, (cap_high, median_val) in enumerate(zip(bx1['caps'][1::2], medians)):
    # Obtener posición Y del bigote superior
    y_high = cap_high.get_ydata()[0]
    x_m = cap_high.get_xdata().mean()
    
    p16 = np.percentile(errors_df.iloc[:, i].values, 16)
    p84 = np.percentile(errors_df.iloc[:, i].values, 84)
    err_low = median_val - p16
    err_high = p84 - median_val
    #std = np.std(errors_df.iloc[:, i].values, ddof=1)

    ax.text(x_m, 
            y_high * 1.3,  # un poco arriba del bigote
            rf'{median_val:.2f}$^{{+{err_high:.2f}}}_{{-{err_low:.2f}}}\,$%', 
            #rf'{median_val:.2f} ± {err_high:.2f}%',
            ha='center', va='bottom', 
            fontsize=24, color='k')

ax.tick_params(axis='x', labelsize=26)
ax.tick_params(axis='y', labelsize=26)
ax.set_ylabel('Error relativo (%)', fontsize=24)
ax.set_yscale('log')
#ax.autoscale(enable=True, axis='y', tight=True)
ax.set_ylim(10**(-6), 10**3)
plt.title(f'Model {PRUEBA}', fontsize=30)
sns.despine(ax=ax, top=True, right=True)
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, f'boxplot_model{PRUEBA}.pdf'))
plt.close()

# Histograms from errors dataframe
n_muestras, n_parametros = errors_df.shape
fig, axes = plt.subplots(1, n_parametros, figsize=(4*n_parametros, 4), sharex=True)

for idx, ax in enumerate(axes):
    values, base = np.histogram(errors_df[LABELS[idx]], bins=30, density=True)
    cumulative = np.cumsum(values) / np.sum(values)
    ax1 = ax.twinx()
    ax.set_yscale('log')
    ax.hist(errors_df[LABELS[idx]], bins=15, histtype='stepfilled', alpha=0.65, color='orange')
    ax1.plot(base[:-1], cumulative, lw=0.8, c='r')
    ax.set_xlabel(fr'$\epsilon$ {plot_labels[idx]} (%)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.8)
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.tick_params(axis='y', colors='red')

fig.suptitle('Relative error by parameter', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(MODEL_PATH + f'Relative_error_histplots_{PRUEBA}.png')
plt.close()

with plt.rc_context({'font.size': 18,
                     'axes.labelsize': 22,
                     'axes.titlesize': 24,
                     'xtick.labelsize': 18,
                     'ytick.labelsize': 18}):
                     
    for n, (label, plot_label) in enumerate(zip(LABELS, labels_for_plot.values())):
        y_true = np.array(original_values[label])
        y_pred = np.array(predicted_values[label])
        residuals = y_true - y_pred

        # 1. Inicializar JointGrid enfocándonos en el margen derecho (distribución del error)
        g = sns.JointGrid(x=y_true, y=residuals, space=0, height=8, ratio=5)

        # 2. Gráfico Central: Dispersión de Residuos (Gris Oxford/Azul Marino Académico)
        g.ax_joint.scatter(y_true, residuals, alpha=0.3, color='#4a5568', edgecolors='none', s=22, label='Residuos')
        g.ax_joint.axhline(y=0, color='#e53e3e', linestyle='--', linewidth=2.5, label=r'$\epsilon = 0$')
        
        # Ajustes del gráfico central
        g.ax_joint.set_xlabel(f'Valor original de {plot_label[0]}', labelpad=20)
        if n == 0 or n == 2:  # Mantener consistencia en la escala si compartes páneles
            g.ax_joint.set_ylabel(r'Residuos ($\epsilon$)', labelpad=20)
        else:
            g.ax_joint.set_ylabel(r'Residuos ($\epsilon$)', labelpad=20)
            
        g.ax_joint.grid(True, linestyle=':', alpha=0.6, color='gray')

        # 3. Gráfico Marginal: Ocultar el superior y configurar el derecho (Histograma + KDE Teórica)
        g.ax_marg_x.set_visible(False)

        # Histograma de densidad
        sns.histplot(
            y=residuals, 
            ax=g.ax_marg_y, 
            color='#2b6cb0', 
            alpha=0.4, 
            kde=False, 
            stat="density", 
            bins=30
        )

        # Ajuste de curva Gaussiana Teórica sobre los residuos reales
        y_axis = np.linspace(residuals.min(), residuals.max(), 200)
        mu_res, std_res = stats.norm.fit(residuals)
        
        # Asimetría por percentiles (Bowley/Robust skewness basada en P25, P50 y P75)
        q25, q50, q75 = np.percentile(residuals, [25, 50, 75])
        denom = q75 - q25
        skew_res = ((q75 - q50) - (q50 - q25)) / denom if denom != 0 else 0.0
        #g.ax_marg_y.plot(stats.norm.pdf(y_axis, mu_res, std_res), y_axis, color='#1a365d', linewidth=2.5, label='Ajuste Normal')

        # Limpieza del panel marginal
        g.ax_marg_y.set_xlabel('Densidad', fontsize=14)
        g.ax_marg_y.grid(False)
        sns.despine(ax=g.ax_marg_y, left=True, bottom=False)

        # 4. Cuadro Estadístico de Rigor Flotante
        texto_metricas = (
            f"$\mu_{{err}} = {mu_res:.4f}$\n"
            f"$\sigma_{{err}} = {std_res:.4f}$\n"
            f"$\mathrm{{SK_B}} = {skew_res:.4f}$"
        )
        props = dict(boxstyle='round,pad=0.6', facecolor='#f7fafc', edgecolor='#cbd5e0', alpha=0.8)
        g.ax_joint.text(0.05, 0.05, texto_metricas, transform=g.ax_joint.transAxes,
                        verticalalignment='bottom', horizontalalignment='left', bbox=props, fontsize=20)

        # Guardar gráfico individual de alta calidad por parámetro
        plt.savefig(os.path.join(MODEL_PATH, f'residual_marginal_{label}_{PRUEBA}.pdf'), bbox_inches='tight', dpi=200)
        plt.close()

print(f"{GREEN}Gráficos de residuos con histograma marginal guardados individualmente.{ENDC}")

# Prueba de Kolmogorov-Smirnov y Skewness para evaluar la normalidad de los residuos
print(f"\n{YELLOW}======== Prueba de Kolmogorov-Smirnov y Skewness por Percentiles (Residuos) ========{ENDC}")
for label in LABELS:
    y_true = np.array(original_values[label])
    y_pred = np.array(predicted_values[label])
    residuals = y_true - y_pred
    mu_res, std_res = stats.norm.fit(residuals)
    q16, q50, q84 = np.percentile(residuals, [16, 50, 84])
    denom = q84 - q16
    skew_res = ((q84 - q50) - (q50 - q16)) / denom if denom != 0 else 0.0
    ks_stat, ks_pvalue = stats.kstest(residuals, 'norm', args=(mu_res, std_res))
    print(f"{CYAN}{label} | Skewness (percentil):{ENDC} {skew_res:.4f} {CYAN}| KS statistic:{ENDC} {ks_stat:.4f} {CYAN}| p-value:{ENDC} {ks_pvalue:.4e}")

print(f"\n{GREEN}Analysis complete. All plots and metrics saved.{ENDC}")

