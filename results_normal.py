import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats
import configparser
from contextlib import redirect_stdout
import os

# Color codes para terminal
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

# Configuración de rutas
main_config = load_config('main_config.ini')
LABELS = [item.strip() for item in main_config['MODEL']['labels'].split(',')]
plot_labels = [r'$\theta_E$', r'$f$', r'$\epsilon_x$', r'$\epsilon_y$']
PRUEBA = main_config['CONFIG']['prueba']
MAIN_PATH = main_config['PATHS']['main_path']
models = {'alexnet': 'alexnet', 'alexnet_original': 'alexnet_original'}
MODEL_PATH = os.path.join(MAIN_PATH, f"{models[PRUEBA]}/")

# --- CAMBIO CLAVE: Leer solo el CSV ---
CSV_PATH = os.path.join(MODEL_PATH, "predictions_vs_original.csv")

def relative_error(Vr, Vo):
    return abs((Vr - Vo) / (Vr + 1e-3)) * 100

def remove_outliers_iqr(df, labels):
    df_clean = df.copy()
    for col in labels:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[(df_clean[col] >= (Q1 - 1.5 * IQR)) & (df_clean[col] <= (Q3 + 1.5 * IQR))]
    return df_clean

def compute_errors(df):
    errors = {
        "theta_E": relative_error(df['theta_E_true'], df['theta_E_pred']),
        "f_axis": relative_error(df['f_true'], df['f_pred']),
        "e1": relative_error(df['e1_true'], df['e1_pred']),
        "e2": relative_error(df['e2_true'], df['e2_pred']),
    }
    return pd.DataFrame(errors)

print(f"{YELLOW}Reading CSV: {CSV_PATH}{ENDC}\n")
df = pd.read_csv(CSV_PATH)

# Calcular errores y limpiar outliers para estadísticas
errors_df = compute_errors(df)
errors_df_clean = remove_outliers_iqr(errors_df, LABELS)
errors_df.to_csv(os.path.join(MODEL_PATH, "errors_tot.csv"))

labels_temp = ['theta_E', 'f', 'e1', 'e2']
output_file = os.path.join(MODEL_PATH, f"metrics_model_{PRUEBA}.txt")
with open(output_file, "w") as f:
    with redirect_stdout(f):
        print(f"{YELLOW}======== R² Values per Parameter ========{ENDC}")
        for label in labels_temp:
            r2 = r2_score(df[f'{label}_true'], df[f'{label}_pred'])
            print(f"{CYAN}{label} | R²:{ENDC} {r2:.4f}")
        
        print(f"\n{YELLOW}======== Error Percentiles (p50, p65, p90) ========{ENDC}")
        for label in LABELS:
            p = np.percentile(errors_df_clean[label], [50, 65, 90])
            print(f"{CYAN}{label}:{ENDC} p50:{p[0]:.2f}% | p65:{p[1]:.2f}% | p90:{p[2]:.2f}%")

# --- Gráfica Original vs Predicho (Simplificada sin incertidumbre de ensamble) ---
def plot_scatter_simple():
    labels_map = {'theta_E': (r'$\theta_E$', 'Radio de Einstein'),
                  'f': (r'$f$', 'Relación axial'),
                  'e1': (r'$\epsilon_x$', 'Elipticidad x'),
                  'e2': (r'$\epsilon_y$', 'Elipticidad y')}
    
    fig, axes = plt.subplots(1, len(LABELS), figsize=(30, 8))
    for i, label in enumerate(labels_map.keys()):
        y_true = df[f'{label}_true'].values
        y_pred = df[f'{label}_pred'].values
        
        # Estadísticas básicas
        residuals = y_pred - y_true
        bias = np.median(residuals)
        r2 = r2_score(y_true, y_pred)
        q16, q84 = np.percentile(residuals, [16, 84])

        axes[i].scatter(y_true, y_pred, alpha=0.3, c='gray', s=10)
        
        # Formateo
        lims = [np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])]
        axes[i].plot(lims, lims, 'r--', lw=3)
        axes[i].set_title(f'{labels_map[label][0]} - {labels_map[label][1]}', fontsize=25)
        axes[i].set_xlabel('Valor original', fontsize=20)
        if i == 0: axes[i].set_ylabel('Valor predicho', fontsize=20)
        
        legend_txt = f"$R^2$={r2:.2f}\n$\mu$={bias:.2f}\n$\sigma^+$:{q84-bias:.2f}\n$\sigma^-$:{bias-q16:.2f}"
        axes[i].legend([legend_txt], loc='upper left', frameon=False, fontsize=18, handlelength=0)
        axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, f'original_vs_predicted_{PRUEBA}.pdf'))
    print(f"{GREEN}Scatter plot guardado.{ENDC}")

plot_scatter_simple()

# --- Gráfica de Errores Relativos y Binned MAE ---
fig, ax = plt.subplots(1, len(LABELS), figsize=(32, 8))
for n, label in enumerate(labels_temp):
    # Usamos los datos limpios para esta visualización
    y_true_clean = df.loc[errors_df_clean.index, f'{label}_true']
    err_vals = errors_df_clean[LABELS[n]].values
    
    median_err = np.median(err_vals)
    bins = np.linspace(y_true_clean.min(), y_true_clean.max(), 10)
    bin_means, bin_edges, _ = stats.binned_statistic(y_true_clean, err_vals, statistic='mean', bins=bins)
    
    ax[n].scatter(y_true_clean, err_vals, c='gray', s=6, alpha=0.2)
    ax[n].plot(bin_edges[:-1], bin_means, marker='o', c='b', lw=3, label='MAE agrupado')
    ax[n].axhline(y=median_err, c='green', linestyle='--', label=f'Mediana: {median_err:.2f}%')
    
    ax[n].set_title(f'Error Relativo: {label}', fontsize=25)
    ax[n].set_xlabel('Valor Original', fontsize=20)
    if n == 0: ax[n].set_ylabel('Error (%)', fontsize=20)
    ax[n].legend(fontsize=15)
    ax[n].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, f'residual_plot_combined_{PRUEBA}.pdf'))
plt.close()

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

print(f"\n{GREEN}Proceso finalizado sin archivos NPY.{ENDC}")