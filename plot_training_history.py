
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import os

def read_config():
    config = configparser.ConfigParser()
    config.read('main_config.ini')
    return config

main_config = read_config()
PRUEBA = main_config['CONFIG']['prueba']
MAIN_PATH = main_config['PATHS']['main_path']
PATH = os.path.join(MAIN_PATH, f'alexnet_{PRUEBA}/')

df = pd.read_csv(PATH + f'training_history_{PRUEBA}.csv')

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(df['loss'], color = 'b', marker='*', markerfacecolor='k',  markeredgecolor='k', markersize=2, linewidth=0.8, label = 'Training')
ax[0].plot(df['val_loss'], color = 'r', linestyle='--', marker='*', markerfacecolor='k',  markeredgecolor='k', markersize=2, linewidth=0.8, label = 'Validation')
ax[0].set_xlabel('Epoch', fontsize=12)
ax[0].set_ylabel('Loss', fontsize=12)
ax[0].set_xticks(range(0, len(df['loss'])+1, 10))
ax[0].legend(frameon=False)

ax[1].plot(df['mae'], color = 'b', marker='*', markerfacecolor='k',  markeredgecolor='k', markersize=2, linewidth=0.8, label = 'Training')
ax[1].plot(df['val_mae'], color = 'r', linestyle='--', marker='*', markerfacecolor='k',  markeredgecolor='k', markersize=2, linewidth=0.8, label = 'Validation')
ax[1].set_xlabel('Epoch', fontsize=12)
ax[1].set_ylabel('MAE', fontsize=12)
ax[1].set_xticks(range(0, len(df['mae'])+1, 10))
ax[1].legend(frameon=False)

plt.suptitle(f'Model {PRUEBA}')
plt.tight_layout()
#plt.show()
plt.savefig(PATH + f'training_history_{PRUEBA}.pdf')
