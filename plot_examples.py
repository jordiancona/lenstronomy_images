import matplotlib.pyplot as plt
import numpy as np
# Importamos los módulos necesarios
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel
# Usamos Extensions, que es la forma estable de calcular curvas
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

# --- 1. CONFIGURACIÓN DE LA MALLA (GRID) ---
num_pix = 250        # Más píxeles para mejor resolución
delta_pix = 0.02     # Píxeles más pequeños (resolución más fina)
half_width = num_pix * delta_pix / 2
extent = [-half_width, half_width, -half_width, half_width]
transform_pix2angle = np.array([[delta_pix, 0], [0, delta_pix]]) # <--- EL CAMBIO ESTÁ AQUÍ

kwargs_grid = {
    'ra_at_xy_0': -half_width, 
    'dec_at_xy_0': -half_width,
    'transform_pix2angle': transform_pix2angle, # Ahora es un array de numpy
    'nx': num_pix, 
    'ny': num_pix
}
grid = PixelGrid(**kwargs_grid)


# --- 2. DEFINICIÓN DE LOS MODELOS FÍSICOS ---

# A) MODELO DE MASA DE LA LENTE (La gravedad invisible)
# Define el Radio de Einstein y la Elipticidad
theta_E = 1.4
e1_mass, e2_mass = 0.25, 0.1  # Una elipticidad notable
lens_model = LensModel(['SIE'])
kwargs_lens = [{'theta_E': theta_E, 'e1': e1_mass, 'e2': e2_mass, 'center_x': 0, 'center_y': 0}]

# B) MODELO DE LUZ DE LA LENTE (La galaxia visible en primer plano)
# ¡NUEVO! Esto es lo que pediste agregar.
lens_light_model = LightModel(['SERSIC'])
# Usamos n_sersic=4, típico de galaxias elípticas masivas.
# La centramos donde está la masa (0,0).
kwargs_lens_light = [{
    'amp': 800, 'R_sersic': 0.8, 'n_sersic': 4, 
    'center_x': 0, 'center_y': 0,
    'e1': 0.1, 'e2': 0.05 # La luz también puede ser elíptica
}]

# C) MODELO DE LUZ DE LA FUENTE (La galaxia de fondo distorsionada)
source_model = LightModel(['SERSIC'])
# La ponemos un poco descentrada para que forme arcos interesantes.
kwargs_source = [{'amp': 3000, 'R_sersic': 0.1, 'n_sersic': 1.5, 'center_x': 0.1, 'center_y': 0.15}]


# --- 3. GENERACIÓN DE LA IMAGEN ---
# Le pasamos los tres modelos: masa, luz de fuente y luz de lente.
image_model = ImageModel(grid, None, lens_model, source_model, lens_light_model)
# Generamos la imagen final sumando todo
image = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light)


# --- 4. CÁLCULO DE ELEMENTOS VISUALES (ANÁLISIS) ---
lens_extras = LensModelExtensions(lens_model)
# Calculamos las curvas críticas (donde la magnificación es teóricamente infinita)
# compute_window debe ser un poco más grande que theta_E
ra_crit, dec_crit, _, _ = lens_extras.critical_curve_caustics(
    kwargs_lens, compute_window=3.0, grid_scale=delta_pix
)


# --- 5. GRAFICADO ---
fig, ax = plt.subplots(figsize=(8, 8))

# A) La imagen simulada (Lente + Arcos de la fuente)
im = ax.imshow(image, origin='lower', extent=extent, cmap='magma', norm='log', vmin=1e-1)

# B) El Radio de Einstein Teórico (Círculo Cian)
# Representa la escala si la lente fuera una esfera perfecta.
circle = plt.Circle((0, 0), theta_E, color='cyan', fill=False, ls='--', lw=2, label=f'Radio de Einstein Teórico ($\theta_E$={theta_E}")')
ax.add_artist(circle)

# C) La Elipticidad Real (Curva Blanca)
# Muestra cómo la masa elíptica deforma el potencial gravitacional.
# Los arcos más brillantes tienden a formarse cerca de esta línea.
for i in range(len(ra_crit)):
    ax.plot(ra_crit[i], dec_crit[i], color='white', lw=1.5, ls='-', label='Curva Crítica (Efecto Elipticidad)' if i==0 else "")

# Decoración
ax.set_title("Simulación Completa: Lente, Fuente y Análisis")
ax.set_xlabel("Arcsec (RA)")
ax.set_ylabel("Arcsec (Dec)")
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

# Añadimos una barra de color logarítmica para ver mejor los detalles tenues
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Flujo (escala logarítmica)')

plt.show()
