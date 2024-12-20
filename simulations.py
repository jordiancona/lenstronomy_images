
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
#from lenstronomy.Util import util
import matplotlib.pyplot as plt

# Configuración básica
kwargs_data = {
    'pixel_scale': 0.05,  # Escala del píxel
    'num_pix': 100,       # Tamaño de la imagen
}

kwargs_psf = {
    'psf_type': 'GAUSSIAN',
    'fwhm': 0.1  # Full Width at Half Maximum
}

# Modelo de masa de la lente
lens_model_list = ['SIE']  # Elipsoide isotérmico singular
kwargs_lens = [{'theta_E': 1.2, 'e1': 0.1, 'e2': 0.05, 'center_x': 0.0, 'center_y': 0.0}]

# Modelo de luz de la galaxia de fondo
light_model_list = ['SERSIC_ELLIPSE']
kwargs_light = [{'amp': 1, 'R_sersic': 0.5, 'n_sersic': 2, 'e1': 0.1, 'e2': -0.1, 'center_x': 0.1, 'center_y': 0.1}]

# Simulación
lens_model = LensModel(lens_model_list)
light_model = LightModel(light_model_list)
sim = SimAPI(numpix = 100, kwargs_single_band =kwargs_data, kwargs_psf=kwargs_psf, lens_model=lens_model, source_model=light_model)
image = sim.image_model(kwargs_lens=kwargs_lens, kwargs_source=kwargs_light)

# Visualización
plt.imshow(image, cmap='inferno')
plt.colorbar()
plt.title('Lente Gravitacional Simulada')
plt.show()
