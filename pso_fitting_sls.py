
import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
import configparser

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
ENDC = '\033[0m'

def print_colored(message, color):
    print(f"{color}{message}{ENDC}")

# Load configuration
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

try:
    main_config = load_config('main_config.ini')
    DELTA_PIX = main_config.getfloat('MODEL', 'delta_pix')
    FILE = main_config['PATHS']['csv_file']
    print_colored("Configuration loaded successfully.", GREEN)
except Exception as e:
    print_colored(f"Error loading configuration: {e}", RED)
    exit(1)

def build_data(image, noise):
    # Simulate data
    kwargs_data = {'image_data': image, 'noise_map': noise, 'pixel_scale': DELTA_PIX}
    return ImageData(**kwargs_data)

def build_psf():
    kwargs_psf = {'psf_type': 'GAUSSIAN',
                  'fwhm': 0.15,
                  'pixel_scale': DELTA_PIX}

    return PSF(**kwargs_psf)

def main():
    try:
        lens_samples = Table.read(FILE)
        print_colored(f"Successfully loaded data from {FILE}", GREEN)
    except FileNotFoundError:
        print_colored(f"Error: The file {FILE} was not found.", RED)
        return

    # Dividir en conjuntos de entrenamiento y prueba
    lens_model_list = ['SIE']
    source_model_list = ['Sersic']

    kwargs_model = {'lens_model': lens_model_list, 
                    'source_light_model': source_model_list,
                    'lens_light_model': []}

    kwargs_lens_init = [{'theta_E': 1.0, 'e1': 0.0, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}]
    kwargs_lower = [{'theta_E': 0.3, 'e1': -0.5, 'e2': -0.5, 'center_x': -1.0, 'center_y': -1.0}]
    kwargs_upper = [{'theta_E': 2.0, 'e1': 0.5, 'e2': 0.5, 'center_x': 1.0, 'center_y': 1.0}]

    kwargs_source_init = [{'re_s': 0.5, 'n_s': 1.0, 'center_x': 0.0, 'center_y': 0.0}]
    kwargs_source_lower = [{'re_s': 0.1, 'n_s': 0.5, 'center_x': -1.0, 'center_y': -1.0}]
    kwargs_source_upper = [{'re_s': 1.0, 'n_s': 4.0, 'center_x': 1.0, 'center_y': 1.0}]

    def build_lens_fitting(image, noise):
        data = build_data(image, noise)
        psf = build_psf()
        lens_model_list = ['SIE']
        source_model_list = ['Sersic']

        kwargs_data_joint = {'image_data': data, 
                            'psf': psf}
        
        kwargs_constraints = {}

        kwargs_likelihood = {'image_likelihood': True}

        kwargs_params = {'lens_model': [kwargs_lens_init, kwargs_lower_lens, kwargs_upper_lens],
                         'source_model': [kwargs_source_init, kwargs_lower_source, kwargs_upper_source]
                         }

        return FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params
        )
    def run_pso(image, noise):
        fitting_sequence = build_lens_fitting(image, noise)

        fitting_list = [['PSO', {'sigma_scale': 0.1, 'n_particles': 100, 'n_iterations': 200}]]

        chain_list = fitting_sequence.fit_sequence(fitting_list)
        kwargs_result = fitting_sequence.best_fit()
        return kwargs_result

    
    results = []
    for i, lens_data in enumerate(lens_samples):

    

