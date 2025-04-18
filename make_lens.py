
try:
    import lenstronomy
except:
    print('Lenstronomy no está instalado')

import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Util import util
from lenstronomy.Data.pixel_grid import PixelGrid
import lenstronomy.Util.image_util as image_util
#from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c

def MakeLens(thetaE, e1, e2, gamma1, gamma2, center_x, center_y):

    # specify the choice of lens models #
    lens_model_list = ['SIE', 'SHEAR']

    # setup lens model class with the list of lens models #
    lensModel = LensModel(lens_model_list = lens_model_list)

    # define parameter values of lens models #
    kwargs_spep = {'theta_E': thetaE, 
                    'e1': e1, 
                    'e2': e2, 
                    'center_x': center_x, 
                    'center_y': center_y}
    
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
    kwargs_lens = [kwargs_spep, kwargs_shear]

    # image plane coordinate #
    theta_ra, theta_dec = np.array(0.), np.array(0.)
    # source plane coordinate #
    beta_ra, beta_dec = lensModel.ray_shooting(theta_ra, theta_dec, kwargs_lens)
    # Fermat potential #
    fermat_pot = lensModel.fermat_potential(x_image = theta_ra, y_image = theta_dec, x_source = beta_ra, y_source=beta_dec, kwargs_lens = kwargs_lens)

    # Magnification #
    mag = lensModel.magnification(theta_ra, theta_dec, kwargs_lens)

    # specifiy the lens model class to deal with #
    solver = LensEquationSolver(lensModel)

    # solve for image positions provided a lens model and the source position #
    theta_ra, theta_dec = solver.image_position_from_source(beta_ra, beta_dec, kwargs_lens)

    # the magnification of the point source images #
    mag = lensModel.magnification(theta_ra, theta_dec, kwargs_lens)

    # set up the list of light models to be used #
    source_light_model_list = ['SERSIC']
    lightModel_source = LightModel(light_model_list=source_light_model_list)

    lens_light_model_list = ['SERSIC_ELLIPSE']
    lightModel_lens = LightModel(light_model_list=lens_light_model_list)

    # define the parameters #
    kwargs_light_source = [{'amp': 100,
                            'R_sersic': 0.1,
                            'n_sersic': 1.5, 
                            'center_x': beta_ra, 
                            'center_y': beta_dec}]

    #e1, e2 = param_util.phi_q2_ellipticity(phi=0.5, q=0.7)
    kwargs_light_lens = [{'amp': 1000,
                            'R_sersic': 0.1,
                            'n_sersic': 2.5,
                            'e1': e1,
                            'e2': e2,
                            'center_x': center_x,
                            'center_y': center_y}]

    # evaluate surface brightness at a specific position #
    flux = lightModel_lens.surface_brightness(x = 0, y = 0, kwargs_list = kwargs_light_lens)

    # unlensed source positon #
    point_source_model_list = ['SOURCE_POSITION']
    pointSource = PointSource(point_source_type_list = point_source_model_list,
                                lens_model = lensModel,
                                fixed_magnification_list = [True])

    kwargs_ps = [{'ra_source': beta_ra, 'dec_source': beta_dec, 'source_amp': 100}]
    # return image positions and amplitudes #
    x_pos, y_pos = pointSource.image_position(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)
    point_amp = pointSource.image_amplitude(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)

    # lensed image positions (solution of the lens equation) #
    point_source_model_list = ['LENSED_POSITION']
    pointSource = PointSource(point_source_type_list = point_source_model_list,
                            lens_model=lensModel,
                            fixed_magnification_list = [False])

    kwargs_ps = [{'ra_image': theta_ra, 'dec_image': theta_dec, 'point_amp': np.abs(mag)*30}]
    # return image positions and amplitudes #
    x_pos, y_pos = pointSource.image_position(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)
    point_amp = pointSource.image_amplitude(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)

    deltaPix = 0.05  # size of pixel in angular coordinates #

    # setup the keyword arguments to create the Data() class #
    ra_at_xy_0, dec_at_xy_0 = -2.5, -2.5 # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
    transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix  # linear translation matrix of a shift in pixel in a shift in coordinates
    kwargs_pixel = {'nx': 100, 'ny': 100,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                    'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                    'transform_pix2angle': transform_pix2angle} 

    pixel_grid = PixelGrid(**kwargs_pixel)
    # return the list of pixel coordinates #
    x_coords, y_coords = pixel_grid.pixel_coordinates
    # compute pixel value of a coordinate position #
    x_pos, y_pos = pixel_grid.map_coord2pix(ra = 0, dec = 0)
    # compute the coordinate value of a pixel position #
    ra_pos, dec_pos = pixel_grid.map_pix2coord(x = 20, y = 10) #20 10

    # PSF
    kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
                  'fwhm': 0.1,  # full width at half maximum of the Gaussian PSF (in angular units)
                  'pixel_size': deltaPix  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
                }

    psf = PSF(**kwargs_psf)
    # return the pixel kernel corresponding to a point source # 
    kernel = psf.kernel_point_source

    # define the numerics #
    kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                    'supersampling_convolution': False}
    # initialize the Image model class by combining the modules we created above #
    imageModel = ImageModel(data_class = pixel_grid,
                            psf_class = psf,
                            lens_model_class = lensModel,
                            source_model_class = lightModel_source,
                            lens_light_model_class = lightModel_lens,
                            point_source_class = None, # in this example, we do not simulate point source.
                            kwargs_numerics = kwargs_numerics)
    
    # simulate image with the parameters we have defined above #
    image = imageModel.image(kwargs_lens = kwargs_lens, kwargs_source = kwargs_light_source,
                            kwargs_lens_light = kwargs_light_lens, kwargs_ps = kwargs_ps)

    # image with noise
    exp_time = 100  # exposure time to quantify the Poisson noise level
    background_rms = 0.1  # background rms value
    poisson = image_util.add_poisson(image, exp_time = exp_time)
    bkg = image_util.add_background(image, sigma_bkd = background_rms)
    image_noisy = image + bkg + poisson

    #f, ax = plt.subplots(1, 1, figsize = (4, 4), sharex = False, sharey = False)
    #ax.matshow(np.log10(image), origin = 'lower', cmap = 'gist_heat')
    #plt.axis('off')
    #axes[1].matshow(np.log10(image_noisy), origin='lower', cmap = 'gray')
    #f.tight_layout()
    #plt.savefig(f'{name}.png', bbox_inches = "tight")
    #plt.close()
    return image