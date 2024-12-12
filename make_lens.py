
try:
    import lenstronomy
except:
    print('Lenstronomy no está instalado')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c, G

background_rms = .5  #  background noise per pixel
exp_time = 100  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 100  #  cutout pixel size per axis 60
pixel_scale = 0.05  #  pixel size in arcsec (area per pixel = pixel_scale**2) 0.05
fwhm = 0.1  # full width at half maximum of PSF 0.05
psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

fwhm = 0.1 # PSF FWHM
kwargs_data = sim_util.data_configure_simple(numPix, pixel_scale, exp_time, background_rms)
data_class = ImageData(**kwargs_data)
kwargs_psf = {'psf_type': 'GAUSSIAN','fwhm': fwhm,'pixel_size': pixel_scale,'truncation': 5}
psf_class = PSF(**kwargs_psf)

f=0.7
sigmav=200.
pa=np.pi/4.0 # position angle in radians
zl=0.3 # lens redshift
zs=1.5 # source redshift

# lens Einstein radius
co = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
dl=co.angular_diameter_distance(zl)
ds=co.angular_diameter_distance(zs)
dls=co.angular_diameter_distance_z1z2(zl,zs)

# compute the Einstein radius
thetaE = 1e6*(4.0*np.pi*sigmav**2/c**2*dls/ds*180.0/np.pi*3600.0).value
# eccentricity computation
e1, e2=(1-f)/(1+f)*np.cos(-2*pa), (1-f)/(1+f)*np.sin(-2*pa)
lens_model_list = ['SIE']
kwargs_sie = {'theta_E': thetaE,
              'center_x': 0,
              'center_y': 0,
              'e1': e1,
              'e2': e2}

kwargs_lens = [kwargs_sie]
lens_model_class = LensModel(lens_model_list = lens_model_list)

# create the light model for the lens (SERSIC_ELLIPSE)
lens_light_model_list = ['SERSIC_ELLIPSE']
kwargs_sersic = {'amp': 3500, # flux of the lens (arbitrary units)
                 'R_sersic': 2., # effective radius
                 'n_sersic': 4, # sersic index
                 'center_x': 0, # x-coordinate
                 'center_y': 0, # y-coordinate
                 'e1': e1,
                 'e2': e2}

kwargs_lens_light = [kwargs_sersic]
lens_light_model_class = LightModel(light_model_list = lens_light_model_list)
# create the light model for the source (SERSIC_ELLIPSE)
source_model_list = ['SERSIC_ELLIPSE']
# set the position of the source
ra_source, dec_source = -0.1 * thetaE, thetaE
kwargs_sersic_ellipse = {'amp': 4000.,
                         'R_sersic': .1,
                         'n_sersic': 3,
                         'center_x': ra_source,
                         'center_y': dec_source,
                         'e1': 0.1,
                         'e2': 0.01}

kwargs_source = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list = source_model_list)

# solve the lens equation and find the image positions
# using the LensEquationSolver class of Lenstronomy.
lensEquationSolver = LensEquationSolver(lens_model_class)
x_image, y_image = lensEquationSolver.image_position_from_source(ra_source,
                                                                 dec_source,
                                                                 kwargs_lens,
                                                                 min_distance=pixel_scale,
                                                                 search_window=numPix * pixel_scale,
                                                                 precision_limit=1e-10, 
                                                                 num_iter_max=100,
                                                                 arrival_time_sort=True,
                                                                 initial_guess_cut=True,
                                                                 verbose=False,
                                                                 x_center=0,
                                                                 y_center=0,
                                                                 num_random=0,
                                                                 non_linear=False,
                                                                 magnification_limit=None)

# compute lensing magnification at image positions
mag = lens_model_class.magnification(x_image,
                                     y_image,
                                     kwargs=kwargs_lens)
mag = np.abs(mag) # ignore the sign of the magnification
# perturb observed magnification due to e.g. micro-lensing
# the noise is generated from a normal distribution
# with mean ’mag’ and standard deviation 0.5

mag_pert = np.random.normal(mag, 0.5, len(mag))

# quasar position in the lens plane
kwargs_ps = [{'ra_image': x_image,
              'dec_image': y_image,
              'point_amp': mag}]

point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list = point_source_list,
                                 fixed_magnification_list=[False])
# create the simulated observation of lens and (lensed)
# source
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
# imageModel includes the details of the instrument, psf, lens,
# and source models
imageModel = ImageModel(data_class, psf_class, lens_model_class,
source_model_class,lens_light_model_class,point_source_class, kwargs_numerics = kwargs_numerics)
# now, the simulated image is saved in image_sim
image_sim = imageModel.image(kwargs_lens, kwargs_source,
kwargs_lens_light, kwargs_ps)
# add noise and background
poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=background_rms)
image_sim = image_sim + poisson + bkg 

data_class.update_data(image_sim)
kwargs_data['image_data'] = image_sim

# plotting the lens system
cmap = mpl.cm.get_cmap("gray").copy()
cmap.set_bad(color='k', alpha=1.)
cmap.set_under('k')

v_min = -4
v_max = 1

f, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=False, sharey=False)

ax = axes
im = ax.matshow(np.log10(image_sim), origin = 'lower', cmap = cmap, extent=[0, 1, 0, 1]) #  vmin=v_min, vmax=v_max, 
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

plt.show()
