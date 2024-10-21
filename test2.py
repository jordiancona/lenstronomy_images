#!/usr/local/bin/python3

import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.cosmology import default_cosmology
import imageio

try:
    import lenstronomy
except:
    print("lenstronomy not installed")

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

file = sys.argv[1]
file_name, file_extension = os.path.splitext(file)

main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
kwargs_lens_main = {'theta_E': 1., 'e1': 0.1, 'e2': 0.1, 'center_x': 0, 'center_y': 0}
kwargs_shear = {'gamma1': 0.05, 'gamma2': 0.05}
lens_model_list = [main_halo_type, 'SHEAR']
kwargs_lens_list = [kwargs_lens_main, kwargs_shear]

subhalo_type = 'TNFW'  # We chose spherical NFW profiles, feel free to chose whatever you want. "TNFW"

# as an example, we render some sub-halos with a very simple distribution to be added on the main lens
num_subhalo = 10  # number of subhalos to be rendered
# the parameterization of the NFW profiles are:
# - Rs (radius of the scale parameter Rs in units of angles)
# - theta_Rs (radial deflection angle at Rs)
# - center_x, center_y, (position of the centre of the profile in angular units)

Rs_mean = 0.1
Rs_sigma = 0.1  # dex scatter
theta_Rs_mean = 0.05
theta_Rs_sigma = 0.1 # dex scatter
r_min, r_max = -2, 2

Rs_list = 10**(np.log10(Rs_mean) + np.random.normal(loc=0, scale=Rs_sigma, size=num_subhalo))
theta_Rs_list = 10**(np.log10(theta_Rs_mean) + np.random.normal(loc=0, scale=theta_Rs_sigma, size=num_subhalo))
center_x_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)
center_y_list = np.random.uniform(low=r_min, high=r_max,size=num_subhalo)

for i in range(num_subhalo):
    lens_model_list.append(subhalo_type)
    kwargs_lens_list.append({'alpha_Rs': theta_Rs_list[i],
                             'Rs': Rs_list[i],
                             'center_x': center_x_list[i],
                             'center_y': center_y_list[i],
                             'r_trunc': 5 * Rs_list[i]
                            })

lensModel = LensModel(lens_model_list)
# we set up a grid in coordinates and evaluate basic lensing quantities on it
x_grid, y_grid = util.make_grid(numPix = 100, deltapix = 0.05)
kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens_list)
# we make a 2d array out of the 1d grid points
kappa = util.array2image(kappa)

z_lens = 0.5
z_source = 2

cosmo = default_cosmology.get()

# class that converts angular to physical units for a specific cosmology and redshift configuration
lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

# here we turn an NFW halo defined as M200 crit and concentration into lensing quantities
M200 = 10**9
concentration = 6
Rs_angle_clump, theta_Rs_clump = lensCosmo.nfw_physical2angle(M=M200, c=concentration)
print(Rs_angle_clump, theta_Rs_clump)

# and here we do the oposite and turn the lensing quantities into physical units
rho0_clump, Rs_clump, c_clump, r200_clump, M200_clump = lensCosmo.nfw_angle2physical(Rs_angle_clump, theta_Rs_clump)
print(rho0_clump, Rs_clump, c_clump, r200_clump, M200_clump)



# find path to data
path = os.getcwd()
dirpath, _ = os.path.split(path)
module_path, _ = os.path.split(dirpath)
ngc_filename = os.path.join(module_path, "lenstronomy/lenstronomy_images/images/images_1/"+file)

ngc_data = imageio.imread(ngc_filename, mode = 'F', pilmode=None) # as_gray = True

# subtract the median of an edge of the image
median = np.median(ngc_data[:200, :200]) # 200, 200
ngc_data -= median

# resize the image to square size (add zeros at the edges of the non-square bits of the image)
nx, ny = np.shape(ngc_data)
n_min = min(nx, ny)
n_max = max(nx, ny)
ngc_square = np.zeros((n_max, n_max))
x_start = int((n_max - nx)/2.)
y_start = int((n_max - ny)/2.)
ngc_square[x_start:x_start+nx, y_start:y_start+ny] = ngc_data

# we slightly convolve the image with a Gaussian convolution kernel of a few pixels (optional)
sigma = 5 # 5
ngc_conv = scipy.ndimage.filters.gaussian_filter(ngc_square, sigma, mode='nearest', truncate=6)

# we now degrate the pixel resoluton by a factor.
# This reduces the data volume and increases the spead of the Shapelet decomposition
factor = 3  # lower resolution of image with a given factor (25)
numPix_large = int(len(ngc_conv)/factor)
n_new = int((numPix_large-1)*factor)
ngc_cut = ngc_conv[0:n_new,0:n_new]
x, y = util.make_grid(numPix=numPix_large-1, deltapix=1)  # make a coordinate grid
ngc_data_resized = image_util.re_size(ngc_cut, factor)  # re-size image to lower resolution

# now we come to the Shapelet decomposition
# we turn the image in a single 1d array
image_1d = util.image2array(ngc_data_resized)  #

n_max = 150  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
beta = 10  # shapelet scale parameter (in units of resized pixels)

# import the ShapeletSet class
shapeletSet = ShapeletSet()

# decompose image and return the shapelet coefficients
coeff_ngc = shapeletSet.decomposition(image_1d, x, y, n_max, beta, 1., center_x=0, center_y=0) 
print(len(coeff_ngc), 'number of coefficients')  # number of coefficients

# reconstruct NGC1300 with the shapelet coefficients
image_reconstructed = shapeletSet.function(x, y, coeff_ngc, n_max, beta, center_x=0, center_y=0)
# turn 1d array back into 2d image
image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image


# we define a very high resolution grid for the ray-tracing (needs to be checked to be accurate enough!)
numPix = 100  # number of pixels (low res of data) 100
deltaPix = 0.05  # pixel size (low res of data)
high_res_factor = 6  # higher resolution factor (per axis) 3
# make the high resolution grid 
theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix*high_res_factor, deltapix=deltaPix/high_res_factor)
# ray-shoot the image plane coordinates (angles) to the source plane (angles)
beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res, theta_y_high_res, kwargs=kwargs_lens_list)

# now we do the same as in Section 2, we just evaluate the shapelet functions in the new coordinate system of the source plane
# Attention, now the units are not pixels but angles! So we have to define the size and position.
# This is simply by chosing a beta (Gaussian width of the Shapelets) and a new center

source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res, coeff_ngc, n_max, beta=.05, center_x=0.2, center_y=0)
# and turn the 1d vector back into a 2d array
source_lensed = util.array2image(source_lensed)  # map 1d data vector in 2d image

f, ax = plt.subplots(1, 1, figsize = (4, 4), sharex = False, sharey = False)
'''
# Reconstructed image of the galaxy
ax = axes[0]
im = ax.matshow(image_reconstructed_2d, origin='lower')
ax.set_title("reconstructed")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)
'''
# Image lensed
#ax = axes[0]
im = ax.matshow(source_lensed, origin = 'lower')
#ax.set_title("lensed source")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)
plt.savefig(f'mock_{file_name}.png')

# and plot the convergence of the lens model
#plt.matshow(np.log10(kappa), origin = 'lower')
#plt.show()
