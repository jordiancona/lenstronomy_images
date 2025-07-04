
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
import lenstronomy
from lenstronomy.Util import util
from lenstronomy.Data.pixel_grid import PixelGrid
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
import astropy.io.fits as fits
from astropy.constants import c
from scipy.optimize import brentq
from dataclasses import dataclass

class sie_lens():
    def __init__(self,co, zl = 0.3, zs = 2.0, sigmav = 200,f = 0.6,pa = 45.0):
        self.sigmav = sigmav 
        self.co = co # cosmological model
        self.zl = zl # lens redshift
        self.zs = zs # source redshift
        self.f = f # axis ratio
        self.pa = pa #*np.pi/180.0 # position angle
        # compute the angular diameter distances:
        self.dl = self.co.angular_diameter_distance(self.zl)
        self.ds = self.co.angular_diameter_distance(self.zs)
        self.dls = self.co.angular_diameter_distance_z1z2(self.zl,self.zs)
        
        # calculates the Einstein radius of the SIS lens in arcsec
        self.theta0 = np.rad2deg((4.0*np.pi*sigmav**2/(c.to("km/s"))**2*self.dls/self.ds).value)*3600.0

    def delta(self,f,phi):
        return np.sqrt(np.cos(phi-self.pa)**2 + self.f**2*np.sin(phi-self.pa)**2)
    
    def kappa(self,x,phi):
        return (np.sqrt(self.f)/2.0/x/self.delta(self.f,phi))

    def gamma(self,x,phi):
        """
        Shear for the SIE lens at position (x,phi) in polar coordinates.
        """
        gamma1, gamma2 = (-self.kappa(x,phi)*np.cos(2.0*phi-self.pa),-self.kappa(x,phi)*np.sin(2.0*phi-self.pa))
        return gamma1, gamma2
    
    def mu(self,x,phi):
        """
        Magnification for the SIE lens at position (x,phi) in polar
        coordinates.
        """
        ga1,ga2 = self.gamma(x,phi)
        ga = np.sqrt(ga1*ga1+ga2*ga2)
        return 1.0/(1.0-self.kappa(x,phi)-ga)/(1.0-self.kappa(x,phi)+ga)
    
    def psi_tilde(self,phi):
        """
        angular part of the lensing potential at the polar angle phi
        """
        if (self.f < 1.0):
            fp=np.sqrt(1.0-self.f**2)
            return np.sqrt(self.f)/fp*(np.sin(phi-self.pa)*np.arcsin(fp*np.sin(phi-self.pa))+np.cos(phi-self.pa)*np.arcsinh(fp/self.f*np.cos(phi-self.pa)))
        else:
            return(1.0)

    def psi(self,x,phi):
        """
        Lensing potential at polar coordinates x,phi
        """
        psi = x*self.psi_tilde(phi)
        return psi

    def alpha(self,phi):
        """
        Deflection angle as a function of the polar angle phi
        """
        fp = np.sqrt(1.0-self.f**2)
        a1 = np.sqrt(self.f)/fp*np.arcsinh(fp/self.f*np.cos(phi))
        a2 = np.sqrt(self.f)/fp*np.arcsin(fp*np.sin(phi))
        return a1,a2

    def cut(self, phi_min = 0, phi_max = 2.0*np.pi, nphi=1000):
        """
        Coordinates of the points on the cut. The arguments phi_min, phi_max, nphi define the range of
        polar angles used.
        """
        phi = np.linspace(phi_min,phi_max,nphi)
        y1_, y2_ = self.alpha(phi)
        y1 = y1_ * np.cos(self.pa) - y2_ * np.sin(self.pa)
        y2 = y1_ * np.sin(self.pa) + y2_ * np.cos(self.pa)
        return -y1,-y2
    
    def tan_caustic(self,phi_min=0,phi_max=2.0*np.pi,nphi=1000):
        """
        Coordinates of the points on the tangential caustic. The arguments phi_min, phi_max, nphi
        define the range ofpolar angles used.
        """
        phi = np.linspace(phi_min,phi_max,nphi)
        delta = np.sqrt(np.cos(phi)**2+self.f**2*np.sin(phi)**2)
        a1,a2=self.alpha(phi)
        y1_=np.sqrt(self.f)/delta*np.cos(phi)-a1
        y2_=np.sqrt(self.f)/delta*np.sin(phi)-a2
        y1 = y1_ * np.cos(self.pa) - y2_ * np.sin(self.pa)
        y2 = y1_ * np.sin(self.pa) + y2_ * np.cos(self.pa)
        return y1,y2
    
    def tan_cc(self,phi_min=0,phi_max=2.0*np.pi,nphi=1000):
        """
        Coordinates of the points on the tangential critical line. The arguments phi_min, phi_max, nphi
        define the range of polar angles used.
        """
        phi = np.linspace(phi_min,phi_max,nphi)
        delta = np.sqrt(np.cos(phi)**2+self.f**2*np.sin(phi)**2)
        r = np.sqrt(self.f)/delta
        x1 = r*np.cos(phi+self.pa)
        x2 = r*np.sin(phi+self.pa)
        return(x1,x2)
    
    def x_ima(self,y1,y2,phi):
        x = y1*np.cos(phi)+y2*np.sin(phi)+(self.psi_tilde(phi+self.pa))
        return x
    
    def phi_ima(self, y1, y2, checkplot = True, eps = 0.001, nphi = 100):
        """
        Solve the lens Equation for a given source position (y1,y2)
        """
        # source position in the frame where the lens major axis is
        # along the £x_2£ axis.
        y1_ = y1 * np.cos(self.pa) + y2 * np.sin(self.pa)
        y2_ = - y1 * np.sin(self.pa) + y2 * np.cos(self.pa)
        # This is Eq.\,\ref{eq:ffunct}
        def phi_func(phi):
            a1,a2=self.alpha(phi)
            func=(y1_+a1)*np.sin(phi)-(y2_+a2)*np.cos(phi)
            return func
        # Evaluate phi_func and the sign of phi_func on an array of
        # polar angles
        U=np.linspace(0.,2.0*np.pi+eps,nphi)
        c = phi_func(U)
        s = np.sign(c)
        phi = []
        xphi = []
        # loop over polar angles
        for i in range(len(U)-1):
        # if two polar angles bracket a zero of phi_func,
        # use Brent’s method to find exact solution
            if s[i] + s[i+1] == 0: # opposite signs
                u = brentq(phi_func, U[i], U[i+1])
                z = phi_func(u)
                if np.isnan(z) or abs(z) > 1e-3:
                    continue
                x = self.x_ima(y1_,y2_,u)
                # append solution to a list if it corresponds to radial
                # distances x>0; discard otherwise (spurious solutions)
                if (x > 0):
                    phi.append(u)
                    xphi.append(x)
            # convert lists to numpy arrays
        xphi = np.array(xphi)
        phi = np.array(phi)
            # returns radii and polar angles of the images. Add position angle
            # to go back to the rotated frame of the lens.
        return xphi, phi+self.pa

@dataclass
class Lenses:
    @classmethod
    def makelens(self, n, f, thetaE, e1, e2, gamma1, gamma2, center_x, center_y):
        self.file_name = f'lens{n+1}'
        
        self.f = f
        self.thetaE = thetaE
        self.e1, self.e2 = e1, e2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.center_x = center_x
        self.center_y = center_y

        # specify the choice of lens models #
        lens_model_list = ['SIE', 'SHEAR']

        # setup lens model class with the list of lens models #
        lensModel = LensModel(lens_model_list = lens_model_list)

        # define parameter values of lens models #
        kwargs_spep = {'theta_E': self.thetaE, 
                       'e1': self.e1, 
                       'e2': self.e2, 
                       'center_x': center_x, 
                       'center_y': center_y}
        
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
        kwargs_lens = [kwargs_spep, kwargs_shear]

        # image plane coordinate #
        # image plane coordinate #
        theta_ra, theta_dec = 1.,.5
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

        ##e1, e2 = param_util.phi_q2_ellipticity(phi=0.5, q=0.7)
        kwargs_light_lens = [{'amp': 1000,
                              'R_sersic': 0.1,
                              'n_sersic': 2.5,
                              'e1': self.e1,
                              'e2': self.e2,
                              'center_x': center_x,
                              'center_y': center_y}]

        # evaluate surface brightness at a specific position #
        flux = lightModel_lens.surface_brightness(x = 1, y = 1, kwargs_list = kwargs_light_lens)

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
                                  lens_model = lensModel,
                                  fixed_magnification_list = [False])

        kwargs_ps = [{'ra_image': theta_ra, 'dec_image': theta_dec, 'point_amp': np.abs(mag)*30}]
        # return image positions and amplitudes #
        x_pos, y_pos = pointSource.image_position(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)
        point_amp = pointSource.image_amplitude(kwargs_ps = kwargs_ps, kwargs_lens = kwargs_lens)

        deltaPix = 0.05  # size of pixel in angular coordinates #

        im_dim = 200
        # setup the keyword arguments to create the Data() class #
        ra_at_xy_0, dec_at_xy_0 = -im_dim*deltaPix/2., -im_dim*deltaPix/2. # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': im_dim, 
                        'ny': im_dim,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 

        pixel_grid = PixelGrid(**kwargs_pixel)
        # return the list of pixel coordinates #
        x_coords, y_coords = pixel_grid.pixel_coordinates
        # compute pixel value of a coordinate position #
        x_pos, y_pos = pixel_grid.map_coord2pix(ra = 0, dec = 0)
        # compute the coordinate value of a pixel position #
        ra_pos, dec_pos = pixel_grid.map_pix2coord(x = 20, y = 10)

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
        self.image = imageModel.image(kwargs_lens = kwargs_lens, kwargs_source = kwargs_light_source,
                                      kwargs_lens_light = kwargs_light_lens, kwargs_ps = kwargs_ps)

        # image with noise
        exp_time = 100  # exposure time to quantify the Poisson noise level
        background_rms = 0.1  # background rms value
        poisson = image_util.add_poisson(self.image, exp_time = exp_time)
        bkg = image_util.add_background(self.image, sigma_bkd = background_rms)
        image_noisy = self.image + bkg + poisson

    # Se crean los archivos FITS a partir de los PNG
    @classmethod
    def Create_FITS(self, path):
        file = self.file_name
        #inbase_name, inbase_ext = os.path.splitext(os.path.basename(file))
        outfile = path + file + '.fits'

        #inimage = imageio.imread(self.path + file, mode = 'F')
        outimage = np.flipud(self.image) #inimage

        file_time = strftime('%Y-%m-%d %H:%M:%S', gmtime())

        outhdr = fits.Header()
        outhdr['DATE'] = file_time
        outhdr['HISTORY'] = 'Generated by fits'
        outhdr['NAME'] = file

        # Lens parameters
        c1 = fits.Card('theta_E', self.thetaE, 'Einstein Radius')
        c2 = fits.Card('f_axis', self.f, 'axial radio')
        c3 = fits.Card('e1', self.e1, 'elipticity1')
        c4 = fits.Card('e2', self.e2, 'elipticity2')

        # Shear components
        c5 = fits.Card('gamma1', self.gamma1, 'first shear component')
        c6 = fits.Card('gamma2', self.gamma2, 'second shear component')

        # Lens Coordinates
        c7 = fits.Card('center_x', self.center_x, 'x coordinate')
        c8 = fits.Card('center_y', self.center_y, 'y coordinate')
        parameters = [c1, c2, c3, c4, c5, c6, c7, c8]

        for parameter in parameters:
            outhdr.append(parameter, end = True)
        
        outlist = fits.ImageHDU(data = outimage, header = outhdr) # .astype('float32')
        outlist.writeto(outfile, overwrite = True)
