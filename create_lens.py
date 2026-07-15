
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
from astropy.table import Table

NUM_PIX = 100  # Número de píxeles
DELTA_PIX = 0.05  # Tamaño del píxel en arcsec

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
    def makelens(self, n, idx, f, thetaE, e1, e2, center_x, center_y):
        self.file_name = f'lens_{n+1}'
        
        self.f = f
        self.thetaE = thetaE
        self.e1_s, self.e2_s = e1, e2
        self.x_s = center_x
        self.y_s = center_y
        
        pa_l = 0.0
        e = (1 - self.f)/(1 + self.f)
        self.e1_l, self.e2_l = e*np.cos(2*pa_l), e*np.sin(2*pa_l)
        # specify the choice of lens models #
        x, y = np.meshgrid(np.linspace(-NUM_PIX/2 * DELTA_PIX, NUM_PIX/2 * DELTA_PIX, NUM_PIX),
                           np.linspace(-NUM_PIX/2 * DELTA_PIX, NUM_PIX/2 * DELTA_PIX, NUM_PIX))
        
        lens_model_list = ['SIE']
        lens_model = LensModel(lens_model_list)
        lens_samples = Table.read('./csst_catalog/csst_wf_single.csv')
        lens = lens_samples[idx]

        re_s = lens['re_s0']
        re_l = lens['re_l']
        
        # Parámetros de la lente
        lens_kwargs = [{
            'theta_E': self.thetaE,
            'e1': self.e1_l,                
            'e2': self.e2_l,               
            'center_x': 0.0,          
            'center_y': 0.0           
        }]
        
        # LUZ DE LA LENTE
        lens_light_model_list = ['SERSIC_ELLIPSE']
        lens_light_model = LightModel(lens_light_model_list)
        
        # Parámetros de la luz de la lente
        lens_light_kwargs = [{
            'amp': 8.,
            'R_sersic': re_l,
            'n_sersic': 4.0,
            'e1': self.e1_l,
            'e2': self.e2_l,
            'center_x': 0.0,
            'center_y': 0.0
        }]
        
        source_light_model_list = ['SERSIC_ELLIPSE']
        source_light_model = LightModel(source_light_model_list)
        
        # Parámetros del perfil Sersic para la fuente
        source_kwargs = [{
            'amp': 50.0,             
            'R_sersic': re_s,          
            'n_sersic': 2.0,          
            'e1': self.e1_s,               
            'e2': self.e2_s,               
            'center_x': self.x_s,          
            'center_y': self.y_s           
        }]
        
        # 1. Calcular la luz de la lente (directa, sin lenteado)
        lens_light_brightness = lens_light_model.surface_brightness(x, y, lens_light_kwargs)
        image_lens_light = lens_light_brightness.reshape(NUM_PIX, NUM_PIX)
        
        # 2. Calcular las posiciones de la fuente lentadas
        x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)
        
        # 3. Calcular el brillo de la fuente en las posiciones lentadas
        source_brightness_lensed = source_light_model.surface_brightness(x_lensed, y_lensed, source_kwargs)
        image_source_lensed = source_brightness_lensed.reshape(NUM_PIX, NUM_PIX)
        
        # 4. CALCULAR LA SUMA CORRECTA: Luz de lente + Fuente lentada
        self.image = image_lens_light + image_source_lensed
        
        # 5. Imagen de la fuente sin lente para comparación
        source_brightness_unlensed = source_light_model.surface_brightness(x, y, source_kwargs)
        image_source_unlensed = source_brightness_unlensed.reshape(NUM_PIX, NUM_PIX)
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
        c3 = fits.Card('e1', self.e1_s, 'source elipticity1')
        c4 = fits.Card('e2', self.e2_s, 'source elipticity2')

        # Lens Coordinates
        c5 = fits.Card('center_x', self.x_s, 'x coordinate')
        c6 = fits.Card('center_y', self.y_s, 'y coordinate')
        parameters = [c1, c2, c3, c4, c5, c6]

        for parameter in parameters:
            outhdr.append(parameter, end = True)
        
        outlist = fits.ImageHDU(data = outimage, header = outhdr) # .astype('float32')
        outlist.writeto(outfile, overwrite = True)
