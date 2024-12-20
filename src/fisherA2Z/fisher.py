import pandas as pd
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from math import fsum
from functools import lru_cache
from numpy.linalg import inv
import sys, scipy
import numdifftools as nd
from scipy.stats import uniform, norm
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from numbers import Number
from sklearn.neighbors import KernelDensity
from itertools import permutations, chain
import pickle
import fisherA2Z

package_path = fisherA2Z.__path__[0]
ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False

def FoM(matrix):
    """
    Calculate the Figure of Merit (FoM) for a given covariance matrix.

    Parameters:
    matrix (ndarray): Covariance matrix.

    Returns:
    float: Square root of the determinant of the matrix.
    """
    return np.sqrt(np.linalg.det(matrix))

def plot_contours(matrix, sigmas, fid, **kwargs):
    """
    Plot 1-sigma or 2-sigma confidence contours based on a covariance matrix.

    Parameters:
    matrix (ndarray): Covariance matrix.
    sigmas (int): Sigma level (1 or 2).
    fid (tuple): Fiducial values for the parameters.

    Returns:
    tuple: Ellipse object, x-limit, and y-limit for the plot.
    """
    prefactor = {1:1.52, 2:2.48}
    prefactor = prefactor[sigmas]
    matrix = np.linalg.inv(matrix)
    s00, s01, s11 = matrix[0][0], matrix[0][1], matrix[1][1]
    a = np.sqrt(
        0.5*(s00 + s11) + np.sqrt(s01**2 + 0.25*(s00-s11)**2)
    )
    b = np.sqrt(
        0.5*(s00 + s11) - np.sqrt(s01**2 + 0.25*(s00-s11)**2)
    )
    b *= prefactor
    a *= prefactor
    theta = np.arctan(2*s01/(s00-s11))/2
    eig = np.linalg.eig(matrix)
    maxeig = eig[1][np.argmax(eig[0])]
    theta = np.arctan2(maxeig[1], maxeig[0])
    el = matplotlib.patches.Ellipse(fid, 2*a, 2*b, angle=-np.degrees(theta), alpha=0.3, **kwargs)
    xlim = np.sqrt(a**2*np.cos(theta)**2 + b**2*np.sin(theta)**2)
    ylim = np.sqrt(a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2)
    return el, xlim, ylim

def marginalize(fisher_matrix, i, j):
    """
    Marginalize a Fisher matrix over two parameters.

    Parameters:
    fisher_matrix (ndarray): Full Fisher matrix.
    i (int): Index of the first parameter.
    j (int): Index of the second parameter.

    Returns:
    ndarray: Marginalized Fisher matrix for the two parameters.
    """
    return np.linalg.inv(np.linalg.inv(fisher_matrix)[np.ix_([i,j], [i,j])]) 

class PhotoZ_core(object):
    """
    Defines the core probability density function (PDF) p(zp | z) for photometric redshifts.
    """

    def __init__(self, zp_support, zbias, sigma_z):
        """
        Initialize the PhotoZ_core object.

        Parameters:
        zp_support (array): Support of the photometric redshifts (zp).
        zbias (float): Bias in the photometric redshift distribution.
        sigma_z (float): Standard deviation of the distribution.
        """
        self.zp_support = zp_support
        self.zbias = zbias
        self.sigma_z = sigma_z

    def get_pdf_given(self, z: Number):
        """
        Compute the PDF p(zp | z) for a given true redshift z.

        Parameters:
        z (Number): True redshift.

        Returns:
        ndarray: PDF values.
        """
        rv = norm()
        scale = self.sigma_z * (1 + z)
        loc = z - self.zbias
        return rv.pdf((np.array(self.zp_support) - loc) / scale) / scale

class Core(object):
    """
    Represents a combined core and outlier population model.
    """

    def __init__(self, zp_support, zbias, sigma_z):
        """
        Initialize the Core object.

        Parameters:
        zp_support (array): Support of the photometric redshifts (zp).
        zbias (float): Bias in the photometric redshift distribution.
        sigma_z (float): Standard deviation of the core distribution.
        """
        self.zp_support = zp_support
        self.core = PhotoZ_core(zp_support, zbias, sigma_z)

    def get_pdf_given(self, z: Number):
        """
        Compute the PDF for a combined population given a true redshift z.

        Parameters:
        z (Number): True redshift.

        Returns:
        ndarray: Normalized PDF values.
        """
        core_pdf = self.core.get_pdf_given(z)
        return core_pdf/np.trapz(core_pdf, self.zp_support)

class SmailZ(object):
    """
    Represents a true redshift distribution following a Smail-type profile.
    """

    def __init__(self, z_support, pdf_values_evaluated_at_zsupport):
        """
        Initialize the SmailZ object.

        Parameters:
        z_support (array): Support of the true redshifts (z).
        pdf_values_evaluated_at_zsupport (array): PDF values evaluated at the z-support.
        """
        self.z_support = z_support
        self.pdf = pdf_values_evaluated_at_zsupport/np.trapz(pdf_values_evaluated_at_zsupport, z_support)

    def get_pdf(self):
        """
        Get the normalized PDF.

        Returns:
        ndarray: PDF values.
        """
        return self.pdf

    def get_pdf_convoled(self, filter_list):
        """
        Convolve the PDF with a set of filter functions.

        Parameters:
        filter_list (list): List of filter functions.

        Returns:
        ndarray: Array of convolved PDFs.
        """
        output_tomo_list = np.array([el * self.pdf for el in filter_list]).T
        output_tomo_list = np.column_stack((self.z_support, output_tomo_list))
        return output_tomo_list

class PhotozModel(object):
    """
    Convolve a joint distribution p(z_s, z_p) with filters to produce tomographic bins.
    """

    def __init__(self, pdf_z, pdf_zphot_given_z, filters):
        """
        Initialize the PhotozModel object.

        Parameters:
        pdf_z (SmailZ): True redshift distribution.
        pdf_zphot_given_z (PhotoZ_core): Core distribution.
        filters (list): List of filter functions.
        """
        self.pdf_zphot_given_z = pdf_zphot_given_z
        self.pdf_z = pdf_z
        self.filter_list = filters

    def get_pdf(self):
        """
        Compute PDFs for all tomographic bins.

        Returns:
        ndarray: Array of tomographic bin PDFs.
        """
        return np.array([self.get_pdf_tomo(el) for el in self.filter_list])

    def get_pdf_tomo(self, filter):
        """
        Compute the PDF for a single tomographic bin.

        Parameters:
        filter (ndarray): Filter function.

        Returns:
        ndarray: PDF values for the tomographic bin.
        """
        z_support = self.pdf_z.z_support
        zp_support = self.pdf_zphot_given_z.zp_support
        z_pdf = self.pdf_z.get_pdf()
        pdf_joint = np.zeros((len(zp_support), len(z_support)))
        for i in range(len(z_support)):
            pdf_joint[:, i] = self.pdf_zphot_given_z.get_pdf_given(z_support[i]) * z_pdf[i] * filter[i]

        pdf_zp = np.zeros((len(zp_support),))
        for i in range(len(zp_support)):
            pdf_zp[i] = np.trapz(pdf_joint[i, :], z_support)

        return pdf_zp

def centroid_shift(unbiased, biased):
    cl_unbiased = np.array(unbiased.ccl_cls['C_ell']).reshape(15, 15).T 
    cl_biased = np.array(biased.ccl_cls['C_ell']).reshape(15, 15).T 
    
    diff_cl = cl_biased - cl_unbiased
    bias_vec = []
    for i, param in enumerate(unbiased.param_order):
        bias_vec.append(sum(diff_cl[idx].dot(
                np.array(unbiased.invcov_list[idx]).dot(unbiased.derivs_sig[param][idx])
                ) for idx in range(len(unbiased.ell))))
    bias_vec = np.array(bias_vec)
    para_bias = np.linalg.inv(fisher).dot(bias_vec) 
    para_bias = {param_order[i]: para_bias[i] for i in range(7)}
    return para_bias


class Fisher:
    def __init__(self, 
                 cosmo, 
                 zvariance=(0.05, 0.05, 0.05, 0.05, 0.05), 
                 zbias=(0,0,0,0,0), 
                 outliers=(0.15, 0.15, 0.15, 0.15, 0.15), 
                 end=None, 
                 probe='3x2pt', 
                 save_deriv = None, overwrite=True, 
                 y1 = False, 
                 step = 0.01,
                 gbias = [1.376695, 1.451179, 1.528404, 1.607983, 1.689579, 1.772899, 1.857700, 1.943754, 2.030887, 2.118943], 
                 ia_params = {'A0': 5.92,'etal':-0.47,'etah': 0.0,'beta': 1.1},
                ):
        """
        Initializes the parameter object for cosmological analysis.

        Parameters:
        -----------
        cosmo : object
            The cosmology object providing initial parameters (e.g., sigma8, Omega_b).
        zvariance : tuple of float, optional
            Initial variances for redshift bins, default is (0.05, 0.05, 0.05, 0.05, 0.05).
        zbias : tuple of float, optional
            Initial biases for redshift bins, default is (0, 0, 0, 0, 0).
        outliers : tuple of float, optional
            Initial outlier fractions for redshift bins, default is (0.15, 0.15, 0.15, 0.15, 0.15).
        end : int, optional
            Number of parameters to consider; defaults to the total number of parameters.
        probe : str, optional
            Type of probe for analysis, default is '3x2pt'.
        save_deriv : str or None, optional
            Path to save derivative outputs, default is None.
        overwrite : bool, optional
            Whether to overwrite existing files, default is True.
        y1 : bool, optional
            Flag for using Year 1 data distributions, default is False.
        step : float, optional
            Step size for numerical derivatives, default is 0.01.
        gbias : list of float, optional
            Galaxy bias values for the redshift bins, default includes 10 values.
        ia_params : dict, optional
            Parameters for intrinsic alignment, with keys:
                - 'A0' (amplitude), 
                - 'etal' (low-z dependence),
                - 'etah' (high-z dependence),
                - 'beta' (scaling exponent).

        Attributes:
        -----------
        save_deriv : str or None
            Path for saving derivatives.
        overwrite : bool
            Overwrite existing files flag.
        zvariance : list of float
            Redshift variances adjusted for redshift scaling.
        zbias : list of float
            Redshift biases adjusted for redshift scaling.
        outliers : list of float
            Outlier fractions for redshift bins.
        gbias : list of float
            Galaxy bias values for each redshift bin.
        IA_interp : object
            Precomputed intrinsic alignment interpolation.
        zmid : numpy.ndarray
            Midpoints of redshift bins from data file.
        dneff : pandas.Series
            Effective number density per redshift bin.
        z : list of float
            Redshift midpoints including a 0 bin.
        funcs_ss, funcs_sp, funcs_pp : dict
            Functions for single, shear-shear, and position-position covariance.
        vals : dict
            Initial parameter values.
        priors : dict
            Parameter priors for Gaussian likelihoods.
        param_order : list of str
            Order of parameters for analysis.
        param_labels : list of str
            Latex-formatted labels for parameters.
        probe : str
            Type of probe used for cosmological analysis.
        end : int
            Total number of parameters.

        Notes:
        ------
        This initialization processes input parameters and prepares 
        required mappings for the likelihood analysis, including functions 
        for derivative computation and pre-loading relevant data files.
        """
    
        self.save_deriv = save_deriv
        self.overwrite = overwrite
        self.zvariance = list(zvariance)
        self.zbias = list(zbias)
        self.y1 = y1
        for i, z in enumerate([0.3, 0.65, 0.95, 1.35, 2.75]):
            self.zbias[i] *= (1+z)
            self.zvariance[i] *= (1+z)
        self.outliers = list(outliers)
        self.has_run = False
        self.intCache = {}
        self.gbias = gbias
        self.IA_interp = pickle.load(open(package_path + '/data/IA_interp.p', 'rb'))
        self.cosmo = cosmo
        self.step = step
        if self.y1:
            df = pd.read_csv(package_path + '/data/nzdist_y1.txt', sep = ' ')
        else:
            df = pd.read_csv(package_path + '/data/nzdist.txt', sep=' ') 
        self.zmid = np.array(list(df['zmid']))
        self.dneff = df['dneff']
        self.z = [0] + self.zmid
        self.funcs_ss = {
            'sigma_8': self.getC_ellOfSigma8_ss,
            'omega_b': self.getC_ellOfOmegab_ss,
            'h': self.getC_ellOfh_ss,
            'n_s': self.getC_ellOfn_s_ss,
            'omega_m': self.getC_ellOfOmegam_ss,
            'w_0': self.getC_ellOfw0_ss,
            'w_a': self.getC_ellOfwa_ss,
            'zbias1': self.getC_ellOfzbias1_ss,
            'zbias2': self.getC_ellOfzbias2_ss,
            'zbias3': self.getC_ellOfzbias3_ss,
            'zbias4': self.getC_ellOfzbias4_ss,
            'zbias5': self.getC_ellOfzbias5_ss,
            'zvariance1': self.getC_ellOfzvariance1_ss,
            'zvariance2': self.getC_ellOfzvariance2_ss,
            'zvariance3': self.getC_ellOfzvariance3_ss,
            'zvariance4': self.getC_ellOfzvariance4_ss,
            'zvariance5': self.getC_ellOfzvariance5_ss,
            'zoutlier1': self.getC_ellOfzoutlier1_ss,
            'zoutlier2': self.getC_ellOfzoutlier2_ss,
            'zoutlier3': self.getC_ellOfzoutlier3_ss,
            'zoutlier4': self.getC_ellOfzoutlier4_ss,
            'zoutlier5': self.getC_ellOfzoutlier5_ss,
            'A0': self.getC_ellOfIA_amp_ss,
            'beta': self.getC_ellOfIA_beta_ss,
            'etal': self.getC_ellOfIA_lowz_ss,
            'etah': self.getC_ellOfIA_highz_ss
        }
        self.funcs_sp = {
            'sigma_8': self.getC_ellOfSigma8_sp,
            'omega_b': self.getC_ellOfOmegab_sp,
            'h': self.getC_ellOfh_sp,
            'n_s': self.getC_ellOfn_s_sp,
            'omega_m': self.getC_ellOfOmegam_sp,
            'w_0': self.getC_ellOfw0_sp,
            'w_a': self.getC_ellOfwa_sp,
            'zbias1': self.getC_ellOfzbias1_sp,
            'zbias2': self.getC_ellOfzbias2_sp,
            'zbias3': self.getC_ellOfzbias3_sp,
            'zbias4': self.getC_ellOfzbias4_sp,
            'zbias5': self.getC_ellOfzbias5_sp,
            'zvariance1': self.getC_ellOfzvariance1_sp,
            'zvariance2': self.getC_ellOfzvariance2_sp,
            'zvariance3': self.getC_ellOfzvariance3_sp,
            'zvariance4': self.getC_ellOfzvariance4_sp,
            'zvariance5': self.getC_ellOfzvariance5_sp,
            'zoutlier1': self.getC_ellOfzoutlier1_sp,
            'zoutlier2': self.getC_ellOfzoutlier2_sp,
            'zoutlier3': self.getC_ellOfzoutlier3_sp,
            'zoutlier4': self.getC_ellOfzoutlier4_sp,
            'zoutlier5': self.getC_ellOfzoutlier5_sp,
            'A0': self.getC_ellOfIA_amp_sp,
            'beta': self.getC_ellOfIA_beta_sp,
            'etal': self.getC_ellOfIA_lowz_sp,
            'etah': self.getC_ellOfIA_highz_sp,
        }
        for i in range(10):
            self.funcs_sp[f'gbias{i+1}'] = self.getC_ellOfgbias_sp_helper
            
        self.funcs_pp = {
            'sigma_8': self.getC_ellOfSigma8_pp,
            'omega_b': self.getC_ellOfOmegab_pp,
            'h': self.getC_ellOfh_pp,
            'n_s': self.getC_ellOfn_s_pp,
            'omega_m': self.getC_ellOfOmegam_pp,
            'w_0': self.getC_ellOfw0_pp,
            'w_a': self.getC_ellOfwa_pp,
        }
        for i in range(10):
            self.funcs_pp[f'gbias{i+1}'] = self.getC_ellOfgbias_pp_helper

        self.vals = {
            'sigma_8': cosmo._params_init_kwargs['sigma8'], 
            'omega_b': cosmo._params_init_kwargs['Omega_b'], 
            'h': cosmo._params_init_kwargs['h'], 
            'n_s': cosmo._params_init_kwargs['n_s'], 
            'omega_m': cosmo._params_init_kwargs['Omega_b']+cosmo._params_init_kwargs['Omega_c'],
            'w_0': cosmo._params_init_kwargs['w0'],
            'w_a': cosmo._params_init_kwargs['wa'],
            'A0': ia_params['A0'],
            'etal':ia_params['etal'],
            'etah':ia_params['etah'],
            'beta':ia_params['beta']
        }
        for i in range(10):
            self.vals[f'gbias{i+1}'] = self.gbias[i]
        for i in range(5):
            self.vals['zbias'+str(i+1)] = self.zbias[i]
            self.vals['zvariance'+str(i+1)] = self.zvariance[i]
            self.vals['zoutlier'+str(i+1)] = self.outliers[i]
        self.priors = {
            'sigma_8': 1/0.2**2, 
            'omega_b': 1/0.003**2, 
            'h': 1/0.125**2,
            'n_s': 1/0.2**2, 
            'omega_m': 1/0.15**2,
            'w_0': 1/0.8**2,
            'w_a': 1/1.3**2,
            'zbias1': 1/0.1**2,
            'zbias2': 1/0.1**2,
            'zbias3': 1/0.1**2,
            'zbias4': 1/0.1**2,
            'zbias5': 1/0.1**2,
            'zvariance1': 1/0.1**2,
            'zvariance2': 1/0.1**2,
            'zvariance3': 1/0.1**2,
            'zvariance4': 1/0.1**2,
            'zvariance5': 1/0.1**2,
            'zoutlier1': 1/0.1**2,
            'zoutlier2': 1/0.1**2,
            'zoutlier3': 1/0.1**2,
            'zoutlier4': 1/0.1**2,
            'zoutlier5': 1/0.1**2,
            'A0': 1/2.5**2,
            'beta': 1/1.0**2,
            'etal': 1/1.5**2,
            'etah': 1/0.5**2,
        }
        for i in range(10):
            string = 'gbias'+str(i+1)
            self.priors[string] = 1/0.9**2
        self.param_order = ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h', 'A0', 'beta', 'etal', 'etah'] + ['zbias'+str(i) for i in range(1, 6)] + ['zvariance'+str(i) for i in range(1,6)] + ['zoutlier'+str(i) for i in range(1,6)] + ['gbias'+str(i) for i in range(1,11)]
        self.param_labels = [r'$\Omega_m$', r'$\sigma_8$', r'$n_s$', r'$w_0$', r'$w_a$', r'$\Omega_b$', r'$h$', r'$A_0$', r'$\beta$', r'$\eta_l$', r'$\eta_h$'] + [r'$z_{bias}$'+str(i) for i in range(1, 6)] + [r'$\std{z}$'+str(i) for i in range(1,6)] + [r'$out$'+str(i) for i in range(1,6)] + [rf'$b_g^{i}$' for i in range(1,11)]
        if end:
            self.end = end
        else:
            self.end = len(self.vals)
        self.probe = probe
        
    def __repr__(self):
        return f'Run status: {self.has_run} \
                with cosmology: {self.cosmo} \
                with Photo-z error model:  \
                - bias:  {self.zbias} \
                - variance {self.zvariance} \
                - outliers: {self.outliers}'
        
    def _makeLensPZ(self):
        """
        Creates lens photometric redshift (PZ) distributions for specific redshift bins
        and assigns galaxy bias values to each bin.

        Parameters:
            None

        Attributes Created or Updated:
            - self.gbias_dict (dict): Maps the centers of the redshift bins to galaxy bias values.
            - self.pdf_z (SmailZ): SmailZ object representing the redshift distribution of galaxies.
            - self.dNdz_dict_lens (dict): Maps the centers of redshift bins to their normalized redshift distributions (P(z)).

        Process:
            1. Divides the redshift range (0.2–1.2) into 10 bins.
            2. Computes the photometric redshift distribution for each bin using a uniform filter and a Smail distribution.
            3. Associates the corresponding galaxy bias (`self.gbias`) with each bin.
        """
        print("Making lens pz")
        bins = np.linspace(0.2, 1.2, 11)
        self.gbias_dict = {}
        bin_centers = [.5*fsum([bins[i]+bins[i+1]]) for i in range(len(bins[:-1]))]
        self.pdf_z = SmailZ(self.zmid, np.array(self.dneff))
        dNdz_dict_lens = {}
        qs = []
        for index, (x,x2) in enumerate(zip(bins[:-1], bins[1:])):
            core = Core(self.zmid, zbias=0, sigma_z=0.03)
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2-x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_lens[bin_centers[index]] = photoz_model.get_pdf()[0]
            self.gbias_dict[bin_centers[index]] = self.gbias[index]
        self.dNdz_dict_lens = dNdz_dict_lens
        
        
    def _NormalizePZ(self, qs, dNdz_dict_source, m=1):
        """
        Normalizes source photometric redshift (PZ) distributions such that the integral 
        over the redshift range (0–4) is unity for each distribution.

        Parameters:
            qs (list or array): Scaling factors for normalizing the distributions.
            dNdz_dict_source (dict): A dictionary where keys represent bin centers and values 
                are redshift distributions (P(z)) for the source galaxies.
            m (float, optional): Additional scaling factor applied uniformly across all 
                distributions (default: 1).

        Returns:
            dict: The updated dictionary with normalized redshift distributions.

        Process:
            1. Scales each redshift distribution by the total sum of the scaling factors (`qs`).
            2. Fits a cubic spline to interpolate each distribution.
            3. Ensures that the integral of each redshift distribution over the range (0–4) equals 1 after normalization.
        """
        for q, k in zip(qs, dNdz_dict_source.keys()):
            dNdz_dict_source[k] = dNdz_dict_source[k]*sum(qs)/q
            f = CubicSpline(self.zmid, dNdz_dict_source[k])
            d = quad(f, 0, 4)[0]
            for k in dNdz_dict_source.keys():
                dNdz_dict_source[k] /= (d*m)
            return dNdz_dict_source
        

    def _makeSourcePZ(self,  implement_outliers=True):
        """
        Generates source photometric redshift (PZ) distributions and accounts for outliers 
        in redshift estimation if enabled.

        Parameters:
            implement_outliers (bool, optional): Determines whether to incorporate outlier 
                distributions into the source PZ (default: True).

        Attributes Created or Updated:
            - self.bins (list): Redshift bin edges for the source galaxies.
            - self.bin_centers (list): Centers of the redshift bins.
            - self.pdf_z (SmailZ): SmailZ object representing the redshift distribution of galaxies.
            - self.dNdz_dict_source (dict): Maps bin centers to their normalized source 
                redshift distributions (P(z)).
            - self.KDEs (list): Preloaded Kernel Density Estimates (KDEs) used to model outliers.
            - self.scores (dict): KDE-derived scores for the redshift range, used to compute 
                outlier distributions.

        Process:
            1. Creates bins and computes their centers based on the redshift midpoints (`self.zmid`).
            2. Calculates the photometric redshift distribution for each bin using a Smail distribution 
               and Core model for photometric errors.
            3. If `implement_outliers` is True, combines the core redshift distribution with an outlier 
               distribution weighted by the outlier fraction for each bin.
            4. Normalizes both the core and outlier distributions to ensure their integrals over the range (0–4) equal 1.
            5. Updates `self.dNdz_dict_source` with the combined (core + outliers) PZ for each bin.

        Notes:
            - The method uses preloaded KDEs stored in a file (`KDEs.p`) to estimate outlier distributions.
            - Core redshift distributions are modeled using uniform filters and the `PhotozModel` class.
        """
        print("Making source pz")
        n = len(self.zmid)
        datapts = ([list(np.ones(int(self.dneff[i]/min(self.dneff)))*self.zmid[i]) for i in range(n)])
        datapts = list(chain.from_iterable(datapts)) # flatten
        bins = datapts[0::int(len(datapts)/5)]
        self.bins = bins
        bin_centers = [.5*fsum([bins[i]+bins[i+1]]) for i in range(len(bins[:-1]))]
        self.bin_centers = bin_centers
        self.pdf_z = SmailZ(self.zmid, np.array(self.dneff))
        self.dNdz_dict_source = {}
        
        self.KDEs = pickle.load(open(package_path + '/data/KDEs.p', 'rb'))
        self.scores = {}
        for index, (x,x2) in enumerate(zip(bins[:-1], bins[1:])):
            bias = self.zbias[index]
            variance = self.zvariance[index]
            outlier = self.outliers[index]
            
            core = Core(self.zmid, zbias=bias, sigma_z=variance)
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2-x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dndz_core = photoz_model.get_pdf()[0]
            #normalize
            fi = CubicSpline(self.zmid, dndz_core)
            dndz_core_norm = dndz_core/quad(fi, 0, 4)[0]
            
            kde = self.KDEs[index]
            self.scores[index] = np.exp(kde.score_samples(np.array(self.zmid).reshape(-1, 1)))
            dndz_outliers = self.scores[index]
            #normalize
            fo = CubicSpline(self.zmid, dndz_core)
            dndz_outliers_norm = dndz_outliers/ quad(fo, 0, 4)[0]
            
            self.dNdz_dict_source[bin_centers[index]] = dndz_core_norm * (1-outlier) + dndz_outliers_norm * outlier

           
    def setPriors(self, priors):
        """
        Sets the inverse square of the provided priors for use in the object's calculations.

        Parameters:
            priors (dict): A dictionary where the keys are parameter names and the values 
                are the prior standard deviations (assumed to be positive).

        Updates:
            - self.priors (float): The computed inverse square of the prior standard deviations 
                for all provided parameters.

        Notes:
            - This implementation overwrites `self.priors` with the computed value for 
              the last key in the input dictionary, which may not align with expected behavior 
              if multiple priors are provided.
        """
        for key in priors.keys():
            self.priors = 1/(priors[key]**2)
                
    def getElls(self, file= package_path + '/data/ell-values.txt'):
        """
        Loads the multipole moment values (ℓ) from a specified file and stores them as a list.

        Parameters:
            file (str): The path to the file containing ell values. Defaults to 
                `package_path + '/data/ell-values.txt'`.

        Updates:
            - self.ell (list): A list of ℓ values loaded from the specified file.

        Notes:
            - The file should be a plain text file with one ℓ value per line.
        """
        ell = pd.read_csv(file, names=['ell'])
        self.ell = list(ell.to_dict()['ell'].values())
        
    def A_h(self, z, etah):
        """
        Computes the amplitude scaling factor \( A_h \) for a given redshift \( z \) 
        and scaling parameter \( \eta_h \).

        Parameters:
            z (float): The redshift value.
            etah (float): The scaling parameter for \( A_h \).

        Returns:
            float: The scaling factor \( A_h \). If \( z > 0.75 \), the factor is 
                   computed as \(((1+z)/(1+0.75))^{\eta_h}\); otherwise, it returns 1.
        """
        if z>0.75:
            return ((1+z)/(1+0.75))**etah
        return 1

        
    def A_l(self, z, etal):
        """
        Computes the amplitude scaling factor \( A_l \) for a given redshift \( z \) 
        and scaling parameter \( \eta_l \).

        Parameters:
            z (float): The redshift value.
            etal (float): The scaling parameter for \( A_l \).

        Returns:
            float: The scaling factor \( A_l \), computed as 
                   \(((1+z)/(1+0.62))^{\eta_l}\).
        """
        return ((1+z)/(1+0.62))**etal
        
    
    def makeShearShearCells(self, cosmo=None):
        """
        Computes the shear-shear angular power spectrum \( C_\ell \) for different redshift bins 
        using the cosmic shear weak lensing formalism.

        Parameters:
            cosmo (CCLCosmology, optional): The cosmology object to be used in the calculation. 
                If not provided, the default cosmology from the object instance (`self.cosmo`) is used.

        Returns:
            list: A list of lists, where each sublist contains the computed \( C_\ell \) values 
                  for a specific redshift bin.

        Updates:
            - self.shearshear_cls (pd.DataFrame): A DataFrame containing the computed shear-shear 
                                                  angular power spectra for each redshift bin and multipole.

        Notes:
            - This method computes the angular power spectra \( C_\ell \) for shear-shear correlation 
              between redshift bins, accounting for intrinsic alignment (IA) biases in the weak lensing tracers.
            - The `self.dNdz_dict_source` should contain the redshift distribution for the sources in each bin.
            - The `self.vals` dictionary should contain the necessary parameters, such as `A0`, `etal`, and `etah`.
        """
        if not cosmo:
            cosmo = self.cosmo
        ccl_cls = pd.DataFrame()
        zbin = 0
        j = 0
        ia0 = self.vals['A0'] * np.array([self.A_l(zi, self.vals['etal']) for zi in self.zmid]) * np.array([self.A_h(zi, self.vals['etah']) for zi in self.zmid])
        lst = list(self.dNdz_dict_source.keys())
        for i, key in enumerate(lst):
            ia = self.getAi(self.vals['beta'], cosmo, dNdz=tuple(self.dNdz_dict_source[key])) * ia0
            lens1 = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_source[key]), ia_bias=(self.zmid, ia))
            for keyj in lst[i:]:
                ia = self.getAi(self.vals['beta'], cosmo, dNdz=tuple(self.dNdz_dict_source[keyj])) * ia0
                lens2 = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_source[keyj]), ia_bias=(self.zmid, ia))
                cls = ccl.angular_cl(cosmo, lens1, lens2, self.ell)
                newdf = pd.DataFrame({'zbin': [int(k) for k in j*np.ones(len(cls))],
                                      'ell': self.ell,
                                      'C_ell': cls})
                ccl_cls = pd.concat((ccl_cls, newdf))
                j += 1


        self.shearshear_cls = ccl_cls.reset_index()
        
        C_ells = []
        for i in set(ccl_cls['zbin']):
            C_ells.append(list(ccl_cls[ccl_cls['zbin']==i]['C_ell']))
        return C_ells
    
    def makePosPosCells(self, cosmo=None):
        """
        Computes the position-position angular power spectrum \( C_\ell \) for different redshift bins 
        using the number counts of galaxies or objects in the lens sample.

        Parameters:
            cosmo (CCLCosmology, optional): The cosmology object to be used in the calculation. 
                If not provided, the default cosmology from the object instance (`self.cosmo`) is used.

        Returns:
            list: A list of lists, where each sublist contains the computed \( C_\ell \) values 
                  for a specific redshift bin.

        Updates:
            - self.pospos_cls (pd.DataFrame): A DataFrame containing the computed position-position 
                                               angular power spectra for each redshift bin and multipole.

        Notes:
            - This method computes the angular power spectra \( C_\ell \) for position-position correlation 
              between redshift bins, using the lens sample's number counts.
            - The `self.dNdz_dict_lens` should contain the redshift distribution for the lenses in each bin.
            - The `self.gbias` should contain the galaxy bias values for each redshift bin.
        """
        if not cosmo:
            cosmo = self.cosmo
        ccl_cls = pd.DataFrame()
        zbin = 0
        j = 0
        lst = list(self.dNdz_dict_lens.keys())
        for i, key in enumerate(lst):
            pos1 = ccl.NumberCountsTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_lens[key]), has_rsd=False, bias=(self.zmid, self.gbias[i]*np.ones_like(self.zmid)))
            cls = ccl.angular_cl(cosmo, pos1, pos1, self.ell)
            newdf = pd.DataFrame({'zbin': [int(k) for k in i*np.ones(len(cls))],
                                  'ell': self.ell,
                                  'C_ell': cls})
            ccl_cls = pd.concat((ccl_cls, newdf))


        self.pospos_cls = ccl_cls.reset_index()
        
        C_ells = []
        for i in set(ccl_cls['zbin']):
            C_ells.append(list(ccl_cls[ccl_cls['zbin']==i]['C_ell']))
        return C_ells


    def makePosShearCells(self, cosmo=None):
        """
        Computes the position-shear angular power spectrum \( C_\ell \) for different redshift bins 
        between lens and source populations.

        This method calculates the angular power spectra between position (lens) and shear (source) 
        tracers for different redshift bins using the appropriate number density and bias values.

        Parameters:
            cosmo (CCLCosmology, optional): The cosmology object to be used in the calculation. 
                If not provided, the default cosmology from the object instance (`self.cosmo`) is used.

        Returns:
            list: A list of lists, where each sublist contains the computed \( C_\ell \) values 
                  for a specific redshift bin.

        Updates:
            - self.posshear_cls (pd.DataFrame): A DataFrame containing the computed position-shear 
                                                 angular power spectra for each redshift bin and multipole.

        Notes:
            - The `self.dNdz_dict_lens` should contain the redshift distribution for the lens (position) sample.
            - The `self.dNdz_dict_source` should contain the redshift distribution for the source (shear) sample.
            - The `self.gbias` should contain the galaxy bias values for each redshift bin of the lens sample.
            - The `self.accept` set defines the allowed combinations of lens and source redshift bins for which 
              the computation of \( C_\ell \) is performed.
            - The method applies the intrinsic alignment (IA) model for both the lens and source tracers.
        """
        if not cosmo:
            cosmo = self.cosmo
        ccl_cls = pd.DataFrame()
        zbin = 0
        j = 0
        llst = list(self.dNdz_dict_lens.keys())
        slst = list(self.dNdz_dict_source.keys())
        self.accept = {(0,1), (0,2), (0,3), (0,4),
                      (1,1), (1,2), (1,3), (1,4),
                      (2,2), (2,3), (2,4),
                      (3,2), (3,3), (3,4),
                      (4,2), (4,3), (4,4),
                      (5,3), (5,4),
                      (6,3), (6,4),
                      (7,3), (7,4),
                      (8,4), 
                      (9,4)
                      }
        for l, key in enumerate(llst):
            pos = ccl.NumberCountsTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_lens[key]), has_rsd=False, bias=(self.zmid, self.gbias[l]*np.ones_like(self.zmid)))
            for s, keyj in enumerate(slst):
                if (l, s) in self.accept:
                    ia0 = self.vals['A0'] * np.array([self.A_l(zi, self.vals['etal']) for zi in self.zmid]) * np.array([self.A_h(zi, self.vals['etah']) for zi in self.zmid])
                    ia = self.getAi(self.vals['beta'], cosmo, dNdz=self.dNdz_dict_source[keyj]) * ia0
                    shear = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_source[keyj]), ia_bias=(self.zmid, ia))
                    cls = ccl.angular_cl(cosmo, pos, shear, self.ell)                

                    newdf = pd.DataFrame({'zbin': [int(o) for o in j*np.ones(len(cls))],
                                          'ell': self.ell,
                                          'C_ell': cls})
                    ccl_cls = pd.concat((ccl_cls, newdf))
                    j += 1


        self.posshear_cls = ccl_cls.reset_index()
        
        C_ells = []
        for i in set(ccl_cls['zbin']):
            C_ells.append(list(ccl_cls[ccl_cls['zbin']==i]['C_ell']))
        return C_ells

    
    def buildCovMatrix(self):
        """
        Builds and loads the inverse covariance matrix based on the selected probe type.

        Depending on the selected probe type (`3x2pt`, `ss`, `sl`, `ll`, or `2x2pt`), 
        this method loads the corresponding inverse covariance matrix from pre-specified 
        data files. It processes the data to reshape it into a square matrix and stores 
        it in the `self.invcov` attribute for later use in the analysis.

        The method reads data from the following sources:
            - `Y10_3x2pt_inv.txt` for the '3x2pt' probe
            - `Y10_shear_shear_inv.txt` or `Y1_shear_shear_inv.txt` for the 'ss' (shear-shear) probe
            - `Y10_shear_pos_inv.txt` for the 'sl' (shear-position) probe
            - `Y10_pos_pos_inv.txt` for the 'll' (position-position) probe
            - `Y10_2x2_inv.txt` for the '2x2pt' probe

        If the `probe` is 'sl' or '2x2pt', the method immediately loads and stores the covariance 
        matrix without reshaping it, and no further processing is done.

        Raises:
            ValueError: If the `probe` type is not one of the expected values ('3x2pt', 'ss', 'sl', 'll', or '2x2pt').

        Updates:
            - `self.invcov` (numpy.ndarray): The reshaped inverse covariance matrix for the selected probe type.

        Notes:
            - The `mat_len` variable determines the size of the inverse covariance matrix, 
              which is different for each probe type.
        """
        print('Getting covariance matrix')
        if self.probe == '3x2pt':
            invcov_SRD = pd.read_csv(package_path + '/data/Y10_3x2pt_inv.txt', 
                                     names=['a','b'], delimiter=' ')
            mat_len = 1000
        elif self.probe == 'ss':
            if self.y1 == False:
                invcov_SRD = pd.read_csv(package_path + '/data/Y10_shear_shear_inv.txt', 
                                         names=['a','b'], delimiter=' ')
            else:
                invcov_SRD = pd.read_csv(package_path + '/data/Y1_shear_shear_inv.txt',
                                        names = ['a', 'b'], delimiter = ' ')
            mat_len = 300
        elif self.probe == 'sl':
            self.invcov = np.loadtxt(package_path + '/data/Y10_shear_pos_inv.txt')
            mat_len = 500
            return
        elif self.probe == 'll':
            invcov_SRD = pd.read_csv(package_path + '/data/Y10_pos_pos_inv.txt', 
                                     names=['a','b'], delimiter=' ')
            mat_len = 200
        elif self.probe == '2x2pt':
            self.invcov = np.loadtxt(package_path + '/data/Y10_2x2_inv.txt')
            return
        
        else:
            raise ValueError('Unkown Probe')
            
            
        self.invcov = np.array(invcov_SRD['b']).reshape(mat_len, mat_len)
        
    def getDerivs(self, param=None):
        """
        Computes the derivatives of the C_ell values with respect to a given parameter.

        This method calculates the derivatives of the C_ell values for different probes (e.g., 'ss', 'sl', 'll', '3x2pt', '2x2pt') 
        with respect to a specific parameter. The derivatives are computed using finite differences, and the results are stored in 
        the `self.derivs_sig` dictionary.

        The method can compute derivatives for a specific parameter if provided, or it can compute the derivatives for all parameters 
        in `self.param_order`. The derivatives are calculated for various types of correlation functions depending on the selected probe.

        Parameters:
            param (str, optional): The specific parameter for which to compute the derivatives. If not provided, derivatives 
                                   are computed for all parameters up to `self.end`.

        Updates:
            - `self.derivs_sig` (dict): A dictionary containing the derivatives of the C_ell values with respect to the specified 
                                         parameter(s). The keys are the parameter names, and the values are the computed derivatives.

        Raises:
            ValueError: If the probe type or parameter is not valid for the specified calculation.

        Notes:
            - The function uses the `nd.Derivative` class to calculate the numerical derivatives.
            - The probe types considered in this function are 'ss', 'sl', 'll', '3x2pt', and '2x2pt'.
            - The method calculates derivatives for shear-shear, shear-position, and position-position correlation functions depending 
              on the probe type.
        """
        print(f'Getting derivatives, number of parameters: {self.end}')
        if not param or not self.has_run:
            self.derivs_sig = {}
            
        if not param:
            params = self.param_order[:self.end]
        else:
            params = [param]
        for var in params:
            print("Getting derivatives of C_ell w.r.t.: ", var)
            zbin = 0
            j = 0
            to_concat = []
            slst = list(self.dNdz_dict_source.keys())
            llst = list(self.dNdz_dict_lens.keys())
            if self.probe in ['ss', '3x2pt']:
                if var not in self.funcs_ss.keys():
                    derivs1 = list(np.zeros_like(self.ShearShearFid))
                else:
                    derivs1 = []
                    for i, key in enumerate(slst):    
                        for keyj in slst[i:]:
                            self.key = key
                            self.keyj = keyj
                            f = nd.Derivative(self.funcs_ss[var], full_output=True, step=self.step)
                            val, info = f(self.vals[var])
                            derivs1.append(val)
                    derivs1 = np.array(derivs1)
                to_concat.append(derivs1)
            
            if self.probe in ['sl', '3x2pt', '2x2pt']:
                if var not in self.funcs_sp.keys():
                    derivs2 = list(np.zeros_like(self.PosShearFid))
                else:
                    derivs2 = []
                    for l, keyl in enumerate(llst):
                        for s, keys in enumerate(slst):
                            if (l, s) in self.accept:
                                self.keyl = keyl
                                self.keys = keys 
                                f = nd.Derivative(self.funcs_sp[var], full_output=True, step=self.step)
                                val, info = f(self.vals[var])
                                derivs2.append(val)
                    derivs2 = np.array(derivs2) 
                to_concat.append(derivs2)
            if self.probe in ['ll', '3x2pt', '2x2pt']:
                if var not in self.funcs_pp.keys():
                    derivs3 = list(np.zeros_like(self.PosPosFid))
                else:
                    derivs3 = []
                    for i, key in enumerate(llst):
                        self.key = key
                        f = nd.Derivative(self.funcs_pp[var], full_output=True, step=self.step)
                        val, info = f(self.vals[var])
                        derivs3.append(val)
                    derivs3 = np.array(derivs3)
                to_concat.append(derivs3)
            self.derivs_sig[var] = np.vstack(tuple(to_concat))
            
    def getFisher(self):
        """
        Computes the Fisher matrix for parameter estimation.

        This method calculates the Fisher matrix, which is used to quantify the information content of the data 
        with respect to a set of parameters. The Fisher matrix is computed by evaluating the derivatives of the 
        C_ell values (stored in `self.derivs_sig`) with respect to each parameter, using the inverse covariance matrix 
        (`self.invcov`) and adding the contributions from prior information.

        The matrix elements are calculated as the dot product of the derivatives with respect to two parameters, 
        weighted by the inverse covariance matrix. The diagonal elements are adjusted by adding prior information 
        (stored in `self.priors`).

        Returns:
            np.ndarray: The Fisher matrix, a square matrix where each element represents the information content 
                        for pairs of parameters.

        Notes:
            - The Fisher matrix is used to estimate the precision with which parameters can be determined from the data.
            - The prior values are added to the diagonal elements of the Fisher matrix.
            - The function assumes that `self.derivs_sig`, `self.invcov`, and `self.priors` are already set.
        """
        print('Building fisher matrix')
        fisher = np.zeros((self.end, self.end))
        derivs = deepcopy(self.derivs_sig)
        for i, var1 in enumerate(self.param_order[:self.end]):
            for j, var2 in enumerate(self.param_order[:self.end]):
                fisher[i][j] = derivs[var1].reshape(-1) @ self.invcov @ derivs[var2].reshape(-1)
                # fisher[i][j] = derivs[var1].flatten().T @ self.invcov @ derivs[var2].flatten()
        for i in range(self.end):
            fisher[i][i] += self.priors[self.param_order[i]]
        return fisher
    
    def makeFidCells(self):
        """
        Constructs fiducial C_ell values for different probes.

        This method computes the fiducial C_ell values for shear-shear, shear-position, and position-position 
        correlation functions depending on the selected probe. It uses the respective functions (`makeShearShearCells`, 
        `makePosShearCells`, and `makePosPosCells`) to generate the C_ell values and stores them in the 
        `self.ShearShearFid`, `self.PosShearFid`, and `self.PosPosFid` attributes.

        The fiducial C_ell values are then combined into a single array (`self.ccl_cls`) that contains all the 
        correlation functions for the given probe.

        Notes:
            - The method only computes the C_ell values for the probes that are selected in the `self.probe` attribute.
            - The probes considered are: '3x2pt', 'ss', 'sl', 'll', and '2x2pt'.
            - The fiducial C_ell values are used as a baseline for calculating the derivatives and Fisher matrix.

        Returns:
            np.ndarray: The stacked C_ell values for all the relevant probes.
        """
        print("Making fiducial c_ells")
        all_cls = []
        if self.probe in ['3x2pt', 'ss']:
            self.ShearShearFid = self.makeShearShearCells()
            all_cls.append(self.ShearShearFid)
        if self.probe in ['3x2pt', 'sl', '2x2pt']:
            self.PosShearFid = self.makePosShearCells()
            all_cls.append(self.PosShearFid)
        if self.probe in ['3x2pt', 'll', '2x2pt']:
            self.PosPosFid = self.makePosPosCells()
            all_cls.append(self.PosPosFid)
        self.ccl_cls = np.vstack(tuple(all_cls))
        
    def process(self):
        """
        Orchestrates the full pipeline of calculations for the analysis.

        This method serves as the main processing function, coordinating the steps necessary to compute the 
        Fisher matrix and the derivatives of the C_ell values. It calls other internal methods to:
        - Compute the source and lens photometric redshift distributions.
        - Retrieve the ell values for angular power spectra.
        - Generate the fiducial C_ell values.
        - Build the covariance matrix.
        - Compute the derivatives of the C_ell values with respect to the model parameters.
        - Calculate the Fisher matrix.

        If derivatives have been previously saved to a file, the method can load them from the specified location, 
        or it can recompute them and optionally overwrite the saved file.

        Attributes modified:
            - `self.derivs_sig`: The computed derivatives of C_ell with respect to the parameters.
            - `self.fisher`: The computed Fisher matrix.
            - `self.has_run`: A flag set to `True` indicating that the processing has completed.

        Notes:
            - If `self.save_deriv` is `None`, derivatives are recalculated and stored in `self.derivs_sig`.
            - If `self.save_deriv` is provided, the method will attempt to load the derivatives from the specified 
              file, unless `self.overwrite` is `True`, in which case the derivatives are recalculated and saved.
        """
        self._makeSourcePZ()
        self._makeLensPZ()
        self.getElls()
        self.makeFidCells()
        self.buildCovMatrix()
        
        if self.save_deriv is None:
            self.getDerivs()
        else:
            if self.overwrite==True:
                self.getDerivs()
                with open(self.save_deriv, 'wb') as output:
                    pickle.dump(self.derivs_sig, output, pickle.HIGHEST_PROTOCOL)
            else:
                self.derivs_sig = pickle.load(open(self.save_deriv,'rb'))
        self.fisher = self.getFisher()
        self.has_run = True
        print('Done')
        
        
#    @lru_cache
    def getAi(self, beta, cosmo, dNdz, Mr_s=-20.70, Q=1.23, alpha_lum=-1.23, phi_0=0.0094, P=-0.3, mlim=25.3, Lp= 1.):
        """ Get the amplitude of the 2-halo part of w_{l+}
        A0 is the amplitude, beta is the power law exponent (see Krause et al. 2016) 
        cosmo is a CCL cosmology object 
        Lp is a pivot luminosity (default = 1)
        dndz is (z, dNdz) - vector of z and dNdz for the galaxies
        (does not need to be normalised) 
        """
        z_input = self.zmid
        dNdz = list(dNdz)
        z = np.array(z_input)

        # Get the luminosity function
        (L, phi_normed) = self.get_phi(z, self.cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim)
        # Pivot luminosity:
        Lp = 1.

        # Get Ai as a function of lens redshift.
        Ai_ofzl = np.zeros(len(z))
        for zi in range(len(z)):
            Ai_ofzl[zi] = scipy.integrate.simps(np.asarray(phi_normed[zi]) * (np.asarray(L[zi]) / Lp)**(beta), np.asarray(L[zi]))
            
        # Integrate over dNdz
        Ai = scipy.integrate.simps(Ai_ofzl * dNdz, z) / scipy.integrate.simps(dNdz, z)
        

        return Ai

    def get_phi(self, z, cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim, Mp=-22.):

        """ This function outputs the Schechter luminosity function with parameters fit in Loveday 2012, following the same procedure as Krause et al. 2015, as a function of z and L 
        The output is L[z][l], list of vectors of luminosity values in z, different at each z due to the different lower luminosity limit, and phi[z][l], a list of luminosity functions at these luminosity vectors, at each z
        cosmo is a CCL cosmology object
        mlim is the magnitude limit of the survey
        Mp is the pivot absolute magnitude.
        other parameteres are the parameters of the luminosity function that are different for different samples, e.g. red vs all. lumparams = [Mr_s, Q, alpha_lum, phi_0, P]
        Note that the luminosity function is output normalized (appropriate for getting Ai)."""


        """ This function outputs the Schechter luminosity function with parameters fit in Loveday 2012, following the same procedure as Krause et al. 2015, as a function of z and L 
        The output is L[z][l], list of vectors of luminosity values in z, different at each z due to the different lower luminosity limit, and phi[z][l], a list of luminosity functions at these luminosity vectors, at each z
        cosmo is a CCL cosmology object
        mlim is the magnitude limit of the survey
        Mp is the pivot absolute magnitude.
        other parameteres are the parameters of the luminosity function that are different for different samples, e.g. red vs all. lumparams = [Mr_s, Q, alpha_lum, phi_0, P]
        Note that the luminosity function is output normalized (appropriate for getting Ai)."""

        # Get the amplitude of the Schechter luminosity function as a function of redshift.
        phi_s = phi_0 * 10.**(0.4 * P * z)

        # Get M_* (magnitude), then convert to L_*
        Ms = Mr_s - Q * (z - 0.1)
        Ls = 10**(-0.4 * (Ms - Mp))

        # Import the kcorr and ecorr correction from Poggianti (assumes elliptical galaxies)
        # No data for sources beyon z = 3, so we keep the same value at higher z as z=3
        (z_k, kcorr, x,x,x) = np.loadtxt(package_path + '/data/kcorr.dat', unpack=True)
        (z_e, ecorr, x,x,x) = np.loadtxt(package_path + '/data/ecorr.dat', unpack=True)
        kcorr_interp = CubicSpline(z_k, kcorr)
        ecorr_interp = CubicSpline(z_e, ecorr)
        kcorr = kcorr_interp(z)
        ecorr = ecorr_interp(z)

        # Get the absolute magnitude and luminosity corresponding to limiting apparent magntiude (as a function of z)
        dl = ccl.luminosity_distance(cosmo, 1./(1.+z))
        Mlim = mlim - (5. * np.log10(dl) + 25. + kcorr + ecorr)
        # Mlim = mlim - (5. * np.log10(dl) + 25. + kcorr)
        Llim = 10.**(-0.4 * (Mlim-Mp))

        L = [0]*len(z)
        for zi in range(0, len(z)):
            L[zi] = np.logspace(np.log10(Llim[zi]), 2., 1000)

        # Now get phi(L,z), where this exists for each z because the lenghts of the L vectors are different.
        phi_func = [0]*len(z)
        for zi in range(0, len(z)):
            phi_func[zi]= np.zeros(len(L[zi]))
            for li in range(0, len(L[zi])):
                phi_func[zi][li] = phi_s[zi] * (L[zi][li] / Ls[zi]) ** (alpha_lum) * np.exp(- L[zi][li] / Ls[zi])

        norm = np.zeros(len(z))
        phi_func_normed = [0]*len(z)
        for zi in range(len(z)):
            norm[zi] = scipy.integrate.simps(phi_func[zi], L[zi])
            phi_func_normed[zi] = phi_func[zi] / norm[zi]

        return (L, phi_func_normed)
        
    def _outlier_helper(self, idx, zoutlier):
        
        dNdz_dict_source = {}
        qs = []
        for index, (x,x2) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            core = Core(self.zmid, zbias=self.zbias[index], sigma_z=self.zvariance[index])
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2-x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[self.bin_centers[index]] = photoz_model.get_pdf()[0]

        for i, b in enumerate(list(sorted(dNdz_dict_source.keys()))):
            f = CubicSpline(self.zmid, dNdz_dict_source[b])
            q = quad(f, 0, 4)[0]
            dNdz_dict_source[b] /= q
            if i==idx:
                dNdz_dict_source[b] = dNdz_dict_source[b]*(1-zoutlier)+self.scores[i]*zoutlier
            else:
                dNdz_dict_source[b] = dNdz_dict_source[b]*(1-self.outliers[i])+self.scores[i]*self.outliers[i]
  
        return dNdz_dict_source

    def getC_ellOfzoutlier1_ss(self, zoutlier):
        index = 0
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_ss(self.cosmo, dNdz_dict_source)
    
    def getC_ellOfzoutlier2_ss(self, zoutlier):
        index = 1
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier3_ss(self, zoutlier):
        index = 2
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier4_ss(self, zoutlier):
        index = 3
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier5_ss(self, zoutlier):
        index = 4
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_ss(self.cosmo, dNdz_dict_source)
    
    def getC_ellOfzoutlier1_sp(self, zoutlier):
        index = 0
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_sp(self.cosmo, dNdz_dict_source)
    
    def getC_ellOfzoutlier2_sp(self, zoutlier):
        index = 1
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier3_sp(self, zoutlier):
        index = 2
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier4_sp(self, zoutlier):
        index = 3
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier5_sp(self, zoutlier):
        index = 4
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper_sp(self.cosmo, dNdz_dict_source)
    
    def _bias_helper(self, idx, zbias):
        dNdz_dict_source = {}
        qs = []
        for index, (x,x2) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if index==idx:
                core = Core(self.zmid, zbias=zbias, sigma_z=self.zvariance[index])
            else:
                core = Core(self.zmid, zbias=self.zbias[index], sigma_z=self.zvariance[index])
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2-x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[self.bin_centers[index]] = photoz_model.get_pdf()[0]

        for i, b in enumerate(list(sorted(dNdz_dict_source.keys()))):
            f = CubicSpline(self.zmid, dNdz_dict_source[b])
            q = quad(f, 0, 4)[0]
            dNdz_dict_source[b] /= q
            dNdz_dict_source[b] = dNdz_dict_source[b]*(1-self.outliers[i])+self.scores[i]*self.outliers[i]

        return dNdz_dict_source
    
    
    def getC_ellOfIA_amp_ss(self, A0):
        return self._helper_ss(self.cosmo, self.dNdz_dict_source, A0=A0)
        
    def getC_ellOfIA_highz_ss(self, etah):
        return self._helper_ss(self.cosmo, self.dNdz_dict_source, etah=etah)
        
    def getC_ellOfIA_lowz_ss(self, etal):
        return self._helper_ss(self.cosmo, self.dNdz_dict_source, etal=etal)
        
    def getC_ellOfIA_beta_ss(self, beta):
        return self._helper_ss(self.cosmo, self.dNdz_dict_source, beta=beta)


    def getC_ellOfIA_amp_sp(self, A0):
        return self._helper_sp(self.cosmo, self.dNdz_dict_source, A0=A0)
        
    def getC_ellOfIA_highz_sp(self, etah):
        return self._helper_sp(self.cosmo, self.dNdz_dict_source, etah=etah)
        
    def getC_ellOfIA_lowz_sp(self, etal):
        return self._helper_sp(self.cosmo, self.dNdz_dict_source, etal=etal)
        
    def getC_ellOfIA_beta_sp(self, beta):
        return self._helper_sp(self.cosmo, self.dNdz_dict_source, beta=beta)

    def getC_ellOfzbias1_ss(self, zbias):
        index = 0
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_ss(self.cosmo, dNdz_dict_source)
    
    def getC_ellOfzbias2_ss(self, zbias):
        index = 1
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias3_ss(self, zbias):
        index = 2
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias4_ss(self, zbias):
        index = 3
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias5_ss(self, zbias):
        index = 4
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias1_sp(self, zbias):
        index = 0
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_sp(self.cosmo, dNdz_dict_source)
    
    def getC_ellOfzbias2_sp(self, zbias):
        index = 1
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias3_sp(self, zbias):
        index = 2
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias4_sp(self, zbias):
        index = 3
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias5_sp(self, zbias):
        index = 4
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def _variance_helper(self, idx, zvar):
        dNdz_dict_source = {}
        qs = []
        for index, (x,x2) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if index==idx:
                core = Core(self.zmid, zbias=self.zbias[index], sigma_z=zvar)
            else:
                core = Core(self.zmid, zbias=self.zbias[index], sigma_z=self.zvariance[index])
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2-x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[self.bin_centers[index]] = photoz_model.get_pdf()[0]


        for i, b in enumerate(list(sorted(dNdz_dict_source.keys()))):
            f = CubicSpline(self.zmid, dNdz_dict_source[b])
            q = quad(f, 0, 4)[0]
            dNdz_dict_source[b] /= q
            dNdz_dict_source[b] = dNdz_dict_source[b]*(1-self.outliers[i])+self.scores[i]*self.outliers[i]
        
        
        return dNdz_dict_source

    def getC_ellOfzvariance1_ss(self, zvariance):
        index = 0
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance2_ss(self, zvariance):
        index = 1
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance3_ss(self, zvariance):
        index = 2
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance4_ss(self, zvariance):
        index = 3
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance5_ss(self, zvariance):
        index = 4
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_ss(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance1_sp(self, zvariance):
        index = 0
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance2_sp(self, zvariance):
        index = 1
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance3_sp(self, zvariance):
        index = 2
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance4_sp(self, zvariance):
        index = 3
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance5_sp(self, zvariance):
        index = 4
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper_sp(self.cosmo, dNdz_dict_source)

    def _helper_ss(self, cosmo, dNdz_dict_source, A0=None, beta=None, etal=None, etah=None):
        if not beta:
            beta = self.vals['beta']
        if not etal:
            etal = self.vals['etal']
        if not etah:
            etah = self.vals['etah']
        if not A0:
            A0 = self.vals['A0']
        ia0 =  A0 * np.array([self.A_l(zi, etal) for zi in self.zmid]) * np.array([self.A_h(zi, etah) for zi in self.zmid])
        ia_lens1 = self.getAi(beta, cosmo, dNdz=tuple(dNdz_dict_source[self.key])) * ia0
        lens1 = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, dNdz_dict_source[self.key]), ia_bias=(self.zmid, ia_lens1))
        ia_lens2 = self.getAi(beta, cosmo, dNdz=tuple(dNdz_dict_source[self.keyj])) * ia0
        lens2 = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, dNdz_dict_source[self.keyj]), ia_bias=(self.zmid, ia_lens2))
        return ccl.angular_cl(cosmo, lens1, lens2, self.ell)

    def _helper_sp(self, cosmo, dNdz_dict_source, A0=None, beta=None, etal=None, etah=None):
        if not beta:
            beta = self.vals['beta']
        if not etal:
            etal = self.vals['etal']
        if not etah:
            etah = self.vals['etah']
        if not A0:
            A0 = self.vals['A0']
        pos = ccl.NumberCountsTracer(self.cosmo, dndz=(self.zmid, self.dNdz_dict_lens[self.keyl]), has_rsd=False, bias=(self.zmid, self.gbias_dict[self.keyl]*np.ones_like(self.zmid)))
        ia0 =  A0 * np.array([self.A_l(zi, etal) for zi in self.zmid]) * np.array([self.A_h(zi, etah) for zi in self.zmid])
        ia = self.getAi(beta, cosmo, dNdz=tuple(dNdz_dict_source[self.keys])) * ia0
        lens = ccl.WeakLensingTracer(cosmo, dndz=(self.zmid, dNdz_dict_source[self.keys]), ia_bias=(self.zmid, ia))
        return ccl.angular_cl(cosmo, pos, lens, self.ell)

    
    def getC_ellOfgbias_pp_helper(self, gbias):
        pos1 = ccl.NumberCountsTracer(self.cosmo, dndz=(self.zmid, self.dNdz_dict_lens[self.key]), has_rsd=False, bias=(self.zmid, gbias*np.ones_like(self.zmid)))
        return ccl.angular_cl(self.cosmo, pos1, pos1, self.ell)

    def getC_ellOfgbias_sp_helper(self, gbias):
        pos1 = ccl.NumberCountsTracer(self.cosmo, dndz=(self.zmid, self.dNdz_dict_lens[self.keyl]), has_rsd=False, bias=(self.zmid, gbias*np.ones_like(self.zmid)))
        ia0 =  self.vals['A0'] * np.array([self.A_l(zi, self.vals['etal']) for zi in self.zmid]) * np.array([self.A_h(zi, self.vals['etah']) for zi in self.zmid])
        ia = self.getAi(self.vals['beta'], self.cosmo, dNdz=tuple(self.dNdz_dict_source[self.keys])) * ia0
        lens1 = ccl.WeakLensingTracer(self.cosmo, dndz=(self.zmid, self.dNdz_dict_source[self.keys]), ia_bias=(self.zmid, ia))
        return ccl.angular_cl(self.cosmo, pos1, lens1, self.ell)
    
    def _helper_pp(self, cosmo):
        pos1 = ccl.NumberCountsTracer(cosmo, dndz=(self.zmid, self.dNdz_dict_lens[self.key]), has_rsd=False, bias=(self.zmid, self.gbias_dict[self.key]*np.ones_like(self.zmid)))
        return ccl.angular_cl(cosmo, pos1, pos1, self.ell)

    def getC_ellOfSigma8_ss(self, sigma8):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=sigma8, n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfOmegab_ss(self, Omega_b):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=Omega_b, h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfh_ss(self, h):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=h, sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfn_s_ss(self, n_s):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=n_s, 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfOmegam_ss(self, Omega_m):
        cosmo = ccl.Cosmology(Omega_c=Omega_m - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfw0_ss(self, w_0):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], w0=w_0, 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfwa_ss(self, w_a):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], wa=w_a, 
            transfer_function='eisenstein_hu')
        return self._helper_ss(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfSigma8_sp(self, sigma8):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=sigma8, n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfOmegab_sp(self, Omega_b):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=Omega_b, h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfh_sp(self, h):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=h, sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfn_s_sp(self, n_s):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=n_s, 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfOmegam_sp(self, Omega_m):
        cosmo = ccl.Cosmology(Omega_c=Omega_m-self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfw0_sp(self, w_0):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], w0=w_0, 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfwa_sp(self, w_a):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], wa=w_a, 
            transfer_function='eisenstein_hu')
        return self._helper_sp(cosmo, dNdz_dict_source=self.dNdz_dict_source)


    def getC_ellOfSigma8_pp(self, sigma8):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=sigma8, n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)

    def getC_ellOfOmegab_pp(self, Omega_b):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=Omega_b, h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)

    def getC_ellOfh_pp(self, h):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=h, sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)

    def getC_ellOfn_s_pp(self, n_s):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=n_s, 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)


    def getC_ellOfOmegam_pp(self, Omega_m):
        cosmo = ccl.Cosmology(Omega_c=Omega_m-self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)


    def getC_ellOfw0_pp(self, w_0):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], w0=w_0, 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)


    def getC_ellOfwa_pp(self, w_a):
        cosmo = ccl.Cosmology(Omega_c=self.vals['omega_m'] - self.vals['omega_b'], Omega_b=self.vals['omega_b'], h=self.vals['h'], sigma8=self.vals['sigma_8'], n_s=self.vals['n_s'], wa=w_a, 
            transfer_function='eisenstein_hu')
        return self._helper_pp(cosmo)