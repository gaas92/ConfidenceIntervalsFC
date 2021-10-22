#ONLY TO FORCE RUN IN CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import tensorflow as tf
import zfit
from zfit import z


import pandas as pd

import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize


import SLSQP_zfit

import scipy
from scipy.stats import binom

zfit.util.cache.clear_graph_cache()

# Model Necessary PDFs
class Angular_PDF(zfit.pdf.BasePDF):
    """ Angular pdf with one parameter from equation (7.2).

    """

    def __init__(self, obs, FH, name="Angular_pdf", ):
    
        params = {'FH': FH}
        super().__init__(obs, params, name=name)

    def _pdf(self, x, norm_range):
    
        cos_l = z.unstack_x(x)
        FH = self.params['FH']

        pdf = 3/2*(1 - FH)*(1 - tf.math.square(tf.math.abs(cos_l))) + FH
    
        return pdf

class bernstein(zfit.pdf.BasePDF):  
    """
    Bernstein_nth Degree
    From a to b
    x-> (x-a/b-a)
    https://en.wikipedia.org/wiki/Bernstein_polynomial
    """

    def __init__(self, coefs, obs, name="Bernstein" ):        
        
        self.degree = len(coefs)-1
        params = dict()
        for indx,c in enumerate(coefs):
            params[f'c{indx}'] = c

        super().__init__(obs, params, name=name+f' Deg. {self.degree}')

    def _unnormalized_pdf(self, x):
        
        x_ = zfit.ztf.unstack_x(x)
        limits = self.norm_range.limit1d
        x_T  = (x_-limits[0])/(limits[1]-limits[0])
        deg = self.degree

        basis = dict()
        
        for i in range(deg+1):
            basis[i] = self.params[f'c{i}']*scipy.special.binom(deg,i)*tf.pow(x_T,i)*tf.pow(1-x_T,deg-i)

        pdf = basis[0]
        
        for i in range(1, deg+1):
            pdf += basis[i]

        return pdf

class JohnsonSU(zfit.models.dist_tfp.WrapDistribution):
    """
    Johnson's S_U distribution callback from tensorflowprobability
    https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JohnsonSU
    """
    _N_OBS = 1
    def __init__(self, gamma, delta, mu, sigma, obs, name="JohnsonSU" ):
        gamma, delta, mu, sigma = self._check_input_params(gamma, delta, mu, sigma)
        params = OrderedDict((('gamma', gamma), ('delta', delta), ('mu', mu), ('sigma', sigma)))
        dist_params = lambda: dict(skewness=gamma.value(), tailweight=delta.value(), loc=mu.value(), scale=sigma.value())
        distribution = tfp.distributions.JohnsonSU
        super().__init__(distribution=distribution, dist_params=dist_params, obs=obs, params=params, name=name)

# Angular Efficency PDF
def angular_efficency(params, ang_obs):
    if 'ChevEffyCoefs' in params['Signal']['angle'].keys():
        efficiency = zfit.pdf.Chebyshev(obs=ang_obs, 
                                    coeffs=params['Signal']['angle']['Chi2EffyCoefs'][1:]) # Chevichev grado variable Codigo original no se porque votan el primer termino 
        
    elif 'BernEffyCoefs' in params['Signal']['angle'].keys():
        efficiency = bernstein(coefs=params['Signal']['angle']['BernEffyCoefs'],  # Bernstein grado variable 
                               obs=ang_obs)  
    else :
        efficiency = None
        print(f'Efficency not implemented yet')
        return -1  

    return efficiency          

# Angular background
def angular_bkg_pdf(params, ang_obs):
    left_coefs  = params['Background']['angle']['2SideBands']['left_sb']
    right_coefs = params['Background']['angle']['2SideBands']['right_sb']
    
    left_bernstein  = bernstein(coefs=left_coefs, obs=ang_obs)
    right_bernstein = bernstein(coefs=right_coefs, obs=ang_obs)
    fangular_bkg = params['Background']['angle']['2SideBands']['fraction']
    background_angular = zfit.pdf.SumPDF([left_bernstein, right_bernstein], fracs=fangular_bkg)

    return background_angular

def complete_angular_pdf(fh, params, ang_obs):
    
    # Angular Signal 
    decay = Angular_PDF(FH=fh, obs=ang_obs)

    # Angular Efficency 
    efficiency = angular_efficency(params=params, ang_obs=ang_obs)

    # Angular Signal 
    decay_eff = zfit.pdf.ProductPDF(pdfs=[decay, efficiency], obs=ang_obs)
    
    # Angular Background
    background_angular = angular_bkg_pdf(params=params, ang_obs=ang_obs)

    s_ini = 437 if not 'yield' in params['Signal'].keys() else int(params['Signal']['yield'])
    b_ini = 706 if not 'yield' in params['Background'].keys() else int(params['Background']['yield'])

    Total = s_ini + b_ini

    S = zfit.Parameter('signalYield_a', s_ini, 0, Total*1.5)
    B = zfit.Parameter('backgroundYield_a', b_ini, 0, Total*1.5)

    signal_extended = decay_eff.create_extended(yield_=S)
    background_extended = background_angular.create_extended(yield_=B)

    complete_ang_pdf = zfit.pdf.SumPDF([signal_extended, background_extended], name='complete_ang_pdf') # zFit-ization

    return complete_ang_pdf

# Mass Signal
def mass_signal_pdf(params, mass_obs):
    if '2Gaussian+CrystalBall' in params['Signal']['mass'].keys():
        mass_signal_params = params['Signal']['mass']['2Gaussian+CrystalBall']
    
        signalCB = zfit.pdf.CrystalBall(mu=mass_signal_params['muCB'],  # Gauss + CB
                                sigma=mass_signal_params['sigmaCB'], 
                                alpha=mass_signal_params['alphaCB'], 
                                n=mass_signal_params['nCB'], 
                                obs=mass_obs, 
                                name='signal_CrystalBall')

        signalG1 = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss1'], 
                          obs=mass_obs, 
                          name='signal_Gaussian1')

        signalG2 = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss2'], 
                          obs=mass_obs, 
                          name='signal_Gaussian2')

        fracs = mass_signal_params['fracCB'], mass_signal_params['fracGauss1'], mass_signal_params['fracGauss2']

        signal_mass = zfit.pdf.SumPDF([signalCB, signalG1, signalG2], fracs=[fracs[0]/sum(fracs), fracs[1]/sum(fracs)], 
                      name='signal_CB+2Gauss')
    
    elif 'Gaussian+CrystalBall' in params['Signal']['mass'].keys():
        mass_signal_params = params['Signal']['mass']['Gaussian+CrystalBall']
    
        signalCB = zfit.pdf.CrystalBall(mu=mass_signal_params['muCB'],  # Gauss + CB
                                sigma=mass_signal_params['sigmaCB'], 
                                alpha=mass_signal_params['alphaCB'], 
                                n=mass_signal_params['nCB'], 
                                obs=mass_obs, 
                                name='signal_CrystalBall')

        signalG = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss'], 
                          obs=mass_obs, 
                          name='signal_Gaussian')

        fracs_ = mass_signal_params['fracCB']

        signal_mass = zfit.pdf.SumPDF([signalCB, signalG], fracs=fracs_, 
                      name='signal_CB+Gauss')
        
    else:
        signal_mass = None
        print(f'{signal_shape} signal shape not implemented yet !!')
        return -1
    
    return signal_mass

# Mass Background 
def mass_background_pdf(params, mass_obs):
    mass_background_params = params['Background']['mass']['Exponential+Gauss']

    
    backExponential = zfit.pdf.Exponential(lambda_=mass_background_params['lambda_'], 
                            obs=mass_obs,
                            name='background_Exponential')

    backBGauss = zfit.pdf.Gauss(mu=mass_background_params['mu'],
                            sigma=mass_background_params['sigma'],
                            obs=mass_obs,
                            name='background_Gaussian')    
    background_mass = zfit.pdf.SumPDF([backExponential, backBGauss], fracs=mass_background_params['fraction_exp'], 
                          name='background_Gauss+Exp')
    return background_mass

def complete_mass_pdf(params, mass_obs):
    
    # Mass Signal
    signal_mass = mass_signal_pdf(params=params, mass_obs=mass_obs)

    # Mass Background 
    background_mass = mass_background_pdf(params=params, mass_obs=mass_obs)

    s_ini = 437 if not 'yield' in params['Signal'].keys() else int(params['Signal']['yield'])
    b_ini = 706 if not 'yield' in params['Background'].keys() else int(params['Background']['yield'])

    Total = s_ini + b_ini

    S = zfit.Parameter('signalYield_m', s_ini, 0, Total*1.5)
    B = zfit.Parameter('backgroundYield_m', b_ini, 0, Total*1.5)

    signal_extended = signal_mass.create_extended(yield_=S)
    background_extended = background_mass.create_extended(yield_=B)

    complete_mass_pdf = zfit.pdf.SumPDF([signal_extended, background_extended], name='complete_mass_pdf') # zFit-ization

    return complete_mass_pdf

# Complete 2-D PDF        
def complete_PDF(mass_obs=None, ang_obs=None, fh=None, params=None, name='complete_pdf',
                 SigYield=None, BkgYield=None):
    """Function that constructs the complete PDF:

    Parameters  
    ----------
    mass_obs : zfit.Space, mandatory
        mass observable space
    ang_obs : zfit.Space, mandatory
        mass observable space
    fh     : zft.Parameter, mandatory
        fh parameter to fit 
    params : dict, mandatory 
        dict with parametes for the PDF,
        params items  can be floating points if we want fix values
        or floating zfit.Parameter if we want to fit them 
    name  : str, optional 
        name for the complete pdf 
    Raises
    ------
    NoParameters
        If parameters dict is not valid
    
    """
    if mass_obs is None:
        print('mass obs is none can´t create PDF !!')
        return -1
    if ang_obs is None:
        print('angular obs is none can´t create PDF !!')
        return -1 
    if params is None:
        print('parameters are none can´t create PDF !!')
        return -1
    
    # Angular Signal 
    decay = Angular_PDF(FH=fh, obs=ang_obs)

    # Angular Efficency 
    efficiency = angular_efficency(params=params, ang_obs=ang_obs)

    # Angular Signal 
    decay_eff = zfit.pdf.ProductPDF(pdfs=[decay, efficiency], obs=ang_obs)
    
    # Angular Background
    background_angular = angular_bkg_pdf(params=params, ang_obs=ang_obs)

    # Mass Signal
    signal_mass = mass_signal_pdf(params=params, mass_obs=mass_obs)

    # Mass Background 
    background_mass = mass_background_pdf(params=params, mass_obs=mass_obs)

    #complete_signal = signal_mass*decay_eff # Antonio Original
    complete_signal = zfit.pdf.ProductPDF(pdfs=[signal_mass, decay_eff], name=name+'_signal')  # zFit-ization

    #complete_background = background_mass*background_angular # Antonio Original
    complete_background = zfit.pdf.ProductPDF(pdfs=[background_mass, background_angular], name=name+'background') # zFit-ization

    if SigYield is None or BkgYield is None:
        s_ini = 437 if not 'yield' in params['Signal'].keys() else int(params['Signal']['yield'])
        b_ini = 706 if not 'yield' in params['Background'].keys() else int(params['Background']['yield'])

        Total = s_ini + b_ini

        S = zfit.Parameter('signalYield', s_ini, 0, Total*1.5)
        B = zfit.Parameter('backgroundYield', b_ini, 0, Total*1.5)
    else :
        S = SigYield
        B = BkgYield
    
    signal_extended = complete_signal.create_extended(yield_=S)
    background_extended = complete_background.create_extended(yield_=B)

    #complete_pdf = signal_extended+background_extended # Antonio Original
    complete_pdf = zfit.pdf.SumPDF([signal_extended, background_extended], name=name) # zFit-ization

    return complete_pdf