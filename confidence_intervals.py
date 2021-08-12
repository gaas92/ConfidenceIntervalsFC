
#ONLY TO FORCE RUN IN CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import tensorflow as tf
import zfit
from zfit import z


import random
import json
import pandas as pd
from time import time

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.optimize import minimize


import SLSQP_zfit
import time

import scipy
from scipy.stats import binom

import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
zfit.util.cache.clear_graph_cache()

# Model PDFs
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
        

    
def cl_function(real_data, FH=0.24, params=None, N=50):
    
    
    """Function that return a 1-CL point for a given:
    FH (true value)
    params (fitted pdf parametes)
    real data (data where the fited parameters where extracted)

    If some of the argements are not pased default values will be used 

    Parameters  
    ----------
    FH : float, optional
        Fitted value (used as True Value)
    params : dict,
        dict with all the fitted parametes for building the base PDF
    real_data: zfit.Data with obs=(BMass, cosThetaKMu)  
        real data in zFit format 
    N : int, optional 
        numer of toy MC to generate 

    Raises
    ------
    NoParameters
        If parameters dict is not valid
    
    NotImplementedError
        If no option is implemented yet 
    """
    
    if params is None:
        print('No parameters Given!!! \nCheck your code!')
        return -1
    
    cos = zfit.Space(obs='cosThetaKMu', limits=[0.0,1.0])
    mass = zfit.Space(obs='BMass', limits=[5.0,6.0])


    fh = zfit.Parameter('F_HH', FH, lower_limit=0.0, upper_limit=3.0)  
    decay = Angular_PDF(FH=fh, obs=cos)

    if 'ChevEffyCoefs' in params['Signal']['angle'].keys():
        efficiency = zfit.pdf.Chebyshev(obs=cos, 
                                    coeffs=params['Signal']['angle']['Chi2EffyCoefs'][1:]) # Chevichev grado variable Codigo original no se porque votan el primer termino 
        
    elif 'BernEffyCoefs' in params['Signal']['angle'].keys():
        efficiency = bernstein(coefs=params['Signal']['angle']['BernEffyCoefs'],  # Bernstein grado variable 
                               obs=cos)  
    else :
        efficiency = None
        print(f'Efficency not implemented yet')
        return -1
    
    #Angular Signal 
    decay_eff = zfit.pdf.ProductPDF(pdfs=[decay, efficiency], obs=cos)

    #Angular sidebands
    left_coefs  = params['Background']['angle']['2SideBands']['left_sb']
    right_coefs = params['Background']['angle']['2SideBands']['right_sb']
    
    left_bernstein  = bernstein(coefs=left_coefs, obs=cos)
    right_bernstein = bernstein(coefs=right_coefs, obs=cos)
    fangular_bkg = params['Background']['angle']['2SideBands']['fraction']
    #background_angular = bernstein(coefs=coef_list, obs=cos) # Suma de dos polinomios de grado variable Codigo original
    
    #Angular Background
    background_angular = zfit.pdf.SumPDF([left_bernstein, right_bernstein], fracs=fangular_bkg)

    if '2Gaussian+CrystalBall' in params['Signal']['mass'].keys():
        mass_signal_params = params['Signal']['mass']['2Gaussian+CrystalBall']
    
        signalCB = zfit.pdf.CrystalBall(mu=mass_signal_params['muCB'],  # Gauss + CB
                                sigma=mass_signal_params['sigmaCB'], 
                                alpha=mass_signal_params['alphaCB'], 
                                n=mass_signal_params['nCB'], 
                                obs=mass, 
                                name='signal_CrystalBall')

        signalG1 = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss1'], 
                          obs=mass, 
                          name='signal_Gaussian1')

        signalG2 = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss2'], 
                          obs=mass, 
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
                                obs=mass, 
                                name='signal_CrystalBall')

        signalG = zfit.pdf.Gauss(mu=mass_signal_params['muGauss'], 
                          sigma=mass_signal_params['sigmaGauss'], 
                          obs=mass, 
                          name='signal_Gaussian')

        fracs_ = mass_signal_params['fracCB']

        signal_mass = zfit.pdf.SumPDF([signalCB, signalG], fracs=fracs_, 
                      name='signal_CB+Gauss')
        
    else:
        signal_mass = None
        print(f'{signal_shape} signal shape not implemented yet !!')
        return -1



    mass_background_params = params['Background']['mass']['Exponential+Gauss']

    
    backExponential = zfit.pdf.Exponential(lambda_=mass_background_params['lambda_'], 
                            obs=mass,
                            name='background_Exponential')

    backBGauss = zfit.pdf.Gauss(mu=mass_background_params['mu'],
                            sigma=mass_background_params['sigma'],
                            obs=mass,
                            name='background_Gaussian')

    background_mass = zfit.pdf.SumPDF([backExponential, backBGauss], fracs=mass_background_params['fraction_exp'], 
                          name='background_Gauss+Exp')


    complete_signal = signal_mass*decay_eff
    complete_background = background_mass*background_angular


    s_ini = 437 if not 'yield' in params['Signal'].keys() else int(params['Signal']['yield'])
    b_ini = 706 if not 'yield' in params['Background'].keys() else int(params['Background']['yield'])

    Total = s_ini + b_ini

    S = zfit.Parameter('signalYield', s_ini, 0, Total)
    B = zfit.Parameter('backgroundYield', b_ini, 0, Total)

    signal_extended = complete_signal.create_extended(yield_=S)
    background_extended = complete_background.create_extended(yield_=B)

    complete_pdf = signal_extended+background_extended
    print(complete_pdf.get_params())
    
    #N = 50 # Number of toy MC
    
    pseudo_data = []
    constAngParams_Full = ({'type': 'ineq', 'fun': lambda x:  x[2]},
                       {'type': 'ineq', 'fun': lambda x:  3-x[2]}) 

    SLSQP_FULL = SLSQP_zfit.SLSQP(constraints=constAngParams_Full)

    
    SLSQP_FULL_profile = SLSQP_zfit.SLSQP() # Without restrictions on POI's
    
    
    Delta_chi2 = []
    sampler = complete_pdf.create_sampler(n=Total, fixed_params=True)
    nll_best = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=sampler)
    nll_profile = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=sampler)

    for i in range(N):
        
        sampler.resample()
        fh.set_value(FH)
        S.set_value(s_ini)
        B.set_value(b_ini)
                
        result_best = SLSQP_FULL.minimize(nll_best)
        best_likelihood = nll_best.value().numpy() 
        
        fh.set_value(FH)
        S.set_value(s_ini)
        B.set_value(b_ini)
        
        result_profile = SLSQP_FULL_profile.minimize(nll_profile, params=(S,B))
        profile_likelihood = nll_profile.value().numpy() 
    
        Delta = profile_likelihood - best_likelihood
        Delta_chi2.append(Delta)
        
        if i%2 ==0:
            zfit.util.cache.clear_graph_cache()
        
    #print(plt.hist(Delta_chi2))
    
    # Delta chi2 data
    
    fh.set_value(FH)
    S.set_value(s_ini)
    B.set_value(b_ini)
    
    nll_obs_best = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=real_data)
    result_obs_best = SLSQP_FULL.minimize(nll_obs_best)
    b_likelihood = nll_obs_best.value().numpy()
    
    #print(b_likelihood)
    
    fh.set_value(FH)
    S.set_value(s_ini)
    B.set_value(b_ini)
        

    nll_obs_profile = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=real_data)
    result_obs_profile = SLSQP_FULL_profile.minimize(nll_obs_profile, params=(S,B))
    p_likelihood = nll_obs_profile.value().numpy()
    
    #print(p_likelihood)
    
    
    Delta_data = p_likelihood - b_likelihood
    
    #print(Delta_data)
    
    factor = []
    
    for i in range(N):
        
        test = Delta_chi2[i]
        
        if test > Delta_data:
            
            factor.append(test)
            
    cl = len(factor)/N
    
    return (FH, cl)