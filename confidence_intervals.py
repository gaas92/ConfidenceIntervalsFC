
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

import scipy.stats as stats
from scipy.optimize import minimize


import SLSQP_zfit

import scipy
from scipy.stats import binom

zfit.util.cache.clear_graph_cache()

from complete_PDF import complete_PDF

def cl_function(real_data, FH=0.24, params=None, N=50, verbose=0):
    
    
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
    s_ini = 437 if not 'yield' in params['Signal'].keys() else int(params['Signal']['yield'])
    b_ini = 706 if not 'yield' in params['Background'].keys() else int(params['Background']['yield'])

    Total = s_ini + b_ini

    S = zfit.Parameter('signalYield', s_ini, 0, Total*1.5)
    B = zfit.Parameter('backgroundYield', b_ini, 0, Total*1.5)

    complete_pdf = complete_PDF(mass_obs=mass, ang_obs=cos, fh=fh, params=params, 
                                SigYield=S, BkgYield=B)
    
    #print(complete_pdf.get_params())
    
    #N = 50 # Number of toy MC
    
    pseudo_data = []
    constAngParams_Full = ({'type': 'ineq', 'fun': lambda x:  x[2]},
                       {'type': 'ineq', 'fun': lambda x:  3-x[2]}) 

    SLSQP_FULL = SLSQP_zfit.SLSQP(constraints=constAngParams_Full)

    
    SLSQP_FULL_profile = SLSQP_zfit.SLSQP() # Without restrictions on POI's
    
    
    Delta_chi2 = []
    zfit.util.cache.clear_graph_cache()
    #sampler = complete_pdf.create_sampler(n=Total, fixed_params=True)
    sampler = complete_pdf.create_sampler(fixed_params=True)

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
        
        if verbose:
            print(f'Toy MC {i} ok')
            if verbose > 1:
                print(f'\tbest_likelihood    = {best_likelihood}')
                print(f'\tprofile_likelihood = {profile_likelihood}')
                print(f'\tDelta              = {Delta}')
        if i%100 ==0:
            zfit.util.cache.clear_graph_cache()
        
    # Delta chi2 data
    zfit.util.cache.clear_graph_cache()
    fh.set_value(FH)
    S.set_value(s_ini)
    B.set_value(b_ini)
    
    nll_obs_best = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=real_data)
    result_obs_best = SLSQP_FULL.minimize(nll_obs_best)
    b_likelihood = nll_obs_best.value().numpy()
       
    fh.set_value(FH)
    S.set_value(s_ini)
    B.set_value(b_ini)     

    nll_obs_profile = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf, data=real_data)
    result_obs_profile = SLSQP_FULL_profile.minimize(nll_obs_profile, params=(S,B))
    p_likelihood = nll_obs_profile.value().numpy()
    
    Delta_data = p_likelihood - b_likelihood
    
    
    factor = []
    
    for i in range(N):
        
        test = Delta_chi2[i]
        
        if test > Delta_data:
            
            factor.append(test)
            
    cl = len(factor)/N
    
    return (FH, cl)