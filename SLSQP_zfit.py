from typing import List, Optional
from collections import OrderedDict

from zfit.minimizers.baseminimizer import BaseMinimizer
from zfit.minimizers.fitresult import FitResult
from zfit.core.interfaces import ZfitLoss
from zfit.core.parameter import Parameter

from scipy.optimize import minimize

import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
import zfit

class SLSQP(BaseMinimizer):
    
    def __init__(self, tolerance=None, verbosity=5, name='SLSQP', 
                 constraints = (),
                 **minimizer_options):
        
        if 'ftol' not in minimizer_options:
            print('ftol not in minizer_options')
            minimizer_options['ftol'] = 10e-8
        else:
            print('ftol = ' + str(minimizer_options['ftol']))
            
            
        self.constraints = constraints
        super().__init__(tolerance=tolerance, 
                         name=name, verbosity=verbosity, 
                         minimizer_options=minimizer_options)
        
        
        
    def _minimize(self, loss: ZfitLoss, params: List[Parameter]):
        
        if params:
            parameters = params
        else:
            parameters = loss.get_params()
        
        start_values = [p.numpy() for p in parameters]
        limits = tuple(tuple((p.lower, p.upper)) for p in parameters)
        
        def func(values):
            #params = loss.get_params()
            with zfit.param.set_values(parameters, values):
                 val = loss.value()
            return val
        
        start_values = zfit.run(parameters)
        minimizer = minimize(
            fun=func,  x0=start_values,
            args=(), method='SLSQP', bounds=limits, 
            constraints=self.constraints, tol=self.tolerance,
            callback=None, 
            options = self.minimizer_options)
        
        self._update_params(params=parameters, values=minimizer.x)

        params = OrderedDict((p, res) for p, res in zip(parameters, minimizer.x))
        fitresult = FitResult(
                              loss = loss, minimizer=minimize,
                              params =params, edm = 1.0, fmin = minimizer.fun,
                              status =minimizer.status,
                              converged = minimizer.success,
                              info = dict(minimizer) )
        
        return fitresult
    
    
    
    
    
    
def create_constraint(model, mu_index=False, sigma_index=False):

    #First look for the indices of the POIs
    if (type(mu_index)!=int and mu_index==False) or (type(sigma_index)!=int and sigma_index==False):
 
        for i,p in enumerate(model.get_params()):
            if 'mu' in p.name.lower() or 'mu_' in p.name.lower():  afb_index = i
            if 'sigma' in p.name.lower() or 'sigma_' in p.name.lower():  fh_index = i

        if str(mu_index)=='False' or str(sigma_index)=='False':
            print('I was not able to find the indices, please fix it here:\n ../scripts/SLSQP_zfit.py')
            raise NotImplementedError
    #Now define the "simple" constraints give the found index
    constAngParams = (
                 {'type': 'ineq', 'fun': lambda x:  x[mu_index]},
                 {'type': 'ineq', 'fun': lambda x:  x[sigma_index]})
    
    print(mu_index, sigma_index)

    return constAngParams
