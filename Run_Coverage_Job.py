import argparse
import json 
from timeit import default_timer as timer
from pathlib import Path

import zfit
zfit.util.cache.clear_graph_cache()

parser = argparse.ArgumentParser(description='Script used to run on HTCondor a full confidence interval using Antonio Cota Method',
                                 epilog='Test example: ./Run_Coverage_Job.py',
                                 add_help=True)

parser.add_argument ('--HELP',    '-H', default=False,  action='store_true', help='Print help message.')
parser.add_argument ('--version', '-v', default='test', help='Version name.')

#parser.add_argument('--Bin',  '-B', default=4,    type=int, help='an integer for the Bin to analyze')
parser.add_argument('--nToy', '-N', default=500, type=int, help='an integer for the number of toy MC to run for every 1-cl calculation (more takes longer but has a smooth 1-cl graph)')
parser.add_argument('--FH', '-FH',  default=0.2, type=float, help='floating fh true-value to calculate Toy & calculate the coverage')
parser.add_argument('--granularity', '-G', default=0.05, type=float, help='floating value for the step size')
parser.add_argument('--range', '-R', default=[0.0, 3.0], type=list, help='list of two values for the FH range')
#parser.add_argument('--Step', '-Step',  default=1, type=int, help='int for the step in the FH region')
parser.add_argument('--Save', '-S', default=1, type=int, help='save in CERNBOX, false only for testing')
parser.add_argument('--verbose', default=0, type=int, help='int for the verbosity level')
parser.add_argument('--nJob', default=1, type=int, help='int for indexing the job (for reference)')


args = parser.parse_args()
if args.HELP:
    parser.print_help()
    exit()

def ConfidenceInterval():
    from confidence_intervals import cl_function
    import pandas as pd
    from complete_PDF import complete_PDF
    import SLSQP_zfit

    with open(f'Bin3/fitParams.json') as f:
        params = json.load(f)

    # Force "TRUE" FH value to given FH value    
    params['Signal']['angle']['FH'] = args.FH
    params['Signal']['angle']['FH_error'] = 0.0
    if args.verbose:
        print('fixed params:')
        print(json.dumps(params, sort_keys=True, indent=4))
    
    #Create PDF for toy MC / "TRUE DATA" 
    cos_true = zfit.Space(obs='cosThetaKMu', limits=[0.0,1.0])
    mass_true = zfit.Space(obs='BMass', limits=[5.0,6.0])


    fh_true = zfit.Parameter('F_H_true', args.FH, lower_limit=0.0, upper_limit=3.0)  
    s_ini = int(params['Signal']['yield'])
    b_ini = int(params['Background']['yield'])

    Total = s_ini + b_ini

    S_true = zfit.Parameter('signalYield_true', s_ini, 0, Total*1.5)
    B_true = zfit.Parameter('backgroundYield_true', b_ini, 0, Total*1.5)

    complete_pdf_true = complete_PDF(mass_obs=mass_true, ang_obs=cos_true, fh=fh_true, params=params, 
                                SigYield=S_true, BkgYield=B_true, name='true')
    
    true_sampler = complete_pdf_true.create_sampler(fixed_params=True)
    true_sampler.resample()

    #loop over FH's & 
    starting_fh = args.range[0]
    idx_ = 0
    results = {'idx':[], 'fh':[], '1-cl':[], 'elapsed_time':[]}
    while starting_fh < args.range[1]:
        starting_fh = round( starting_fh + args.granularity, 4)
        if starting_fh > args.range[1]:
            starting_fh = args.range[1]
        start = timer()
        fh, one_cl = cl_function(FH=starting_fh, params=params, real_data=true_sampler, 
                                 N=args.nToy, verbose=args.verbose, index=idx_)
        end = timer()
        results['idx'].append(int(idx_))
        results['fh'].append(float(fh))
        results['1-cl'].append(float(one_cl))
        results['elapsed_time'].append(round((end - start), 5))

        idx_+=1


    true_data_df = true_sampler.to_pandas()

    results_df = pd.DataFrame.from_dict(results)
    Path(f"Coverage_results/{args.nJob}").mkdir(parents=True, exist_ok=True)

    true_data_df.to_csv(f'Coverage_results/{args.nJob}/true_toyMC.csv', index=False)
    results_df.to_csv(f'Coverage_results/{args.nJob}/results.csv', index=False)

    #perform Fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model=complete_pdf_true, data=true_sampler)
    constAngParams_Full = ({'type': 'ineq', 'fun': lambda x:  x[2]},
                       {'type': 'ineq', 'fun': lambda x:  3-x[2]}) 
    

    SLSQP = SLSQP_zfit.SLSQP(constrains=constAngParams_Full)
    Minuit = zfit.minimize.Minuit(use_minuit_grad=True)

    SLSQP_result = SLSQP.minimize(nll)
    Minuit_result = Minuit.minimize(nll)

    SLSQP_result.hesse()
    Minuit_result.hesse()

    SLSQP_params = SLSQP_result.params
    Minuit_params = Minuit_result.params

    fh_minuit = Minuit_params[fh_true]['value']
    fh_e_minuit = Minuit_params[fh_true]['minuit_hesse']['error']

    fh_slsqp = SLSQP_params[fh_true]['value']
    fh_e_slsqp = SLSQP_params[fh_true]['hesse_np']['error']

    print('CI ok')

    return {'fh_minuit':fh_minuit, 'fh_e_minuit':fh_e_minuit,
            'fh_slsqp':fh_slsqp, 'fh_e_slsqp':fh_e_slsqp }


if __name__ == "__main__":
    if args.Save == 1 :
        None
    else :
        print('False Save')
    start = timer()
    minimizer_values = ConfidenceInterval()
    end = timer()
    print(f'Full 1-cl curve ok,  took {(end - start):.4f} seconds')

    #Save log

    with open(f'Coverage_results/{args.nJob}/log.txt', 'w') as f:
        f.write('Log for the job \n')
        f.write(f'Minimizer results: \n')
        f.write(f'\tFH Minuit: {minimizer_values["fh_minuit"]}\n')
        f.write(f'\tFH Minuit error: {minimizer_values["fh_e_minuit"]}\n')
        f.write(f'\tFH SLSQP: {minimizer_values["fh_slsqp"]}\n')
        f.write(f'\tFH SLSQP error: {minimizer_values["fh_e_slsqp"]}\n')

        f.write(f'job took {(end - start):.4f} seconds \n')
        f.write(f'nToy MC per 1-cl point: {args.nToy} \n')
        f.write(f'n of FH ponts for coverage: {int( (args.range[1] - args.range[0])/args.granularity )}')