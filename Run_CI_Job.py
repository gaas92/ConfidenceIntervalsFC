import argparse
import json 
from timeit import default_timer as timer

import zfit
zfit.util.cache.clear_graph_cache()


parser = argparse.ArgumentParser(description='Script used to run on HTCondor confidence intervals using Antonio Cota Method',
                                 epilog='Test example: ./Run_CI_Job.py',
                                 add_help=True)

parser.add_argument ('--HELP',    '-H', default=False,  action='store_true', help='Print help message.')
parser.add_argument ('--version', '-v', default='test', help='Version name.')

parser.add_argument('--Bin',  '-B', default=4,    type=int, help='an integer for the Bin to analyze')
parser.add_argument('--nToy', '-N', default=1000, type=int, help='an integer for the number of toy MC to run for every 1-cl calculation (more takes longer but has a smooth 1-cl graph)')
parser.add_argument('--FH', '-FH',  default=0.0, type=float, help='floating fh true-value to calculate 1-cl')
parser.add_argument('--Save', '-S', default=1, type=int, help='save in CERNBOX, false only for testing')

args = parser.parse_args()
if args.HELP:
    parser.print_help()
    exit()


def analyzeFH():
    from confidence_intervals import cl_function
    import pandas as pd
    with open(f'Bin{args.Bin}/fitParams.json') as f:
        params = json.load(f)
    

    real_data_df = pd.read_csv(f'Bin{args.Bin}/Data.csv')
    real_data = zfit.Data.from_pandas(real_data_df)

    fh, one_cl = cl_function(FH=args.FH, params=params, real_data=real_data)
    

    if args.Save == 1 :
        None
    else :
        print('False Save')
        print(f'FH = {fh} , 1-cl = {one_cl}')


if __name__ == "__main__":
    if args.Save == 1 :
        None
    else :
        print('False Save')
    start = timer()
    analyzeFH()
    end = timer()
    print(f'took {(end - start):.4f} seconds')
    print('main ok')