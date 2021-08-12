import argparse

parser = argparse.ArgumentParser(description='Script used to run on HTCondor confidence intervals using Antonio Cota Method',
                                 epilog='Test example: ./Run_CI_Job.py',
                                 add_help=True)

parser.add_argument ('--HELP', '-H', default=False, action='store_true', help='Print help message.')
parser.add_argument ('--version', '-v', default='test', help='Version name.')

parser.add_argument('--Bin', '-B', default=4, type=int, help='an integer for the Bin to analyze')
parser.add_argument('--nToy', '-N', default=1000, type=int, help='an integer for the number of toy MC to run for every 1-cl calculation (more takes longer but has a smooth 1-cl graph)')


args = parser.parse_args()
if args.HELP:
    parser.print_help()
    exit()

def analyzeBin(nBin):
    

if __name__ == "__main__":
    print(f'analyzing bin: {args.Bin}')
    print('main ok')