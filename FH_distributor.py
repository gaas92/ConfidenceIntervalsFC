import json

import argparse

parser = argparse.ArgumentParser(description='Script used to run on HTCondor confidence intervals using Antonio Cota Method',
                                 epilog='Test example: python FH_distributor.py',
                                 add_help=True)

parser.add_argument ('--HELP',    '-H', default=False,  action='store_true', help='Print help message.')
parser.add_argument ('--version', '-v', default='test', help='Version name.')

parser.add_argument('--Nfh',     '-N', default=50,   type=int, help='an integer for the Number of jobs/FH points to generate')
parser.add_argument('--Range',   '-R', default='full',    type=str, choices=['full', '1sigma', '2sigma', '3sigma'], help='str to define the FH range to generate')
args = parser.parse_args()

if args.HELP:
    parser.print_help()
    exit()



if __name__ == "__main__":
    
    for i in range(1,8):
        print(f'in Bin {i}')
        f = open(f'Bin{i}/fitParams.json',)
        data = json.load(f)
        fh = data['Signal']['angle']['FH']
        fhe = data['Signal']['angle']['FH_error']
        print(f'FH = {fh} +/- {fhe}')
        if args.Range == 'full':
            range_ = [0.0, 3.0]
        elif args.Range == '1sigma':
            range_ = [0.0 if fh - fhe < 0 else fh - fhe, 
                      3.0 if fh + fhe > 3 else fh + fhe]
        elif args.Range == '2sigma':
            range_ = [0.0 if fh - 2*fhe < 0 else fh - 2*fhe, 
                      3.0 if fh + 2*fhe > 3 else fh + 2*fhe]
        elif args.Range == '3sigma':
            range_ = [0.0 if fh - 3*fhe < 0 else fh - 3*fhe, 
                      3.0 if fh + 3*fhe > 3 else fh + 3*fhe]
        else :
            range_ = []
            print('range not valid !')
            break
        
        f.close()
        step_size = (range_[1] - range_[0]) / args.Nfh

        step_list = []
        for j in range(args.Nfh+1):
            this_element = f'{j} {range_[0] + j*step_size }'
            step_list.append(this_element)

        with open(f'Bin{i}/stepsFile.txt', 'w') as f:
            for item in step_list:
                f.write("%s\n" % item)

    print('main ok')