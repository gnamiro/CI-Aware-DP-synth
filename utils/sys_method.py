import os
import argparse

from Constraints import *
parser = argparse.ArgumentParser()
parser.add_argument('--algo-vals', nargs='+', type=str, help='Save the names of methods that we want to create a directory for')
parser.add_argument('--eps-vals', nargs='+', type=str, help='Save the epsilon values')
# parser.add_argument('--plot-evals', action='store_true', help='Plot evaluation metrics')


# eps = [0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.5, 0.75, 1, 10, 100, 1000, 10000, 10000000]
# eps = [0.001, 0.01, 0.1, 0.5, 0.75, 1, 10, 100, 1000, 10000000]
def create_dir(algorithms, cv, eps, main_folder):
    for i in range(cv):
        for alg in algorithms:
            # if alg not in ['otclean', 'vanilla_mst', 'ours_cmi', 'ours_mmd', 'ours_tvd_L1', 'ours_tvd_L2', 'vanilla_greedy', 'vanilla_opt', 'independent_coupling']:
            for e in eps:
                for f in FRACTIONS:   
                # print(e)
                    os.makedirs(f'{main_folder}/cs={i}/frac={f}/{alg}/eps={e}', exist_ok=True)
                    # for mech in ['KNN', 'MF']:
                        # os.makedirs(f'{main_folder}/cs={i}/frac={f}/{mech}/{alg}/eps={e}', exist_ok=True)
            # else:
            #     os.makedirs(f'{main_folder}/cs={i}/{alg}', exist_ok=True)


if __name__ == '__main__':
    args = parser.parse_args()
    algorithms =  args.algo_vals
    # eps = args.eps_vals
    # main_folder = f'../{data_path}'
    main_folder = f'{DATA_PATH}'
    # algorithms = ['mst', 'fair_greedy', 'fair_opt']
    create_dir(algorithms=algorithms, cv=5, eps=EPS, main_folder=main_folder)
