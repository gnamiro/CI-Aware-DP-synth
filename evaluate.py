import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

# from functions import bucketize, bucketize_with_map
from evaluations.algorithms import *

# from utils.preprocess import *
from utils.visualization import plot_comparison_on_measures, plot_comparison_on_measures_no_privacy
from utils.evaluations import *
from utils.preprocess import *

import sys
import os

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(utils_path)

from Constraints import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no-save-evals', action='store_false', help='Save the evaluation metrics into excel')
parser.add_argument('--no-plot-evals', action='store_false', help='Plot evaluation metrics')
parser.add_argument('--method', type=str, default='LR', help="Evaluation ML method: LR, MLP, XGB, RF, SVM, NB, DT")
parser.add_argument('--with-private', action='store_true', help="Include Vanilla mst")
parser.add_argument('--eps', type=int, default=None, help='Epsilon value')

args = parser.parse_args()

if __name__ == "__main__":
    cv = 5
    eps = args.eps
    

    dfs = {}

    dfs['original'] = {'Original': pd.DataFrame(columns=MEASURES), 
                    'Dropped': pd.DataFrame(columns=MEASURES), 
                    'mst': pd.DataFrame(columns=MEASURES),
                    'hard': pd.DataFrame(columns=MEASURES),
                    'privCI': pd.DataFrame(columns=MEASURES),
                    'greedy': pd.DataFrame(columns=MEASURES),
                    'opt': pd.DataFrame(columns=MEASURES),
                    }
    # print(dfs)
    print("Starting the preprocessing of the datasets...")
    # We don't need preprocessing because we don't have any categorical attribute in this case!!
    ## MST postprocessing 

    
   
    print("Preprocessing is done...\n Now the data is ready to evaluate the performance of the methods!")

    evaluate(DATA_PATH, 'original', 'Original', dfs, cv, ML_ALGO, 0)

    if args.with_private:

        print("Starting Vanilla MST v1 Performance Evaulation...")
        print("Vanilla MST")
        file_name = 'results_mst_'
        evaluate_original(DATA_PATH, 'original', 'mst', 'mst', file_name, dfs, cv, ML_ALGO, 0, eps)
        print("********"*10)

        print("********"*10)
        print("Vanilla MST + Reg")
        file_name = 'results_mst_hard_'
        evaluate_original(DATA_PATH, 'original', 'hard', 'hard_constraint', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, eps)
        print("********"*10)

        print("********"*10)
        print("Vanilla MST + Reg")
        file_name = 'results_privCI_'
        evaluate_original(DATA_PATH, 'original', 'privCI', 'privCI', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, eps)
        print("********"*10)

        
        print("********"*10)
        print("Vanilla Greedy")
        file_name = 'results_greedy_'
        evaluate_original(DATA_PATH, 'original', 'greedy', 'greedy', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, eps)
        print("********"*10)

        # print("********"*10)
        # print("Vanilla OPT")
        # file_name = 'results_opt_'
        # evaluate_original(DATA_PATH, 'original', 'opt', 'opt', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, eps)
        # print("Vanilla MST Evaluation finished... \n Starting OTClean Performance Evaulation...")
    
    # SAVE results
    # if args.no_save_evals:
    #     save_data(dfs, f"res/{db_name}/{ML_ALGO}/{db_name}_{ML_ALGO}_eps={eps}_results_no_privacy.xlsx")

    #Visualization
    # if args.no_plot_evals:
    #     # if args.with_otclean:
    #     #     plot_comparison_on_MEASURES(dfs, MEASURES, True)
        # else:
    # plot_comparison_on_MEASURES_no_privacy(dfs, MEASURES, False, eps)
    #     # plot_comparison_on_MEASURES(dfs, MEASURES, False)

    
