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
from results_io import ensure_results_table

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no-save-evals', action='store_false', help='Save the evaluation metrics into excel')
parser.add_argument('--no-plot-evals', action='store_false', help='Plot evaluation metrics')
parser.add_argument('--method', type=str, default='LR', help="Evaluation ML method: LR, MLP, XGB, RF, SVM, NB, DT")
parser.add_argument('--with-private', action='store_true', help="Include Vanilla mst")
# parser.add_argument('--e', type=str, default=None, help='epsilon value')

args = parser.parse_args()

if __name__ == "__main__":
    cv = 5
    # eps = args.eps
    ensure_results_table(csv_path, methods=methods, cv=CV)

    for e in EPS:
        dfs = {}

        dfs['original'] = {'Original': pd.DataFrame(columns=MEASURES), 
                        'Dropped': pd.DataFrame(columns=MEASURES), 
                        'MST': pd.DataFrame(columns=MEASURES),
                        'HC': pd.DataFrame(columns=MEASURES),
                        'PrivCI': pd.DataFrame(columns=MEASURES),
                        'PrefairG': pd.DataFrame(columns=MEASURES),
                        'PrefairE': pd.DataFrame(columns=MEASURES),
                        }
        # print(dfs)
        print("Starting the preprocessing of the datasets...")
        # We don't need preprocessing because we don't have any categorical attribute in this case!!
        ## MST postprocessing 

        ML_ALGO = args.method
    
        print("Preprocessing is done...\n Now the data is ready to evaluate the performance of the methods!")

        evaluate(DATA_PATH, 'original', 'Original', dfs, cv, ML_ALGO, 0, eps=None)

        if args.with_private:

            print("Starting Vanilla MST v1 Performance Evaulation...")
            print("Vanilla MST")
            file_name = 'results_mst_'
            evaluate_original(DATA_PATH, 'original', 'MST', 'mst', file_name, dfs, cv, ML_ALGO, 0, e)
            print("********"*10)

            print("********"*10)
            print("Vanilla MST + Reg")
            file_name = 'results_mst_hard_'
            evaluate_original(DATA_PATH, 'original', 'HC', 'hard_constraint', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, e)
            print("********"*10)

            print("********"*10)
            print("Vanilla MST + Reg")
            file_name = 'results_privCI_'
            evaluate_original(DATA_PATH, 'original', 'PrivCI', 'privCI', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, e)
            print("********"*10)

            
            print("********"*10)
            print("Vanilla Greedy")
            file_name = 'results_greedy_'
            evaluate_original(DATA_PATH, 'original', 'PrefairG', 'greedy', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, e)
            print("********"*10)

            print("********"*10)
            print("Vanilla OPT")
            file_name = 'results_opt_'
            evaluate_original(DATA_PATH, 'original', 'PrefairE', 'opt', file_name, dfs, cv, ML_ALGO, DROP_PROTECTED_ATTRIBUTE, e)
            print("Vanilla MST Evaluation finished... \n Starting OTClean Performance Evaulation...")
        
    
