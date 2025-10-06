
import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, 'src')
MECH_DIR = os.path.join(THIS_DIR, 'mechanisms')

for p in [SRC_DIR, MECH_DIR]:
    if p not in sys.path:
        sys.path.append(p)

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)
from Constraints import *


import pandas as pd
import numpy as np
from mbi import Dataset

import mst as mst
import our_mst as cmst
import mst_constraint as hard_mst

import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='HC', help="PrivCI with different methods: PrivCI or HC")



# MST parameters
degree = 2
num_marginals = None
max_cells = 100000
delta = 1e-9


## Define privacy budget here
# eps = [0.1, 1, 10]

## Define cross validation folds here
cv = [0, 1, 2, 3, 4]
if __name__ == "__main__":
    args = parser.parse_args()
    type = args.method
    print(f"[INFO] You are using the {type} method!")

    for i in cv:
        for e in EPS:

            for cmi_ratio in np.arange(0.001, 0.02, 0.05): # How much you want to decrease the CMI compared to the original CMI
                dataset = f'{DATA_PATH}/cs={i}/train.csv'
                domain = f'{DATA_PATH}/domain.json'

                degree = 100 # PrivCI lambda coeff degree 

                print(f"\n\n[INFO] Our method --> cv:{i} and eps:{e} with cmi: {cmi_ratio}, degree: {degree}")
                data = Dataset.load(dataset, domain)

                if type.lower() == 'privci':
                    # PrivCI
                    # In case you are using PrivCI, make sure to apply the constraints in the 
                    # `calculate_regularizer_loss` method in src/mbi/estimation.py file
                    model, decode_fn, logs = cmst.MST(data, e, delta, 
                                                cmi_value=cmi_ratio, 
                                                degree=degree, 
                                                proc_attr=PROTECTED_ATTR)

                else:
                    # Hard Constraint
                    model, decode_fn = hard_mst.MST(data, e, delta, 
                                                    protected=PROTECTED_ATTRS, 
                                                    outcome=OUTCOME, 
                                                    admissible=ADMISSIBLE_ATTRS, 
                                                    inadmissible=INADMISSIBLE_ATTRS)

                synth = model.synthetic_data()
                synth = decode_fn(synth)
                data = synth.df
                if type.lower() == 'privci':
                    save_path = f"{DATA_PATH}/cs={i}/privCI/eps={str(e)}/results_privCI_{i}.csv"
                else:
                    save_path = f"{DATA_PATH}/cs={i}/hard_constraint/eps={str(e)}/results_mst_hard_{i}.csv"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                data.to_csv(save_path, index=False);
                
