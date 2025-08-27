
import sys
import os
sys.path.append('./mechanisms')
sys.path.append('./src/mbi')

import pandas as pd
import numpy as np
from mbi import Dataset

import mst as mst
import our_mst as cmst
import mst_constraint as hard_mst

import time




# MST parameters
degree = 2
num_marginals = None
max_cells = 100000
delta = 1e-9


## Define you dataset and constraints here
data_path = '../data/dutch'
TARGET_ATTR = 'occupation'
protected_attr = 'sex'
admissible = ['economic_status', 'household_position', 'household_size']
inadmissible = ['edu_level', 'age', 'marital_status', 'country_birth', 'citizenship', 'prev_residence_place']
outcome = [TARGET_ATTR]
protected = [protected_attr]
CONSTRAINT = [protected, outcome, admissible]


## Define your type of method here --> privci or HC
type = 'HC'


## Define privacy budget here
eps = [0.1, 1, 10]

## Define cross validation folds here
cv = [0, 1, 2, 3, 4]


for i in cv:
    for e in eps:

        for cmi_ratio in np.arange(0.01, 0.02, 0.05): # How much you want to decrease the CMI compared to the original CMI
            dataset = f'{data_path}/cs={i}/train.csv'
            domain = f'{data_path}/domain.json'

            degree = 500 # PrivCI lambda coeff degree 

            print(f"[INFO] cv:{i} and eps:{e} with cmi: {cmi_ratio}, degree: {degree}")
            data = Dataset.load(dataset, domain)

            if type.lower() == 'privci':
                # PrivCI
                # In case you are using PrivCI, make sure to apply the constraints in the 
                # `calculate_regularizer_loss` method in src/mbi/estimation.py file
                model, decode_fn, logs = cmst.MST(data, e, delta, 
                                            cmi_value=cmi_ratio, 
                                            degree=degree, 
                                            proc_attr=protected_attr)

            else:
                # Hard Constraint
                model, decode_fn = hard_mst.MST(data, e, delta, 
                                                protected=protected, 
                                                outcome=outcome, 
                                                admissible=admissible, 
                                                inadmissible=inadmissible)

            synth = model.synthetic_data()
            synth = decode_fn(synth)
            data = synth.df
            if type.lower() == 'privci':
                save_path = f"{data_path}/cs={i}/privCI/eps={str(e)}/results_privCI_{i}.csv"
            else:
                save_path = f"{data_path}/cs={i}/hard_constraint/eps={str(e)}/results_mst_hard_{i}.csv"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path, index=False);
            
