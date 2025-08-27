
import sys
sys.path.append('./Prefair/mechanisms')

import mst_fair_greedy as fairMST
import mst_fair_optimal as fairMSTOpt

import numpy as np
from mbi import Dataset

import time

import sys
import os

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils'))
sys.path.append(utils_path)

from Constraints import *
from utility_test import test_utility
import numpy as np


degree = 2
num_marginals = None
max_cells = 100000

eps = [0.1, 1, 10]
type = 'G' # Define your type of Prefair here --> G or E
cv = 0
for i in range(0, 5):
    for e in eps:
        print('cv:', str(i), " -- ", 'eps:', str(e))
        dataset = f'{DATA_PATH}/cs={i}/train.csv'
        domain = f'{DATA_DOMAIN_PATH}'
        data = Dataset.load(dataset, domain)
        start_time = time.time()

        if type.lower() == 'g':
            synth = fairMST.MST(data, e, DELTA, CONSTRAINT[1], CONSTRAINT[2]) ## Prefair Greedy
        else:
            synth = fairMSTOpt.MST(data, e, DELTA, CONSTRAINT[1], CONSTRAINT[2], CONSTRAINT[0]) ## Prefair Exponential 
        end_time = time.time()
        print(f"[DEBUG] the time it took for cv:{i}, eps:{e} --> time: {end_time - start_time}")
        
        if type.lower() == 'g':
            save_path = f"{DATA_PATH}/cs={i}/greedy/eps=" +str(e)+"/results_greedy_" + str(i) + ".csv" ## Prefair Greedy
        else:
            save_path = f"{DATA_PATH}/cs={i}/opt/eps=" +str(e)+"/results_opt_" + str(i) + ".csv" ## Prefair Exponential

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        synth.df.to_csv(save_path, index=False)
