
import sys
import os
sys.path.append('./mechanisms')

import pandas as pd
import numpy as np
from mbi import Dataset
import mst as mst

import time

degree = 2
num_marginals = None
max_cells = 100000

delta = 1e-9
cv = [0, 1, 2, 3, 4]

# FRACTIONS = [0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
eps = [0.1, 1, 10]

summary_records = []
data_path = '../data/dutch'

for i in cv:
    for e in eps:
        dataset = f'{data_path}/cs={i}/train.csv'
        domain = f'{data_path}/domain.json'

        print(f"[INFO] cv:{i} and eps:{e}")
        data = Dataset.load(dataset, domain)
        start_time = time.time()
        model, decode_fn = mst.MST(data, e, delta)
        synth = model.synthetic_data()
        synth = decode_fn(synth)
        data = synth.df

        elapsed = time.time()

        save_path = f"{data_path}/cs={i}/mst/eps={str(e)}/results_mst_{i}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False);
