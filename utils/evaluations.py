import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
from preprocess import *
from evaluations.algorithms import *

import sys
import os

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

from Constraints import *
from results_io import set_metric

def evaluate(path, algorithm, method, dfs, cv, ml_algo, proc_dop, eps=None):
    for i in range(cv):
        train_path, test_path = f'{path}/cs={i}/train.csv',  f'{path}/cs={i}/test.csv'
        num_iter = 1
        new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=proc_dop, dist=0)
        set_metric(csv_path, method_name=method, fold_num=i, epsilon=eps, metric_name=f'{ml_algo}_AUC', value=new_df['AUC'][0])
        dfs[algorithm][method] = pd.concat([dfs[algorithm][method], new_df], axis=0)
    print(dfs[algorithm][method].mean(axis=0))
    print(dfs[algorithm][method].std(axis=0))

def evaluate_original(path, algorithm, method, folder, file_name, dfs, cv, ml_algo, proc_drop, eps=None):

    for i in range(cv):
        print(f"for {algorithm} -- fold {i} evaluating the experiment")
        train_path, test_path = None, None
        if eps is None:
            train_path, test_path = f'{path}/cs={i}/{folder}/{file_name}{i}.csv', f'{path}/cs={i}/test.csv'
        else:
            train_path, test_path = f'{path}/cs={i}/{folder}/eps={eps}/{file_name}{i}.csv', f'{path}/cs={i}/test.csv'

        print(method, file_name, train_path, test_path)     
        num_iter = i
        new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=proc_drop, dist=0)

        set_metric(csv_path, method_name=method, fold_num=i, epsilon=eps, metric_name=f'{ml_algo}_AUC', value=new_df['AUC'][0])

        dfs[algorithm][method] = pd.concat([dfs[algorithm][method], new_df], axis=0)
        print("((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))")
    print(dfs[algorithm][method].mean(axis=0))
    print(dfs[algorithm][method].std(axis=0))

def evaluate_ot(path, algorithm, method, dfs, cv, ml_algo, eps):
    for i in range(cv):
        # train_path = f'{path}/cs={i}/otclean/train_dummified_dist.csv'
        train_path = f'{path}/cs={i}/otclean/eps={eps}/clean_train_dist.csv'
        test_path  = f'{path}/cs={i}/train_label.csv' if method == 'otclean_normal' else f'{path}/cs={i}/otclean/eps={eps}/otclean_test.csv'
        num_iter = 1
        new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=0, dist=1)

        dfs[algorithm][method] = pd.concat([dfs[algorithm][method], new_df], axis=0)
    print(dfs[algorithm][method].mean(axis=0))
    print(dfs[algorithm][method].std(axis=0))

# def evaluate_ot_my(path, algorithm, method, dfs, cv, ml_algo):
#     for i in range(1, cv+1):
#         train_path = f'{path}/data/{i}/otclean_my_{i}_dist.csv'
#         test_path  = f'{path}/cs={i}/test.csv' if method == 'otclean_normal' else f'{path}/data/{i}/clean_my_test.csv'
#         num_iter = 1
#         new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=drop_protected_attribute, dist=1)

#         dfs[algorithm][method] = pd.concat([dfs[algorithm][method], new_df], axis=0)
#     print(dfs[algorithm][method].mean(axis=0))
#     print(dfs[algorithm][method].std(axis=0))


def evaluate_dp(path, algorithm, file_name, dfs, cv, ml_algo, proc_drop):
    # print(dfs[algorithm])
    for e in eps:
        print('------------------------------------------------------')
        print(f'---------------------epsilon:{e}---------------------')
        print(f'---------------------epsilon:{e}---------------------')
        for i in range(cv):
            # print(algorithm ,e)
            test_path, train_path = f'{path}/cs={i}/test.csv',  f'{path}/cs={i}/{algorithm}/eps={e}/{file_name}{i}.csv'
            num_iter = 1
            new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=proc_drop, dist=0)

            dfs[algorithm][e] = pd.concat([dfs[algorithm][e], new_df], axis=0)
        print(dfs[algorithm][e].mean(axis=0))
        print(dfs[algorithm][e].std(axis=0))
    return dfs

def evaluate_dp_ot(path, algorithm, method, dfs, cv, ml_algo):
    dfs_method = 'mst_otclean_normal' if method == 'normal' else 'mst_otclean_clean'
    for e in eps:
        print('------------------------------------------------------')
        print(f'---------------------epsilon:{e}---------------------')
        print(f'---------------------epsilon:{e}---------------------')
        for i in range(cv):
            print(f'------------------------cv={i}--------------------------')
            train_path = f'{path}/cs={i}/{algorithm}/eps={e}/clean_train_dist.csv'
            test_path  = f'{path}/cs={i}/test.csv' if method == 'normal' else f'{path}/cs={i}/{algorithm}/eps={e}/otclean_test.csv'
        
            num_iter = 1
            new_df = ml_predict(train_path, test_path, ml_algo, num_iter, proc_drop=drop_protected_attribute, dist=1)

            dfs[dfs_method][e] = pd.concat([dfs[dfs_method][e], new_df], axis=0)
        print(dfs[dfs_method][e].mean(axis=0))
        print(dfs[dfs_method][e].std(axis=0))
    return dfs
