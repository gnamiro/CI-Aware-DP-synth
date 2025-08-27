import numpy as np
import pandas as pd
import itertools
import os
import pickle

import ot
from tqdm import tqdm

# Assumed imports from your existing codebase
from Constraints import *
from visualization import plot_distance

# Device setup for torch (optional)
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
import ot  # Python Optimal Transport library
import itertools
import json


import argparse

debugging = True

def convert_to_dist(df, selected_columns, target_domain):
    df = df[selected_columns]
    
    # Get probabilities
    p_x_plain = dict(df.value_counts(normalize=True))
    
    initial_distribution = []
    for x in target_domain:
        initial_distribution.append(p_x_plain.get(x, 0))
    
    return np.array(initial_distribution)


def calculate_domain(df, selected_columns):
    unique_values = [df[col].unique() for col in selected_columns]
    all_combinations = list(itertools.product(*unique_values))

    all_combinations_flat = [
        tuple(itertools.chain(*map(tuple, [comb])))
        for comb in all_combinations
    ]

    domain_df = pd.DataFrame(all_combinations_flat, columns=selected_columns)
    return domain_df

def sinkhorn_distance(orig_dist, synth_dist, domain, epsilon=0.1):
    orig_dist += 1e-8
    synth_dist += 1e-8
    orig_dist /= np.sum(orig_dist)
    synth_dist /= np.sum(synth_dist)
    print("orig_dist sum:", np.sum(orig_dist), "min:", np.min(orig_dist), "max:", np.max(orig_dist))
    print("synth_dist sum:", np.sum(synth_dist), "min:", np.min(synth_dist), "max:", np.max(synth_dist))
    print("non-zero entries in orig_dist:", np.count_nonzero(orig_dist))
    print("non-zero entries in synth_dist:", np.count_nonzero(synth_dist))



    domain_tensor = torch.from_numpy(domain).float()
    C = torch.cdist(domain_tensor, domain_tensor, p=2).cpu().numpy().astype(np.float64)
    print("Cost matrix C:")
    print("Shape:", C.shape)
    print("Min:", np.min(C), "Max:", np.max(C), "Mean:", np.mean(C))
    C = (C + C.T) / 2  # Ensure symmetry 
    C /= np.max(C)  # Normalize distances, avoid minimal values

    try:
        dist = ot.sinkhorn2(orig_dist, synth_dist, C, reg=epsilon, stopThr=1e-6)
    except Exception as e:
        print("Sinkhorn failed:", e)
        dist = np.nan

    return dist

def wasserstein_distance_dist(orig_dist, synth_dist, domain):
    domain_tensor = torch.from_numpy(domain).to(device)
    M = torch.cdist(domain_tensor, domain_tensor, p=2).cpu().numpy().astype(np.float64)

    orig_dist /= np.sum(orig_dist)
    synth_dist /= np.sum(synth_dist)

    return ot.lp.emd2(orig_dist, synth_dist, M, numItermax=10000000)


def kl_divergence(p_dist, q_dist, epsilon=1e-12):
    # measure the KL divergence from q to p
    p = np.array(p_dist) + epsilon
    q = np.array(q_dist) + epsilon

    kl_pq = np.sum(p * np.log(p / q))
    # kl_qp = np.sum(q * np.log(q / p))

    return kl_pq


def total_variation(p_dist, q_dist):
    return 0.5 * np.sum(np.abs(p_dist - q_dist))


def mmd_rbf(p_dist, q_dist, domain_array, gamma=1.0):
    """
    Compute MMD (Maximum Mean Discrepancy) between two distributions using an RBF kernel.

    Parameters:
        p_dist (np.array): Probability distribution over the domain (e.g., original).
        q_dist (np.array): Another probability distribution (e.g., synthetic).
        domain_array (np.ndarray): Shared domain, shape (n_samples, n_features).
        gamma (float): RBF kernel parameter (1 / (2 * sigma^2)).

    Returns:
        float: MMD distance.
    """
    # Compute kernel matrices
    K = rbf_kernel(domain_array, domain_array, gamma=gamma)

    # MMD^2 formula using distributions as weights
    mmd2 = (
        p_dist @ K @ p_dist
        + q_dist @ K @ q_dist
        - 2 * p_dist @ K @ q_dist
    )

    return np.sqrt(max(mmd2, 0))  # Ensure non-negativity

#*************************************
#*******Wasserstein dist********
#*************************************
def format_to_1(x):
    if len(x) == 4:
        return x[0] + x[1] + x[2] + x[3]
    return x[0] + x[1] + x[2]

def normalize(df, independence_columns, normalization_coeffs={'S':1.0,'N':1.0,'Y':1.0,'A':1.0}, divide_by_std=True):
    if divide_by_std:
        std = df.std()
        for i in df.columns:
            if std[i] != 0:
                df[i] = df[i].apply(lambda x: x / std[i])

    for i in independence_columns[2]:
        df[i] = df[i].apply(lambda x: x * normalization_coeffs['A'])#0.2
    for i in independence_columns[1]:
        df[i] = df[i].apply(lambda x: x * normalization_coeffs['Y'])#0.5
    for i, col in enumerate(independence_columns[0]):
        if i == 0:
            df[col] = df[col].apply(lambda x: x * normalization_coeffs['S'])
        else:
            df[col] = df[col].apply(lambda x: x * normalization_coeffs['N'])
    
    for i in df.columns:
        if i != 'TEMP':
            df[i] = df[i].apply(lambda x: np.round(x,4))
    return df

def wasserstein_distance_(df_1, df_2, independence_columns):
    df_1 = df_1[format_to_1(independence_columns)]
    df_2 = df_2[format_to_1(independence_columns)]
    
    p_x_1 = dict(df_1.value_counts(normalize=True))
    p_x_2 = dict(df_2.value_counts(normalize=True))

    df_1_domain_unnormalized = list(p_x_1.keys())
    df_2_domain_unnormalized = list(p_x_2.keys())

    std_1 = df_1.std()
    std_2 = df_2.std()
    df_1 = normalize(df_1, independence_columns)
    df_2 = normalize(df_2, independence_columns)
    p_x_1 = dict(df_1.value_counts(normalize=True))
    p_x_2 = dict(df_2.value_counts(normalize=True))

    df_1_domain = list(p_x_1.keys())
    df_2_domain = list(p_x_2.keys())

    distribution_1 = []
    for x in df_1_domain:
        distribution_1.append(p_x_1[tuple(x)])

    distribution_2 = []
    for x in df_2_domain:
        distribution_2.append(p_x_2[tuple(x)])

    df_1_domain_tensor = torch.tensor(df_1_domain).to(device)
    df_2_domain_tensor = torch.tensor(df_2_domain).to(device)
    M = torch.cdist(df_1_domain_tensor, df_2_domain_tensor, p=2).cpu().numpy().astype(np.float64)

    return ot.lp.emd2(distribution_1, distribution_2, M, numItermax=1000000)

def wasserstein_distance_fixed_domain(df_1, df_2, domain, independence_columns):
    df_1 = df_1[format_to_1(independence_columns)]
    df_2 = df_2[format_to_1(independence_columns)]
    
    # Round or normalize only if used in first version
    df_1 = normalize(df_1.copy(), independence_columns, divide_by_std=False)
    df_2 = normalize(df_2.copy(), independence_columns, divide_by_std=False)

    domain = [tuple(x) for x in domain]
    
    p_x_1 = df_1.value_counts(normalize=True)
    p_x_2 = df_2.value_counts(normalize=True)

    dist_1 = np.array([p_x_1.get(tuple(x), 0.0) for x in domain])
    dist_2 = np.array([p_x_2.get(tuple(x), 0.0) for x in domain])

    domain_tensor = torch.tensor(domain).float().to(device)
    M = torch.cdist(domain_tensor, domain_tensor, p=2).cpu().numpy()

    return ot.lp.emd2(dist_1, dist_2, M)

def empirical_wasserstein(df1, df2, attributes, n_bins=100, method='histogram'):
    """
    Compute Wasserstein distance using empirical distributions
    
    Parameters:
    n_bins (int): Number of bins for histogram-based methods
    method (str): 'histogram' or 'kde' (Kernel Density Estimate)
    """
    distances = []
    
    for attr in attributes:
        # Handle NaNs
        data1 = df1[attr].dropna().values
        data2 = df2[attr].dropna().values
        
        # Create empirical distributions
        if method == 'histogram':
            # Quantile-based bin edges for better coverage
            combined = np.concatenate([data1, data2])
            bin_edges = np.quantile(combined, np.linspace(0, 1, n_bins+1))
            
            hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
            hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
        elif method == 'kde':
            # Kernel Density Estimation
            kde1 = gaussian_kde(data1)
            kde2 = gaussian_kde(data2)
            x = np.linspace(min(combined), max(combined), n_bins)
            hist1 = kde1(x)
            hist2 = kde2(x)
        else:
            raise ValueError("Invalid method. Choose 'histogram' or 'kde'")

        # Compute Wasserstein distance between distributions
        distances.append(wasserstein_distance(hist1, hist2))
    
    return sum(distances)  # Sum of 1D distances



def weights_to_dist(weighted_df, selected_columns, full_domain):
    df = weighted_df[selected_columns]
    weights = weighted_df['repaired_weights'].values
    point_tuples = [tuple(row) for row in df.values]
    
    weight_dict = {}
    for pt, w in zip(point_tuples, weights):
        weight_dict[pt] = weight_dict.get(pt, 0) + w
    
    dist = []
    for x in full_domain:
        dist.append(weight_dict.get(x, 0))
    
    dist = np.array(dist, dtype=np.float64)
    dist /= dist.sum()  # Normalize
    return dist


def load_datasets_for_fold(data_path, i, eps):
    """
    Load datasets for cross-validation iteration `i` across multiple `j` variations.
    """
    if eps is None:
        datasets = {
                    "orig": pd.read_csv(f"{data_path}/cs={i}/train_label.csv"),
                    "greedy": pd.read_csv(f"{data_path}/cs={i}/greedy/results_greedy_{i}.csv"),
                    "opt": pd.read_csv(f"{data_path}/cs={i}/opt/results_opt_{i}.csv"),
                    'mst': pd.read_csv(f"{data_path}/cs={i}/mst/results_mst_{i}.csv"),
                    # 'ours_cmi': pd.read_csv(f"{data_path}/cs={i}/ours_cmi/results_mst_pmd_cmi_{i}.csv"), 
                    # 'ours_mmd': pd.read_csv(f"{data_path}/cs={i}/ours_mmd/results_mst_pmd_mmd_{i}.csv"), 
                    'ours_tvd_L2': pd.read_csv(f"{data_path}/cs={i}/tvd_L2/results_mst_pmd_tvd_L2_{i}.csv"),
                    'mst+otclean': pd.read_csv(f"{data_path}/cs={i}/otclean/clean_train_dist.csv"),
                }
        return datasets
    else:
        datasets = {
                    # "orig": pd.read_csv(f"{data_path}/cs={i}/train_label.csv"),
                    "greedy": pd.read_csv(f"{data_path}/cs={i}/greedy/eps={eps}/results_greedy_{i}.csv"),
                    # "greedy_2": pd.read_csv(f"{data_path}/cs={i}/greedy/eps={eps}/results_greedy_same_size_{i}.csv"),
                    # "opt": pd.read_csv(f"{data_path}/cs={i}/opt/eps={eps}/results_opt_{i}.csv"),
                    # "opt_2": pd.read_csv(f"{data_path}/cs={i}/opt/eps={eps}/results_opt_same_size_{i}.csv"),
                    'mst': pd.read_csv(f"{data_path}/cs={i}/mst/eps={eps}/results_mst_{i}.csv"),
                    # 'mst_2': pd.read_csv(f"{data_path}/cs={i}/mst/eps={eps}/results_mst_same_size_{i}.csv"),
                    # 'ours_cmi': pd.read_csv(f"{data_path}/cs={i}/cmi/eps={eps}/results_mst_cmi_{i}.csv"), 
                    # 'ours_cmi_2': pd.read_csv(f"{data_path}/cs={i}/cmi/eps={eps}/results_mst_cmi_weighted_{i}.csv"), 
                    # 'ours_cmi_3': pd.read_csv(f"{data_path}/cs={i}/cmi/eps={eps}/results_mst_cmi_weighted_optim_{i}.csv"), 
                    # # 'ours_mmd': pd.read_csv(f"{data_path}/cs={i}/ours_mmd/eps={eps}/results_mst_pmd_mmd_{i}.csv"), 
                    # 'ours_tvd_L2': pd.read_csv(f"{data_path}/cs={i}/tvd_L2/eps={eps}/results_mst_pmd_tvd_L2_{i}.csv"),
                    # 'ours_tvd_L2_2': pd.read_csv(f"{data_path}/cs={i}/tvd_L2/eps={eps}/results_mst_tvd_L2_same_size_{i}.csv"),
                    'privCI': pd.read_csv(f"{data_path}/cs={i}/privCI/eps={eps}/results_privCI_{i}.csv"),
                    # 'ours_tvd_L2': pd.read_csv(f"{data_path}/cs={i}/tvd_L2/eps={eps}/results_tvd_L2_same_space_{i}.csv"),
                    'hard_constraint': pd.read_csv(f"{data_path}/cs={i}/hard_constraint/eps={eps}/results_mst_hard_{i}.csv"),
                    # 'mst+otclean': pd.read_csv(f"{data_path}/cs={i}/otclean/eps={eps}/clean_train_dist.csv"),
                    # 'mst+otclean+': pd.read_csv(f"{data_path}/cs={i}/otclean/eps={eps}/clean_train_dist.csv"),
                }
        return datasets
import json

def evaluate_datasets(data_path, cv=5, distance_metric='wasserstein', eps=None):
    results = {
        # "orig": [],
        "mst": [],
        # "mst_2": [],
        # "ours_tvd_L2": [],
        # "ours_tvd_L2_2": [],
        "hard_constraint": [],
        "privCI": [],
        # "ours_cmi": [],
        # "ours_cmi_2": [],
        # "ours_cmi_3": [],
        "greedy": [],
        # "greedy_2": [],
        # "opt": [],
        # "opt_2": [],
        # "mst+otclean": [],
        # "mst+otclean+": []
    }

    # Load domain from JSON file
    with open(f"{data_path}/domain.json", "r") as f:
        domain_info = json.load(f)

    selected_columns = PROTECTED_ATTRS + INADMISSIBLE_ATTRS + OUTCOME + ADMISSIBLE_ATTRS
    assert set(selected_columns).issubset(domain_info), \
        f"Some selected columns are missing in domain.json: {set(selected_columns) - set(domain_info)}"

    domain_ranges = [list(range(domain_info[col])) for col in selected_columns]
    target_domain = list(itertools.product(*domain_ranges))
    domain_array = np.array(target_domain, dtype=np.float64)

    for i in range(cv):
        print(f"Evaluating fold {i}...")

        datasets = load_datasets_for_fold(data_path, i, eps)

        test_df = pd.read_csv(f"{data_path}/cs={i}/test.csv")
        train_df = pd.read_csv(f"{data_path}/cs={i}/train.csv")
        mst_df = pd.read_csv(f"{data_path}/cs={i}/mst/eps={eps}/results_mst_{i}.csv")
        orig_df = pd.concat([train_df, test_df], ignore_index=True)

        orig_dist = convert_to_dist(orig_df, selected_columns, target_domain)
        train_dist = convert_to_dist(train_df, selected_columns, target_domain)
        mst_dist = convert_to_dist(mst_df, selected_columns, target_domain)

        for model_name, synth_df in datasets.items():
            # if model_name == 'mst+otclean':
            #     synth_dist = weights_to_dist(synth_df, selected_columns, target_domain)
            # elif model_name == 'mst+otclean+':
            #     synth_dist = convert_to_dist(synth_df, selected_columns, target_domain)
            # else:
            synth_dist = convert_to_dist(synth_df, selected_columns, target_domain)

            if distance_metric == 'w':
                distance = wasserstein_distance_dist(train_dist, synth_dist, domain_array)
            elif distance_metric == 'Wa':
                distance = wasserstein_distance_(train_df, synth_df, CONSTRAINT)
            elif distance_metric == 'kl':
                distance = kl_divergence(train_dist, synth_dist)
            elif distance_metric == 'tvd':
                distance = total_variation(train_dist, synth_dist)
            elif distance_metric == 'MMD':
                distance = mmd_rbf(mst_dist, synth_dist, domain_array)
            elif distance_metric == 'sinkhorn':
                distance = sinkhorn_distance(train_df, synth_df, domain_array)
            else:
                raise ValueError("Unsupported distance metric")

            print(f"Fold {i} | {model_name} | {distance_metric}: {distance:.4f}")
            results[model_name].append(distance)

    return results



if __name__ == '__main__':
    # Example usage

    parser = argparse.ArgumentParser(description='Run Cross-Validation for CI Tests.')
    
    parser.add_argument('--dist_type', type=str, choices=['kl', 'w', 'tvd', 'Wa', 'MMD'], default='W', help='Distance type')
    parser.add_argument('--eps', type=float, default=None, help='Epsilon value')
    
    args = parser.parse_args()

    metric = args.dist_type
    eps = args.eps

    # for metric in distance_metrics:
    print(f"Running {metric} distance evaluation...")
    summary_results = evaluate_datasets(DATA_PATH, cv=5, distance_metric=metric, eps=eps)

    print("\nResults Summary")
    for model, stats in summary_results.items():
        print(f"{model} -- Mean: {np.mean(stats):.4f}, Std: {np.std(stats):.4f}")

    # plot_distance(summary_results, metric, f'res/{db_name}/eps={args.eps}/{db_name}_{metric}.jpg')
