import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys
import seaborn as sns

from Constraints import *

# Define colors for specific methods (slightly different shades)
method_colors = {
    "mst": "#1F77B4",         # Blue
    "mst_reg": "#FF7F0E",     # Orange
    "fair_greedy": "#2CA02C", # Green
    "fair_opt": "#D62728",    # Red
    "original": "#9467BD",    # Purple
    "vanilla_mst": "#8C564B", # Brown
    "extra_1": "#E377C2",     # Pink
    "extra_2": "#7F7F7F"      # Gray
}


def plot_comparison_on_measures(dfs, measures, OTOnly):
    index = np.arange(len(eps))
    if not os.path.exists('res/' + db_name + '/' + ML_algo):
        os.makedirs('res/' + db_name + '/' + ML_algo)

    for measure in measures:
        plt.figure(figsize=(13, 6))
        plt.title(f'Mean Cross Validation Results for {measure}')

        if not OTOnly:
            for method, eps_df in dfs.items():
                if method == 'original':
                    continue
                mean_values = []
                std_values = []

                for epsilon, df in eps_df.items():

                    mean = df[measure].mean()
                    std = df[measure].std()

                    mean_values.append(mean)
                    std_values.append(std)

                mean_values = np.array(mean_values)
                std_values = np.array(std_values)

                 # Use the specified color for each method
                color = method_colors.get(method, "gray")  # Default to gray if method not in dictionary
                plt.errorbar(
                    eps,
                    mean_values,
                    yerr=std_values,
                    label=method,
                    fmt="-o",
                    alpha=0.6,
                    color=color
                )
                if measure == 'ROD':
                    plt.ylim(0, 8)
                # plt.fill_between(eps, mean_values - std_values, mean_values + std_values, alpha=0.2)

        # original dataset for different algorithms
        # orignal_df = dfs['original']['_']
        # plt.errorbar(eps, np.mean(orignal_df[measure])*np.ones(len(index)), yerr=np.std(orignal_df[measure]), linestyle='--', color='black', label='original', linewidth=2)
        # plt.fill_between(eps, np.mean(orignal_df[measure])*np.ones(len(index)) - np.std(orignal_df[measure]), np.mean(orignal_df[measure])*np.ones(len(index)) + np.std(orignal_df[measure]), alpha=0.2, hatch='/')
        
        # vanilla_mst = dfs['original']['vanilla_mst']
        # plt.errorbar(eps, np.mean(vanilla_mst[measure])*np.ones(len(index)), yerr=np.std(vanilla_mst[measure]), linestyle='--', color='blue', label='vanilla_mst', linewidth=2)
        # # plt.fill_between(eps, np.mean(vanilla_mst[measure])*np.ones(len(index)) - np.std(vanilla_mst[measure]), np.mean(vanilla_mst[measure])*np.ones(len(index)) + np.std(vanilla_mst[measure]), alpha=0.2, hatch='\\')
        
        # vanilla_mst = dfs['original']['vanilla_mst_reg']
        # plt.errorbar(eps, np.mean(vanilla_mst[measure])*np.ones(len(index)), yerr=np.std(vanilla_mst[measure]), linestyle='--', color='blue', label='vanilla_mst', linewidth=2)
        
        # vanilla_greedy = dfs['original']['vanilla_greedy']
        # plt.errorbar(eps, np.mean(vanilla_greedy[measure])*np.ones(len(index)), yerr=np.std(vanilla_greedy[measure]), linestyle='--', color='red', label='vanilla_greedy', linewidth=2)
        # # plt.fill_between(eps, np.mean(vanilla_greedy[measure])*np.ones(len(index)) - np.std(vanilla_greedy[measure]), np.mean(vanilla_greedy[measure])*np.ones(len(index)) + np.std(vanilla_greedy[measure]), alpha=0.2, hatch='|')
        
        # vanilla_opt = dfs['original']['vanilla_opt']
        # plt.errorbar(eps, np.mean(vanilla_opt[measure])*np.ones(len(index)), yerr=np.std(vanilla_opt[measure]), linestyle='--', color='purple', label='vanilla_opt', linewidth=2)
        # # plt.fill_between(eps, np.mean(vanilla_opt[measure])*np.ones(len(index)) - np.std(vanilla_opt[measure]), np.mean(vanilla_opt[measure])*np.ones(len(index)) + np.std(vanilla_opt[measure]), alpha=0.2, hatch='-')
        
        # indep_coupling = dfs['original']
        # plt.errorbar(eps, np.mean(indep_coupling[measure])*np.ones(len(index)), yerr=np.std(indep_coupling[measure]), linestyle='--', color='orange', label='indep_coupling', linewidth=2)
        # # plt.fill_between(eps, np.mean(indep_coupling[measure])*np.ones(len(index)) - np.std(indep_coupling[measure]), np.mean(indep_coupling[measure])*np.ones(len(index)) + np.std(indep_coupling[measure]), alpha=0.2, hatch='+')

        # otclean_normal = dfs['original']['otclean_normal']
        # plt.errorbar(eps, np.mean(otclean_normal[measure])*np.ones(len(index)), yerr=np.std(otclean_normal[measure]), linestyle='dotted', color='green', label='OTCLEAN normal test', linewidth=2)
        # # plt.fill_between(eps, np.mean(otclean_normal[measure])*np.ones(len(index)) - np.std(otclean_normal[measure]), np.mean(otclean_normal[measure])*np.ones(len(index)) + np.std(otclean_normal[measure]), alpha=0.2, hatch='x')

        # otclean_clean = dfs['original']['otclean_clean']
        # plt.errorbar(eps, np.mean(otclean_clean[measure])*np.ones(len(index)), yerr=np.std(otclean_clean[measure]), linestyle='-.', color='yellow', label='OTCLEAN clean test', linewidth=2)
        # # plt.fill_between(eps, np.mean(otclean_clean[measure])*np.ones(len(index)) - np.std(otclean_clean[measure]), np.mean(otclean_clean[measure])*np.ones(len(index)) + np.std(otclean_clean[measure]), alpha=0.2, hatch='*')

        plt.legend()
        plt.xlabel('Epsilon')
        plt.ylabel(measure)
        plt.xscale('log')
        plt.xticks(eps, eps, rotation=90)
        plt.tight_layout()

        # plt.show()
        plt.savefig(f'res/{db_name}/{ML_algo}/{db_name}_{ML_algo}_{measure}_privacy.pdf')

def plot_comparison_on_measures_no_privacy(dfs, measures, OTOnly, eps=None):
    if not os.path.exists('res/' + db_name + '/' + f'eps={eps}/' + ML_algo):
        os.makedirs('res/' + db_name + '/' + f'eps={eps}/' + ML_algo)
    
    # Color-blind friendly palette
    colors = sns.color_palette("colorblind")

    for measure in measures:
        plt.figure(figsize=(13, 6))
        plt.title(f'Mean Cross Validation Results for {measure}')

        mechanism_means = []
        mechanism_stds = []
        mechanism_names = []
        
        if not OTOnly:
            for method, eps_df in dfs.items():
                if method == 'original':
                    continue
                
                mean_values = []
                std_values = []

                for epsilon, df in eps_df.items():
                    mean = df[measure].mean()
                    std = df[measure].std()

                    mean_values.append(mean)
                    std_values.append(std)

                mean_values = np.array(mean_values)
                std_values = np.array(std_values)

                mechanism_means.append(np.mean(mean_values))
                mechanism_stds.append(np.mean(std_values))
                mechanism_names.append(method)

        # original dataset for different algorithms
        # original_datasets = ['_', 'vanilla_mst', 'torch_vanilla_mst', 'torch_vanilla_mst_reg', 'vanilla_greedy', 'vanilla_opt', 'otclean_normal', 'otclean_clean']
        # original_datasets = ['_', 'vanilla_mst', 'vanilla_mst_one_clique_reg', 'vanilla_mst_reg', 'vanilla_mst_reg_re', 'vanilla_mst_reg_42', 'vanilla_mst_reg_7', 'vanilla_mst_reg_2', 'vanilla_mst_reg_21', 'vanilla_greedy', 'vanilla_opt', 'otclean_normal', 'otclean_clean']
        # original_datasets = ['_', 'vanilla_mst', 'vanilla_mst_reg_pmd', 'results_mst_pmd_reg_42', 'results_mst_pmd_reg_42_1e_4', 'results_mst_pmd_reg_21_5', 'results_mst_pmd_reg_101_50', 'vanilla_greedy', 'vanilla_opt', 'otclean_normal', 'otclean_clean']
        # original_datasets = ['Original', 'Dropped', 'vanilla_mst', 'vanilla_mst_v1_fixed_lr', 'mst_reg_cmi_v2', 'vanilla_mst_reg_tvd_L1_JIT', 'vanilla_greedy', 'vanilla_opt']
        # original_datasets = ['Original', 'Dropped', 'mst', 'Ours_cmi_reg', 'Ours_mmd', 'Ours_tvd_L2', 'vanilla_greedy', 'vanilla_opt']
        original_datasets = ['Original', 'Dropped', 'mst', 'tvd_L2', 'greedy', 'opt', 'otclean_normal']
        # original_datasets = ['_', 'vanilla_mst', 'vanilla_mst_reg']
        for i, dataset in enumerate(original_datasets):
            data = dfs['original'][dataset]
            mechanism_means.append(np.mean(data[measure]))
            mechanism_stds.append(np.std(data[measure]))
            mechanism_names.append(dataset)
        
        x = np.arange(len(mechanism_names))
        
        bars = plt.bar(x, mechanism_means, yerr=mechanism_stds, color=colors[:len(mechanism_names)], capsize=5)
        plt.xticks(x, mechanism_names, rotation=45, ha='right')
        plt.ylabel(measure)
        plt.xlabel('Mechanism')
        
        # Add mean values on top of each bar
        for bar, mean in zip(bars, mechanism_means):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{mean:.2f}', va='bottom')  # Adjust va to place the text above the bar
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'res/{db_name}/eps={eps}/{ML_algo}/{db_name}_{ML_algo}_{measure}_eps={eps}_SN_Y_A_train_test_Auto_diff.pdf') ## Change this name based on experiment type
        # plt.savefig(f'res/{db_name}/{ML_algo}/{db_name}_{ML_algo}_{measure}_SN_Y_A_train_train_Auto_diff.pdf') ## Change this name based on experiment type
        plt.close()

def plot_distance(dist_dict, title, save_path):
    means = {method: np.mean(values) for method, values in dist_dict.items()}
    stds = {method: np.std(values) for method, values in dist_dict.items()}

    methods = list(means.keys())
    mean_values = list(means.values())
    std_values = list(stds.values())

    plt.figure(figsize=(10, 5))
    plt.errorbar(methods, mean_values, yerr=std_values, fmt='o', capsize=5, capthick=2)
    
    # Add mean values as text above the points
    for i, method in enumerate(methods):
        plt.text(i, mean_values[i] + std_values[i] + 0.005, f'{mean_values[i]:.6f}', 
                 ha='center', va='bottom', fontsize=10, color='blue')

    plt.xlabel('Method')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
