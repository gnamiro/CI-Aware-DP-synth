from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from IPython.display import display
from sklearn import metrics
import itertools
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
import torch
import ot
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import random
import math
import psutil
import functools
from collections import defaultdict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_feature_ranking(data, y_column_name, model_type=RandomForestClassifier()):
    y = data[y_column_name]
    X = data.drop(columns=[y_column_name])

    rfe = RFE(model_type, n_features_to_select=1)
    rfe.fit(X, y)

    feature_selected = {}
    # Selected features are assigned rank 1.
    importance_ranking = rfe.ranking_
    feature_names = list(X.columns)

    for i in range(len(feature_names)):
        feature_selected[feature_names[i]] = importance_ranking[i]

    return feature_selected

def convert_list_to_str(l):
    def f(a):
        r = ''
        for e in a:
            r += str(int(e))
        return int('0b' + r, 2)#int(r)
    res = []
    for i, element in enumerate(l):
        res.append(f(element))
    return res

def plot_3d_dist(b, target_domain, file_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    s = [10000*b[n] for n in range(len(target_domain))]

    dim_1 = convert_list_to_str(np.array(target_domain)[:,0])
    dim_2 = convert_list_to_str(np.array(target_domain)[:,1])
    dim_3 = convert_list_to_str(np.array(target_domain)[:,2])

    img = ax.scatter(dim_1, dim_2, dim_3, c=b, cmap=plt.hot(), s=s)
    fig.colorbar(img)
    plt.savefig(file_name)
    plt.show()


# def conditional_prob(data, target_data, **kwargs):
#     total_count = len(data)
#     target_counts = target_data.value_counts()
#     probabilities = target_counts / total_count
#     return probabilities


def plot_conditional_dist(df, target_attr, protected, inadmissible, admissible, method_name, dist=False, weights=0, plot=True):
    hist_SNA = plot_conditional_dist_helper(df, target_attr, protected + inadmissible + admissible, method_name, 'S,N,A', dist, weights, plot)
    plot_conditional_dist_helper(df, target_attr, protected + admissible, method_name, 'S,A', dist, weights, plot)
    plot_conditional_dist_helper(df, target_attr, protected, method_name, 'S', dist, weights, plot)
    plot_conditional_dist_helper(df, target_attr, admissible, method_name, 'A', dist, weights, plot)
    return hist_SNA


def plot_conditional_dist_helper(df, target_attr, conditioning_attrs, method_name, directory, dist=False, weights=0, plot=True):
    plt.figure(figsize=(15, 6))
    # hue = df[['S', 'N', 'A']].apply(
    #     lambda row: f"({int(row.S)}, {int(row.N)}, {int(row.A)})", axis=1)
    # hue.name = 'S, N, A'
    # ax_1 = sns.histplot(data=df, x=hue, hue=target_attr, multiple="dodge")

    if dist:
        df['count'] = weights
        conditional_density = df.copy()
        conditional_density = conditional_density.groupby(conditioning_attrs + [target_attr]).agg({'count': 'sum'})
    else:
        conditional_density = df.groupby(conditioning_attrs, as_index=False)[target_attr].value_counts()
        # display(conditional_density)
        domains = []
        for i in range(len(conditioning_attrs)):
            domains.append(conditional_density[conditioning_attrs[i]].drop_duplicates().values.tolist())
        domains.append(conditional_density[target_attr].drop_duplicates().values.tolist())
        for element in itertools.product(*domains):
            if not (conditional_density[conditional_density.columns[:len(element)]] == element).all(1).any():
                conditional_density.loc[conditional_density.shape[0]] = list(element) + [0]
    conditional_density = conditional_density.sort_values(by=[target_attr] + conditioning_attrs).reset_index(drop=True)

    if dist:
        conditional_probs = df.groupby(conditioning_attrs + [target_attr])[['count']].sum()
        conditional_probs['count'] = conditional_probs['count'] / conditional_probs.groupby(conditioning_attrs)['count'].sum()
        conditional_probs = conditional_probs.rename({'count': 'proportion'}, axis=1)
        conditional_probs = conditional_probs.reset_index()
    else:
        conditional_probs = df.groupby(conditioning_attrs, as_index=False)[target_attr].value_counts(normalize=True)
    conditional_probs = conditional_probs.sort_values(by=[target_attr] + conditioning_attrs)
    if plot:
        def f(a):
            if type(a) == str:
                return a
            else:
                return int(a)
        if len(conditioning_attrs) == 3:
            x = conditional_probs[conditioning_attrs].apply(
                lambda row: f"({f(row[0])}, {f(row[1])}, {f(row[2])})", axis=1)
            x.name = conditioning_attrs[0]+','+conditioning_attrs[1]+','+conditioning_attrs[2]
        elif len(conditioning_attrs) == 2:
            x = conditional_probs[conditioning_attrs].apply(
                lambda row: f"({f(row[0])}, {f(row[1])})", axis=1)
            x.name = conditioning_attrs[0]+','+conditioning_attrs[1]
        elif len(conditioning_attrs) == 1:
            x = conditional_probs[conditioning_attrs].apply(
                lambda row: f"({f(row[0])})", axis=1)
            x.name = conditioning_attrs[0]
    
        ax = sns.barplot(data=conditional_probs, x=x, y='proportion', hue=target_attr)
        for i, g in enumerate(ax.patches):
            ax.annotate(int(conditional_density.loc[i,'count']),
                    (g.get_x() + g.get_width() / 2., g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')
        plt.title('Distribution of Y over ' + x.name)
        plt.savefig('Outputs/Y,' + directory + '/' + method_name + '.png')
        plt.show()

    return conditional_probs


    # plt.figure(figsize=(10, 6))
    # conditional_density = df.groupby(['S', 'A'], as_index=False)[target_attr].value_counts()
    # conditional_density = conditional_density.sort_values(by=['Y', 'S', 'A']).reset_index(drop=True)
    # conditional_probs = df.groupby(['S', 'A'], as_index=False)[target_attr].value_counts(normalize=True)
    # x = conditional_probs[['S', 'A']].apply(
    #     lambda row: f"({int(row.S)}, {int(row.A)})", axis=1)
    # x.name = 'S, A'
    # ax = sns.barplot(data=conditional_probs, x=x, y='proportion', hue=target_attr)
    # for i, g in enumerate(ax.patches):
    #     ax.annotate(conditional_density.loc[i,'count'],
    #                (g.get_x() + g.get_width() / 2., g.get_height()),
    #                ha = 'center', va = 'center',
    #                xytext = (0, 9),
    #                textcoords = 'offset points')
    # plt.title('Distribution of Y over S, A')
    # plt.savefig('Outputs/Y,S,A/' + method_name + '.png')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # conditional_density = df.groupby('S', as_index=False)[target_attr].value_counts()
    # conditional_density = conditional_density.sort_values(by=['Y', 'S']).reset_index(drop=True)
    # conditional_probs = df.groupby('S', as_index=False)[target_attr].value_counts(normalize=True)
    # ax = sns.barplot(data=conditional_probs, x='S', y='proportion', hue=target_attr)
    # for i, g in enumerate(ax.patches):
    #     ax.annotate(conditional_density.loc[i,'count'],
    #                (g.get_x() + g.get_width() / 2., g.get_height()),
    #                ha = 'center', va = 'center',
    #                xytext = (0, 9),
    #                textcoords = 'offset points')
    # plt.title('Distribution of Y over S')
    # plt.savefig('Outputs/Y,S/' + method_name + '.png')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # conditional_density = df.groupby('A', as_index=False)[target_attr].value_counts()
    # conditional_density = conditional_density.sort_values(by=['Y', 'A']).reset_index(drop=True)
    # conditional_probs = df.groupby('A', as_index=False)[target_attr].value_counts(normalize=True)
    # ax = sns.barplot(data=conditional_probs, x='A', y='proportion', hue=target_attr)
    # for i, g in enumerate(ax.patches):
    #     ax.annotate(conditional_density.loc[i,'count'],
    #                (g.get_x() + g.get_width() / 2., g.get_height()),
    #                ha = 'center', va = 'center',
    #                xytext = (0, 9),
    #                textcoords = 'offset points')
    # plt.title('Distribution of Y over A')
    # plt.savefig('Outputs/Y,A/' + method_name + '.png')
    # plt.show()


def roc_c(X_test, y_test, model, method, num_thresholds=1000, plot=True):
    # Compute predicted probabilities on the test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Compute the ROC curve
    fpr, tpr = [1], [1]
    last_fpr_tpr = (1,1)
    for threshold in np.linspace(0, 1, num_thresholds):
        y_binary = [1 if score >= threshold else 0 for score in y_pred_prob]
        fpr_temp, tpr_temp, thresholds = roc_curve(y_test, y_binary)
        if len(fpr_temp) > 2 and (fpr_temp[1], tpr_temp[1]) != last_fpr_tpr:
            # print(threshold, fpr_temp[1], tpr_temp[1])
            fpr.append(fpr_temp[1])
            tpr.append(tpr_temp[1])
            last_fpr_tpr = (fpr_temp[1], tpr_temp[1])
    fpr.append(0)
    tpr.append(0)
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    # Compute the Area Under the Curve (AUC)
    roc_auc = metrics.auc(fpr, tpr)
    # roc_auc = roc_auc_score(y_test, y_pred_prob)
    # Plot the ROC curve
    if plot:
        plt.figure()
        lw = 2  # Line width
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.6f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        file_name = 'Outputs/ROC_curves/roc_curve_' + method + '.png'
        plt.savefig(file_name)
        plt.show()

    return roc_auc

def extracting_domain(df, independence_columns):
    # z_domain = np.unique(np.round(np.array(df["education-num"]),1), axis=0)
    # x1_domain = np.unique(np.round(np.array(df["sex"]),1), axis=0)
    # x2_domain = np.unique(np.round(np.array(df["income"]),1), axis=0)
    z_domain = df[independence_columns[2]].drop_duplicates().values.tolist()
    x1_domain = df[independence_columns[0]].drop_duplicates().values.tolist()
    x2_domain = df[independence_columns[1]].drop_duplicates().values.tolist()

    dim = [len(z_domain), len(x1_domain), len(x2_domain)]

    domains = [z_domain, x1_domain, x2_domain]
    return x1_domain, x2_domain, z_domain, dim, domains


def compute_target_domain(x1_domain, x2_domain, z_domain):
    target_domain = []
    for k in z_domain:
        for j in x2_domain:
            for i in x1_domain:
                target_domain.append([i, j, k])
    return target_domain


def compute_target_domain_unsaturated(x1_domain, x2_domain, z_domain, w_domain):
    target_domain = []
    for k in z_domain:
        for j in x2_domain:
            for i in x1_domain:
                for w in w_domain:
                    target_domain.append([i, j, k, w])
    return target_domain


def compute_x_domain(x1_domain, x2_domain, z_domain, x_domain_plain):
    x_domain = []
    for k in z_domain:
        for j in x2_domain:
            for i in x1_domain:
                val = format_to_1([i,j,k])
                if val in x_domain_plain:
                    x_domain.append(val)
    return x_domain


def compute_x_domain_unsaturated(x1_domain, x2_domain, z_domain, w_domain, x_domain_plain):
    x_domain = []
    for k in z_domain:
        for j in x2_domain:
            for i in x1_domain:
                for w in w_domain:
                    val = [i,j,k,w]
                    val = val[0] + val[1] + val[2] + val[3]
                    if val in x_domain_plain:
                        x_domain.append([i,j,k,w])
    return x_domain


def compute_target_domain_marginal(x1_domain, z_domain, plain=True):
    target_domain = []
    for k in z_domain:
        for i in x1_domain:
            if plain:
                target_domain.append(i+k)
            else:
                target_domain.append([i, k])
    return target_domain


def format_to_1(x):
    if len(x) == 4:
        return x[0] + x[1] + x[2] + x[3]
    return x[0] + x[1] + x[2]


def format_to_4(x_plain, idx1, idx2, idx3):
    return [x_plain[0:idx1],
            x_plain[idx1:idx1 + idx2],
            x_plain[idx1 + idx2: idx1 + idx2 + idx3],
            x_plain[idx1 + idx2 + idx3:]]


def format_to_3(x_plain, independence_columns):
    return [x_plain[0:len(independence_columns[0])],
            x_plain[len(independence_columns[0]):len(independence_columns[0]) + len(independence_columns[1])],
            x_plain[len(independence_columns[0]) + len(independence_columns[1]):]]


def format_to_2(x_plain, independence_columns):
    return [x_plain[0:len(independence_columns[0])],
            x_plain[len(independence_columns[0]):]]


def to_tuple(lst):
    return tuple(tuple(i) for i in lst)


def generate_initial_distribution(x_domain, target_domain, p_x):
    initial_distribution = []
    for x in target_domain:
        if x in x_domain:  # .tolist():
            initial_distribution.append(p_x[to_tuple(x)])
        else:
            initial_distribution.append(0)
    return initial_distribution


def generate_repaired_dataset(df, Gs, df_without_W, x_domain, target_domain, independence_columns):
    df_result = df.copy()
    num_repaired_records = 0

    cols = format_to_1(independence_columns)
    axis0_indices, axis1_indices = np.where(Gs > 0.5 / df.shape[0])
    for k in range(len(axis0_indices)):
        i, j = axis0_indices[k], axis1_indices[k]
        num_records = round(Gs[i][j] * df.shape[0])
        source_indexes = \
        np.where((np.array(df_without_W) == np.array(format_to_1(x_domain[i]), dtype=object)).all(axis=1))[0]
        if not (x_domain[i] == target_domain[j]):
            for h in range(min(num_records, len(source_indexes))):
                # print(source_indexes[h], x_domain[i], target_domain[j])
                num_repaired_records += 1
                for w, attr in enumerate(format_to_1(target_domain[j])):
                    df_result.iloc[source_indexes[h], df_result.columns.get_loc(cols[w])] = attr

    # num_mapping_records = 0
    # cols = format_to_1(independence_columns)
    # for i in range(len(x_domain)):
    #   source_indexes = np.where((np.array(df_without_W) == np.array(format_to_1(x_domain[i]), dtype=object)).all(axis=1))[0]
    #   counter = 0
    #   for j in range(len(target_domain)):
    #     num_records = round(Gs[i][j]*df.shape[0])
    #     if num_records >= 1 and not (x_domain[i] == target_domain[j]):
    #       # print(x_domain[i], target_domain[j], Gs[i][j]*48842)
    #       for k in range(num_records):
    #         for w, attr in enumerate(format_to_1(target_domain[j])):
    #           if counter+k < len(source_indexes):
    #             df_result.iloc[source_indexes[counter+k], df_result.columns.get_loc(cols[w])] = attr
    #         # df_result.iloc[source_indexes[counter+k], df_result.columns.get_loc('sex')] = target_domain[j][0]
    #         # df_result.iloc[source_indexes[counter+k], df_result.columns.get_loc('income')] = target_domain[j][1]
    #         # df_result.iloc[source_indexes[counter+k], df_result.columns.get_loc('education-num')] = target_domain[j][2]
    #       counter += num_records
    #   num_mapping_records += counter

    df_result.to_csv("Boston_repair.csv", index=False)
    print("Num of repaired records:", num_repaired_records)

    return df_result


def generate_most_likely_repair(df, Gs, df_without_W, x_domain, target_domain, k_records, independence_columns):
    df_result = df.copy()
    num_repaired_records = 0
    repaired_indices = []

    cols = format_to_1(independence_columns)
    axis1_indices = np.argmax(Gs, axis=1)
    most_important_mappings = {}
    for i in range(Gs.shape[0]):
        most_important_mappings[i] = Gs[i][axis1_indices[i]]
    most_important_mappings = sorted(most_important_mappings.items(), key=lambda x: x[1], reverse=True)
    # ind = np.unravel_index(np.argsort(Gs, axis=None), Gs.shape)
    for count in range(len(most_important_mappings)):#range(len(ind[0])):#range(Gs.shape[0]):
        i = most_important_mappings[count][0]#ind[0][len(ind[0])-1-count]
        j = axis1_indices[i]#ind[1][len(ind[0])-1-count]
        source_indexes = \
        np.where((np.array(df_without_W) == np.array(format_to_1(x_domain[i]), dtype=object)).all(axis=1))[0]
        # if not (x_domain[i] == target_domain[j]):
        for k in range(len(source_indexes)):
            if source_indexes[k] not in repaired_indices:
                print(source_indexes[k], x_domain[i], target_domain[j])
                num_repaired_records += 1
                repaired_indices.append(source_indexes[k])
                for w, attr in enumerate(format_to_1(target_domain[j])):
                    df_result.iloc[source_indexes[k], df_result.columns.get_loc(cols[w])] = attr
                    df_without_W.iloc[source_indexes[k], df_without_W.columns.get_loc(cols[w])] = attr
                if num_repaired_records == k_records:
                    break
        if num_repaired_records == k_records:
            break

    df_result.to_csv("Boston_repair.csv", index=False)
    print("Num of repaired records:", num_repaired_records)

    return df_result, repaired_indices


def generate_most_likely_repair_2(df, Gs, df_without_W, x_domain, target_domain, k_records, dirty_records, before_repair, clean_df):
    df_result = df.copy()
    num_repaired_records = 0
    repaired_indices = []

    marginal_prob = np.sum(Gs, axis=1)
    most_important_mappings = {}
    for i, x in enumerate(x_domain):
        most_important_mappings[i] = marginal_prob[i] - Gs[i][target_domain.index(x)]
    most_important_mappings = sorted(most_important_mappings.items(), key=lambda x: x[1], reverse=True)
    # ind = np.unravel_index(np.argsort(Gs, axis=None), Gs.shape)
    for count in range(len(most_important_mappings)):#range(len(ind[0])):#range(Gs.shape[0]):
        i = most_important_mappings[count][0]#ind[0][len(ind[0])-1-count]
        source_indexes = \
        np.where((np.array(df_without_W) == np.array(format_to_1(x_domain[i]), dtype=object)).all(axis=1))[0]
        temp = \
            np.where((np.array(clean_df) == np.array(format_to_1(x_domain[i]), dtype=object)).all(axis=1))[0]
        # if k_records == 2000:
        #     print("num_records", len(source_indexes), "record:", x_domain[i], "initial prob:", marginal_prob[i], "exiting mass:",
        #       most_important_mappings[count][1])
        #     axis1_indices = np.argsort(-Gs[i])[:20]#np.argpartition(Gs[i], -4)[-4:]
        #     print(len(temp))
        #     print("top 20 repairs:")
        #     for k in range(20):
        #         # print(axis1_indices)
        #         print(target_domain[axis1_indices[k]], end=", ")
        #         print("mass:", Gs[i][axis1_indices[k]])
        # if not (x_domain[i] == target_domain[j]):
        num_dirty = 0
        before_rep_dict = {}
        for k in range(len(source_indexes)):
            if source_indexes[k] not in repaired_indices:
                # print(source_indexes[k], x_domain[i], target_domain[j])
                # if k_records == 2000:
                #     if source_indexes[k] in dirty_records:
                #         num_dirty += 1
                #         print(str(num_dirty) + ".", "dirty  ", end="")
                #         print("idx:", source_indexes[k], end=", ")
                #         print("clean value:", before_repair[dirty_records.index(source_indexes[k])].to_numpy())
                #         if before_repair[dirty_records.index(source_indexes[k])].to_numpy()[0] not in before_rep_dict.keys():
                #             before_rep_dict[before_repair[dirty_records.index(source_indexes[k])].to_numpy()[0]] = 1
                #         else:
                #             before_rep_dict[before_repair[dirty_records.index(source_indexes[k])].to_numpy()[0]] += 1
                num_repaired_records += 1
                repaired_indices.append(source_indexes[k])
                if num_repaired_records == k_records:
                    break
        # if k_records == 2000:
        #     before_rep_dict = dict(sorted(before_rep_dict.items(), key=lambda x: x[1], reverse=True))
        #     print(before_rep_dict)
        #     print("Percentage of dirty records:", num_dirty/len(source_indexes))
        if num_repaired_records == k_records:
            break

    df_result.to_csv("Boston_repair.csv", index=False)
    # print("Num of repaired records:", num_repaired_records)

    return df_result, repaired_indices


def generate_dataset_sampling(df, distribution, indices, target_domain):
    df_result = pd.DataFrame(columns=df.columns)

    idx = 0
    for i, mass in enumerate(list(distribution)):
        num_records = round(mass)
        for j in range(num_records):
            df_result.loc[idx] = format_to_1(target_domain[indices[i]])
            idx += 1
    print(idx)

    df_result.to_csv("adult_repair.csv", index=False)
    return df_result


def sampling(columns, distribution, target_domain, num_samples):
    df_result = pd.DataFrame(columns=columns)
    # for i in range(100):
    #   df_result = df_result.append(df, ignore_index=True)

    target_domain_copy = []
    for x in target_domain:
        target_domain_copy.append(format_to_1(x))
    target_domain_copy = np.array(target_domain_copy)

    domain_size = target_domain_copy.shape[0]
    target_domain_copy = target_domain_copy[distribution >= 0.5/domain_size, :]
    distribution = distribution[distribution >= 0.5/domain_size]

    distribution = distribution / distribution.sum(0)

    target_dist = [0] * target_domain_copy.shape[0]
    for i in range(num_samples):
        idx = np.random.choice(range(target_domain_copy.shape[0]), p=distribution)
        target_dist[idx] += 1
        df_result.loc[i] = target_domain_copy[idx]

    # custm = stats.rv_discrete(values=(range(target_domain_copy.shape[0]), distribution))
    # for j in range(100):
    #     print(j)
    #     R = np.array(custm.rvs(size=df.shape[0]))
    #     for i in range(target_domain_copy.shape[0]):
    #       target_dist[i] = len(np.where(R == i)[0])
    #       df_result.loc[j*df.shape[0] + np.where(R == i)[0]] = target_domain_copy[i]
    # for i, idx in enumerate(R):
    #   target_dist[idx] += 1
    #   df_result.loc[i] = target_domain_copy[idx]

    # df_result.to_csv("adult_repair.csv", index=False)
    return df_result

def kl(w, z):
    return (w.t() @ torch.log(w / z) - torch.ones(1, w.shape[0], dtype=torch.float64, device=device) @ w + torch.ones(1, z.shape[0], dtype=torch.float64, device=device) @ z)[0].cpu().numpy().astype(np.float64)


def marginalize(b, dim):
    res = []
    for i in range(dim[0]):
        slice = b[i*dim[1]*dim[2]:(i+1)*dim[1]*dim[2]].copy()
        for j in range(dim[1]):
            demarginalized_val = 0
            for k in range(dim[2]):
                demarginalized_val += slice[k*dim[1] + j]
            res.append(demarginalized_val)
    res = np.array(res)
    return res


def group_edu(x):
    # if x <= 5:
    #     return 5
    # elif x == 6 or x== 7:
    #     return 6.5
    # elif x == 8 or x== 9:
    #     return 8.5
    # elif x == 10 or x== 11:
    #     return 10.5
    # elif x == 12:
    #     return 12
    # elif x >= 13:
    #     return 13
    # else:
    #     return x
    
    # if x <= 8:
    #     return 8
    # elif x <= 10:
    #     return 9
    # elif x <= 12:
    #     return 11
    # elif x == 13:
    #     return 13
    # elif x == 14:
    #     return 14
    # else:
    #     return 15

    if x <= 5:
        return 5
    elif x >= 13:
        return 13
    else:
        return x


def age_cut(x):
    if x <= 25:
        return 20
    elif x <= 35:
        return 30
    elif x <= 45:
        return 40
    elif x <= 60:
        return 50
    # elif x <= 60:
    #     return 55
    else:
        return 70
    # if x > 70:
    #     return 70
    # else:
    #     return x
    
def hours_cut(x):
    if x <= 20:
        return 20
    elif x <= 35:
        return 30
    elif x <= 40:
        return 40
    elif x <= 45:
        return 45
    elif x <= 60:
        return 60
    # elif x <= 55:
    #     return 50
    # elif x <= 70:
    #     return 60
    else:
        return 70


def occupation_cut(x):
    if x in ['Exec-managerial', 'Prof-specialty']:
        return 'NOC-1'
    if x in ['Tech-support', 'Sales', 'Protective-serv']:
        return 'NOC-2'
    elif x in ['Craft-repair', 'Transport-moving']:
        return 'NOC-3'
    elif x in ['Adm-clerical', 'Machine-op-inspct', 'Farming-fishing', 'Armed-Forces']:
        return 'NOC-4'
    elif x in ['Handlers-cleaners', 'Other-service']:
        return 'NOC-5'
    elif x in ['Priv-house-serv']:
        return 'NOC-6'
    elif x in ['?']:
        return 'NOC-7'
    

def bucketize_country(x):
    nationality_hierarchy = {
        # High income
        'United-States': 4, 'Germany': 4, 'Canada': 4, 'Japan': 4, 'France': 4, 'Ireland': 4,
        'England': 4, 'Italy': 4, 'Taiwan': 4, 'Portugal': 4, 'Scotland': 4, 'Holand-Netherlands': 4,

        # Upper-middle income
        'Poland': 3, 'China': 3, 'Mexico': 3, 'Cuba': 3, 'Philippines': 3, 'India': 3, 'Hungary': 3,
        'Iran': 3, 'Greece': 3, 'Hong': 3, 'Jamaica': 3, 'Puerto-Rico': 3, 'Columbia': 3, 'Yugoslavia': 3, 'Peru': 3,

        # Lower-middle income
        'El-Salvador': 2, 'Vietnam': 2, 'Nicaragua': 2, 'Haiti': 2, 'Honduras': 2,
        'Guatemala': 2, 'Dominican-Republic': 2, 'Thailand': 2, 'Ecuador': 2,

        # Low income or Unknown/Unspecified
        'Cambodia': 1, 'Laos': 1, 'Trinadad&Tobago': 1,
        '?': 1, 'South': 1, 'Outlying-US(Guam-USVI-etc)': 1}
    return nationality_hierarchy[x]


# def reduce_domain_size(df):
    
#     # df['age'] = df['age'].apply(lambda x: age_cut(x))

#     # df['hours-per-week'] = df['hours-per-week'].apply(lambda x: hours_cut(x))

#     # df['education-num'] = df['education-num'].apply(lambda x: group_edu(x))

#     # df['occupation'] = df['occupation'].apply(lambda x: occupation_cut(x))
    
#     df['marital-status'] = df['marital-status'].apply(lambda x: 'Never-married' if x == 'Never-married' else 'Married' if 'Married' in x else 'Seperated')

#     df['age'] = df['age'].apply(lambda x: age_cut(x))
#     # df['age'] = df['age'].apply(lambda x: np.floor(x / 10.0) * 10.0)

#     df['hours-per-week'] = df['hours-per-week'].apply(lambda x: hours_cut(x))
#     # df['hours-per-week'] = df['hours-per-week'].apply(lambda x: np.floor(x / 10.0) * 10.0)

#     # df['Original-education-num'] = df['education-num'].apply(lambda x: 5 if x<=5 else 14 if x>=14 else x)
#     df['education-num'] = df['education-num'].apply(lambda x: group_edu(x))

#     df['occupation'] = df['occupation'].apply(lambda x: occupation_cut(x))

#     # df['bucketized-native-country'] = df['native-country'].apply(lambda x: bucketize_country(x))

#     return df

def reduce_domain_size(df):

    def convert_marital_status(status):
      if status in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
          return 'married'
      elif status in ['Never-married', 'Separated', 'Widowed']:
          return 'single'
      else:
          return 'divorced'

    df['marital-status'] = df['marital-status'].apply(convert_marital_status)

    # df['native-country'] = df['native-country'].replace('Outlying-US(Guam-USVI-etc)' , 'US Minor Islands')

    df['age'] = df['age'].apply(lambda x: age_cut(x))
    # df['age'] = df['age'].apply(lambda x: np.floor(x / 10.0) * 10.0)

    df['hours-per-week'] = df['hours-per-week'].apply(lambda x: hours_cut(x))
    # df['hours-per-week'] = df['hours-per-week'].apply(lambda x: np.floor(x / 10.0) * 10.0)

    # df['Original-education-num'] = df['education-num'].apply(lambda x: 5 if x<=5 else 14 if x>=14 else x)
    df['education-num'] = df['education-num'].apply(lambda x: group_edu(x))

    df['occupation'] = df['occupation'].apply(lambda x: occupation_cut(x))


    # income_mapping = {'<=50K': 0, '>50K': 1}
    # df['income'] = df['income'].map(income_mapping)

    return df

def make_feature_numerical(df, feature, feature_domain):
    df[feature] = df[feature].astype('category')
    df[feature] = df[feature].cat.reorder_categories(feature_domain, ordered=True)
    df[feature] = df[feature].cat.codes


def merge_columns(df, cols):
    res = df[cols[0]].astype(str)
    for c in cols[1:]:
        res += df[c].astype(str)
    return res


def feature_importance(df, target):
    std = df.std()
    for i in df.columns:
        if i != target:
            df[i] = df[i].apply(lambda x: x / std[i])

    lr = LogisticRegression(max_iter=10000)
    clf = lr.fit(df[df.columns.drop(target)], df[target], sample_weight=np.ones(df.shape[0]))
    print(clf.coef_[0])


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


def unnormalize(df, std, independence_columns, normalization_coeffs={'S':1.0,'N':1.0,'Y':1.0,'A':1.0}, multiply_by_std=True):
    for i, col in enumerate(independence_columns[0]):
        if i == 0:
            df[col] = df[col].apply(lambda x: x / normalization_coeffs['S'])
        else:
            df[col] = df[col].apply(lambda x: x / normalization_coeffs['N'])
    for i in independence_columns[1]:
        df[i] = df[i].apply(lambda x: x / normalization_coeffs['Y'])
    for i in independence_columns[2]:
        df[i] = df[i].apply(lambda x: x / normalization_coeffs['A'])
    
    if multiply_by_std:
        for i in df.columns:
            if std[i] != 0:
                df[i] = df[i].apply(lambda x: x * std[i])
    
    for i in df.columns:
        if std[i] != 0:
            df[i] = df[i].apply(lambda x: np.round(x, 0))
    return df


def wasserstein_distance_dists(dist_1, dist_2, domain):
    domain_tensor = torch.from_numpy(domain).to(device)
    M = torch.cdist(domain_tensor, domain_tensor, p=2).cpu().numpy().astype(np.float64)
    return np.sum(np.multiply(ot.lp.emd(dist_1/dist_1.sum(0), dist_2/dist_2.sum(0), M, numItermax=1000000),M)) * dist_1.sum(0)
    # return ot.lp.emd2(dist_1, dist_2, M, numItermax=1000000)


def wasserstein_between_hists(hists, iteration_number, columns):
    temp_df = pd.get_dummies(hists["Original data"])
    domain = temp_df[columns].to_numpy().astype(np.float64)
    original_hist = hists["Original data"]['proportion'].to_numpy()
    result_table = []
    for method in hists.keys():
        # import pdb;pdb.set_trace()
        if method != "Original data":
            method_hist = hists[method]['proportion'].to_numpy()
            result_table.append([method, wasserstein_distance_dists(original_hist, method_hist, domain)])
    with open('Outputs/Wasserstein between histograms_' + str(iteration_number) + '.txt', 'w') as f:
            print(tabulate(result_table, headers=["", "Original histogram"]), file=f)


def kl_between_hists(hists, iteration_number):
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    original_hist = hists["Original data"]['proportion'].to_numpy()
    result_table = []
    for method in hists.keys():
        # import pdb;pdb.set_trace()
        if method != "Original data":
            method_hist = hists[method]['proportion'].to_numpy()
            result_table.append([method, kl_divergence(original_hist, method_hist)])
    with open('Outputs/KL between histograms_' + str(iteration_number) + '.txt', 'w') as f:
            print(tabulate(result_table, headers=["", "Original histogram"]), file=f)


def wasserstein_distance(df_1, df_2, independence_columns):
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

    # M = []
    # for i in df_1_domain:
    #  l = []
    #  for j in df_2_domain:
    #    cost = 0
    #    for k in range(len(i)):
    #      if type(i[k]) == str and i[k] != j[k]:
    #        cost += 1
    #      elif type(i[k]) != str:
    #        cost += (i[k] - j[k])**2
    #    l.append(math.sqrt(cost))
    #  M.append(l)
    df_1_domain_tensor = torch.tensor(df_1_domain).to(device)
    df_2_domain_tensor = torch.tensor(df_2_domain).to(device)
    M = torch.cdist(df_1_domain_tensor, df_2_domain_tensor, p=2).cpu().numpy().astype(np.float64)
    #
    # Gs = ot.lp.emd(distribution_1, distribution_2, M, numItermax=1000000)
    # df_1 = unnormalize(df_1, std_1, independence_columns)
    # df_2 = unnormalize(df_2, std_2, independence_columns)
    # plot_marginal_mapping(Gs, df_1_domain_unnormalized, df_2_domain_unnormalized, df_1, independence_columns, 0, "CAP")
    # plot_marginal_mapping(Gs, df_1_domain_unnormalized, df_2_domain_unnormalized, df_1, independence_columns, 1, "CAP")
    # plot_marginal_mapping(Gs, df_1_domain_unnormalized, df_2_domain_unnormalized, df_1, independence_columns, 2, "CAP")

    return ot.lp.emd2(distribution_1, distribution_2, M, numItermax=1000000)


def kl_divergence_dfs(df_1, df_2, independence_columns):
    p_x_1 = dict(df_1.value_counts(normalize=True))
    p_x_2 = dict(df_2.value_counts(normalize=True))

    df_1_domain = list(p_x_1.keys())
    df_2_domain = list(p_x_2.keys())

    distribution_1 = []
    distribution_2 = []
    for x in df_1_domain:
        if x in df_2_domain:
            distribution_1.append(p_x_1[tuple(x)])
            distribution_2.append(p_x_2[tuple(x)])

    # for x in df_2_domain:
    #     if x not in df_1_domain:
    #         distribution_2.append(p_x_2[tuple(x)])
    #         distribution_1.append(0)
    
    distribution_1 = np.array(distribution_1)
    distribution_2 = np.array(distribution_2)
    
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    return kl_divergence(distribution_1, distribution_2)


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def consistency_score(X, y, n_neighbors=5):
  std = X.std()
  for i in X.columns:
    X[i] = X[i].apply(lambda x: x / std[i])
  X = X.to_numpy()
  y = y.to_numpy()
  nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
  nbrs.fit(X)
  indices = nbrs.kneighbors(X, return_distance=False)

  # compute consistency score
  return 1 - abs(y - y[indices].mean(axis=1)).mean()


def conditional_mutual_information(data, X, Y, Z, delta=1):

  cmi = 0

  P_Z = data.groupby(Z).size()
  P_Z = P_Z / P_Z.sum()

  P_XZ = data.groupby(X + Z).size()
  P_XZ = P_XZ / P_XZ.sum()

  P_YZ = data.groupby(Y + Z).size()
  P_YZ = P_YZ / P_YZ.sum()

  P_XYZ = data.groupby(X + Y + Z).size()
  P_XYZ = P_XYZ / P_XYZ.sum()

  for ind in P_XYZ.index:
    x_ind = ind[:len(X)]
    y_ind = ind[len(X):len(X + Y)]
    z_ind = ind[len(X + Y):]

    xz_ind = x_ind + z_ind
    yz_ind = y_ind + z_ind
    xyz_ind = ind

    z_ind = pd.MultiIndex.from_tuples([z_ind], names=Z) if len(Z) != 1 else pd.Index(z_ind, name=Z[0])
    xz_ind = pd.MultiIndex.from_tuples([xz_ind], names=X + Z)
    yz_ind = pd.MultiIndex.from_tuples([yz_ind], names=Y + Z)
    xyz_ind = pd.MultiIndex.from_tuples([xyz_ind], names=X + Y + Z)

    cmi += delta * P_XYZ[xyz_ind].item() * np.log2(
      P_Z[z_ind].item() * P_XYZ[xyz_ind].item() / (P_XZ[xz_ind].item() * P_YZ[yz_ind].item()))

  return cmi


def list_of_combs(arr):
    """returns a list of all subsets of a list"""

    combs = []
    for i in range(0, len(arr) + 1):
        listing = [list(x) for x in combinations(arr, i)]
        combs.extend(listing)
    return combs


def ROD(df,att,Y_features,X_features,inadmisable,priv,minor,positive=1):
    sub_sets = list_of_combs(inadmisable)
    dood = 1.1
    for adm_set in sub_sets:
        grouped=df.groupby(X_features+adm_set)
        total=len(df.index)
        total=0
        bias = 0
        val=0
        observed=0
        for name, group in grouped:
            mean=group.groupby(att)[Y_features].mean()
            count = group.groupby(att)[Y_features].count()
            size = len(group.index)
            #print(mean)
            if size>0:
                if len(mean.values)==2:
                            min_id=0
                            priv_eff=0
                            if mean.values[0]==mean.values[1]:
                                val=1
                            else:
                                if mean.index[0]==priv:
                                    priv_eff=mean.values[0]
                                    min_id=1

                                elif mean.index[1]==priv:
                                    priv_eff=mean.values[1]
                                    min_id=0
                                min_eff=mean.values[min_id]
                                max_ratio=max(mean.values[0],mean.values[1])
                                min_ratio=min(mean.values[0],mean.values[1])
                                if positive==0:
                                    min_eff=1-min_eff
                                    priv_eff=1-priv_eff

                                val = (min_ratio) / (max_ratio)
                            #print(val)
                            size = len(group.index)
                            #print(val,size)
                            total=total+size
                            bias=bias+(val*size)
                            observed=1

        #print('##########')
        #print(bias )
        if observed:
            if total!=0:
               #print(bias/total)
               bias=float(bias)/total
            if bias<dood:
                    dood=float(bias)
    print(dood)
    return float(dood)


class ContinTable(object):
    #  Contingacy Table Class
    def __init__(self):
        self.matrix = None   # A continacy table as Matrix
        self.dim=None        # Contingaxy table dimentions
        self.Ymrn=None       # Y margin
        self.Xmrn = None     # X margin
        pass


    def data_to_cnt(self, data=None, X=None, Y=None):

    # Input: A panda dataframe and two set of attributes
    # Outpout: Creates a contingacy table and its marginals
             self.table= pd.crosstab([data[att] for att in X],[data[att] for att in Y],margins=False)
             #print(self.table)
             self.matrix=np.asmatrix(self.table.values)
             self.dim=list(self.matrix.shape)
             self.Xmrn, self.Ymrn, self.total = self.get_margins(self.matrix)
             self.col_index=list(self.table.columns)
             self.row_index=list(self.table.index)


    def get_margin(self, matrix=None):
    # Returns a vector consists of the sum of columns
        matrix=np.asmatrix(matrix  )
        return np.asmatrix(np.array([e.sum() for e in matrix]))


    def get_margins(self, matrix=None):
    # Returns margins of a contingacy table
        Xmrn = self.get_margin(matrix)
        Ymrn = self.get_margin(matrix.T)
        total = Xmrn.sum()
        return Xmrn, Ymrn,total

def OR_multi(df, att, Y_features, X_features, inadmisable, priv, minor, positive=1):
    sub_sets = list_of_combs(inadmisable)
    print(sub_sets)
    dood = 1.1
    res = 0
    for adm_set in sub_sets:
        grouped = df.groupby(X_features + adm_set)
        total = len(df.index)
        total = 0
        bias = []
        val = 0
        observed = 0
        pvaluea = 0
        pvalue = 0
        i = 0
        prob = []
        tables = []
        values = defaultdict(list)
        
        for name, group in grouped:
            count = group.groupby(att if isinstance(att, list) else [att]).count()
            print("****"*8)
            print(f"a subclass in admissibles --> {name}")
            print(f"sex and prediction values:: ")
            print(group[['sex', 'Predicted']])
            count = count.values
            tbl = ContinTable()
            tbl.data_to_cnt(group, att if isinstance(att, list) else [att], [Y_features])
            table = tbl.matrix
            size = len(group.index)
            
            if len(count) >= 2:
                if (count[0])[0] > 5 and (count[1])[0] > 5:
                    print("here we print inside len count 2 and gets its values")
                    print(count)
                    table2 = sm.stats.Table.from_data(group[att + [Y_features] if isinstance(att, list) else [att, Y_features]])
                    print("table contents: --> ")
                    print(table2.table_orig)
                    if table.shape == (2, 2):
                        print("now let's see what is inide table2")
                        print(table2.table)
                        
                        print("ROD: ")
                        rod = sm.stats.StratifiedTable([table2.table]).oddsratio_pooled
                        print(rod)
                        if rod >= 1.5:
                            total_count = table2.table_orig.sum()
                            print(f"total count of the table: {total_count}")
                            # Convert to probability matrix by dividing each element by the total count
                            probability_matrix = table2.table_orig / total_count
                            print(probability_matrix)
                            probabilities = {
                                "P(0,0)": probability_matrix.iloc[0, 0],
                                "P(0,1)": probability_matrix.iloc[0, 1],
                                "P(1,0)": probability_matrix.iloc[1, 0],
                                "P(1,1)": probability_matrix.iloc[1, 1],
                            }
                            print("Extracted Probabilities:", probabilities)
                            values[name].append(rod)
                        res += 1
                        tables.insert(0, table2.table + 1)
                    else:
                        tables.insert(0, np.asarray([[float(size / 4), float(size / 4)], 
                                                     [float(size / 4), float(size / 4)]]))
            print()
        
        orr = 1
        pval = 1
        print(f"Total results: {res}")
        print(values)
        if len(tables) > 1:
            st = sm.stats.StratifiedTable(tables)
            orr = st.oddsratio_pooled
            pval = st.test_null_odds()
        
        return orr, pval


def OR(df,att,Y_features,X_features,inadmisable,priv,minor,positive=1):
    sub_sets = list_of_combs(inadmisable)
    # print(sub_sets)
    dood = 1.1
    res = 0
    for adm_set in sub_sets:
        # print(adm_set)
        grouped=df.groupby(X_features+adm_set)
        # print(X_features)
        total=len(df.index)
        total=0
        bias = []
        val=0
        observed=0
        pvaluea=0
        pvalue=0
        i=0
        prob=[]
        tables=[]
        values = defaultdict(list)
        for name, group in grouped:
            count=group.groupby(att).count()
            
            count=count.values
            tbl = ContinTable()
            tbl.data_to_cnt(group, [att], [Y_features])
            table=tbl.matrix
            size=len(group.index)
            if len(count)>=2:
                if (count[0])[0]>5 and (count[1])[0]>5:
                    table2 = sm.stats.Table.from_data(group[[att,Y_features]]) 
                    if table.shape==(2,2):
                        rod = sm.stats.StratifiedTable([table2.table]).oddsratio_pooled
                        if rod > 1 or rod < 1:
                            # print("****"*8)
                            # print(f"a subclass in admissibles --> {name}")
                            # print("now let's see what is inide table2")
                            # print("table contents: --> ")
                            # print(table2.table_orig)
                            # print("now let's see what is inide table2")
                            # print(table2.table)
                            # print(rod)
                            # print()
                            
                            # print("ROD: ")
                            
                            # total_count = table2.table_orig.sum()
                            # print(f"total count of the table: {total_count}")
                            # Convert to probability matrix by dividing each element by the total count
                            # probability_matrix = table2.table_orig / total_count
                            # print(probability_matrix)
                            # probabilities = {
                            #     "P(0,0)": probability_matrix.iloc[0, 0],
                            #     "P(0,1)": probability_matrix.iloc[0, 1],
                            #     "P(1,0)": probability_matrix.iloc[1, 0],
                            #     "P(1,1)": probability_matrix.iloc[1, 1],
                            # }
                            # print("Extracted Probabilities:", probabilities)
                            values[name].append(rod)
                        res += 1
                        tables.insert(0,table2.table+1)
                        # print(table2.table)

                    else:
                        tables.insert(0,np.asarray([[float(size/4), float(size/4)], [float(size/4), float(size/4)]]))

        orr=1
        pval=1
        # print('##########')
        # print(f"Total results: {res}")
        # print(values)
        if len(tables)>1:
            st = sm.stats.StratifiedTable(tables)
            # print(st.summary())
            orr=st.oddsratio_pooled
            pval=st.test_null_odds()
        '''
        bias=   [float(x) for x in bias]
        prob = [float(x) / sum(prob) for x in prob]
        print(sum(prob) )
        i=0
        orr=0
        for item in bias:
            orr=orr+item*prob[i]
            i=i+1
        #bias=weighted_median(bias,prob)
        print(orr)
        '''
        # print("ROD values -->", orr)
        # print("Number of (2, 2) shapes -->", res)
        # print("****"*8)
        return orr,pval
    

def add_extra_features_to_dist_old(df, cols, extra_feature, Gs, x_domain, target_domain, dist):
    feature_domain = df[extra_feature].drop_duplicates().values.tolist()
    p_y_x = dict(df.groupby(cols)[extra_feature].value_counts(normalize=True))
    p_x = dict(df[cols].value_counts(normalize=True))
    dict_p_y_x = {}
    for feature_val in feature_domain:
        dict_p_y_x[feature_val] = []
        for x in x_domain:
            if tuple(x+[feature_val]) in p_y_x.keys():
                dict_p_y_x[feature_val].append(p_y_x[tuple(x+[feature_val])])
            else:
                dict_p_y_x[feature_val].append(0)
        dict_p_y_x[feature_val] = np.array(dict_p_y_x[feature_val])
    new_target_domain, new_dist = [], []
    for i, record in enumerate(target_domain):
        for feature_val in feature_domain:
            full_record = record + [feature_val]
            new_val = np.inner(dict_p_y_x[feature_val], Gs[:,target_domain.index(record)])
            new_dist.append(new_val)
            new_target_domain.append(full_record)
    new_dist = np.array(new_dist)
    return new_target_domain, new_dist


def add_extra_features_to_dist(df, cols, extra_feature, Gs, x_domain, target_domain, dist):
    feature_domain = df[extra_feature].drop_duplicates().values.tolist()
    p_y_x = dict(df.groupby(cols)[extra_feature].value_counts(normalize=True))
    p_x = dict(df[cols].value_counts(normalize=True))
    dict_p_y_x = {}
    for feature_val in feature_domain:
        dict_p_y_x[tuple(feature_val)] = []
        for x in x_domain:
            if tuple(x+feature_val) in p_y_x.keys():
                dict_p_y_x[tuple(feature_val)].append(p_y_x[tuple(x+feature_val)])
            else:
                dict_p_y_x[tuple(feature_val)].append(0)
        dict_p_y_x[tuple(feature_val)] = np.array(dict_p_y_x[tuple(feature_val)])
    new_target_domain, new_dist = [], []
    for i, record in enumerate(target_domain):
        for feature_val in feature_domain:
            full_record = record + feature_val
            new_val = np.inner(dict_p_y_x[tuple(feature_val)], Gs[:,target_domain.index(record)])
            new_dist.append(new_val)
            new_target_domain.append(full_record)
    new_dist = np.array(new_dist)
    return new_target_domain, new_dist


def prediction_consistency(test_data, classifier, train_features, protected_attribute):
    n = len(test_data)
    dataset = test_data[train_features]
    protected_loc = dataset.columns.get_loc(protected_attribute)
    pc = 0
    for i in range(n):
        original_record = dataset.iloc[i]
        flipped_record = original_record.copy()
        flipped_record.iloc[0, protected_loc] = 1 - flipped_record.loc[0, protected_loc]
        if classifier.predict(original_record) == classifier.predict(flipped_record):
            pc += 1
    pc /= n
    return pc


def education_distance(ed1, ed2):
    if ed1 == ed2:
        return 0.0

    education_hierarchy = {
        'Doctorate': 15,
        'Prof-school': 14,
        'Masters': 13,
        'Bachelors': 12,
        'Assoc-acdm': 11,
        'Assoc-voc': 10,
        'Some-college': 9,
        'HS-grad': 8,
        '12th': 7,
        '11th': 6,
        '10th': 5,
        '9th': 4,
        '7th-8th': 3,
        '5th-6th': 2,
        '1st-4th': 1,
        'Preschool': 0,
    }

    # distance = abs(education_hierarchy.get(ed1, -1) - education_hierarchy.get(ed2, -1))
    distance = abs(ed1 - ed2)

    return distance / 8.0  # Normalizing to a scale of 0 to 1

def marital_status_distance(ms1, ms2):
    if ms1 == ms2:
        return 0.0

    marital_status_hierarchy = {
        'Never-married': 0,
        'Married-civ-spouse': 1,
        'Married-AF-spouse': 2,
        'Married-spouse-absent': 3,
        'Separated': 4,
        'Divorced': 5,
        'Widowed': 6,
    }
    # marital_status_hierarchy = {
    #     'Never-married': 0,
    #     'Married': 1,
    #     'Separated': 2
    # }

    distance = abs(marital_status_hierarchy.get(ms1, -1) - marital_status_hierarchy.get(ms2, -1))

    return distance / 6.0  # Normalizing to a scale of 0 to 1


def occupation_distance(oc1, oc2):
    if oc1 == oc2:
        return 0.0

    # occupation_hierarchy = {
    #     'Exec-managerial': 14, #1
    #     'Prof-specialty': 13,
    #     'Tech-support': 12, #2
    #     'Sales': 11,
    #     'Protective-serv': 10,
    #     'Craft-repair': 9, #3
    #     'Transport-moving': 8,
    #     'Adm-clerical': 7, #4
    #     'Machine-op-inspct': 6,
    #     'Farming-fishing': 5,
    #     'Armed-Forces': 1,
    #     'Handlers-cleaners': 4, #
    #     'Other-service': 3,
    #     'Priv-house-serv': 2, #
    #     '?': 0, #
    # }
    occupation_hierarchy = {
        'NOC-1': 0,
        'NOC-2': 1,
        'NOC-3': 2,
        'NOC-4': 3,
        'NOC-5': 4,
        'NOC-6': 5,
        'NOC-7': 6,
    }

    distance = abs(occupation_hierarchy.get(oc1, -1) - occupation_hierarchy.get(oc2, -1))

    return distance / 6.0  # Normalizing to a scale of 0 to 1

def sex_distance(sex1, sex2):
    return 0.0 if sex1 == sex2 else 1.0

def income_distance(income1, income2):
    return 0.0 if income1 == income2 else 1.0

def hours_per_week_distance(value1, value2):
    min_value = 1
    max_value = 99
    threshold1 = 20
    threshold2 = 40

    if max_value == min_value:  # Prevent division by zero
        return 0.0

    raw_distance = abs(int(value1) - int(value2))

    if raw_distance > threshold2:
        return min((raw_distance - threshold2) * 3 + (threshold2 - threshold1) * 2 + threshold1, max_value - min_value) / (max_value - min_value)
    elif raw_distance > threshold1:
        return min((raw_distance - threshold1) * 2 + threshold1, max_value - min_value) / (max_value - min_value)
    else:
        return raw_distance / (max_value - min_value)


def age_distance(value1, value2):
    min_value = 17
    max_value = 90

    if max_value == min_value:  # Prevent division by zero
        return 0.0

    raw_distance = abs(int(value1) - int(value2))

    d = raw_distance / (max_value - min_value)
    if (raw_distance > 10):
      d = 1
    return d



def record_distance(r1, r2, include_target_sensitive=True):
    total_distance = 0.0
    # if include_target_sensitive:
    #     total_distance += (sex_distance(r1['sex'], r2['sex']) * 0.1) ** 2
    #     total_distance += (income_distance(r1['income'], r2['income']) * 0.2) ** 2
    # total_distance += (marital_status_distance(r1['marital-status'], r2['marital-status']) * 0.1) ** 2
    # total_distance += (education_distance(r1['education-num'], r2['education-num']) * 0.15) ** 2
    # total_distance += (occupation_distance(r1['occupation'], r2['occupation']) * 0.2) ** 2
    # total_distance += (hours_per_week_distance(r1['hours-per-week'], r2['hours-per-week']) * 0.1) ** 2
    # total_distance += (age_distance(r1['age'], r2['age']) * 0.15) ** 2
    
    total_distance += (0.04 if r1['marital-status'] != r2['marital-status'] else 0) ** 2
    total_distance += (education_distance(r1['education-num'], r2['education-num']) * 0.15) ** 2
    total_distance += (occupation_distance(r1['occupation'], r2['occupation']) * 0.2) ** 2
    total_distance += (hours_per_week_distance(r1['hours-per-week'], r2['hours-per-week']) * 0.1) ** 2
    total_distance += (age_distance(r1['age'], r2['age']) * 0.15) ** 2
    # total_distance += (2 if r1['occupation'] != r2['occupation'] else 0) ** 2
    # total_distance += abs(r1['age'] - r2['age']) ** 2
    # total_distance += abs(r1['education-num'] - r2['education-num']) ** 2
    # total_distance += abs(r1['hours-per-week'] - r2['hours-per-week']) ** 2

    # total_distance += (marital_status_distance(r1['marital-status'], r2['marital-status'])) ** 2
    # total_distance += (education_distance(r1['education-num'], r2['education-num']) * 1.5) ** 2
    # total_distance += (occupation_distance(r1['occupation'], r2['occupation']) * 2) ** 2
    # total_distance += (hours_per_week_distance(r1['hours-per-week'], r2['hours-per-week'])) ** 2
    # total_distance += (age_distance(r1['age'], r2['age']) * 1.5) ** 2   
    return total_distance ** 0.5


def individual_fairness(data, num_pairs_to_sample, threshold, target_attr, method_name):

    # List to store DataFrames of similar pairs
    similar_pairs_dfs = []
    violations = 0
    severities = []
    sampled_pairs = random.sample([(i, j) for i in range(data.shape[0]-1) for j in range(i+1, data.shape[0])], num_pairs_to_sample)

    # Loop over the sampled pairs to check the Lipschitz condition
    for i, j in sampled_pairs:
    # for _ in range(num_pairs_to_sample):
    #     i, j = sorted(random.sample(range(data.shape[0]), 2))  # Sort indices to avoid duplicate pairs with reversed indices
        r1, r2 = data.iloc[i], data.iloc[j]

        # Create a tuple representing the pair
        pair = (i, j)

        distance = record_distance(r1, r2, include_target_sensitive=False)

        # Check the conditions and whether the pair is new
        if distance < threshold and r1[target_attr] != r2[target_attr] and (r1['sex'] != r2['sex']):
            violations += 1
            severities.append(threshold - distance)
            # print(f"Record {i} and Record {j} are close with a distance of {distance}")
            # print("Record 1 values:")
            # print(r1)
            # print("Record 2 values:")
            # print(r2)

            # Creating a DataFrame for the similar pair with distance as a new column
            pair_df = pd.DataFrame([r1, r2])
            pair_df['distance'] = distance

            # Appending this pair DataFrame to the list
            similar_pairs_dfs.append(pair_df)

            # Adding an empty row between pairs
            similar_pairs_dfs.append(pd.DataFrame([[None]*len(data.columns)], columns=data.columns))

    if len(similar_pairs_dfs) > 0:
        # Concatenating all pair DataFrames and saving to CSV
        result_df = pd.concat(similar_pairs_dfs)
        result_df.to_csv('violation-pairs_' + method_name + '.csv', index=False)
    return violations / num_pairs_to_sample, sum(severities)


def plot_AUC_ROD(cross_valid_result, cross_valid_fairness, methods, classifier, directory):
    data = []
    positions = []
    for method in methods:
        df_method = cross_valid_result[classifier].loc[(cross_valid_result[classifier]['method'] == method) & (cross_valid_result[classifier]['Measure of performance'] == 'AUC')]
        data.append(df_method['value'])

        df2_method = cross_valid_fairness[classifier].loc[(cross_valid_fairness[classifier]['method'] == method) & (cross_valid_fairness[classifier]['Measure of fairness'] == 'ROD')]
        positions.append(math.log(df2_method['value'].mean()))

    plt.close()
    fig, ax = plt.subplots(figsize=(12.5, 10))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(50)

    # Creating plot
    box = ax.boxplot(data, positions=positions, manage_ticks=False, widths=0.1, patch_artist=True)

    # fill with colors
    colors_dark = sns.color_palette("dark")[0:len(methods)+3]
    colors_deep = sns.color_palette("deep")[0:len(methods)+3]
    colors_bright = sns.color_palette("bright")[0:len(methods)]
    colors = [colors_dark[0],colors_bright[2],colors_bright[3],colors_deep[7],colors_deep[8],colors_deep[9],colors_dark[6]]
    patches = []
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patches.append(patch)
    
    ax.legend(handles=patches, labels=methods, loc='lower right', fontsize=26.5)
    ax.set_xlabel('ROD')
    ax.set_ylabel('AUC')
    plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    # show plot
    plt.savefig(directory + '/AUC_ROD_' + classifier + '.pdf')
    plt.show()
    plt.close()


def track_memory_and_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # in MB
        return_values = None
        
        try:
            # Run the function and capture return values
            return_values = func(*args, **kwargs)
        finally:
            end_memory = process.memory_info().rss / (1024 * 1024)  # in MB
            memory_usage = end_memory - start_memory
            
        return memory_usage, return_values
    
    return wrapper