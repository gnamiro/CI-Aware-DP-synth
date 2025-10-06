import numpy as np
import pandas as pd
from collections import Counter
import math

import itertools

import argparse

from Constraints import *
from results_io import ensure_results_table, set_metric
from visualization import plot_distance

import dit
from dit import Distribution
from dit.multivariate import entropy
from dit.divergences import kullback_leibler_divergence

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from scipy.stats import chi2_contingency, chi2
# import pingouin as pg
# 
from sklearn.kernel_approximation import RBFSampler
from scipy.stats import gamma
from scipy.linalg import eigh
from scipy import stats
from tqdm import tqdm

from joblib import Parallel, delayed

from pgmpy.estimators.CITests import chi_square, modified_log_likelihood, power_divergence, log_likelihood

from sklearn.cross_decomposition import CCA

import statsmodels.api as sm
from scipy.stats import chi2

from pgmpy.estimators.CITests import chi_square

def test_conditional_independence_chi(df, constraint, significance_level=0.05):
    """
    Test conditional independence of X ⊥ Y | Z for categorical data.
    
    Parameters:
    - df: DataFrame containing the data.
    - X_cols: List of columns representing variable set X.
    - Y_cols: List of columns representing variable set Y.
    - Z_cols: List of columns representing conditioning set Z.
    - significance_level: Threshold for rejecting independence (default: 0.05).
    
    Returns:
    - Dictionary with p-value and whether independence holds.
    """
    data = df.copy()
    X_cols, Y_cols, Z_cols = constraint
    
    # Combine columns into single categorical variables
    data['X'] = data[X_cols].astype(str).agg('_'.join, axis=1)
    data['Y'] = data[Y_cols].astype(str).agg('_'.join, axis=1)
    Z = []
    if Z_cols:
        data['Z'] = data[Z_cols].astype(str).agg('_'.join, axis=1)
        Z = ['Z']
    
    # Perform chi-square conditional independence test
    p_value = chi_square(X='X', Y='Y', Z=Z, data=data, boolean=False)
    
    return [p_value]




def _get_predictions_XGBoost(X, Y, Z, data, **kwargs):
    """
    Function to get predictions using XGBoost for `ci_pillai`.
    """

    xgb_params = {
        "enable_categorical": None,
        "reg_alpha": kwargs.get("reg_alpha", 1.0),     # L1 regularization (sparsity)
        "reg_lambda": kwargs.get("reg_lambda", 1.0),   # L2 regularization (ridge)
        "max_depth": kwargs.get("max_depth", 3),       # Prevent deep overfitting trees
        "seed": kwargs.get("seed"),
        "random_state": kwargs.get("seed"),
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ImportError(
            e.msg
            + ". xgboost is required for using pillai_trace test. Please install using: pip install xgboost"
        )
    xgb_params['enable_categorical'] = True
    
    # Step 2: Convert to categorical if needed
    data[X] = data[X].astype("category")
    data[Y] = data[Y].astype("category")

    if data.loc[:, X].dtype == "category":
        print('X categorical', end=' ')
        clf_x = XGBClassifier(**xgb_params)
        x, x_cat_index = pd.factorize(data.loc[:, X])
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict_proba(data.loc[:, Z])
    else:
        print("Not here")
        clf_x = XGBRegressor(**xgb_params)
        x = data.loc[:, X]
        x_cat_index = None
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict(data.loc[:, Z])

    if data.loc[:, Y].dtype == "category":
        print("Y categorical")
        clf_y = XGBClassifier(**xgb_params)
        y, y_cat_index = pd.factorize(data.loc[:, Y])
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict_proba(data.loc[:, Z])
    else:
        print("Not here")
        clf_y = XGBRegressor(**xgb_params)
        y = data.loc[:, Y]
        y_cat_index = None
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict(data.loc[:, Z])

    return (pred_x, pred_y, x_cat_index, y_cat_index)


def _get_predictions_LR(X, Y, Z, data, **kwargs):
    """
    Function to get predictions using Linear or Logistic Regression for `ci_pillai`.
    """

    from sklearn.linear_model import LogisticRegression, LinearRegression
    import numpy as np
    import pandas as pd

    # Step 1: Ensure categorical types are marked
    data[X] = data[X].astype("category")
    data[Y] = data[Y].astype("category")

    # ---------- Predict X ----------
    if data[X].dtype.name == "category":
        print("X categorical", end=" ")
        x, x_cat_index = pd.factorize(data[X])
        clf_x = LogisticRegression(max_iter=1000)
        clf_x.fit(data[Z], x)
        pred_x = clf_x.predict_proba(data[Z])  # shape: [n_samples, n_classes]
    else:
        print("X continuous", end=" ")
        clf_x = LinearRegression()
        x = data[X].values
        x_cat_index = None
        clf_x.fit(data[Z], x)
        pred_x = clf_x.predict(data[Z])  # shape: [n_samples,]
    
    # ---------- Predict Y ----------
    if data[Y].dtype.name == "category":
        print("Y categorical")
        y, y_cat_index = pd.factorize(data[Y])
        clf_y = LogisticRegression(max_iter=1000)
        clf_y.fit(data[Z], y)
        pred_y = clf_y.predict_proba(data[Z])
    else:
        print("Y continuous")
        clf_y = LinearRegression()
        y = data[Y].values
        y_cat_index = None
        clf_y.fit(data[Z], y)
        pred_y = clf_y.predict(data[Z])

    return pred_x, pred_y, x_cat_index, y_cat_index


def pillai_trace(X, Y, Z, data, boolean=True, **kwargs):
    """
    A mixed-data residualization based conditional independence test[1].

    Uses XGBoost estimator to compute LS residuals[2], and then does an
    association test (Pillai's Trace) on the residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \bot Y | Z

    Y: str
        The second variable for testing the independence condition X \bot Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \bot Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional conditional independences for categorical and ordinal data." Proceedings of the AAAI Conference on Artificial Intelligence.
    [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for Covariates. Journal of the American Statistical Association.
    [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate general linear hypothesis. Computational Statistics & Data Analysis.
    """
    # Step 1: Test if the inputs are correct
    # print(data.info())
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 1.1: If no conditional variables are specified, use a constant value.
    if len(Z) == 0:
        Z = ["cont_Z"]
        data = data.assign(cont_Z=np.ones(data.shape[0]))
    
    if not pd.api.types.is_numeric_dtype(data[X]):
        data[X] = data[X].astype("category")

    if not pd.api.types.is_numeric_dtype(data[Y]):
        data[Y] = data[Y].astype("category")

    # Step 2: Get the predictions
    pred_x, pred_y, x_cat_index, y_cat_index = _get_predictions_XGBoost(X, Y, Z, data, **kwargs)

    # Step 3: Compute the residuals
    # print(data.loc[:, X].dtype)
    if data.loc[:, X].dtype == "category":
        x = pd.get_dummies(data.loc[:, X]).loc[
            :, x_cat_index.categories[x_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_x = (x - pred_x).iloc[:, :-1]
    else:
        res_x = data.loc[:, X] - pred_x

    # print(data.loc[:, Y].dtype)
    if data.loc[:, Y].dtype == "category":
        y = pd.get_dummies(data.loc[:, Y]).loc[
            :, y_cat_index.categories[y_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_y = (y - pred_y).iloc[:, :-1]
    else:
        res_y = data.loc[:, Y] - pred_y

    # Step 4: Compute Pillai's trace.
    if isinstance(res_x, pd.Series):
        res_x = res_x.to_frame()
    if isinstance(res_y, pd.Series):
        res_y = res_y.to_frame()

    cca = CCA(scale=False, n_components=min(res_x.shape[1], res_y.shape[1]))
    res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

    cancor = []
    for i in range(min(res_x.shape[1], res_y.shape[1])):
        cancor.append(np.corrcoef(res_x_c[:, [i]].T, res_y_c[:, [i]].T)[0, 1])

    coef = (np.array(cancor) ** 2).sum()

    # Step 5: Compute p-value using f-approximation [3].
    s = min(res_x.shape[1], res_y.shape[1])
    df1 = res_x.shape[1] * res_y.shape[1]
    df2 = s * (data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
    f_stat = (coef / df1) * (df2 / (s - coef))
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Step 6: Return
    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value, res_x, res_y  # <<== added res_x and res_y


def pillai_test(df, constraint, significance_level=0.05):
    """
    Test conditional independence of X ⊥ Y | Z for categorical data.
    
    Parameters:
    - df: DataFrame containing the data.
    - X_cols: List of columns representing variable set X.
    - Y_cols: List of columns representing variable set Y.
    - Z_cols: List of columns representing conditioning set Z.
    - significance_level: Threshold for rejecting independence (default: 0.05).
    
    Returns:
    - Dictionary with p-value and whether independence holds.
    """
    X_cols, Y_cols, Z_cols = constraint
    data = df.copy()
    
    # Combine columns into single categorical variables
    data['X'] = data[X_cols].astype(str).agg('_'.join, axis=1)
    data['Y'] = data[Y_cols].astype(str).agg('_'.join, axis=1)
    Z = []
    if Z_cols:
        data['Z'] = data[Z_cols].astype(str).agg('_'.join, axis=1)
        Z = ['Z']
    
    # Perform chi-square conditional independence test
    # p_value = modified_log_likelihood(X='X', Y='Y', Z=Z_cols, data=data, boolean=False)
    coef, p_value, res_x, res_y = pillai_trace(X='X', Y='Y', Z=Z_cols, data=data, boolean=False)
    
    return [p_value, res_x, res_y, coef, 'Pilla']

def test_conditional_independence(df, constraint, significance_level=0.05):
    """
    Test conditional independence of X ⊥ Y | Z for categorical data.
    
    Parameters:
    - df: DataFrame containing the data.
    - X_cols: List of columns representing variable set X.
    - Y_cols: List of columns representing variable set Y.
    - Z_cols: List of columns representing conditioning set Z.
    - significance_level: Threshold for rejecting independence (default: 0.05).
    
    Returns:
    - Dictionary with p-value and whether independence holds.
    """
    X_cols, Y_cols, Z_cols = constraint
    data = df.copy()
    
    # Combine columns into single categorical variables
    data['X'] = data[X_cols].astype(str).agg('_'.join, axis=1)
    data['Y'] = data[Y_cols].astype(str).agg('_'.join, axis=1)
    Z = []
    if Z_cols:
        data['Z'] = data[Z_cols].astype(str).agg('_'.join, axis=1)
        Z = ['Z']
    
    # Perform chi-square conditional independence test
    p_value = modified_log_likelihood(X='X', Y='Y', Z=Z_cols, data=data, boolean=False)
    # p_value = pillai_trace(X='X', Y='Y', Z=Z_cols, data=data, boolean=False)
    
    return [p_value]



def sklearn_cmi(dist, constraint):
    """
    Compute Conditional Mutual Information (CMI) using sklearn's mutual_info_classif.

    Args:
        dist (pd.DataFrame): Dataframe containing dataset.
        constraint (list of lists): Defines [X], [Y], [Z] attributes.

    Returns:
        float: Estimated CMI(X; Y | Z).
    """

    X = dist[constraint[0]].values  
    Y = dist[constraint[1]].values.flatten() 
    Z = dist[constraint[-1]].values  

    # print(f"X shape: {X.shape}, Y shape: {Y.shape}, Z shape: {Z.shape}")


    YZ = np.column_stack((Y, Z)) 
    # print(f"YZ shape after stacking: {YZ.shape}")

    YZ_combined = np.array(["".join(map(str, row)) for row in YZ])  
    YZ_encoded = LabelEncoder().fit_transform(YZ_combined)

    Z_combined = np.array(["".join(map(str, row)) for row in Z])  
    Z_encoded = LabelEncoder().fit_transform(Z_combined)

    # print(f"X shape: {X.shape}, YZ_encoded shape: {YZ_encoded.shape}, Z_encoded shape: {Z_encoded.shape}")

    MI_X_YZ = mutual_info_classif(X, YZ_encoded)
    MI_X_Z = mutual_info_classif(X, Z_encoded)

    # print("MI(X;YZ):", MI_X_YZ)
    # print("MI(X;Z):", MI_X_Z)
    # print("()()()" * 8)

    CMI = MI_X_YZ - MI_X_Z
    return CMI

def conditional_mutual_information_entropy(dist, X, Y, Z):
    """Computes CMI(X; Y | Z) using entropy formula"""
    return entropy(dist, X+Z, rv_mode='indexes') + entropy(dist, Y+Z, rv_mode='indexes') - entropy(dist, X+Y+Z, rv_mode='indexes') - entropy(dist, Z, rv_mode='indexes')


def align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ):
    """
    Align the shapes of prob_Z, prob_XZ, and prob_YZ to match prob_XYZ for broadcasting.

    Parameters:
    - prob_Z: P(Z), smallest array (e.g., (7, 9, 6, 5)).
    - prob_XZ: P(X, Z), next largest array (e.g., (3, 7, 9, 6, 5)).
    - prob_YZ: P(Y, Z), next largest array (e.g., (2, 3, 7, 9, 6, 5)).
    - prob_XYZ: P(X, Y, Z), full joint probability array.

    Returns:
    - Aligned versions of prob_Z, prob_XZ, and prob_YZ for broadcasting with prob_XYZ.
    """
    shape_Z = prob_Z.shape
    shape_XZ = prob_XZ.shape
    shape_YZ = prob_YZ.shape
    shape_XYZ = prob_XYZ.shape

    # Determine how many dimensions each component (X, Y, Z) has in prob_XYZ
    num_X = len(shape_XYZ) - len(shape_YZ)  # Number of X dimensions
    num_Y = len(shape_YZ) - len(shape_Z)                                   # Number of Y dimensions
    num_Z = len(shape_Z)                                                   # Number of Z dimensions

    # Adjust the shapes dynamically
    aligned_prob_Z = prob_Z.reshape((1,) * num_X + (1,) * num_Y + shape_Z)  # Pad Z for X and Y dimensions
    aligned_prob_XZ = prob_XZ.reshape(shape_XZ[:num_X] + (1,) * num_Y + shape_XZ[num_X:])  # Pad XZ for Y dimensions
    aligned_prob_YZ = prob_YZ.reshape((1,) * num_X + shape_YZ[:num_Y] + shape_YZ[num_Y:])  # Pad YZ for X dimensions

    return aligned_prob_Z, aligned_prob_XZ, aligned_prob_YZ


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
      P_Z[z_ind].item() * P_XYZ[xyz_ind].item() / (P_XZ[xz_ind].item() * P_YZ[yz_ind].item() + 1e-12) + 1e-12)


  return cmi


def calculate_conditional_mutual_information(dataset, constraint):
    """
    Calculate conditional mutual information (CMI) for a generalized constraint using DataFrame operations.

    Parameters:
    - dataset: A pandas DataFrame containing the data.
    - constraint: A list of three elements, where each element is a list of attributes 
                  (e.g., [X_list, Y_list, Z_list]).
    
    Returns:
    - float: Conditional mutual information based on the specified constraints.
    """
    assert len(constraint) == 3, "Constraint must have three elements (e.g., [X, Y, Z])"
    
    flatten_constrained = [attr for proj in constraint for attr in proj]
    
    # Projection Z
    _count_Z = dataset.groupby(constraint[-1]).size() / len(dataset)
    
    # Projection X + Z
    _count_XZ = dataset.groupby(constraint[0] + constraint[-1]).size() / len(dataset)
    
    # Projection Y + Z
    _count_YZ = dataset.groupby(constraint[1] + constraint[-1]).size() / len(dataset)
    
    # Projection X + Y + Z
    _count_XYZ = dataset.groupby(flatten_constrained).size() / len(dataset)

    prob_Z = _count_Z.reset_index(name='prob_Z')
    prob_XZ = _count_XZ.reset_index(name='prob_XZ')
    prob_YZ = _count_YZ.reset_index(name='prob_YZ')
    prob_XYZ = _count_XYZ.reset_index(name='prob_XYZ')

    merged = prob_XYZ.merge(prob_XZ, how='left', on=constraint[0] + constraint[-1])
    merged = merged.merge(prob_YZ, how='left', on=constraint[1] + constraint[-1])
    merged = merged.merge(prob_Z, how='left', on=constraint[-1])

    epsilon = 1e-12
    merged.fillna(epsilon, inplace=True)

    merged['cmi'] = merged['prob_XYZ'] * np.log2(
        merged['prob_XYZ'] * merged['prob_Z'] /
        (merged['prob_XZ'] * merged['prob_YZ'] + epsilon) + epsilon
    )

    regularizer_loss = max(merged['cmi'].sum(), 0)
    
    return float(regularizer_loss)


def calculate_cmi_using_dit(dataset, constraint):
    # Step 1: Extract attribute names from the constraint
    protected_attrs, target_attrs, admissible_attrs = constraint  # Unpacking constraint

    # Step 2: Create the filtered dataset with only the relevant attributes
    attributes = protected_attrs + target_attrs + admissible_attrs  # Maintain order
    dataset = dataset[attributes]

    # Step 3: Convert dataset to tuples for probability calculation
    data_tuples = [tuple(row) for row in dataset.values]
    df_counts = pd.DataFrame(data_tuples).value_counts(normalize=True)  # Compute empirical probabilities
    probabilities = df_counts.values
    unique_tuples = [tuple(x) for x in df_counts.index]  # Extract unique tuples

    # Step 4: Create a `dit` probability distribution
    dist = Distribution(unique_tuples, probabilities)

    # Step 5: Map attribute names to their respective indices in `dataset`
    col_order = list(dataset.columns)  # Get dataset column order

    protected_indices = [col_order.index(attr) for attr in protected_attrs]
    target_indices = [col_order.index(attr) for attr in target_attrs]
    admissible_indices = [col_order.index(attr) for attr in admissible_attrs]

    # Step 6: Compute Conditional Mutual Information (CMI)
    return conditional_mutual_information_entropy(dist, protected_indices, target_indices, admissible_indices)



def manual_entropy(probabilities):
    """Compute entropy given probability distribution."""
    probabilities = probabilities[probabilities > 0]  # Avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_conditional_mutual_information_entropy(dataset, constraint):
    """
    Calculate conditional mutual information (CMI) using entropy-based formulation.

    Parameters:
    - dataset: A pandas DataFrame containing the data.
    - constraint: A list of three elements, where each element is a list of attributes 
                  (e.g., [X_list, Y_list, Z_list]).
    
    Returns:
    - float: Conditional mutual information based on entropy computation.
    """
    assert len(constraint) == 3, "Constraint must have three elements (e.g., [X, Y, Z])"
    
    flatten_constrained = [attr for proj in constraint for attr in proj]

    # Compute probability distributions from dataset
    _count_XYZ = dataset.groupby(flatten_constrained).size() / len(dataset)
    _count_XZ = dataset.groupby(constraint[0] + constraint[-1]).size() / len(dataset)
    _count_YZ = dataset.groupby(constraint[1] + constraint[-1]).size() / len(dataset)
    _count_Z = dataset.groupby(constraint[-1]).size() / len(dataset)

    # Compute entropies
    H_XYZ = manual_entropy(_count_XYZ.values)
    H_XZ = manual_entropy(_count_XZ.values)
    H_YZ = manual_entropy(_count_YZ.values)
    H_Z = manual_entropy(_count_Z.values)

    # Compute CMI using entropy difference formula
    CMI = H_XZ + H_YZ - H_XYZ - H_Z

    return max(CMI, 0)  # Ensure non-negative result

def conditional_independence_chi2(df, constraint):
    X, Y, Z = constraint

    X_col = X[0]
    Y_col = Y[0]
    Z_cols = Z  

    strata = df.groupby(Z_cols)

    total_chi2 = 0
    total_df = 0

    for _, sub_data in strata:
        contingency = pd.crosstab(sub_data[X_col], sub_data[Y_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue  # Not enough categories to compute chi2

        chi2_stat, p, dof, expected = chi2_contingency(contingency)

        total_chi2 += chi2_stat
        total_df += dof

    final_p_value = chi2.sf(total_chi2, total_df)

    return total_chi2, total_df, final_p_value


def conditional_independence_g_test(df, constraint):
    X, Y, Z = constraint

    X_col = X[0]
    Y_col = Y[0]
    Z_cols = Z

    strata = df.groupby(Z_cols)

    total_g = 0
    total_df = 0

    for _, sub_data in strata:
        contingency = pd.crosstab(sub_data[X_col], sub_data[Y_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        g_stat, p, dof, expected = chi2_contingency(contingency, lambda_="log-likelihood")

        total_g += g_stat
        total_df += dof

    final_p_value = chi2.sf(total_g, total_df)

    return total_g, total_df, final_p_value


# Define a function to compute CMI using different methods
def compute_cmi(df, constraint):
    """
    Compute CMI using three different methods: 
    - `calculate_conditional_mutual_information`
    - `calculate_cmi_using_dit`
    - `cmi` with k=4 (or k=5 for mst_reg variations)
    """
    cmi_1 = calculate_conditional_mutual_information(df, constraint)
    cmi_2 = calculate_cmi_using_dit(df, constraint)
    # cmi_3 = knncmi.cmi(constraint[0], constraint[1], constraint[2], 4, df)
    cmi_3 = conditional_mutual_information(df, constraint[0], constraint[1], constraint[2])
    cmi_4 = calculate_conditional_mutual_information_entropy(df, constraint)
    return cmi_1, cmi_2, cmi_3, cmi_4

def combine_attributes(df, columns):
    # Combine the values into a string representation
    combined_series = df[columns].astype(str).agg('_'.join, axis=1)
    # print(f"{columns},\n {combined_series}")
    
    # Encode it as a single numeric feature
    encoder = LabelEncoder()
    combined_encoded = encoder.fit_transform(combined_series)
    
    return combined_encoded

def partial_corr_test(data, constraint, method='spearman'):
    """
    Compute partial correlation using pingouin.partial_corr()

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    constraint : list
        [X_cols, Y_cols, Z_cols]. Each element is a list of column names.
    method : str
        Correlation method. Options: 'pearson' (default), 'spearman', 'kendall'.

    Returns
    -------
    result : dict
        Dictionary containing partial correlation coefficient, p-value, and details.
    """

    X_cols, Y_cols, Z_cols = constraint

    # # Validate input: pingouin.partial_corr only works for SINGLE X and SINGLE Y!
    # if len(X_cols) != 1 or len(Y_cols) != 1:
    #     print(X_cols, Y_cols)
    
    combined_X = combine_attributes(data, X_cols)
    combined_Y = combine_attributes(data, Y_cols)
    df_combined = data.copy()
    df_combined['combined_X'] = combined_X
    df_combined['combined_Y'] = combined_Y


    x_var = 'combined_X'
    y_var = 'combined_Y'

    # Combine Z variables into a single list
    covar = Z_cols if len(Z_cols) > 0 else None

    # Compute partial correlation using pingouin
    corr_result = pg.partial_corr(data=df_combined,
                                  x=x_var,
                                  y=y_var,
                                  covar=covar,
                                  method=method)

    # Extract values
    r = corr_result['r'].values[0]
    pval = corr_result['p-val'].values[0]

    # print(f"Partial correlation (r): {r:.4f}, p-value: {pval:.4g}")

    # return {
    #     'r': r,
    #     'pval': pval
    # }
    return [r, None, pval]


def run_permutation_single(data, constraint, grouped_indices, Y_cols, Y_data):
    """
    Runs a single permutation and calculates CMI.
    """
    Y_perm = Y_data.copy()

    for idx in grouped_indices:
        if len(idx) > 1:
            permuted_idx = np.random.permutation(idx)
            Y_perm[idx, :] = Y_data[permuted_idx, :]

    data_perm = data.copy()
    data_perm[Y_cols] = Y_perm

    # Compute CMI on permuted data
    T_perm_i = calculate_conditional_mutual_information(data_perm, constraint)
    return T_perm_i

def permutation_test_cmi_parallel(data, constraint, n_permutations=1000, random_state=42, n_jobs=-1, verbose=True):
    """
    Parallel permutation test for conditional independence using CMI.

    Args:
        data: pd.DataFrame
        constraint: list of lists [X_cols, Y_cols, Z_cols]
        n_permutations: number of permutations
        random_state: seed for reproducibility
        n_jobs: number of parallel workers (-1 = use all cores)
    """
    np.random.seed(random_state)

    X_cols, Y_cols, Z_cols = constraint

    # Compute the observed Conditional Mutual Information
    T_obs = calculate_conditional_mutual_information(data, constraint)
    if verbose:
        print(f"Observed CMI: {T_obs:.4f}")

    # Prepare a numpy array for Y columns
    Y_data = data[Y_cols].values.copy()

    if len(Z_cols) == 0:
        grouped_indices = [np.arange(len(data))]
    else:
        grouped = data.groupby(Z_cols, sort=False).groups  # faster, no sorting
        grouped_indices = list(grouped.values())
    print(f"Total groups: {len(grouped_indices)}")
    print(f"Groups with size > 1: {sum(len(g) > 1 for g in grouped_indices)}")
    # print(grouped_indices)
    

    # Run permutations in parallel
    T_perm = Parallel(n_jobs=n_jobs)(
        delayed(run_permutation_single)(data, constraint, grouped_indices, Y_cols, Y_data)
        for _ in tqdm(range(n_permutations))
    )

    T_perm = np.array(T_perm)

    if verbose:
        print(f"Observed CMI: {T_obs:.4f}")
        print(f"Max Permuted CMI: {np.max(T_perm):.4f}, Min Permuted CMI: {np.min(T_perm):.4f}")
        print(f"Mean Permuted CMI: {np.mean(T_perm):.4f}")
        print(f"Standard Deviation of Permuted CMI: {np.std(T_perm):.4f}")


    # p-value
    p_value = (np.sum(T_perm >= T_obs) + 1) / (n_permutations + 1)


    if verbose:
        print(f"Permutation test p-value: {p_value:.4f}")

    return [p_value]


def test_conditional_independence_loglinear(df, constraint, significance_level=0.05):
    """
    Test conditional independence using log-linear models.
    
    Parameters:
    - df: DataFrame containing the data.
    - X_cols: List of columns representing variable set X.
    - Y_cols: List of columns representing variable set Y.
    - Z_cols: List of columns representing conditioning set Z.
    - significance_level: Threshold for rejecting independence (default: 0.05).
    
    Returns:
    - Dictionary with p-value and whether independence holds.
    """
    X_cols, Y_cols, Z_cols = constraint
    data = df.copy()
    
    # Combine variables into single categorical features
    data['X'] = data[X_cols].astype(str).agg('_'.join, axis=1)
    data['Y'] = data[Y_cols].astype(str).agg('_'.join, axis=1)
    data['Z'] = data[Z_cols].astype(str).agg('_'.join, axis=1) if Z_cols else pd.Series(['constant']*len(data))
    
    # Create a contingency table
    contingency_table = data.groupby(['X', 'Y', 'Z']).size().unstack(fill_value=0)
    
    # Flatten the table and reset index
    observed_counts = contingency_table.stack().reset_index(name='count')
    
    # Fit models
    # Null model: No X-Y interaction (conditional independence)
    formula_null = 'count ~ X + Y + Z + X:Z + Y:Z'
    # Alternative model: Includes X-Y interaction
    formula_alt = 'count ~ X*Y + X:Z + Y:Z'
    
    # Fit Poisson GLM (equivalent to log-linear model)
    null_model = sm.GLM.from_formula(formula_null, data=observed_counts, family=sm.families.Poisson()).fit()
    alt_model = sm.GLM.from_formula(formula_alt, data=observed_counts, family=sm.families.Poisson()).fit()
    
    # Likelihood ratio test
    lr_stat = -2 * (null_model.llf - alt_model.llf)
    df_diff = alt_model.df_model - null_model.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    
    return {
        'p_value': p_value,
        'independent': p_value > significance_level
    }

def load_datasets(data_path, i, eps):
    """
    Load datasets for cross-validation iteration `i` across multiple `j` variations.
    """
    if eps is None:
        datasets = {
            "orig": pd.read_csv(f"{data_path}/cs={i}/train.csv")
                }
        return datasets
    else:
        datasets = {
                    "Original": pd.read_csv(f"{data_path}/cs={i}/train.csv"),
                    "PrefairG": pd.read_csv(f"{data_path}/cs={i}/greedy/eps={eps}/results_greedy_{i}.csv"),
                    "PrefairE": pd.read_csv(f"{data_path}/cs={i}/opt/eps={eps}/results_opt_{i}.csv"),
                    'MST': pd.read_csv(f"{data_path}/cs={i}/mst/eps={eps}/results_mst_{i}.csv"),
                    'PrivCI': pd.read_csv(f"{data_path}/cs={i}/privCI/eps={eps}/results_privCI_{i}.csv"),
                    'HC': pd.read_csv(f"{data_path}/cs={i}/hard_constraint/eps={eps}/results_mst_hard_{i}.csv"),
                }
        return datasets


def run_cross_validation(data_path, cv=5, test_type='CMI', eps=None):
    """
    Run cross-validation and compute mean/std for each model across multiple `j` runs.
    
    test_type: 'CMI', 'Chi', or 'G'
    """
    results = {

    }

    # The constraint structure you provided: X, Y, Z
    independence_columns_ = [PROTECTED_ATTRS, [TARGET_ATTR], ADMISSIBLE_ATTRS]
    # independence_columns_ = [protected_attributes, [target_attr], admissibles]

    # Choose which test function to use
    if test_type == 'CMI':
        test_func = compute_cmi
    elif test_type == 'Chi':
        test_func = test_conditional_independence_chi
    elif test_type == 'log':
        test_func = test_conditional_independence
    elif test_type == 'G':
        test_func = conditional_independence_g_test
    elif test_type == 'Perm':
        test_func = permutation_test_cmi_parallel
    elif test_type == 'partial':
        test_func = partial_corr_test
    elif test_type == 'pillai':
        test_func = pillai_test
    else:
        raise ValueError("Invalid test type. Choose 'CMI', 'Chi', or 'G'.")

    for i in range(cv):
        print("=" * 40)
        print(f'Cross Validation Fold {i} -- epsilon {eps}')

        datasets = load_datasets(data_path, i, eps)

        ci_values = {}
        for name, dataset in datasets.items():
            result = test_func(dataset, independence_columns_)
            if test_type in ['Chi', 'Perm', 'CMI', 'pillai', 'log']:
                ci_values[name] = result
            else:
                ci_values[name] = [result[0], result[2]]  # p-value only for Chi/G tests

        # Print results
        print(f'\nResults for test type: {test_type}')
        for name, item in ci_values.items():
            epsilon = None if name == "Original" else eps
            if item[-1] == 'pillai':
                print(f"{name}: {test_type}, P-value: {item[1]}")
            if len(item) == 4:
                print(f"{name}: {test_type} by Joint: {item[0]} -- {test_type} by entropy (DIT): {item[1]}, diff: {item[1]==item[3]}")
                set_metric(csv_path, method_name=name, fold_num=i, epsilon=epsilon, metric_name='CMI', value=item[0])
            elif len(item) == 3:
                print(f"{name}: {test_type} T_observed: {item[1]} -- T_perm: {item[2]}, P Value: {item[0]}")
            elif len(item) == 2:
                print(f"{name}: {test_type}: {item[0]}, P-value: {item[1]}")
                set_metric(csv_path, method_name=name, fold_num=i, epsilon=epsilon, metric_name='CHI_P_Val', value=item[1])
            else:
                print(f"{name}: {test_type}, P-value: {item[0]}")
                set_metric(csv_path, method_name=name, fold_num=i, epsilon=epsilon, metric_name='CHI', value=item[0][1])



    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Cross-Validation for CI Tests.')
    
    parser.add_argument('--test_type', type=str, choices=['CMI', 'Chi', 'G', 'Perm', 'partial', 'KCI', 'pillai', 'log'], default='CMI', help='Conditional Independence Test type')
    # parser.add_argument('--eps', type=str, default=None, help='Epsilon value')
      
    args = parser.parse_args()
    print(f"for datapath: {DATA_PATH}")

    ensure_results_table(csv_path, methods=methods, cv=CV)
    for e in EPS:
        dist_dict = run_cross_validation(
            data_path=DATA_PATH,
            cv=CV,
            test_type=args.test_type,
            eps=e
        )
