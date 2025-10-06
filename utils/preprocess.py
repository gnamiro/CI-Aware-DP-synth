'''
TODO 1: Remove preprocessing for adult dataset and job dataset.
            - Now we have split_df which mainly does the job.
            - For converting we should impelement them in the preprocessing.py in the root dir
TODO 2: Change the main method:
    - Make it Functional
    - Add Arg params
'''


import sys
import os
import argparse

import pickle
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from Constraints import *
from functions import encode_columns, data_process, make_noisy, best_model, bucketize, Inject_miss_berno

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

    

    
def preprocessing_adult(df, sex_map, income_map, df_map, main_folder, save_map_path):
    
    income_mapping = {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.':1}
    sex_mapping = {'Male': 1, 'Female':0}
    # race_mapping = {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4}
    df['income'] = df['income'].map(income_mapping)
    df['sex'] = df['sex'].map(sex_mapping).astype(int)
    # df['race'] = df['race'].map(race_mapping)
    split_df(df, df_map, main_folder, save_map_path)

def convert_type(df, columns, source_type, target_type):
    for col in columns:
        if df[col].dtype == source_type:
            df[col] = df[col].astype(target_type)
        else:
            raise('Wrong dtype for column')
    
    return df
   
def preprocessing_for_rec(df, degree_map, df_map, main_folder, save_map_path):
    degree_mapping = {'bachelor': 1, 'master':0} #Priv 1, UnPriv 0
    if degree_map:
        df['degree'] = df['degree'].map(degree_mapping)
    df = df.loc[~df['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]
    df['is-recid'] = df['is-recid'].apply(lambda x: 1-x)
    df['age-cat'] = df['age-cat'].apply(lambda x: adjustAge(x))
    df['priors-count'] = df['priors-count'].apply(lambda x: quantizePrior_old(x))

    # print(df.isnull().sum())

    bool_columns = df.select_dtypes(include=['bool'])
    df = convert_type(df,bool_columns.columns, bool, int)
    # print(df.isnull().sum())
    # Select columns with object data type
    split_df(df, df_map, main_folder, save_map_path)
    return df 


def adjustAge(x):
    if x == 'Less than 25':
        return 0
    elif x == '25 - 45':
        return 1
    else:
        return 2


def quantizePrior_old(x):
    if x <=0:
        return 0
    elif 1<=x<=3:
        return 1
    else:
        return 2

def quantizePrior_old(x):
    if x <=0:
        return 0
    elif 1<=x<=3:
        return 1
    else:
        return 2

def quantizePrior(x):
    if x <=0:
        return 0
    # elif x>=5:
    #     return 5
    else:
        return x

def adjustAge(x):
    if x == 'Less than 25':
        return 0
    elif x == '25 - 45':
        return 1
    else:
        return 2

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


def group_edu(x):
    if x <= 5:
        return 5
    elif x >= 13:
        return 13
    else:
        return x


def reduce_domain_size(df):
    
    df['age'] = df['age'].apply(lambda x: age_cut(x))

    df['hours-per-week'] = df['hours-per-week'].apply(lambda x: hours_cut(x))

    df['education-num'] = df['education-num'].apply(lambda x: group_edu(x))

    df['occupation'] = df['occupation'].apply(lambda x: occupation_cut(x))
    df['marital-status'] = df['marital-status'].apply(lambda x: 'Never-married' if x == 'Never-married' else 'Married' if 'Married' in x else 'Seperated')


    income_mapping = {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.':1}
    sex_mapping = {'Male': 1, 'Female':0}
    df['income'] = df['income'].map(income_mapping)
    df['sex'] = df['sex'].map(sex_mapping).astype(int)

    return df


def load_dataset(filename):
    dataframe = None
    if filename == 'Adult':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        dataframe = pd.read_csv(url, header=None,
                         names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                "occupation",
                                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                                "native-country", "income"], sep=',\s*', engine='python')
        dataframe = reduce_domain_size(dataframe)

    elif filename == 'Compas':
        dataframe = pd.read_csv(f'{DATA_PATH}/compas-scores-two-years.csv')
        dataframe = dataframe.loc[~dataframe['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]
        dataframe['is-recid'] = dataframe['is-recid'].apply(lambda x: 1-x)
        dataframe['age-cat'] = dataframe['age-cat'].apply(lambda x: adjustAge(x))
        dataframe['priors-count'] = dataframe['priors-count'].apply(lambda x: quantizePrior_old(x))
    return dataframe[PROTECTED_ATTRS+ADMISSIBLE_ATTRS+INADMISSIBLE_ATTRS+OUTCOME]

def load_mushroom_subset():
    df = pd.read_csv(f'{DATA_PATH}/agaricus-lepiota.data', header=None)
    df.columns = [
        'poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    return df

def load_bank_dataset():
    df = pd.read_csv(f"./data/Bank/bank_undersampled.csv")
    return df

def load_nursery_dataset(nursery):
    # data (as pandas dataframes) 
    X = nursery.data.features 
    y = nursery.data.targets 

    df = pd.DataFrame(X, columns=nursery.data.feature_names)
    df['class'] = y
    df = df[df['class'] != df['class'].value_counts().index[-1]]
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    return df_encoded


def dump_domain(df):
    print(df.info())
    # domain_info = {col: int(df[col].max()) for col in df.columns}
    domain_info = {col: int(df[col].nunique()) for col in df.columns}
    print(f'[INFO] domain of dataset: {domain_info}')
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(f"{DATA_PATH}/domain.json", 'w') as f:
        json.dump(domain_info, f, indent=4)
    
    print(f"Domain information saved in {DATA_PATH}/domain.json")



def data_split(data):
    from sklearn.model_selection import StratifiedKFold

    data = data[ADMISSIBLE_ATTRS + PROTECTED_ATTRS + INADMISSIBLE_ATTRS + [TARGET_ATTR]]
    label_encoders = {}
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))  # ensure all values are strings
        label_encoders[col] = le  # save encoder if you need to inverse_transform later

    X = data[ADMISSIBLE_ATTRS + INADMISSIBLE_ATTRS + PROTECTED_ATTRS]
    Y = data[TARGET_ATTR]
    print("[INFO] Starting data split and preprocessing...")
    print("[INFO] Attributes: ")
    print(f"\t[DEBUG] X shape: {data.nunique()}")
    original_data = data.copy()
    validation_size = 0.1
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        print(f"[INFO] Processing split {i + 1}/5")
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        X_train[TARGET_ATTR] = y_train
        X_test[TARGET_ATTR] = y_test

        split_path = f"{DATA_PATH}/cs={i}"
        os.makedirs(split_path, exist_ok=True)

        X_train.to_csv(f"{split_path}/train.csv", index=False)
        X_test.to_csv(f"{split_path}/test.csv", index=False)

        print(f"[INFO] Saved train/test data for split {i + 1}")

def data_cleaning_preprocess(data):
    data =  data[ADMISSIBLE_ATTRS+ [PROTECTED_ATTR]+[TARGET_ATTR]]
    X = data[ADMISSIBLE_ATTRS+ [PROTECTED_ATTR]].values
    Y = data[TARGET_ATTR].values
    cv = RepeatedStratifiedKFold(n_splits=CV, n_repeats=1, random_state=42)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
        print(f"[INFO] Processing CV split {i + 1}/5")
        
        train = data.iloc[train_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)

        model_name, best_params, accuracy = best_model_info(train, test)
        best_model_metadata = {
            'Model Name': model_name,
            'Best Parameters': best_params,
            'Accuracy': accuracy
        }

        split_path = f'{DATA_PATH}/cs={i}'
        os.makedirs(f'{split_path}/model_info', exist_ok=True)

        with open(f'{split_path}/model_info/best_model_info.json', 'w') as f:
            json.dump(best_model_metadata, f, indent=4)

        train.to_csv(f"{split_path}/train.csv", index=False)
        test.to_csv(f"{split_path}/test.csv", index=False)
        print(f"[INFO] Saved train/test data and model info for split {i}")
        
        train_copy = train.copy()
        for f in FRACTIONS:
            dirty_train = make_noisy(Data_clean_train=train_copy, INDEPENDENT=PROTECTED_ATTR,
                                      TARGET=TARGET_ATTR, fraction=f)
            # make_noisy(Data_clean_train,INDEPENDENT, TARGET , fraction= f)


            frac_path = f'{split_path}/frac={f}'
            os.makedirs(frac_path, exist_ok=True)
            dirty_train.to_csv(f"{frac_path}/dirty_train.csv", index=False)
            print(f"[INFO] Saved noisy dataset for split {i}, fraction {f}")


def car_preprocess(data):
    FILE_NAME = 'CAR'
    data, _ = encode_columns(data, 'car')
    _, _, data = data_process(data, FILE_NAME, TARGET_ATTR) # check for missing and plot correlation heatmap
    dump_domain(data)
    # num_classes = len(list(data[TARGET_ATTR].unique()))
    data_split(data)
    # data_cleaning_preprocess(data)

def boston_preprocess(data):
    [data, MAP_df_val] = bucketize(data, TARGET_ATTR, n_cols = 5, n_target =4)
    [data, MAP_df_val] = encode_columns(data, 'boston')
    ENCODE =  False
    FILE_NAME = 'BOSTON'
    [X,Y, data] = data_process(data, FILE_NAME, TARGET_ATTR) 
    dump_domain(data)
    data_split(data)
    # data_cleaning_preprocess(data)

def mushroom_preprocess(data):
    from sklearn.preprocessing import LabelEncoder
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col])
    
    dump_domain(data)
    data_split(data)

def pass_preprocess(data):
    discretize_numeric = True
    label_col="pass_bar"
    num_bins=5

    np.random.seed(42)
    rng = np.random.default_rng(42)

    data.replace(
        to_replace=["unknown", "Unknown", "NA", "na", "NaN", ""],
        value=np.nan,
        inplace=True,
    )

    # 3) Convert string TRUE/FALSE to 1/0 (preserve NaNs)
    for col in data.columns:
        s = data[col]
        # Only consider object-like or mixed types
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            non_na_str = s.dropna().astype(str).str.lower()
            if not non_na_str.empty and non_na_str.isin({"true", "false"}).all():
                data[col] = (
                    s.astype(str).str.lower().map({"true": 1, "false": 0})
                    .astype("Int64")  # nullable int
                )
    # 4) Strip quotes/spaces from string columns WITHOUT turning NaN into "nan"
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip().str.strip('"').str.strip("'")

    # 5) Discretize numeric features (excluding label)
    if discretize_numeric:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)
        for col in numeric_cols:
            try:
                # qcut can create NaNs for constant columns; handled later
                data[col] = pd.qcut(data[col], q=num_bins, labels=False, duplicates="drop")
            except Exception as e:
                print(f"[cleaner] Could not discretize {col}: {e}")

    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        try:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,  # sklearn >= 1.1
            )
            data[cat_cols] = enc.fit_transform(data[cat_cols]).astype("int64")
        except TypeError:
            # Fallback for older sklearn: encode unknown as -1, then convert NaN to -1
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            arr = enc.fit_transform(data[cat_cols])
            # Replace NaN from encoder with -1
            arr = np.where(np.isnan(arr), -1, arr)
            data[cat_cols] = arr.astype("int64")

    # 7) Impute NaN/-1 using values drawn from OBSERVED categories in each column
    #    (prevents introducing unseen categories, e.g., 2 in a binary column)
    for col in data.columns:
        s = data[col]
        # Work with numeric-like columns (after encoding/discretization most are numeric/Int64)
        mask = s.isna() | (s == -1)
        if mask.any():
            valid = s[~s.isna() & (s != -1)].unique()
            if valid.size > 0:
                data.loc[mask, col] = rng.choice(valid, size=int(mask.sum()))
            else:
                # If the whole column is missing, default to 0
                data.loc[mask, col] = 0

    constant_cols = [c for c in data.columns if data[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"[cleaner] Dropping constant columns: {constant_cols}")
        data = data.drop(columns=constant_cols)

    # 8) Convert all columns to plain int (after imputation there should be no NaN/-1 carryover)
    data = data.astype(int)

    # 9) Drop leaky columns
    LEAKY_COLS = [
        "bar1", "bar2", "bar1_yr", "bar2_yr", "bar", "bar_passed",
        "Dropout", "dnn_bar_pass_prediction"
    ]
    data = data.drop(columns=[c for c in LEAKY_COLS if c in data.columns], errors="ignore")


    return data


def split_data(data):
    dump_domain(data)
    data_split(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for DP synthetic data generation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to preprocess"
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    if dataset_name.lower() == 'adult':
        df = load_dataset('Adult')
        ord_enc = OrdinalEncoder()
        for column in BINARY_COLS:
            df.loc[:, column] = ord_enc.fit_transform(df[[column]])
        df[BINARY_COLS] = df[BINARY_COLS].apply(pd.to_numeric)
        df.to_csv(f"{DATA_PATH}/adult.csv")
        print(df.info())
        split_data(df)

    elif dataset_name.lower() == 'compas':
        df = load_dataset(dataset_name)
        ord_enc = OrdinalEncoder()
        for column in BINARY_COLS:
            df.loc[:, column] = ord_enc.fit_transform(df[[column]])
        df[BINARY_COLS] = df[BINARY_COLS].apply(pd.to_numeric)
        df.to_csv(f'{DATA_PATH}/compas.csv')
        split_data(df)
    
    elif dataset_name.lower() == 'dutch':
        df = pd.read_csv(f'{DATA_PATH}/dutch.csv')
        df = df[ADMISSIBLE_ATTRS+PROTECTED_ATTRS+INADMISSIBLE_ATTRS+OUTCOME]
        split_data(df)
    
    elif dataset_name.lower() == 'law':
        df = pd.read_csv(f'{DATA_PATH}/bar_pass_1.csv')
        df = df[ADMISSIBLE_ATTRS+PROTECTED_ATTRS+INADMISSIBLE_ATTRS+OUTCOME]
        # df = pass_preprocess(df) # --> Preparation for pass dataset

        split_data(df)