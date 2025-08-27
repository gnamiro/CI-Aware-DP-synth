

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import warnings
import xgboost as xgb
warnings.filterwarnings("ignore", category=UserWarning)
# from ci_sinkhorn_implementation import repair, cost_matrix, conditional_mutual_information, cost_matrix_ME_, repair2, repair3, cost_matrix2, hamming
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score




from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def best_model(X_train_in,y_train_in ,X_test_in, y_test_in,WE=False,w=''):
    
    classifiers = {
        'Logistic Regression': (LogisticRegression(), {'solver': ['liblinear','sag','saga','newton-cg'], 'C': [ 0.1, 1, 10, 100], 'max_iter': [100000], 'random_state':[42]}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [None, 10, 20, 30 , 40 , 50], 'random_state':[42]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [20, 50, 100, 200, 500], 'random_state':[42]}),
        'Support Vector Classifier': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'random_state':[42]}),
    }

    X_test = X_test_in.copy()
    y_test = y_test_in.copy()

    X_train = X_train_in.copy()
    y_train = y_train_in.copy()



    results = {}

    for clf_name, (clf, param_grid) in classifiers.items():
        grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5)
        if not WE:
            grid_search.fit(X_train, y_train)
        else:
            grid_search.fit(X_train, y_train,sample_weight =w)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[clf_name] = {
            'Best Params': best_params,
            'Accuracy': accuracy,
        }


    # for clf_name, metrics in results.items():

        
    #     print(f"Classifier: {clf_name}")
    #     print(f"Best Parameters: {metrics['Best Params']}")
    #     print(f"Accuracy: {metrics['Accuracy']:.4f}")
    #     print("\n")
        
    acc_max = 0

    for k in results.keys():
        if results[k]['Accuracy'] > acc_max:
            acc_max = results[k]['Accuracy']
            best_model = k
            best_param = results[k]['Best Params']
    best_param['random_state'] = 42
    best_model_name = best_model

    # if best_model == 'Logistic Regression':
    #     best_param['multi_class'] ='ovr'
    #     best_model = LogisticRegression(**best_param)

    # elif best_model == 'Decision Tree':
    #     best_model = DecisionTreeClassifier(**best_param)
        
    # elif best_model == 'Random Forest':
    #     best_model = RandomForestClassifier(**best_param)

    # elif best_model == 'Support Vector Classifier':
    #     best_param['probability'] = True
    #     best_param['decision_function_shape'] = 'ovr'
    #     best_model = SVC(**best_param)

    return best_model_name, best_param, acc_max





def select_model(MODE, EXP):
    if MODE == 'car':
        return SVC(C=10, kernel='rbf', random_state=42, probability=True, decision_function_shape='ovr')
    if MODE =='boston':
        if EXP in [0,2]:
            return SVC(C=10, kernel='linear', random_state=42, probability=True, decision_function_shape='ovr')
        elif EXP ==1:
            return RandomForestClassifier(n_estimators=50,random_state=42, probability=True, decision_function_shape='ovr')
        elif EXP ==3:
            # return KNeighborsClassifier(n_neighbors=3, decision_function_shape='ovr')
            return SVC(C=10, kernel='linear', random_state=42, probability=True, decision_function_shape='ovr')



def auc_score_calc(model, X_test_in, y_test_in, category_num):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    try:
        y_score = model.decision_function(X_test_in)
    except:
        y_score = model.predict_proba(X_test_in)
    if category_num ==2:
        y_pred  = model.predict(X_test_in)
        roc_auc = roc_auc_score(y_test_in, y_pred)
        return roc_auc


    for i in range(category_num):
    # Create binary labels for the current class
        y_true = (y_test_in == i).astype(int)
        # Calculate ROC curve and AUC for the current class
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc

def auc_score_calc(model, X_test_in, y_test_in, category_num):
    try:
        y_score = model.decision_function(X_test_in)
    except:
        y_score = model.predict_proba(X_test_in)

    # Binary classification case
    if category_num == 2:
        y_pred = model.predict(X_test_in)
        return roc_auc_score(y_test_in, y_pred)

    # Multiclass case
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    supports = dict()

    for i in range(category_num):
        y_true = (y_test_in == i).astype(int)
        supports[i] = np.sum(y_true)
        if np.sum(y_true) == 0:
            roc_auc[i] = 0.0
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute weighted AUC
    total_support = sum(supports.values())
    weighted_auc = sum(supports[i] * roc_auc[i] for i in roc_auc) / total_support if total_support > 0 else 0.0
    return weighted_auc





def data_process(df,FILE_NAME,TARGET):

    Data  = df.copy()

    # step3: Impute missing values with the mean of each column
    parity = Data.shape[0]
    Data.dropna(axis=0, inplace=True)
    Data = Data.fillna(Data.mean())
    if Data.shape[0] != parity:
        print('MISSING VALUES!')

    Y = Data[TARGET]
    X = Data.drop(TARGET, axis=1)

    ## Generate Heatmap for data
    # correlation_matrix = Data.corr()
    # plt.figure(figsize=(14, 12))  # Optional: Adjust the figure size

    # # Generate the heatmap
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    # plt.title('Correlation Heatmap')
    # plt.savefig('HeatMap/'+FILE_NAME+'_heatmap.png')
    # plt.close()
    return [X,Y, Data]


#**************************************************
#***************** ADD NOISE **********************
#**************************************************
def get_sorted_indices(arr):
    sorted_arr = sorted(arr)
    indices = [sorted_arr.index(x) for x in arr]
    return indices

def classify_number(number):
    if number == 0:
        return -1
    elif number % 2 == 0:
        return -1
    else:
        return 1
    
def sigmoid(x):
    return 1/ (1 + np.exp(-x))


# def add_correlated_noise(data, col_name, ref_col_name, noise_scale = 1 , fraction =1):
#     ref_col = data[ref_col_name]
#     ref_col_mean = ref_col.mean()
#     ref_col_std = ref_col.std()
#     data_with_noise = data.copy()
#     sign =np.array(ref_col.apply(classify_number))
#     p = 1 - fraction

#     noise_values = np.random.normal(ref_col_mean, ref_col_std, len(data_with_noise)) * noise_scale * np.random.choice([-1, 1], size=len(data_with_noise)) #+ (np.max(ref_col)/2)*sigmoid(np.array(ref_col))*sign

#     noise = np.array(sorted(noise_values))
#     noise = np.round(noise,0)
#     sorted_inds = get_sorted_indices(ref_col)
#     index_value_dict = dict()
#     for i in range(len(ref_col)):
#         index_value_dict[i] = sorted_inds[i]
#     sorted_vals = sorted(index_value_dict.items(), key=lambda x: x[1])
#     unique_indices = [0] * len(sorted_inds)
#     for i in range(len(sorted_vals)):
#         unique_indices[sorted_vals[i][0]] = i
#     sorted_noise = [noise[i] for i in unique_indices]
#     n = len(sorted_noise)
#     mask = np.random.choice([0, 1], size=n, p=[p, (1 - p)]) # p percent are ones
#     sorted_noise = sorted_noise * mask
#     # data_with_noise['noise'] = sorted_noise
#     data_with_noise.loc[:, col_name] = data_with_noise[col_name] + sorted_noise
#     return data_with_noise

def add_correlated_noise(data, col_names, ref_col_name, noise_scale=1, fraction=1):
    if isinstance(col_names, str):
        col_names = [col_names]
        
    ref_col = data[ref_col_name]
    ref_col_mean = ref_col.mean()
    ref_col_std = ref_col.std()
    data_with_noise = data.copy()
    sign = np.array(ref_col.apply(classify_number))
    p = 1 - fraction

    # Generate one noise pattern to be shared (for joint noise)
    noise_values = np.random.normal(ref_col_mean, ref_col_std, len(data)) * noise_scale
    noise_values *= np.random.choice([-1, 1], size=len(data))

    # Sort noise by reference column
    noise = np.round(np.array(sorted(noise_values)), 0)
    sorted_inds = get_sorted_indices(-ref_col)
    index_value_dict = {i: sorted_inds[i] for i in range(len(ref_col))}
    sorted_vals = sorted(index_value_dict.items(), key=lambda x: x[1])
    unique_indices = [0] * len(sorted_inds)
    for i in range(len(sorted_vals)):
        unique_indices[sorted_vals[i][0]] = i
    sorted_noise = [noise[i] for i in unique_indices]

    # Apply noise based on mask
    n = len(sorted_noise)
    mask = np.random.choice([0, 1], size=n, p=[p, 1 - p])
    sorted_noise = np.array(sorted_noise) * mask

    for col_name in col_names:
        data_with_noise[col_name] = data_with_noise[col_name] + sorted_noise

    return data_with_noise



def inject_strong_y_conditioned_noise(data, X, Y, fraction=0.1, strength=10.0, seed=None):
    """
    Injects strong, stochastic correlation between Y and X into a small fraction of data.
    This targets early-fraction CMI increase, by rewriting X with Y-conditioned categorical samples.
    """
    if seed is not None:
        np.random.seed(seed)

    df = data.copy()
    domain_X = sorted(df[X].unique())
    n_classes = len(domain_X)

    # Build Y-conditioned distributions
    y_vals = sorted(df[Y].unique())
    y_distributions = {}

    for i, y_val in enumerate(y_vals):
        # INVERT the peak: assign Y=0 → high X values, Y=1 → low X values
        peak = int((1 - i / (len(y_vals) - 1)) * (n_classes - 1))  # flipped
        logits = -np.abs(np.arange(n_classes) - peak) * strength
        probs = np.exp(logits) / np.sum(np.exp(logits))
        y_distributions[y_val] = probs


    # Select a fraction of rows to corrupt
    idx = df.index.to_numpy()
    n_corrupt = int(fraction * len(df))
    selected = np.random.choice(idx, size=n_corrupt, replace=False)

    # Replace X with samples from Y-conditioned categorical
    for i in selected:
        y_i = df.loc[i, Y]
        probs = y_distributions[y_i]
        df.at[i, X] = np.random.choice(domain_X, p=probs)

    return df



#**************************************************
#******************* FUNCTIONS ********************
#**************************************************
def column_combination(col_names, n):
    res = []
    for row in col_names:
        remain = col_names.copy()
        remain.remove(row)
        subsets = list(itertools.combinations(remain, n))
        for row2 in subsets:
            res.append([row,row2])
    return res
    

def cond_mutual_info(data, X, Y, Z, delta=1):
    # I( X ; Y | Z)
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



def bucketize(data, TARGET, n_cols =40 , n_target =5):
    MAP_df_val = {}
        

    df = data.copy()


    col = list(df.columns)
    # col.remove('MEDV')

    for row in col:
        n = n_cols
        if row == TARGET:
            n = n_target
    # Specify the step size
        

        ROUND = True
        if df[row].dtype =='float64':
            ROUND =False

        TYPE = df[row].dtype
        min_value = df[row].min()
        max_value = df[row].max()*1.1

        step = round((max_value -  min_value)/n)

        df['Bucket'] = pd.cut(df[row], bins=n)

        if ROUND:
            bucket_means = round(df.groupby('Bucket',observed= False)[row].mean())
            df['Bucket'] = df['Bucket'].map(bucket_means)
            df[row] = df['Bucket']
            df[row]=df[row].astype(int)
        else:
            bucket_means = (df.groupby('Bucket',observed= False)[row].mean())
            df['Bucket'] = df['Bucket'].map(bucket_means)
            df[row] = df['Bucket']


        df = df.drop('Bucket', axis=1)

    Data  = df.copy()
    Data[TARGET] = Data[TARGET].astype('float64')
    label_encoder = LabelEncoder()
    label_encoder.fit(Data[TARGET])  
    Data.loc[:,TARGET] = label_encoder.transform(Data[TARGET])
    classes = label_encoder.classes_
    labels = label_encoder.transform(classes)
    class_to_label = {classes[i]: labels[i] for i in range(len(classes))}
    MAP_df_val[TARGET] = class_to_label

    for row in data.columns:
        Data[row]= Data[row].astype(data[row].dtype)
    Data[TARGET] = Data[TARGET].astype('int')

    return [Data, MAP_df_val]





def encode_columns(df, MODE ):
    MAP_df_val = {}
    Data = df.copy()

    for col in Data.columns:
        if MODE=='car' and Data[col].dtype != 'object': 
            continue
        # if MODE == 'car' and col in ['doors']: # let see what happens if I decrease the domain size!
        #     Data[col] = Data[col].replace({
        #         '2': 'few',
        #         '3': 'few',
        #         '4': 'many',
        #         '5more': 'many'
        #     })
        if MODE in ['boston', 'mushroom'] :
            Data[col] = Data[col].astype('category')

        label_encoder = LabelEncoder()  
        # label_encoder.fit(Data[col])  
        Data.loc[:,col] = label_encoder.fit_transform(Data[col])

        classes = label_encoder.classes_
        labels = label_encoder.transform(classes)
        class_to_label = {classes[i]: labels[i] for i in range(len(classes))}
        MAP_df_val[col] = class_to_label
        if MODE == 'boston':
            Data[col] = Data[col].astype(int)


    missing_keys=[]
    for row in MAP_df_val.keys():
        if '?' in MAP_df_val[row].keys():
            missing_keys.append(row)
    for row in missing_keys:
        value_to_replace = MAP_df_val[row]['?']
        Data[row].replace(value_to_replace, np.nan, inplace =True)
        mean_col = Data[row].mean()
        Data[row].fillna(round(mean_col), inplace=True)
        Data[row] = Data[row].astype(int)
    
    # for col in Data.columns:

    return [Data, MAP_df_val]





def check_CMI(X,Data,TARGET,FILE_NAME, N_init, INN):
    col = list(X.columns)

    pairs=[]

    for n in range(len(col)):
        if n!= N_init : continue
        pairs.append(column_combination(col,n))
    tot= 0
    for subpair in pairs:
        for row in subpair: 
            tot+=1
    print(tot)
    CMI_result = []
    TOT = 0
    for subpair in pairs:
        for row in subpair: 
            Given_attr = list(row[1])
            if row[0] != INN : continue
            TOT +=1
    print(TOT)
    for subpair in pairs:
        for row in subpair: 
    
            Given_attr = list(row[1])
            if row[0] != INN : continue
            check_attr = [row[0]]

            cmi = cond_mutual_info(Data,check_attr,[TARGET],Given_attr)
            CMI_result.append([check_attr[0],Given_attr,cmi] )
            print(check_attr[0],', LABEL','|',Given_attr, ':', round(CMI_result[-1][2],5))


            CMI_result.sort(key=lambda x:x[2], reverse=True)
            with open('CMI_TXT/'+FILE_NAME+'_CMI.txt', 'w') as file:
                for row in CMI_result:
                    file.write(str(row[0])+' , LABEL | ' + str(row[1])+ ' : ' + str(round(row[2],5)) + '\n')
    return CMI_result




# def make_noisy(Data_clean_train,INDEPENDENT, TARGET, fraction =1, scale =1):
#     Data_dirty_train = add_correlated_noise(Data_clean_train,INDEPENDENT,TARGET,scale, fraction)


#     # Define the range [a, b] and the target range [c, d]
#     a, b = np.min(Data_dirty_train[INDEPENDENT]), np.max(Data_dirty_train[INDEPENDENT])
#     c, d = np.min(Data_clean_train[INDEPENDENT]), np.max(Data_clean_train[INDEPENDENT])
#     def map_values(value):
#         return round(((value - a) / (b - a)) * (d - c) + c)

#     Data_dirty_train[INDEPENDENT] = Data_dirty_train[INDEPENDENT].apply(map_values)
#     return Data_dirty_train

def make_noisy(Data_clean_train, INDEPENDENT, TARGET, fraction=1, scale=1):
    Data_dirty_train = add_correlated_noise(Data_clean_train, INDEPENDENT, TARGET, scale, fraction)

    if isinstance(INDEPENDENT, str):
        INDEPENDENT = [INDEPENDENT]

    for col in INDEPENDENT:
        a, b = np.min(Data_dirty_train[col]), np.max(Data_dirty_train[col])
        c, d = np.min(Data_clean_train[col]), np.max(Data_clean_train[col])

        def map_values(value):
            return round(((value - a) / (b - a)) * (d - c) + c) if b > a else value

        Data_dirty_train[col] = Data_dirty_train[col].apply(map_values)

    return Data_dirty_train



# def classify_number(x):
#     return 1 if x > 0 else -1 if x < 0 else 0

# def get_sorted_indices(values):
#     return sorted(range(len(values)), key=lambda k: values.iloc[k])

# def add_conditional_noise(data, target_col, independent_col, admissible_cols, noise_scale=1.0, fraction=1.0):
#     """
#     Adds noise to the target_col conditioned on the values of admissible_cols.
#     Noise is generated using the distribution of independent_col within each group.

#     Parameters:
#         data: pandas DataFrame
#         target_col: column name (string) to which noise is added
#         independent_col: column (string) used to generate the noise
#         admissible_cols: column name or list of names to group by
#         noise_scale: multiplier for noise magnitude
#         fraction: fraction of rows in each group that receive noise

#     Returns:
#         DataFrame with noisy target_col
#     """
#     data_with_noise = data.copy()

#     group_keys = admissible_cols if isinstance(admissible_cols, list) else [admissible_cols]

#     for _, group_data in data.groupby(group_keys):
#         indices = group_data.index
#         ref_col = group_data[independent_col]

#         ref_mean = ref_col.mean()
#         ref_std = ref_col.std() if ref_col.std() > 0 else 1.0  # avoid std=0

#         sign = np.array(ref_col.apply(classify_number))
#         p = 1 - fraction
#         noise_values = np.random.normal(ref_mean, ref_std, len(group_data)) * \
#                        noise_scale * np.random.choice([-1, 1], size=len(group_data))
#         noise = np.round(np.array(sorted(noise_values)), 0)

#         sorted_inds = get_sorted_indices(ref_col)
#         index_value_dict = {i: sorted_inds[i] for i in range(len(ref_col))}
#         sorted_vals = sorted(index_value_dict.items(), key=lambda x: x[1])
#         unique_indices = [0] * len(sorted_inds)
#         for i in range(len(sorted_vals)):
#             unique_indices[sorted_vals[i][0]] = i
#         sorted_noise = [noise[i] for i in unique_indices]

#         mask = np.random.choice([0, 1], size=len(sorted_noise), p=[p, (1 - p)])
#         sorted_noise = np.array(sorted_noise) * mask

#         data_with_noise.loc[indices, target_col] = data_with_noise.loc[indices, target_col] + sorted_noise

#     return data_with_noise

# def make_noisy(data_clean_train, independent, target, admissible, fraction=1.0, scale=1.0):
#     """
#     Adds noise to the target attribute conditioned on the independent and admissible ones,
#     then maps the resulting target values back to the original domain.

#     Parameters:
#         data_clean_train: original clean DataFrame
#         independent: name of the independent variable
#         target: name of the target variable
#         admissible: grouping columns for applying conditional noise
#         fraction: fraction of data to apply noise to
#         scale: magnitude of noise

#     Returns:
#         DataFrame with noisy target values and full target domain
#     """
#     data_dirty_train = add_conditional_noise(
#         data_clean_train,
#         target_col=target,
#         independent_col=independent,
#         admissible_cols=admissible,
#         noise_scale=scale,
#         fraction=fraction
#     )

#     # Map noisy target back to clean domain
#     a, b = np.min(data_dirty_train[target]), np.max(data_dirty_train[target])
#     c, d = np.min(data_clean_train[target]), np.max(data_clean_train[target])

#     def map_values(value):
#         return round(((value - a) / (b - a)) * (d - c) + c) if b != a else c

#     data_dirty_train[target] = data_dirty_train[target].apply(map_values)

#     return data_dirty_train





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import copy


def check_NN(Data, TARGET, INDEPENDENT, CONDITION_col, batchsize,hidden_size,lr, df_blind_repair , weights_blind_repair,TRAIN_repair, TRAIN_dirty ):
    REPAIR_NN = TRAIN_repair

    # Check if GPU is available, and use it if available
    device = torch.device("cpu")


    # Split data into training and testing sets
    X = Data[CONDITION_col + [INDEPENDENT]]
    X_train, X_test, y_train, y_test = train_test_split(np.array(X).copy(),
                                                        np.array(Data[TARGET]) , test_size=0.2, random_state=42)

    data_clean_train = pd.DataFrame(X_train, columns= X.columns)
    data_clean_train[TARGET] = y_train
    data_dirty_train = make_noisy(data_clean_train,INDEPENDENT, TARGET)
    X_train_dirty = data_dirty_train[CONDITION_col + [INDEPENDENT]]
    X_train_dirty = torch.tensor(np.array(X_train_dirty), dtype=torch.float32).to(device)



    # Convert data to PyTorch tensors and move to GPU if available
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)

    ################################### X_train = copy.deepcopy(X_train_dirty)
    if TRAIN_dirty:
        X_train = copy.deepcopy(X_train_dirty)

    if REPAIR_NN:
        repair_data = np.array(df_blind_repair[CONDITION_col + [INDEPENDENT]])
        repair_data_y = np.array(df_blind_repair[TARGET])

        X_train = torch.tensor(repair_data, dtype=torch.float32).to(device)
        y_train = torch.tensor(repair_data_y, dtype=torch.int64).to(device)
        sw_weight = torch.tensor(weights_blind_repair, dtype=torch.float64).to(device)


    




    # Create DataLoader for training and testing data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)


    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, len(list(Data[TARGET].unique()))), 
        nn.Softmax(dim=1)).to(device)

    # Choose a Loss Function and Optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    num_epochs = 550  # You can start with a large number of epochs
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_test_loss = float('inf')
    patience = 10  # Number of epochs without improvement to wait for early stopping
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0

        if REPAIR_NN:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                weighted_loss = torch.sum(loss * sw_weight) / torch.sum(sw_weight)
                weighted_loss.backward()
                optimizer.step()


                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_loss += weighted_loss.item()  # Use weighted loss here


        else:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_loss += loss.item()

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_accuracy)

        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0

        with torch.no_grad():


            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                test_loss += loss.item()

            test_accuracy = correct_test / total_test
            test_losses.append(test_loss / len(test_loader))
            test_accs.append(test_accuracy)

            # Early stopping based on validation loss
            if test_losses[-1] < best_test_loss:
                best_test_loss = test_losses[-1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    # print(f'Early stopping after {epoch + 1} epochs.')
                    break

    print(round(test_accuracy,5))




def My_repair(Data_in, INDEPENDENT, TARGET, CONDITION_col, MODE, TH = 1e-3, init_cmi =False, final_cmi =False, TYPE=1,bg_coef = 0,C_dist = [],COST =False, cost_dict = {} , BIG = 1000, MM = [], MMM =[]):
    df = Data_in.copy()
    independence_columns = [[INDEPENDENT], [TARGET], CONDITION_col]
    if MODE == 'BLIND':
        normalization_coeffs = {'S':1.0,'N':1.0,'Y':1.0,'A':1.0}
    else:
        # normalization_coeffs = {'S':0.000000000000000000000000000000000001,'N':1300,'Y':1300,'A':1100}
        normalization_coeffs = {'S':0.000000000000000000000000000000000000000000000000000000000000000000000000000001,'N':20000,'Y':2000,'A':1412} ## BOSTON
        
        #fig5
        if bg_coef != 0 : 
            normalization_coeffs = {'N':0.0000001,'S':1,'Y':10,'A':10}


    coeff_1=[1/150]#150
    coeff_2=[100]#100

    
    threshold=TH


    if TYPE == -1:
        df_repaired, weights_repaired, _, _, _, _ = repair(df,independence_columns, 'fairness', normalization_coeffs,
                                                                    return_dist=True, repair_y=False, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
                                                                    custom_cost=False, verbose=True, convergence_threshold=threshold, 
                                                                    target_attr=[TARGET], cost_function=cost_matrix, C_DIST = C_dist)#hamming
    

    elif TYPE == 0: 

        df_repaired, weights_repaired, _, _, _, _ = repair(df,independence_columns, 'data cleaning', normalization_coeffs,
                                                                    return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
                                                                    custom_cost=True, verbose=True, convergence_threshold=threshold, 
                                                                    target_attr=[TARGET], cost_function=cost_matrix, C_DIST = C_dist)#hamming
    

    elif TYPE ==1:
        df_repaired, weights_repaired, _, _, _, _ = repair(df,independence_columns, 'data cleaning', normalization_coeffs,
                                                                    return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
                                                                    custom_cost=True, verbose=True, convergence_threshold=threshold, 
                                                                    target_attr=[TARGET], cost_function=hamming, C_DIST = C_dist)#hamming
    else: 
        df_repaired, weights_repaired, _, _, _, _ = repair(df,independence_columns, 'data cleaning', normalization_coeffs,
                                                                    return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
                                                                    custom_cost=True, verbose=True, convergence_threshold=threshold, 
                                                                    target_attr=[TARGET], cost_function=cost_matrix2, C_DIST = C_dist)#hamming

    # else:
    #     # df_repaired, weights_repaired, _, _, _, _ = repair_custom_cost(clean, dirty, INDEPENDENT, df,independence_columns, 'data cleaning', normalization_coeffs,
    #     #                                                         return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
    #     #                                                         custom_cost=True, verbose=True, convergence_threshold=threshold, 
    #     #                                                         target_attr=[TARGET])


    #     df_repaired, weights_repaired, _, _, _, _ = repair2(df,independence_columns, 'data cleaning', normalization_coeffs,
    #                                                             return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
    #                                                             custom_cost=True, verbose=True, convergence_threshold=threshold, 
    #                                                             target_attr=[TARGET], cost_function=cost_matrix_ME_, cost_dict=cost_dict, BIG =BIG)


        # df_repaired, weights_repaired, _, _, _, _ = repair3(df,independence_columns, 'data cleaning', normalization_coeffs,
        #                                                         return_dist=True, repair_y=True, coeff_1=coeff_1, coeff_2=coeff_2, saturated=True,
        #                                                         custom_cost=True, verbose=True, convergence_threshold=threshold, 
        #                                                         target_attr=[TARGET], MM = MM, MMM =MMM)




    for col in df_repaired.columns:
        df_repaired[col] = df_repaired[col].astype(int)

    initial_CMI = -1
    final_CMI = -1
    if init_cmi:
        initial_CMI = conditional_mutual_information(df, independence_columns[0],independence_columns[1],independence_columns[2])
    if final_cmi:
        final_CMI = conditional_mutual_information(df_repaired, independence_columns[0],independence_columns[1],independence_columns[2])

    return df_repaired, weights_repaired , initial_CMI, final_CMI







def Inject_missing(Data_IN, TARGET, MISS_METHOD, MISS_col, LABEL_dep, fraction):
    Data_in = Data_IN.copy()
    num_rows_to_replace = int(fraction * len(Data_in))

    if MISS_METHOD == 'MAR':
        tmp =Data_in[MISS_col][Data_in[TARGET]==LABEL_dep] ## 20 percent of rows that Lebel is LABEL_dep
        num_rows_to_replace = int(fraction * len(tmp))
        # if num_rows_to_replace > len(tmp):
        #     num_rows_to_replace = len(tmp)
        #     print('WARNINGL: ALL ROWS are missing in MAR')
        indices_to_replace = np.random.choice(tmp.index, num_rows_to_replace, replace=False)
        Data_in.loc[indices_to_replace, MISS_col] = np.nan

    elif MISS_METHOD == 'MCAR':
        num_rows_to_replace = int(fraction * len(Data_in))
        indices_to_replace = np.random.choice(Data_in.index, num_rows_to_replace, replace=False)
        Data_in.loc[indices_to_replace, MISS_col] = np.nan

    elif MISS_METHOD == 'MNAR':
        # tmp =Data_in[MISS_col][Data_in[MISS_col]==LABEL_dep] ## 20 percent of rows that Lebel is LABEL_dep
        # num_rows_to_replace = int(fraction * len(tmp))
        # # if num_rows_to_replace > len(tmp):
        # #     num_rows_to_replace = len(tmp)
        # #     print('WARNINGL: ALL ROWS are missing in MAR')
        # indices_to_replace = np.random.choice(tmp.index, num_rows_to_replace, replace=False)
        # Data_in.loc[indices_to_replace, MISS_col] = np.nan

        p_init  = dict(Data_in[MISS_col].value_counts() / len(Data_in))

        random_state = np.random.RandomState(42)
        all_K = p_init.keys()

        random_numbers = random_state.rand(len(all_K)) * fraction 

        my_sum = 0
        p_final = {}
        for i,k in enumerate(all_K):
            if i == len(all_K)-1 : break
            my_sum += p_init[k]*random_numbers[i]
            p_final[k] = random_numbers[i]

        p_final[list(all_K)[-1]] =  (fraction - my_sum) /  p_init[list(all_K)[-1]]


        for idx in Data_in.index:
            val = Data_in.loc[idx][MISS_col]
            p_final[val]
            action = [np.random.rand() < p_final[val] for _ in range(1)][0]

            if action : 
                Data_in.loc[idx, MISS_col] = np.nan
                
        # np.sum(Data_in['doors'].value_counts()) / len(Data_in)

    return Data_in



def clac_p_berno(MISS_COND, val):
    if MISS_COND == [[]]: 
        return 0.5
    x= 0
    for row in MISS_COND:
        k = val[row[0]]
        x += row[1][k]
    # return (np.exp(x)-1)/(np.exp(len(MISS_COND))-1)
    # return (x/len(MISS_COND))
    return (np.arctan(x)/np.arctan(len(MISS_COND)))



def Inject_miss_berno(Data_IN, MISS_col, MISS_COND, fraction):
    
    Data_in  = Data_IN.copy()
    total_nans =  int(len(Data_in) * fraction) 
    number_nans = Data_in[MISS_col].isna().sum()

    while number_nans < total_nans: 
        Data_in = Data_in.sample(frac=1)
        valid_indexes = Data_in[Data_in[MISS_col].notna()].index
        
        for early,idx in enumerate(valid_indexes):
            
            p_berno = clac_p_berno(MISS_COND, Data_in.loc[idx])
            if early == len(valid_indexes)-1 and p_berno == 0 :
                number_nans = total_nans
                break
            if p_berno == 0: continue

            action = [np.random.rand() < p_berno for _ in range(1)][0]
            if action : 
                Data_in.loc[idx, MISS_col] = np.nan
            number_nans = Data_in[MISS_col].isna().sum()
            break

    
    return Data_in


# Use groupby and apply to select random rows
def get_random_row(group):
    random_row_index = group.sample(1 , random_state = 42).index[0]  # Get the index of the random row
    return random_row_index








def myrepair_(Data_dirty_in, VAR_in):
    
    [model_clean_orig_in,num_classes,X_test, y_test, CONDITION_col, INDEPENDENT, TARGET ] = VAR_in

    Data_repaired_train_BLIND, weights_repaired_blind , CMI_dirty, CMI_blind_repair = My_repair(Data_dirty_in.copy(), INDEPENDENT,TARGET, CONDITION_col, 'BLIND', 1e-2 ,False, False, TYPE = 0)
    model_repair_blind = copy.deepcopy(model_clean_orig_in)
    model_repair_blind = model_repair_blind.fit(
                                Data_repaired_train_BLIND[CONDITION_col + [INDEPENDENT]],
                                Data_repaired_train_BLIND[TARGET].astype('category'), sample_weight= weights_repaired_blind)
    y_repair_blind_pred = model_repair_blind.predict(X_test)
    acc_repair_blind = accuracy_score(y_test.astype('category'), y_repair_blind_pred)
    auc_repair_blind = auc_score_calc(model_repair_blind, X_test, y_test, num_classes)
    f1 = f1_score(y_test.astype('category'), y_repair_blind_pred.astype(int), average='weighted')


    return auc_repair_blind, f1
    

def mystat(auc_dirty_std, auc_repair_std):

    means_dirty = np.average(np.average(auc_dirty_std, axis =2),axis=1)
    std_dirty = np.std(np.std(auc_dirty_std, axis =2),axis=1)
    means_repair = np.average(np.average(auc_repair_std, axis =2),axis=1)
    std_repair = np.std(np.std(auc_repair_std, axis =2),axis=1)

    return [means_dirty,std_dirty, means_repair,std_repair ]


def xgb_impute(data, column):
    y = data[column]
    X = data.drop(column, axis=1)

    X1 = X[y.notnull()]
    y1 = y[y.notnull()]

    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X1, y1)
    predicted = model.predict(X[y.isnull()])

    data.loc[data[column].isnull(), column] = predicted
    data[column] = data[column].round(0)
    return data


def mytrain(Data_dirty_train_in,VAR_in ): 
    [model_clean_orig_in,num_classes,X_test, y_test, CONDITION_col, INDEPENDENT, TARGET ] = VAR_in

    model_dirty = copy.deepcopy(model_clean_orig_in)
    model_dirty = model_dirty.fit(Data_dirty_train_in[CONDITION_col + [INDEPENDENT]], Data_dirty_train_in[TARGET].astype('category'))
    y_dirty_pred = model_dirty.predict(X_test)
    acc = accuracy_score(y_test.astype('category'), y_dirty_pred)
    auc = auc_score_calc(model_dirty, X_test, y_test, num_classes)
    f1 = f1_score(y_test.astype('category'), y_dirty_pred.astype(int), average='weighted')

    return auc, f1


# from ci_sinkhorn_implementation import format_to_3, generate_initial_distribution, normalize, compute_target_domain, extracting_domain, unnormalize, NMF

def my_dist(df, independence_columns):

    x1_domain, x2_domain, z_domain, dim, _ = extracting_domain(df, independence_columns)
    target_domain = compute_target_domain(x1_domain, x2_domain, z_domain)

    plain_columns = independence_columns[0] + independence_columns[1] + independence_columns[2]
    df_without_W = df[plain_columns]

    x_domain_plain = df_without_W.drop_duplicates().values.tolist()
    x_domain = []
    for x_plain in x_domain_plain:
        x = format_to_3(x_plain, independence_columns)
        x_domain.append(x)
 

    p_x_plain = dict(df_without_W.value_counts(normalize=True))
    p_x = {}
    for key in p_x_plain:
        p_x[tuple(format_to_3(key, independence_columns))] = p_x_plain[key]

    initial_distribution = generate_initial_distribution(x_domain, target_domain, p_x)

    slice_size = dim[1] * dim[2]

    initial_distribution2 = np.array(initial_distribution)
    initial_distribution2 = initial_distribution2 / initial_distribution2.sum(0)
    initial_distribution2 = initial_distribution2[initial_distribution2 > 0] * df.shape[0]
    

    b = []
    model = NMF(solver='mu', n_components=1, init='random', beta_loss='kullback-leibler')
    for count, i in enumerate(z_domain):
        
        X = np.reshape(initial_distribution[count * slice_size:(count + 1) * slice_size], (dim[1], dim[2]),order='F')
        W = model.fit_transform(X)
        H = model.components_
        b += list((W * H).flatten('F'))

    b = np.array(b, dtype=np.float64)


    dist_result = np.array(b)
    dist_result = dist_result / dist_result.sum(0)
    dist_result = dist_result[dist_result > 0] * df.shape[0]
    

    return dist_result, initial_distribution2



from scipy.stats import wasserstein_distance


def EMD(df1, df2, INDEPENDENT, TARGET, CONDITION_col):
    independence_columns = [[INDEPENDENT], [TARGET], CONDITION_col]

    dist2,b =my_dist(df2.copy(), independence_columns)
    dist1,b =my_dist(df1.copy(), independence_columns)

    emd = wasserstein_distance(dist2, dist1)

    return emd
