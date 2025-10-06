# TODO 1: Apply L2 regularization on the synthetic data during training with models (especially LR)

import sys
import os

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import catboost
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio
from Constraints import *
from tools import undummify, roc_c
from CI_test import compute_cmi, test_conditional_independence_chi, test_conditional_independence

# from utils.CMI import compute_cmi
tf.random.set_seed(42)
# Set random seed for scikit-learn
random_state = check_random_state(42)
np.random.seed(42)

def build_nn_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

models = {
    # 'LR': LogisticRegression(class_weight='balanced'),
    'LR': LogisticRegression(),
    'XGB': xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    'RF': RandomForestClassifier(n_estimators=200),
    'SVM': SVC(probability=True), # It's important to note that probability estimates from SVM may not always be well-calibrated, especially in high-dimensional or imbalanced datasets.
    'EnsembleSVC': BaggingClassifier(estimator=SVC(probability=True), n_estimators=10, random_state=42),
    'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with 128 and 64, 32 neurons
            activation='relu',  # Activation function (ReLU works well)
            solver='adam',  # Optimizer (Adam is adaptive)
            alpha=0.0001,  # L2 regularization (helps prevent overfitting)
            batch_size='auto',  # Uses mini-batches
            learning_rate='adaptive',  # Adjusts learning rate dynamically
            max_iter=500,  # Maximum number of iterations
            random_state=42,
            early_stopping=True,  # Stops training when validation loss stops improving
            verbose=False  # Prints training progress
        ),
    'CAT': catboost.CatBoostClassifier(verbose=0, random_state=42)
}

PRIV, UNPRIV = 1, 0

def constraint_attribs(df, independence_columns):
    # print(independence_columns)
    undummified_columns = independence_columns.copy()
    new_columns = []
    for i, list_of_columns in enumerate(independence_columns):
        new_columns.append([])
        for ind_column in list_of_columns:
            for col in df:
                if col.startswith(ind_column):
                    new_columns[i].append(col)
    independence_columns = new_columns
    # print(independence_columns)
    return independence_columns
    

def ml_predict(path, test_path, model_name, num_iter, proc_drop, dist):
    accuracy_vals                       = 0
    accuracy_parity_vals                = 0
    demographic_parity_vals             = 0
    TPRBalance_vals                     = 0
    TNRBalance_vals                     = 0
    conditional_demographic_parity_vals = 0
    conditional_TPRBalance_vals         = 0
    conditional_TNRBalance_vals         = 0
    auc_vals                            = 0
    rod_vals                            = 0 # 1/dom(Admissible)
    _rod_vals                           = 0
    f1_scores                           = 0
    avg_f1_scores                       = 0
    precision_vals                      = 0
    recall_scores                       = 0
    equalized_odds                      = 0
    equi_opps                           = 0
    pred_prevs                          = []
    true_prevs                          = []
    dem_pars                            = 0
    status                              = []
    

    # trainpath = f'{path}{i}.csv' if num_iter > 1 else path
    trainpath = path
    train = pd.read_csv(trainpath, encoding="utf-8")
    train_cols = train.columns.drop('repaired_weights') if dist else train.columns
    # train_cols = train.columns
    train_lables = train.pop(TARGET_ATTR)
    train_weights = train.pop('repaired_weights') if dist else None
    if proc_drop:
        train.pop(PROTECTED_ATTR)
        if proc_drop == 2:
            train.drop(columns=INADMISSIBLE_ATTRS, inplace=True)

    


    testpath = test_path
    test = pd.read_csv(testpath, encoding="utf-8")
    test = test[train_cols]
    test_lables = test.pop(TARGET_ATTR)
    if proc_drop:
        test_sex = test.pop(PROTECTED_ATTR)
        if proc_drop == 2:
            test.drop(columns=INADMISSIBLE_ATTRS, inplace=True)



    if model_name == "NN":
        # Initialize & Train the Neural Network
        input_dim = train.shape[1]
        model = build_nn_model(input_dim)
        if dist:
            # model.fit(train, train_lables, sample_weight=train_weights)
            model.fit(train, train_lables, 
            sample_weight=train_weights, 
            epochs=30, batch_size=32, validation_split=0.1)
        else:
            model.fit(train, train_lables, epochs=30, batch_size=32, verbose=1, validation_split=0.1)
            # model.fit(train, train_lables)

        # Get predictions
        predictions = (nn_model.predict(test) > 0.5).astype(int).flatten()

    else:
        model = models[model_name]

        # 
        if dist:
            model.fit(train, train_lables, sample_weight=train_weights)
        else:
            model.fit(train, train_lables)
        
        predictions = model.predict(test)

    auc = 0
    
    auc = roc_c(test, test_lables, model, method='none', plot=False)

    
    if proc_drop:
        test = pd.concat([test, test_sex], axis=1)
    test[TARGET_ATTR] = test_lables
    test['Predicted'] = predictions
    # print("----Test-----"*8)
    # print(predictions)
    # print("----Test-----"*8)

    test = test.round({'Predicted': 0})
    test["Predicted"]=test["Predicted"].astype(int)
    test_without_onehot = undummify(test)
    # race_test = test[test[PROTECTED_ATTR)s[0]].isin([0, 1])]

    tn, fp, fn, tp = confusion_matrix(test[TARGET_ATTR], test["Predicted"]).ravel()
    f1_scores = f1_score(test[TARGET_ATTR], test['Predicted'])
    avg_f1_scores = f1_score(test[TARGET_ATTR], test['Predicted'], average='macro') #F! SCore

    precision_vals = precision_score(test[TARGET_ATTR], test['Predicted'], average='macro')
    recall_scores = recall_score(test[TARGET_ATTR], test['Predicted'], average='macro')


    test['ACC'] = ((test['Predicted'] == 1) & (test[TARGET_ATTR] == 1)) | ((test['Predicted'] == 0) & (test[TARGET_ATTR] == 0))
    test['TP'] = ((test['Predicted'] == 1) & (test[TARGET_ATTR] == 1))
    test['TN'] = ((test['Predicted'] == 0) & (test[TARGET_ATTR] == 0))
    test['FP'] = ((test['Predicted'] == 1) & (test[TARGET_ATTR] == 0))
    test['FN'] = ((test['Predicted'] == 0) & (test[TARGET_ATTR] == 1))


    accuracy = test['ACC'].sum()/test.shape[0]

    accuracy_vals = accuracy
    auc_vals = auc

    
    
    # equi_odd = equalized_odds_difference(test_without_onehot[TARGET_ATTR], test_without_onehot['Predicted'], sensitive_features=test_without_onehot[PROTECTED_ATTRS[0]])
    # equalized_odds = equi_odd

    
    # dem_prity = demographic_parity_difference(test_without_onehot[TARGET_ATTR],
    #                                             test_without_onehot['Predicted'],
                                                # sensitive_features=test_without_onehot[PROTECTED_ATTRS[0]])
    # dem_pars = dem_prity

    
    print("Accuracy: " + str(accuracy_vals))
    print("f1 score " + str(f1_scores))
    print("Macro f1 score " + str(avg_f1_scores))
    print("Macro precision score " + str(precision_vals))
    print("Macro recall score " + str(recall_scores))
    print("ROC AUC Score: "+ str(auc_vals))
    print("ROD -- " + str(_rod_vals))
    print("Equalized Odd: " + str(equalized_odds))
    print("Demographic Parity: " + str(dem_pars))

    # stats = np.array([[accuracy_vals, auc_vals, equalized_odds, _rod_vals, f1_scores,
    #                     tp / (tp + fn), fp / (fp + tn),
    #                     avg_f1_scores, precision_vals, recall_scores, dem_pars]])\
    stats = np.array([[auc]])
    # parts = path.split("/")
    # filename = parts[-2] + "_" + parts[-1]
    df = pd.DataFrame(stats, columns=MEASURES)
    # print(df.head())
    return(df)
