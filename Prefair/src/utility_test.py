import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc
import json

#############################################
##########           CAR           ##########   
#############################################
# TARGET_ATTR = 'class'
# PROTECTED_ATTR = 'doors'
# INADMISSIBLE_ATTRS = []
# ADMISSIBLE_ATTRS = ['buying', 'maint', 'safety','persons']
# # binary_columns = []
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = []
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] 

TARGET_ATTR = 'poisonous'
PROTECTED_ATTR = 'gill-color-reduced'
INADMISSIBLE_ATTRS = []
ADMISSIBLE_ATTRS = ['cap-shape', 'ring-number', 'ring-type', 'gill-size', 'bruises']
OUTCOME = [TARGET_ATTR]
EXTRA_FEATURES = []
PROTECTED_ATTRS = ['gill-color-reduced'] # Independent attributes
CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
#############################################
#############     Boston         ############   
#############################################
# TARGET_ATTR = 'MEDV'
# PROTECTED_ATTR = 'AGE'
# INADMISSIBLE_ATTRS = []
# # ADMISSIBLE_ATTRS = ['CRIM', 'B', 'INDUS', 'CHAS', 'NOX', 'RM', 'ZN', 'RAD', 'TAX', 'PTRATIO']
# # ADMISSIBLE_ATTRS = ['CRIM', 'B', 'CHAS', 'RM', 'ZN', 'PTRATIO']
# ADMISSIBLE_ATTRS = ['CRIM', 'CHAS', 'RM', 'LSTAT']
# binary_columns = []
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = []
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] # Condtion attributes

#############################################
#############      Bank          ############   
#############################################
# TARGET_ATTR = 'y'
# PROTECTED_ATTR = 'age'
# INADMISSIBLE_ATTRS = []
# ADMISSIBLE_ATTRS = ['duration','contact','housing','pdays','balance']
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = []
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] # Condtion attributes


#############################################
#############     Mushroom        ###########  
#############################################
# TARGET_ATTR = 'poisonous'
# PROTECTED_ATTR = 'cap-shape'
# INADMISSIBLE_ATTRS = []
# ADMISSIBLE_ATTRS = ['gill-color-reduced', 'ring-number', 'ring-type', 'gill-size', 'bruises']
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = []
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] 

#####################################################
############      Boston Missing         ############   
#####################################################
# DB_NAME = 'Boston_Miss'
# TARGET_ATTR = 'MEDV'
# PROTECTED_ATTR = 'TAX'
# ADMISSIBLE_ATTRS =['CRIM', 'INDUS', 'NOX', 'RM','PTRATIO']
# MISS_COL = 'TAX'
# OUTCOME = [TARGET_ATTR]
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] # Condtion attributes
# EXTRA_FEATURES = []
#############################################
############       Syn          #############   
#############################################
# TARGET_ATTR = 'Y'
# PROTECTED_ATTR = 'X'
# INADMISSIBLE_ATTRS = []
# ADMISSIBLE_ATTRS = ['Z1', 'Z2']
# # binary_columns = []
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = []
# PROTECTED_ATTRS = [PROTECTED_ATTR] # Independent attributes
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS] # Condtion attributes

def best_model(model_name, params):
    params = {}
    # params['multi_class'] ='ovr'
    model = LogisticRegression()
    return model


# def auc_score_calc(model, X_test_in, y_test_in, category_num):

#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     try:
#         y_score = model.decision_function(X_test_in)
#     except:
#         y_score = model.predict_proba(X_test_in)
#     if category_num ==2:
#         y_pred  = model.predict(X_test_in)
#         roc_auc = roc_auc_score(y_test_in, y_pred)
#         return roc_auc


#     for i in range(category_num):
#     # Create binary labels for the current class
#         y_true = (y_test_in == i).astype(int)
#         # Calculate ROC curve and AUC for the current class
#         fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     return roc_auc

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


def test_utility(data_path, cv, synth_df):
    model_info_path = f'{data_path}/cs={cv}/model_info/best_model_info.json'
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['Model Name']
    params = model_info['Best Parameters']
    model = best_model(model_name, params)

    train_df = pd.read_csv(f"{data_path}/cs={cv}/train.csv")
    # val_df = pd.read_csv(f"{data_path}/cs={cv}/val.csv")
    test_df = pd.read_csv(f"{data_path}/cs={cv}/test.csv")
    orig_df = pd.concat([train_df, test_df], ignore_index=True)
    num_classes = orig_df[TARGET_ATTR].nunique()

    X_train, y_train, X_test, y_test = synth_df[PROTECTED_ATTRS + ADMISSIBLE_ATTRS], synth_df[TARGET_ATTR], test_df[PROTECTED_ATTRS + ADMISSIBLE_ATTRS], test_df[TARGET_ATTR].values
    model.fit(X_train, y_train)
    return auc_score_calc(model, X_test, y_test, num_classes)
    # auc =  auc_score_calc(model, X_test, y_test, num_classes)
    # return np.mean(list(auc.values())) if num_classes > 2 else auc
