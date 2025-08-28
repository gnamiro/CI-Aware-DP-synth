
####################################
###########    Adult    ############
####################################
# DB_NAME = 'Adult_S_Y'
# FILE_NAME = 'clean_adult'
# TARGET_ATTR = 'income'
# OUTCOME = [TARGET_ATTR]
# PROTECTED_ATTR = 'sex'
# PROTECTED_ATTRS = ['sex']
# INADMISSIBLE_ATTRS = ['marital-status']
# ADMISSIBLE_ATTRS = ['occupation', 'education-num', 'hours-per-week', 'age']
# BINARY_COLS = ['sex', 'income']

# # independence_columns = [[protected_attribute]+inadmissibles, outcome , admissibles]
# CONSTRAINT = [[PROTECTED_ATTR], OUTCOME , ADMISSIBLE_ATTRS]
# EXTRA_FEATURES = INADMISSIBLE_ATTRS
# # data_path = 'data/Adult_S_Y'
# DATA_PATH = f'data/{DB_NAME}'
# DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'

####################################
##########    Compas    ############
####################################
# DB_NAME = 'Compas_S_Y'
# TARGET_ATTR = 'is-recid'
# PROTECTED_ATTR = 'race'
# INADMISSIBLE_ATTRS = ['age-cat', 'priors-count']
# ADMISSIBLE_ATTRS = ['c-charge-degree']
# BINARY_COLS = ['is-recid', 'race', 'c-charge-degree']
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = INADMISSIBLE_ATTRS
# PROTECTED_ATTRS = [PROTECTED_ATTR]
# # independence_columns = [protected_attributes, outcome, admissibles]
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
# # independence_columns = [protected_attributes, inadmissibles, admissibles]
# DATA_PATH = f'./data/{DB_NAME}'
# DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'

####################################
##########    Dutch     ############
####################################
#### Prefair
FILE_NAME = "dutch"
DB_NAME = 'dutch'
TARGET_ATTR = 'occupation'
PROTECTED_ATTR = 'sex'
ADMISSIBLE_ATTRS = ['edu_level', 'economic_status', 'cur_eco_activity']
INADMISSIBLE_ATTRS = ['household_position', 'household_size', 'age', 'marital_status', 'country_birth', 'citizenship', 'prev_residence_place']
OUTCOME = [TARGET_ATTR]
EXTRA_FEATURES = INADMISSIBLE_ATTRS
PROTECTED_ATTRS = [PROTECTED_ATTR]
CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
# independence_columns = [protected_attributes, inadmissibles, admissibles]
DATA_PATH = f'./data/{DB_NAME}'
DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'


DELTA = 1e-9
DROP_PROTECTED_ATTRIBUTE = 0 # Used in evaluation to exclude the protected attribute from classification task!

ALGORITHMS = ['mst', 'greedy', 'opt', 'privCI', 'mst_hard']
CV = 5

MEASURES = ['Accuracy', 'AUC', 'Equalized_Odd', 'ROD', 
                                    'F1_score', 'TPR', 'FPR', 'Macro_F1', 
                                    'Macro_Precision', 'Macro_Recall', 'Demo_Par']



ML_ALGO = 'MLP'
