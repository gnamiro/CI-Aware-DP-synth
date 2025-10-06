
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
DB_NAME = 'Compas'
TARGET_ATTR = 'is-recid'
PROTECTED_ATTR = 'race'
INADMISSIBLE_ATTRS = ['age-cat', 'priors-count']
ADMISSIBLE_ATTRS = ['c-charge-degree']
BINARY_COLS = ['is-recid', 'race', 'c-charge-degree']
OUTCOME = [TARGET_ATTR]
EXTRA_FEATURES = INADMISSIBLE_ATTRS
PROTECTED_ATTRS = [PROTECTED_ATTR]
# independence_columns = [protected_attributes, outcome, admissibles]
CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
# independence_columns = [protected_attributes, inadmissibles, admissibles]
DATA_PATH = f'./data/{DB_NAME}'
DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'

####################################
##########    Dutch     ############
####################################
# FILE_NAME = "dutch"
# DB_NAME = 'dutch'
# TARGET_ATTR = 'occupation'
# PROTECTED_ATTR = 'sex'
# ADMISSIBLE_ATTRS = ['edu_level', 'economic_status', 'cur_eco_activity']
# INADMISSIBLE_ATTRS = ['household_position', 'household_size', 'age', 'marital_status', 'country_birth', 'citizenship', 'prev_residence_place']
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = INADMISSIBLE_ATTRS
# PROTECTED_ATTRS = [PROTECTED_ATTR]
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
# # independence_columns = [protected_attributes, inadmissibles, admissibles]
# DATA_PATH = f'./data/{DB_NAME}'
# DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'

####################################
##########    Dutch     ############
####################################
# FILE_NAME = "Law"
# DB_NAME = 'Law'
# TARGET_ATTR = "pass_bar"
# PROTECTED_ATTR = 'race'
# ADMISSIBLE_ATTRS = ["decile3", "decile1", "grad"]
# INADMISSIBLE_ATTRS = ["indxgrp2", "cluster", "tier"]
# OUTCOME = [TARGET_ATTR]
# EXTRA_FEATURES = INADMISSIBLE_ATTRS
# PROTECTED_ATTRS = [PROTECTED_ATTR]
# CONSTRAINT = [PROTECTED_ATTRS, OUTCOME, ADMISSIBLE_ATTRS]
# # independence_columns = [protected_attributes, inadmissibles, admissibles]
# DATA_PATH = f'./data/{DB_NAME}'
# DATA_DOMAIN_PATH = f'{DATA_PATH}/domain.json'

###########################
##### Privacy Setting #####
DELTA = 1e-9
EPS = [0.1, 0.5, 1, 5, 10]
OUR_METHOD = 'HC' # HC or PrivCI
PREFAIR_METHOD = 'G' # G, E
###########################


DEFAULT_COLS = ["method_name", "fold_num", "epsilon", "CMI", "CHI", "W-dist", "KL", "TVD", "LR_AUC", "MLP_AUC"]
csv_path = f"results/{DB_NAME}.csv"
methods = ["Original", "MST", "PreFairG", "PreFairE", "PrivCI", 'HC']

DROP_PROTECTED_ATTRIBUTE = 0 # TODO: will be removed in future versions

ALGORITHMS = ['mst', 'greedy', 'opt', 'privCI', 'mst_hard']
CV = 5
MEASURES = ['AUC']
