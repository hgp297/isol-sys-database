import pandas as pd
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

pickle_path = './data/initial/fuller/'

from db import Database

# generate holder dfs
raw_input = None
tfp_designs = None
lrb_designs = None
mf_designs = None
cbf_designs = None
retained_designs = None
generated_designs = None
ops_analysis = None

# generate db object with methods
dummy_obj = Database(10)

# how many seeds used?
seed_num = 100

for i in range(seed_num):
    file_str = 'structural_db_seed_'+ str(i+1) +'.pickle'
    cur_obj = pd.read_pickle(pickle_path+file_str)
    
    # concatenate the 
    if ops_analysis is None:
        ops_analysis = cur_obj.ops_analysis.copy()
    else:
        ops_analysis = pd.concat([ops_analysis, cur_obj.ops_analysis], axis=0)
        
    if retained_designs is None:
        retained_designs = cur_obj.retained_designs.copy()
    else:
        retained_designs = pd.concat([retained_designs, cur_obj.retained_designs], axis=0)
        
    if generated_designs is None:
        generated_designs = cur_obj.generated_designs.copy()
    else:
        generated_designs = pd.concat([generated_designs, cur_obj.generated_designs], axis=0)
        
    if raw_input is None:
        raw_input = cur_obj.raw_input.copy()
    else:
        raw_input = pd.concat([raw_input, cur_obj.raw_input], axis=0)
        
ops_analysis = ops_analysis.reset_index(drop=True)
ops_analysis = ops_analysis.drop(columns=['index'])

# save to dummy object
dummy_obj.ops_analysis = ops_analysis
dummy_obj.raw_input = raw_input
dummy_obj.generated_designs = generated_designs
dummy_obj.retained_designs = retained_designs

dummy_obj.n_points = ops_analysis.shape[0]
dummy_obj.n_generated = generated_designs.shape[0]

import pickle
final_path = '../data/'
with open(final_path+'structural_db_complete.pickle', 'wb') as f:
    pickle.dump(dummy_obj, f)