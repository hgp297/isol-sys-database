import pandas as pd
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

run_case = 'cbf_lrb_constructable'

pickle_path = './data/validation/'+run_case+'/'

# TODO: make directory if not present

from db import Database

# generate holder dfs
ida_df = None

# generate db object with methods
dummy_obj = Database(10)

# how many row used?
row_num = 200

for i in range(row_num):
    # try to run 200 rows, stop whenever we stop finding rows
    file_str = 'row_'+ str(i) +'.pickle'
    try:
        cur_obj = pd.read_pickle(pickle_path+file_str)
    except:
        break
    
    # concatenate the dataframe
    if ida_df is None:
        ida_df = cur_obj.copy()
    else:
        ida_df = pd.concat([ida_df, cur_obj], axis=0)
        
ida_df = ida_df.reset_index(drop=True)

# save to dummy object
dummy_obj.ida_results = ida_df

import pickle
final_path = '../data/validation/'+run_case+'/'

import os
if os.path.exists(final_path):
    pass
else:
    os.makedirs(final_path)
    
with open(final_path+run_case+'.pickle', 'wb') as f:
    pickle.dump(dummy_obj, f)