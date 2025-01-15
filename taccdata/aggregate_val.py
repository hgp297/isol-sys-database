import pandas as pd
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

# run_case = 'mf_lrb_enhanced'

def agg_data(run_case):
    pickle_path = './data/validation/'+run_case+'/'
    
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
        csv_str = 'row_'+ str(i) +'.csv'
        
        # TODO: something weird happens here if new numpy is used to pickle?
        try:
            cur_obj = pd.read_pickle(pickle_path+file_str)
        except ModuleNotFoundError:
            cur_obj = pd.read_csv(pickle_path+csv_str)
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
        

agg_data('mf_tfp_moderate')
agg_data('cbf_tfp_moderate')
# agg_data('mf_lrb_moderate')
agg_data('mf_tfp_enhanced')
agg_data('cbf_tfp_enhanced')
# agg_data('mf_lrb_enhanced')