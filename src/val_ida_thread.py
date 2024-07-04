############################################################################
#               Tool to run IDA in parallel 

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: July 2024

# Description:  function to be callable from command line to run any row of the df

# Open issues:  This requires us to know the length of ida_df
#               Also need to save everything back to the correct pickle after finishing

############################################################################

def ida_run_row(row_num, run_case_str):
    
    # design_dict = {
    #     'gap_ratio' : 0.6,
    #     'RI' : 2.25,
    #     'T_ratio': 5.0,
    #     'zeta_e': 0.25,
    #     'isolator_system' : 'LRB',
    #     'superstructure_system' : 'CBF',
    #     'k_ratio' : 15
    # }

    import json
    input_path = './inputs/'
    with open(input_path+run_case_str+'.in') as f: 
        data = f.read() 
    design_dict = json.loads(data)
    
    # use db to prepare all IDA runs, then grab the assigned row
    from db import prepare_ida_util
    import pandas as pd
    ida_df = prepare_ida_util(design_dict)
    
    assert row_num <= len(ida_df)-1
    
    # prepare the output path
    output_path = './outputs/'+run_case_str+'/row_'+str(row_num)+'_output/'
    import os
    import shutil
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # run one single run
    thread_row = ida_df.iloc[row_num]
    gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/'
    from experiment import run_nlth
    print('========= Run %d of %d ==========' % 
          (row_num+1, len(ida_df)))
    bldg_result = run_nlth(thread_row, gm_path, output_path)
    db_results = pd.DataFrame(bldg_result).T
    
    # store the csv results
    data_path = '../data/validation/'+run_case_str+'/'
    
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    output_str = 'row_'+str(row_num)
    db_results.to_csv(data_path+output_str+'.csv', index=False)
    
    import pickle
    with open(data_path+output_str+'.pickle', 'wb') as f:
        pickle.dump(db_results, f)
    
    
import argparse
parser = argparse.ArgumentParser(
    description='Create db with IDA ready to run, then run one row of the IDA df.')
parser.add_argument('idx', metavar='i', type=int, nargs='?',
                    help='Index of the IDA df to be ran')
parser.add_argument('run_case', metavar='s', type=str, nargs='?',
                    help='String of run case to help organize output file')

args = parser.parse_args()
ida_run_row(args.idx, args.run_case)