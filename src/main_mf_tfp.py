############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2024

# Description:  Main workfile for TFP-MF database, reworked in new framework

# Open issues:  

############################################################################

# from db import Database

# main_obj = Database(100, n_buffer=8, seed=130, 
#                     struct_sys_list=['MF'], isol_wts=[1, 0])

# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)

# main_obj.scale_gms(repeat=11)

#%% troubleshoot fatal case

# # run 110/400
# # run 364/400
# # troubleshoot building
# run = main_obj.retained_designs.iloc[364]
# from building import Building

# bldg = Building(run)
# bldg.model_frame()
# bldg.apply_grav_load()

# T_1 = bldg.run_eigen()

# bldg.provide_damping(80, method='SP',
#                                   zeta=[0.05], modes=[1])

# dt = 0.005
# ok = bldg.run_ground_motion(run.gm_selected, 
#                         run.scale_factor*1.0, 
#                         dt, T_end=60.0)

# from plot_structure import plot_dynamic
# plot_dynamic(run)

#%% analyze database

# main_obj.analyze_db('tfp_mf_db_stack.csv', save_interval=5)

# # Pickle the main object
# import pickle
# with open('../data/tfp_mf_db_stack.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% DoE

# you could either read the csv or unpickle
# or chain this straight from the analyzed main_obj

# pickle_path = '../data/'

# import pickle

# with open(pickle_path+"tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
# main_obj.calculate_collapse()
# main_obj.perform_doe(n_set=50,batch_size=5)

# import pickle
# with open('../data/tfp_mf_db_doe_loocv_batch_strict.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
#%%
# TODO: solve the optimization problem by hand

# create a db with 10 sets of gap ratios, repeat the remaining variables
# repeat each set 30 times
# design structures
# randomly select ground motion for each
#%% load DoE

from db import Database
pickle_path = '../data/'

import pickle
import pandas as pd

with open(pickle_path+"tfp_mf_db_doe_loocv.pickle", 'rb') as picklefile:
    main_obj = pickle.load(picklefile)
    
#%% run validation

validation_path = '../data/validation/'
# TODO: is there a way to pipe this straight from GP? and organize depending on target
sample_dict = {
    'gap_ratio' : 1.0,
    'RI' : 2.0,
    'T_ratio': 3.0,
    'zeta_e': 0.2
}

design_df = pd.DataFrame(sample_dict, index=[0])

main_obj.prepare_idas(design_df)
main_obj.analyze_ida('ida_baseline.csv')

import pickle
with open(validation_path+'tfp_mf_db_ida_baseline.pickle', 'wb') as f:
    pickle.dump(main_obj, f)