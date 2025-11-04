############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Main workfile

# Open issues:  

############################################################################

# from db import Database

# main_obj = Database(16, seed=26)

# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)

# main_obj.scale_gms()

#%%
# from building import Building
# run = main_obj.retained_designs.iloc[0]
# bldg = Building(run)
# bldg.model_frame()
# bldg.apply_grav_load()
# T_1 = bldg.run_eigen()
# bldg.provide_damping(80, method='SP', zeta=[0.05], modes=[1])
# dt = 0.005
# ok = bldg.run_ground_motion(run.gm_selected, 
#                         run.scale_factor*1.0, 
#                         dt, T_end=60.0)

# from plot_structure import plot_dynamic
# plot_dynamic(run)
#%% generate analyze database

# main_obj.analyze_db('structural_db_mixed_tol.csv', save_interval=5)

# # Pickle the main object
# import pickle
# with open('../data/structural_db_mixed_tol.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% database gen in parallel

# from db import Database
# num = 4
# seed = 105
# main_obj = Database(n_points=num, seed=seed)
# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)
# main_obj.scale_gms()
# output_dir = './outputs/seed_'+str(seed)+'_output/'
# main_obj.analyze_db('structural_db_seed_'+str(seed)+'.csv', save_interval=5,
#                     output_path=output_dir)
# import pickle
# with open('../data/structural_db_seed_'+str(seed)+'.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

# current workflow (parallel on tacc)
# change_taskfile.sh writes calls on gen_db_thread (specifying n runs) <- change to main_obj calls here
# gen_task is written
# run_control_sbatch then runs on tacc
# collect_gen brings data from $SCRATCH back to $WORK
# local update.sh copies data to this repo (taccdata), aggregate file unites the df


#%% calculate maximum pelicun losses

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_complete_spectracomments.pickle")

# main_obj.calc_cmp_max(main_obj.ops_analysis,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_complete_spectracomments_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% run pelicun

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_complete_spectracomments.pickle")

# max_obj = pd.read_pickle(pickle_path+"loss/structural_db_complete_spectracomments_max_loss.pickle")
# df_loss_max = max_obj.max_loss

# main_obj.run_pelicun(main_obj.ops_analysis, collect_IDA=False,
#                 cmp_dir='../resource/loss/', max_loss_df=df_loss_max,
#                 edp_mode='generate')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_complete_spectracomments_fixededp_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% validate design

# current workflow: an analysis file generates the .in input files for validation
# change_valfile then creates the run case name and writes the IDA calls to val_task
# everything else is handled by val_control_sbatch -> val_task -> val_ida_thread -> .in


# %% run pelicun on validation (deterministic on the IDA) and calculate max loss
import pandas as pd
import pickle

# #### cbf tfp
# run_case = 'cbf_tfp_moderate_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

    
# #### mf tfp
# run_case = 'mf_tfp_moderate_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
    
# #### cbf lrb
# run_case = 'cbf_lrb_moderate_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
    
# #### mf lrb
# run_case = 'mf_lrb_moderate_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
    
    
# ###### enhanced
    
#### cbf tfp
run_case = 'cbf_tfp_enhanced_spectracomments_fixededp'
validation_path = '../data/validation/'+run_case+'/'
loss_path = '../data/validation/'+run_case+'/'

main_obj = pd.read_pickle(validation_path+run_case+".pickle")
df_val = main_obj.ida_results

main_obj.calc_cmp_max(main_obj.ida_results,
                cmp_dir='../resource/loss/',
                validation_run=True)

with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
    pickle.dump(main_obj, f)
    
df_loss_max = main_obj.max_loss
    
main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
                cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
    pickle.dump(main_obj, f)

    
# #### mf tfp
# run_case = 'mf_tfp_enhanced_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
    
#### cbf lrb
run_case = 'cbf_lrb_enhanced_spectracomments_fixededp'
validation_path = '../data/validation/'+run_case+'/'
loss_path = '../data/validation/'+run_case+'/'

main_obj = pd.read_pickle(validation_path+run_case+".pickle")
df_val = main_obj.ida_results

main_obj.calc_cmp_max(main_obj.ida_results,
                cmp_dir='../resource/loss/',
                validation_run=True)

with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
    pickle.dump(main_obj, f)
    
df_loss_max = main_obj.max_loss
    
main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
                cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
    pickle.dump(main_obj, f)
    
    
# #### mf lrb
# run_case = 'mf_lrb_enhanced_spectracomments_fixededp'
# validation_path = '../data/validation/'+run_case+'/'
# loss_path = '../data/validation/'+run_case+'/'

# main_obj = pd.read_pickle(validation_path+run_case+".pickle")
# df_val = main_obj.ida_results

# main_obj.calc_cmp_max(main_obj.ida_results,
#                 cmp_dir='../resource/loss/',
#                 validation_run=True)

# with open(loss_path+run_case+'_max_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
    
# df_loss_max = main_obj.max_loss
    
# main_obj.run_pelicun(main_obj.ida_results, collect_IDA=True,
#                 cmp_dir='../resource/loss/', max_loss_df = df_loss_max)

# with open(loss_path+run_case+'_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)