############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Main workfile

# Open issues:  

############################################################################

from db import Database

main_obj = Database(400)

main_obj.design_bearings(filter_designs=True)
main_obj.design_structure(filter_designs=True)

main_obj.scale_gms()

#%% troubleshoot

# failed CBFs in 100 set: 10, 22, 38
# 10 solved with smaller time step
# 22 solved with strong ghosts + Broyden
# 38 solved with smaller time step, without convergence adds

# solution ideas for 22: run through time-stepping loops, increase time-step
# attempting a non-zero gravity spring for 22

# troubleshoot building
run = main_obj.retained_designs.iloc[8]
from building import Building

bldg = Building(run)
bldg.model_frame(convergence_mode=False)
bldg.apply_grav_load()

T_1 = bldg.run_eigen()

bldg.provide_damping(80, method='SP',
                                  zeta=[0.05], modes=[1])

dt = 0.005
ok = bldg.run_ground_motion(run.gm_selected, 
                        run.scale_factor*1.0, 
                        dt, T_end=60.0)

#%%

# failed CBFs in 200 set: 7? 25
# fatal condition: dt = 0.005, convergence_mode=False, Broyden in algo
# ran in loop

# fatal crash ONLY happens in loop
# crashes after Broyden works and loop returns to Newton for single step

# all runs seem to solve with dt really small, but we don't want this

# # troubleshoot building
# run = main_obj.retained_designs.iloc[25]
# from building import Building

# bldg = Building(run)
# bldg.model_frame(convergence_mode=False)
# bldg.apply_grav_load()

# T_1 = bldg.run_eigen()

# bldg.provide_damping(80, method='SP',
#                                   zeta=[0.05], modes=[1])

# dt = 0.005
# ok = bldg.run_ground_motion(run.gm_selected, 
#                         run.scale_factor*1.0, 
#                         dt, T_end=60.0)


#%% pushover

# bldg = Building(run)
# bldg.model_frame(convergence_mode=True)
# bldg.apply_grav_load()

# T_1 = bldg.run_eigen()

# bldg.provide_damping(80, method='SP',
#                                   zeta=[0.05], modes=[1])

# bldg.run_pushover(max_drift_ratio=0.1)

# from plot_structure import plot_pushover
# plot_pushover(bldg)

#%% dynamic run

from plot_structure import plot_dynamic
plot_dynamic(run)

#%% ground motion spectrum

# from gms import plot_spectrum
# plot_spectrum(run)


#%% animation

# from plot_structure import animate_gm
# fig, animate, n_ani = animate_gm(bldg)
# import matplotlib.animation as animation
# animation.FuncAnimation(fig, animate, n_ani, interval=1/4, blit=True)

#%% generate analyze database

# main_obj.analyze_db('structural_db_mixed.csv', save_interval=5)

# # Pickle the main object
# import pickle
# with open('../data/structural_db_mixed.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% run pelicun

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_mixed.pickle")

# main_obj.run_pelicun(main_obj.ops_analysis, collect_IDA=False,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_mixed_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% calculate maximum pelicun losses

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"tfp_mf_db_doe_prestrat.pickle")

# main_obj.calc_cmp_max(main_obj.doe_analysis,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'tfp_mf_db_doe_loss_max.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)

#%% calculate maximum pelicun losses

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_mixed.pickle")

# main_obj.calc_cmp_max(main_obj.ops_analysis,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_mixed_loss_max.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)
