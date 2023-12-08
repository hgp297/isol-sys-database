############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Main workfile

# Open issues:  

############################################################################

from db import Database

main_obj = Database(100)

main_obj.design_bearings(filter_designs=True)
main_obj.design_structure(filter_designs=True)

main_obj.scale_gms()

#%% troubleshoot

# # cbf lrb
# run = main_obj.retained_designs.loc[299]

# # cbf tfp
# run = main_obj.retained_designs.loc[714]

# # mf lrb
# run = main_obj.retained_designs.loc[704]

# # mf tfp
# run = main_obj.retained_designs.loc[68]

# troubleshoot building
run = main_obj.retained_designs.iloc[0]
from building import Building

bldg = Building(run)
bldg.model_frame()
bldg.apply_grav_load()

T_1 = bldg.run_eigen()

bldg.provide_damping(80, method='SP',
                                  zeta=[0.05], modes=[1])

dt = 0.001
ok = bldg.run_ground_motion(run.gm_selected, 
                        run.scale_factor*1.5, 
                        dt, T_end=60.0)

# from experiment import run_nlth
# res = run_nlth(troubleshoot_run)


#%% pushover

# bldg = Building(run)
# bldg.model_frame()
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

# main_obj.analyze_db('structural_db_dbe.csv', save_interval=5)


#%%
# # plot distribution of parameters

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.close('all')
# fig, axs = plt.subplots(2, 2, figsize=(13, 13))

# lrbs = main_obj.lrb_designs
# tfps = main_obj.tfp_designs
# import pandas as pd
# df_plot = pd.concat([lrbs, tfps], axis=0)

# sns.histplot(data=df_plot, x="Q", kde=True, 
#               hue='isolator_system',ax=axs[0, 0])
# sns.histplot(data=df_plot, x="k_ratio", kde=True, 
#               hue='isolator_system',ax=axs[0, 1])
# sns.histplot(data=df_plot, x="T_m", kde=True, 
#               hue='isolator_system',ax=axs[1, 0])
# sns.histplot(data=df_plot, x="zeta_e", kde=True, 
#               hue='isolator_system',ax=axs[1, 1])

# # plt.legend()
# plt.show()

#%%

# # plot distribution of parameters

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.close('all')
# fig, axs = plt.subplots(1, 1, figsize=(9, 9))

# lrbs = main_obj.lrb_designs
# tfps = main_obj.tfp_designs

# lrbs['strain_ratio'] = (lrbs['D_m']*lrbs['moat_ampli'])/lrbs['t_r']
# lrbs['dm_check'] = (lrbs['D_m']*lrbs['moat_ampli'])/lrbs['d_bearing']
# lrbs['amp'] = lrbs['moat_ampli']
# # import pandas as pd
# # df_plot = pd.concat([lrbs, tfps], axis=0)

# sns.histplot(data=lrbs, x="strain_ratio", kde=True, ax=axs)

# plt.show()
