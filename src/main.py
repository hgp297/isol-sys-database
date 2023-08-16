############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Main workfile

# Open issues:  

############################################################################

from db import Database

main_obj = Database(50)

main_obj.design_bearings(filter_designs=True)

main_obj.design_structure()

test_mf_tfp = main_obj.mf_designs.iloc[0]
test_cbf = main_obj.cbf_designs.iloc[0]
test_mf_lrb = main_obj.mf_designs.iloc[-1]

main_obj.scale_gms()


# # test build one building (MF, TFP only)
# from building import Building
# mf_tfp_bldg = Building(test_mf_tfp)
# mf_tfp_bldg.model_frame()

#%%

main_obj.analyze_db('just_a_test.csv')

#%%

# from building import Building
# from bearing import Bearing
# tfp_example = Bearing(test_mf_tfp)

# # test build one building (MF, LRB)
# mf_lrb_bldg = Building(test_mf_lrb)
# mf_lrb_bldg.model_frame()
# mf_lrb_bldg.apply_grav_load()
# mf_lrb_bldg.provide_damping(80, method='SP',
#                        zeta=[0.05], modes=[1])

# dt = 0.005
# mf_lrb_bldg.run_ground_motion('RSN3905_TOTTORI_OKY002EW', 1.0, dt)

# # test build CBF
# cbf_bldg = Building(test_cbf)
# cbf_bldg.model_frame()
# cbf_bldg.apply_grav_load()
# cbf_bldg.provide_damping(80, method='SP',
#                           zeta=[0.05], modes=[1])

# sample_lrb = main_obj.lrb_designs.iloc[0]
# from design import design_LRB
# test = design_LRB(sample_lrb)

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
