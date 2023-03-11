############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Test file

# Open issues:  

############################################################################

from structure_database import Database

main_obj = Database(5000)

main_obj.design_bearings(filter_designs=True)

main_obj.design_structure()

test_mf = main_obj.mf_designs.iloc[0]
test_cbf = main_obj.tfp_designs.iloc[3]

# test build one building (MF, TFP only)
from building import Building
mf_bldg = Building(test_mf)
mf_bldg.model_frame()

# sample_lrb = main_obj.lrb_designs.loc[87]
# from design import design_LRB
# test = design_LRB(sample_lrb)

#%%
# plot distribution of parameters

import seaborn as sns
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(13, 13))

lrbs = main_obj.lrb_designs
tfps = main_obj.tfp_designs

sns.histplot(data=lrbs, x="Q", kde=True, label="LRB", ax=axs[0, 0])
sns.histplot(data=lrbs, x="k_ratio", kde=True, label="LRB", ax=axs[0, 1])
sns.histplot(data=lrbs, x="T_m", kde=True, label="LRB", ax=axs[1, 0])
sns.histplot(data=lrbs, x="zeta_e", kde=True, label="LRB", ax=axs[1, 1])

sns.histplot(data=tfps, x="Q", kde=True, label="TFP", ax=axs[0, 0])
sns.histplot(data=tfps, x="k_ratio", kde=True, label="TFP", ax=axs[0, 1])
sns.histplot(data=tfps, x="T_m", kde=True, label="TFP", ax=axs[1, 0])
sns.histplot(data=tfps, x="zeta_e", kde=True, label="TFP", ax=axs[1, 1])

plt.legend()
plt.show()