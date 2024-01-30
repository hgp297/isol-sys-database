############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2024

# Description:  Main workfile for TFP-MF database, reworked in new framework

# Open issues:  

############################################################################

from db import Database

main_obj = Database(400, n_buffer=8, seed=130, 
                    struct_sys_list=['MF'], isol_wts=[1, 0])

main_obj.design_bearings(filter_designs=True)
main_obj.design_structure(filter_designs=True)

main_obj.scale_gms()

#%% analyze database

main_obj.analyze_db('tfp_mf_db.csv', save_interval=5)

# Pickle the main object
import pickle
with open('../data/tfp_mf_db.pickle', 'wb') as f:
    pickle.dump(main_obj, f)