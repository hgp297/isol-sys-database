############################################################################
#               Create database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Main workfile

# Open issues:  

############################################################################

# from db import Database

# main_obj = Database(400)

# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)

# main_obj.scale_gms()

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


#%% run pelicun

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_mixed_tol.pickle")

# main_obj.run_pelicun(main_obj.ops_analysis, collect_IDA=False,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_mixed_loss.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)


#%% calculate maximum pelicun losses

# import pandas as pd
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_mixed_tol.pickle")

# main_obj.calc_cmp_max(main_obj.ops_analysis,
#                 cmp_dir='../resource/loss/')

# import pickle
# loss_path = '../data/loss/'
# with open(loss_path+'structural_db_mixed_loss_max.pickle', 'wb') as f:
#     pickle.dump(main_obj, f)



#%% validate design

from db import Database
pickle_path = '../data/'

import pandas as pd

main_obj = pd.read_pickle(pickle_path+"structural_db_mixed_tol.pickle")
    
validation_path = '../data/validation/'

# TODO: is there a way to pipe this straight from GP? and organize depending on target
sample_dict = {
    'gap_ratio' : 0.6,
    'RI' : 2.25,
    'T_ratio': 2.16,
    'zeta_e': 0.25
}

design_df = pd.DataFrame(sample_dict, index=[0])

main_obj.prepare_idas(design_df)
main_obj.analyze_ida('ida_10.csv')

import pickle
with open(validation_path+'tfp_mf_db_ida_10.pickle', 'wb') as f:
    pickle.dump(main_obj, f)