# from db import Database

# main_obj = Database(20)

# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)

# main_obj.scale_gms()

#%%

# # run info
# import pandas as pd
# import numpy as np
# import loss


# idx = pd.IndexSlice
# pd.options.display.max_rows = 30

# # and import pelicun classes and methods
# from pelicun.assessment import Assessment

# # get database
# # initialize, no printing outputs, offset fixed with current components
# PAL = Assessment({
#     "PrintLog": False, 
#     "Seed": 985,
#     "Verbose": False,
#     "DemandOffset": {"PFA": 0, "PFV": 0}
# })

# # generate structural components and join with NSCs
# P58_metadata = PAL.get_default_metadata('loss_repair_DB_FEMA_P58_2nd')

# # data = pd.read_csv('../data/tfp_mf_db.csv')
# pickle_path = '../data/'
# main_obj = pd.read_pickle(pickle_path+"structural_db_complete.pickle")
# data = main_obj.ops_analysis
# run = data.iloc[67]


# floors = run.num_stories
# area = run.L_bldg**2 # sq ft

# # lab, health, ed, res, office, retail, warehouse, hotel
# fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
# bldg_usage = [fl_usage]*floors

# area_usage = np.array(fl_usage)*area

# loss_obj = loss.Loss_Analysis(run)
# loss_obj.nqe_sheets()
# loss_obj.normative_quantity_estimation(bldg_usage, P58_metadata)


# additional_frag_db = pd.read_csv('../resource/loss/custom_component_fragilities.csv',
#                                   header=[0,1], index_col=0)
# loss_obj.process_EDP()
# [cmp, dmg, loss, loss_cmp, agg, 
#   collapse_rate, irr_rate] = loss_obj.estimate_damage(
#       custom_fragility_db=additional_frag_db, mode='generate')

# components_sample = loss_obj.components
# components_sample = components_sample[['Component', 'Comment', 'Theta_0','Units']]
# components_sample = components_sample.groupby(['Component']).sum()
# print(components_sample.to_latex(float_format="{:.1f}".format))
# components_sample.to_csv('./sample_cmp.csv')