############################################################################
#               Script to try a single design

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2024

# Description:  

############################################################################

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

# TODO: T_ratio will need to be conditioned away from RI

import pandas as pd

test_design = pd.Series({
    'S_1' : 1.0,
    'T_ratio' : 3.0,
    'gap_ratio' : 1.1,
    'k_ratio' : 8.0,
    'RI' : 0.9,
    'L_bldg' : 150.0,
    'h_bldg': 60.0,
    'superstructure_system' : 'MF',
    'isolator_system' : 'TFP',
    'num_frames': 2
})

test_design['num_bays'] = round(test_design['L_bldg']/ 30.0)
test_design['num_stories'] = round(test_design['h_bldg'] / 14.0)


test_design['L_bay'] = (test_design['L_bldg'] / 
                           test_design['num_bays'])
test_design['h_story'] = (test_design['h_bldg'] / 
                             test_design['num_stories'])
test_design['S_s'] = 2.2815

# TODO: Tfbe needs good regression
test_design['T_fbe'] = 0.05*test_design['h_bldg']**(0.75)
test_design['T_m'] = test_design['T_ratio'] * test_design['T_fbe']

# TODO: either iterate on Q or sample based on T_m/k_ratio density
test_design['Q'] = 0.07
design_df = test_design.to_frame().T

import design as ds
all_tfp_designs = design_df.apply(lambda row: ds.design_TFP(row),
                               axis='columns', result_type='expand')

all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                           'T_e', 'k_e', 'zeta_e', 'D_m']

tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                  (all_tfp_designs['R_1'] <= 50.0) &
                                  (all_tfp_designs['R_2'] <= 180.0) &
                                  (all_tfp_designs['zeta_e'] <= 0.25)]
            