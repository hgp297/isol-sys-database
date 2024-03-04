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
    'T_ratio' : 3.3,
    'gap_ratio' : 1.1,
    'RI' : 0.9,
    'zeta_m' : 0.25,
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
# Cu Ta
test_design['T_fbe_r'] = 1.4*0.028*test_design['h_bldg']**(0.8)
# test_design['T_fbe_r'] = 0.078*test_design['h_bldg']**(0.75)
test_design['T_m'] = test_design['T_ratio'] * test_design['T_fbe_r']

pi = 3.14159

# from ASCE Ch. 17, get damping multiplier
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
from numpy import interp
B_m = interp(test_design['zeta_m'], zetaRef, BmRef)

lim = (1/8*test_design['S_1']/B_m*pi/test_design['T_m'])

test_design['Q'] = lim-0.01

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

design_df = pd.concat([design_df, tfp_designs], axis=1)

from loads import define_lateral_forces, define_gravity_loads

design_df[['W', 
       'W_s', 
       'w_fl', 
       'P_lc',
       'all_w_cases',
       'all_Plc_cases']] = design_df.apply(lambda row: define_gravity_loads(row),
                                           axis='columns', result_type='expand')

# assumes that there is at least one design
design_df[['wx', 
       'hx', 
       'h_col', 
       'hsx', 
       'Fx', 
       'Vs',
       'T_fbe']] = design_df.apply(lambda row: define_lateral_forces(row),
                                   axis='columns', result_type='expand')
                                   
all_mf_designs = design_df.apply(lambda row: ds.design_MF(row, db_string='../../resource/'),
                               axis='columns', 
                               result_type='expand')

all_mf_designs.columns = ['beam', 'column', 'flag']


# keep the designs that look sensible
mf_designs = all_mf_designs.loc[all_mf_designs['flag'] == False]
mf_designs = mf_designs.dropna(subset=['beam','column'])
 
mf_designs = mf_designs.drop(['flag'], axis=1)
            