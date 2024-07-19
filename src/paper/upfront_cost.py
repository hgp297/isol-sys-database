############################################################################
#               Figure generation (plotting, ML, inverse design)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: April 2024

# Description:  Main file which imports the structural database and starts the
# loss estimation

# Open issues:  (1) 

############################################################################
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from doe import GP

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None  

plt.close('all')

main_obj = pd.read_pickle("../../data/loss/structural_db_parallel_loss.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df_raw = main_obj.ops_analysis
df_raw = df_raw.reset_index(drop=True)

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

# df = df.drop(columns=['index'])
# df = df_whole.head(100).copy()

df['max_drift'] = df.PID.apply(max)
df['log_drift'] = np.log(df['max_drift'])

df['max_velo'] = df.PFV.apply(max)
df['max_accel'] = df.PFA.apply(max)

df['T_ratio'] = df['T_m'] / df['T_fb']
df['T_ratio_e'] = df['T_m'] / df['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df['Bm'] = np.interp(df['zeta_e'], zetaRef, BmRef)

df['gap_ratio'] = (df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2)

df_loss = main_obj.loss_data

max_obj = pd.read_pickle("../../data/loss/structural_db_parallel_max_loss.pickle")
df_loss_max = max_obj.max_loss

#%%
# make a generalized 2D plotting grid, defaulted to gap and Ry
# grid is based on the bounds of input data
def make_2D_plotting_space(X, res, x_var='gap_ratio', y_var='RI', 
                           all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                           third_var_set = None, fourth_var_set = None,
                           x_bounds=None, y_bounds=None):
    
    if x_bounds == None:
        x_min = min(X[x_var])
        x_max = max(X[x_var])
    else:
        x_min = x_bounds[0]
        x_max = x_bounds[1]
    if y_bounds == None:
        y_min = min(X[y_var])
        y_max = max(X[y_var])
    else:
        y_min = y_bounds[0]
        y_max = y_bounds[1]
    xx, yy = np.meshgrid(np.linspace(x_min,
                                     x_max,
                                     res),
                         np.linspace(y_min,
                                     y_max,
                                     res))

    rem_vars = [i for i in all_vars if i not in [x_var, y_var]]
    third_var = rem_vars[0]
    fourth_var = rem_vars[-1]
       
    xx = xx
    yy = yy
    
    if third_var_set is None:
        third_var_val= X[third_var].median()
    else:
        third_var_val = third_var_set
    if fourth_var_set is None:
        fourth_var_val = X[fourth_var].median()
    else:
        fourth_var_val = fourth_var_set
    
    
    X_pl = pd.DataFrame({x_var:xx.ravel(),
                         y_var:yy.ravel(),
                         third_var:np.repeat(third_var_val,
                                             res*res),
                         fourth_var:np.repeat(fourth_var_val, 
                                              res*res)})
    X_plot = X_pl[all_vars]
                         
    return(X_plot)


#%% normalize DVs and prepare all variables
df['bldg_area'] = df['L_bldg']**2 * (df['num_stories'] + 1)

df['replacement_cost'] = 600.0*(df['bldg_area'])
df['total_cmp_cost'] = df_loss_max['cost_50%']
df['cmp_replace_cost_ratio'] = df['total_cmp_cost']/df['replacement_cost']
df['median_cost_ratio'] = df_loss['cost_50%']/df['replacement_cost']
df['cmp_cost_ratio'] = df_loss['cost_50%']/df['total_cmp_cost']

# but working in parallel (2x faster)
df['replacement_time'] = df['bldg_area']/1000*365
df['total_cmp_time'] = df_loss_max['time_l_50%']
df['cmp_replace_time_ratio'] = df['total_cmp_time']/df['replacement_time']
df['median_time_ratio'] = df_loss['time_l_50%']/df['replacement_time']
df['cmp_time_ratio'] = df_loss['time_l_50%']/df['total_cmp_time']

df['replacement_freq'] = df_loss['replacement_freq']

df[['B_50%', 'C_50%', 'D_50%', 'E_50%']] = df_loss[['B_50%', 'C_50%', 'D_50%', 'E_50%']]

df['impacted'] = pd.to_numeric(df['impacted'])

cost_var = 'cmp_cost_ratio'
time_var = 'cmp_time_ratio'
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

mask = df['B_50%'].isnull()

df['B_50%'].loc[mask] = df_loss_max['B_50%'].loc[mask]
df['C_50%'].loc[mask] = df_loss_max['C_50%'].loc[mask]
df['D_50%'].loc[mask] = df_loss_max['D_50%'].loc[mask]
df['E_50%'].loc[mask] = df_loss_max['E_50%'].loc[mask]


#%% subsets

df_no_impact = df[df['impacted'] == 0]

# remove the singular outlier point
from scipy import stats
df_no_impact = df_no_impact[np.abs(stats.zscore(df_no_impact['cmp_cost_ratio'])) < 10].copy()

df_tfp = df_no_impact[df_no_impact['isolator_system'] == 'TFP']
df_lrb = df_no_impact[df_no_impact['isolator_system'] == 'LRB']
df_cbf = df_no_impact[df_no_impact['superstructure_system'] == 'CBF']
df_mf = df_no_impact[df_no_impact['superstructure_system'] == 'MF']


#%% maximum structural component loss

# plt.close('all')

df_cbf = df[df['superstructure_system'] == 'CBF']
df_mf = df[df['superstructure_system'] == 'MF']

df_cbf_max = df_loss_max[df['superstructure_system'] == 'CBF']
df_mf_max = df_loss_max[df['superstructure_system'] == 'MF']

cmap = plt.cm.tab10

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.scatter(df_cbf['Vs'], df_cbf_max['B_50%'], alpha=0.7, color=cmap(1), label='CBF')
ax.scatter(df_mf['Vs'], df_mf_max['B_50%'], alpha=0.7, color=cmap(0), label='MF')

ax.set_ylabel("Structural component cost (\$)", fontsize=axis_font)
ax.set_xlabel('$V_s$ (kip)', fontsize=axis_font)

ax.legend()


#%% Calculate upfront cost of data

def moment_frame_cost(df, steel_per_unit=1.25):
    n_bays = df.num_bays
    
    # ft
    L_beam = df.L_bay
    h_story = df.h_story
    
    all_beams = df.beam
    all_cols = df.column
    
    # sum of per-length-weight of all floors
    col_wt = [float(member.split('X',1)[1]) for member in all_cols]
    beam_wt = [float(member.split('X',1)[1]) for member in all_beams] 
    
    # col_all_wt = np.array(list(map(sum, col_wt)))
    # beam_all_wt = np.array(list(map(sum, beam_wt)))
    
    # find true steel costs
    n_frames = 4
    n_cols = 4*n_bays
    
    floor_col_length = np.array(n_cols*h_story, dtype=float)
    floor_beam_length = np.array(L_beam * n_bays * n_frames, dtype=float)
        
    floor_col_wt = col_wt*floor_col_length 
    floor_beam_wt = beam_wt*floor_beam_length
    
    bldg_wt = sum(floor_col_wt) + sum(floor_beam_wt)
    
    steel_cost = steel_per_unit*bldg_wt
    return(steel_cost)

# TODO: this does not include gravity frames
def braced_frame_cost(df, brace_db, steel_per_unit=1.25):
    n_bays = df.num_bays
    
    # ft
    L_beam = df.L_bay
    h_story = df.h_story
    
    from math import atan, cos
    theta = atan(h_story/(L_beam/2))
    L_brace = (L_beam/2)/cos(theta)
    
    all_beams = df.beam
    all_cols = df.column
    all_braces = df.brace
    
    n_braced = int(round(n_bays/2.25))
    n_braced = max(n_braced, 1)
    
    # sum of per-length-weight of all floors
    col_wt = [float(member.split('X',1)[1]) for member in all_cols]
    beam_wt = [float(member.split('X',1)[1]) for member in all_beams] 
    brace_wt = [brace_db.loc[brace_db['AISC_Manual_Label'] == brace_name]['W'].item() 
                    for brace_name in all_braces]
    
    n_frames = 4
    
    # in CBF, only count the big frames
    n_cols = 4*(n_braced+1)
    
    floor_col_length = np.array(n_cols*h_story, dtype=float)
    floor_beam_length = np.array(L_beam * n_braced * n_frames, dtype=float)
    floor_brace_length = np.array(L_brace * n_braced * n_frames, dtype=float)
        
    floor_col_wt = col_wt*floor_col_length 
    floor_beam_wt = beam_wt*floor_beam_length
    floor_brace_wt = brace_wt*floor_brace_length
    
    bldg_wt = sum(floor_col_wt) + sum(floor_beam_wt) + sum(floor_brace_wt)
    steel_cost = steel_per_unit*bldg_wt
    
    return(steel_cost)

def calc_steel_cost(df, brace_db, steel_per_unit=1.25):
    superstructure_system = df.superstructure_system
    
    if superstructure_system == 'MF':
        return moment_frame_cost(df, steel_per_unit=steel_per_unit)
    else:
        return braced_frame_cost(df, brace_db, steel_per_unit=steel_per_unit)
    
    
db_string = '../../resource/'
brace_db = pd.read_csv(db_string+'braceShapes.csv', index_col=None, header=0)  

df['steel_cost'] = df.apply(
       lambda row: calc_steel_cost(
           row, brace_db=brace_db,
           steel_per_unit=1.25),
       axis='columns', result_type='expand')

#%% steel weight calc plot

# plt.close('all')

df_cbf = df[df['superstructure_system'] == 'CBF']
df_mf = df[df['superstructure_system'] == 'MF']


cmap = plt.cm.tab10

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.scatter(df_cbf['Vs'], df_cbf['steel_cost'], alpha=0.7, color=cmap(1), label='CBF')
ax.scatter(df_mf['Vs'], df_mf['steel_cost'], alpha=0.7, color=cmap(0), label='MF')

ax.set_ylabel("Steel cost (\$)", fontsize=axis_font)
ax.set_xlabel('$V_s$ (kip)', fontsize=axis_font)

ax.legend()