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

# import pickle
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from doe import GP

# #%% collapse fragility def
# import numpy as np
# from scipy.stats import norm
# inv_norm = norm.ppf(0.84)
# # collapse as a probability
# from scipy.stats import lognorm
# from math import log, exp

# plt.rcParams["text.usetex"] = True
# x = np.linspace(0, 0.15, 200)
# mu = log(0.1)- 0.25*inv_norm
# sigma = 0.25;

# ln_dist = lognorm(s=sigma, scale=exp(mu))
# p = ln_dist.cdf(np.array(x))

# plt.close('all')
# fig = plt.figure(figsize=(8,10))
# ax = fig.add_subplot(2, 1, 1)
# # ax = plt.subplots(1, 1, figsize=(8,6))

# ax.plot(x, p, label='Collapse', color='blue')

# mu_irr = log(0.01)
# ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
# p_irr = ln_dist_irr.cdf(np.array(x))

# ax.plot(x, p_irr, color='red', label='Irreparable')

# axis_font = 20
# subt_font = 18
# xleft = 0.15
# ax.set_ylim([0,1])
# ax.set_xlim([0, xleft])
# ax.set_ylabel('Limit state probability', fontsize=axis_font)
# # ax.set_xlabel('Drift ratio', fontsize=axis_font)

# ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
# ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
# ax.text(0.105, 0.52, r'PID = 0.078', fontsize=axis_font, color='blue')
# ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

# ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
# ax.hlines(y=0.84, xmin=0.1, xmax=xleft, color='blue', linestyle=":")
# ax.text(0.105, 0.87, r'PID = 0.10', fontsize=axis_font, color='blue')
# ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

# lower= ln_dist.ppf(0.16)
# ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
# ax.hlines(y=0.16, xmin=lower, xmax=xleft, color='blue', linestyle=":")
# ax.text(0.105, 0.19, r'PID = 0.061', fontsize=axis_font, color='blue')
# ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


# ax.hlines(y=0.5, xmin=0.0, xmax=exp(mu_irr), color='red', linestyle=":")
# lower = ln_dist_irr.ppf(0.16)
# ax.hlines(y=0.16, xmin=0.0, xmax=lower, color='red', linestyle=":")
# upper = ln_dist_irr.ppf(0.84)
# ax.hlines(y=0.84, xmin=0.0, xmax=upper, color='red', linestyle=":")
# ax.plot([lower], [0.16], marker='*', markersize=15, color="red", linestyle=":")
# ax.plot([0.01], [0.5], marker='*', markersize=15, color="red", linestyle=":")
# ax.plot([upper], [0.84], marker='*', markersize=15, color="red", linestyle=":")
# ax.vlines(x=upper, ymin=0, ymax=0.84, color='red', linestyle=":")
# ax.vlines(x=0.01, ymin=0, ymax=0.5, color='red', linestyle=":")
# ax.vlines(x=lower, ymin=0, ymax=0.16, color='red', linestyle=":")

# ax.text(0.005, 0.19, r'RID = 0.007', fontsize=axis_font, color='red')
# ax.text(0.005, 0.87, r'RID = 0.013', fontsize=axis_font, color='red')
# ax.text(0.005, 0.53, r'RID = 0.010', fontsize=axis_font, color='red')

# ax.set_title('a) MF replacement fragility definition', fontsize=axis_font)
# ax.grid()
# label_size = 16
# clabel_size = 12

# ax.legend(fontsize=label_size, loc='upper center')

# # cbf
# inv_norm = norm.ppf(0.9)
# # collapse as a probability

# x = np.linspace(0, 0.08, 200)
# mu = log(0.05)- 0.55*inv_norm
# sigma = 0.55;

# ln_dist = lognorm(s=sigma, scale=exp(mu))
# p = ln_dist.cdf(np.array(x))

# # plt.close('all')
# ax = fig.add_subplot(2, 1, 2)

# ax.plot(x, p, label='Collapse', color='blue')

# mu_irr = log(0.005)
# ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
# p_irr = ln_dist_irr.cdf(np.array(x))

# ax.plot(x, p_irr, color='red', label='Irreparable')

# axis_font = 20
# subt_font = 18
# xleft = 0.08
# ax.set_ylim([0,1])
# ax.set_xlim([0, xleft])
# ax.set_ylabel('Limit state probability', fontsize=axis_font)
# ax.set_xlabel('Drift ratio', fontsize=axis_font)

# ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
# ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
# ax.text(0.055, 0.52, r'PID = 0.025', fontsize=axis_font, color='blue')
# ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

# ax.vlines(x=0.05, ymin=0, ymax=0.9, color='blue', linestyle=":")
# ax.hlines(y=0.9, xmin=0.05, xmax=xleft, color='blue', linestyle=":")
# ax.text(0.055, 0.84, r'PID = 0.05', fontsize=axis_font, color='blue')
# ax.plot([0.05], [0.9], marker='*', markersize=15, color="blue", linestyle=":")

# lower= ln_dist.ppf(0.16)
# ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
# ax.hlines(y=0.16, xmin=lower, xmax=xleft, color='blue', linestyle=":")
# ax.text(0.055, 0.19, r'PID = 0.014', fontsize=axis_font, color='blue')
# ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


# ax.hlines(y=0.5, xmin=0.0, xmax=exp(mu_irr), color='red', linestyle=":")
# lower = ln_dist_irr.ppf(0.16)
# ax.hlines(y=0.16, xmin=0.0, xmax=lower, color='red', linestyle=":")
# upper = ln_dist_irr.ppf(0.84)
# ax.hlines(y=0.84, xmin=0.0, xmax=upper, color='red', linestyle=":")
# ax.plot([lower], [0.16], marker='*', markersize=15, color="red", linestyle=":")
# ax.plot([0.005], [0.5], marker='*', markersize=15, color="red", linestyle=":")
# ax.plot([upper], [0.84], marker='*', markersize=15, color="red", linestyle=":")
# ax.vlines(x=upper, ymin=0, ymax=0.84, color='red', linestyle=":")
# ax.vlines(x=0.005, ymin=0, ymax=0.5, color='red', linestyle=":")
# ax.vlines(x=lower, ymin=0, ymax=0.16, color='red', linestyle=":")

# ax.text(0.005, 0.19, r'RID = 0.0037', fontsize=axis_font, color='red')
# ax.text(0.005, 0.87, r'RID = 0.0067', fontsize=axis_font, color='red')
# ax.text(0.005, 0.53, r'RID = 0.005', fontsize=axis_font, color='red')

# ax.set_title('b) CBF replacement fragility definition', fontsize=axis_font)
# ax.grid()
# label_size = 16
# clabel_size = 12

# ax.legend(fontsize=label_size, loc='lower center')
# fig.tight_layout()
# plt.show()
# plt.savefig('./dissertation_figures/replacement_def.eps')

#%%

# from db import Database

# main_obj = Database(100)

# main_obj.design_bearings(filter_designs=True)
# main_obj.design_structure(filter_designs=True)

# main_obj.scale_gms()

#%%

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

main_obj = pd.read_pickle("../../data/loss/structural_db_complete_normloss.pickle")
    
main_obj.calculate_collapse()

df_raw = main_obj.ops_analysis
df_raw = df_raw.reset_index(drop=True)

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 5].copy()

# df = df.drop(columns=['index'])
# df = df_whole.head(100).copy()

df['max_drift'] = df.PID.apply(max)
df['log_drift'] = np.log(df['max_drift'])

df['max_velo'] = df.PFV.apply(max)
df['max_accel'] = df.PFA.apply(max)

# df['T_ratio'] = df['T_m'] / df['T_fb']
df['T_ratio_e'] = df['T_m'] / df['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df['Bm'] = np.interp(df['zeta_e'], zetaRef, BmRef)

df['gap_ratio'] = (df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2)

df_loss = main_obj.loss_data

max_obj = pd.read_pickle("../../data/loss/structural_db_complete_max_loss.pickle")
df_loss_max = max_obj.max_loss

#%% normalize DVs and prepare all variables

def loss_percentages(df_main, df_loss, df_max):
    df_main['bldg_area'] = df_main['L_bldg']**2 * (df_main['num_stories'] + 1)

    df_main['replacement_cost'] = 600.0*(df_main['bldg_area'])
    df_main['total_cmp_cost'] = df_max['cost_50%']
    df_main['cmp_replace_cost_ratio'] = df_main['total_cmp_cost']/df_main['replacement_cost']
    df_main['median_cost_ratio'] = df_loss['cost_50%']/df_main['replacement_cost']
    df_main['cmp_cost_ratio'] = df_loss['cost_50%']/df_main['total_cmp_cost']

    # but working in parallel (2x faster)
    df_main['replacement_time'] = df_main['bldg_area']/1000*365
    df_main['total_cmp_time'] = df_max['time_l_50%']
    df_main['cmp_replace_time_ratio'] = df_main['total_cmp_time']/df_main['replacement_time']
    df_main['median_time_ratio'] = df_loss['time_l_50%']/df_main['replacement_time']
    df_main['cmp_time_ratio'] = df_loss['time_l_50%']/df_main['total_cmp_time']

    df_main['replacement_freq'] = df_loss['replacement_freq']

    df_main[['B_50%', 'C_50%', 'D_50%', 'E_50%']] = df_loss[['B_50%', 'C_50%', 'D_50%', 'E_50%']]

    df_main['impacted'] = pd.to_numeric(df_main['impacted'])

    mask = df_loss['B_50%'].isnull()

    df_main['B_50%'].loc[mask] = df_max['B_50%'].loc[mask]
    df_main['C_50%'].loc[mask] = df_max['C_50%'].loc[mask]
    df_main['D_50%'].loc[mask] = df_max['D_50%'].loc[mask]
    df_main['E_50%'].loc[mask] = df_max['E_50%'].loc[mask]
    
    return(df_main)
    
df = loss_percentages(df, df_loss, df_loss_max)

cost_var = 'cmp_cost_ratio'
time_var = 'cmp_time_ratio'
repl_var= 'replacement_freq'
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

db_string = '../../resource/'
brace_db = pd.read_csv(db_string+'braceShapes.csv', index_col=None, header=0)  

df['system'] = df['superstructure_system'] +'-' + df['isolator_system']
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

def make_design_space(res, var_list=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                      fixed_var=None, bound_dict=None):
    
    if bound_dict is None:
        bound_dict = {
            'gap_ratio': (0.6, 2.0),
            'RI': (0.5, 2.25),
            'T_ratio': (2.0, 11.0),
            'zeta_e': (0.1, 0.25),
            'k_ratio': (5.0, 12.0)}
    
    fixed_val = {
        'gap_ratio': 1.0,
        'RI': 2.0,
        'T_ratio': 4.0,
        'zeta_e': 0.15,
        'k_ratio': 10.0
        }
    
    if fixed_var is None:
        xx, yy, uu, vv = np.meshgrid(np.linspace(*bound_dict[var_list[0]], res),
                                     np.linspace(*bound_dict[var_list[1]], res),
                                     np.linspace(*bound_dict[var_list[2]], res),
                                     np.linspace(*bound_dict[var_list[3]], res))
        
        X_space = pd.DataFrame({var_list[0]:xx.ravel(),
                             var_list[1]:yy.ravel(),
                             var_list[2]:uu.ravel(),
                             var_list[3]:vv.ravel()})
    else:
        fixed_val_single = fixed_val[fixed_var]
        excluded_idx = var_list.index(fixed_var)
        my_args = tuple(np.linspace(*bound_dict[var_list[i]], res) 
                                    for i in range(len(var_list)) 
                                    if i!=excluded_idx)
        
        xx, yy, uu, vv = np.meshgrid(*my_args,
                                     fixed_val_single)
        
        var_list.pop(excluded_idx)
        
        # TODO: this is unordered
        X_space = pd.DataFrame({var_list[0]:xx.ravel(),
                             var_list[1]:yy.ravel(),
                             var_list[2]:uu.ravel(),
                             fixed_var:vv.ravel()})
    return(X_space)

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
    
    # only 2 lateral frames
    n_frames = 4
    n_cols = 4*n_bays
    
    floor_col_length = np.array(n_cols*h_story, dtype=float)
    floor_beam_length = np.array(L_beam * n_bays * n_frames, dtype=float)
        
    floor_col_wt = col_wt*floor_col_length 
    floor_beam_wt = beam_wt*floor_beam_length
    
    bldg_wt = sum(floor_col_wt) + sum(floor_beam_wt)
    
    steel_cost = steel_per_unit*bldg_wt
    return(steel_cost)

def braced_frame_cost(df, brace_db, steel_per_unit=1.25):
    n_bays = df.num_bays
    
    # ft
    L_beam = df.L_bay
    h_story = df.h_story
    n_stories = df.num_stories
    
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
    
    # only 2 lateral frames
    n_frames = 4
    
    # in CBF, only count the big frames
    n_cols = 4*(n_braced+1)
    
    floor_col_length = np.array(n_cols*h_story, dtype=float)
    floor_beam_length = np.array(L_beam * n_braced * n_frames, dtype=float)
    floor_brace_length = np.array(L_brace * n_braced * n_frames, dtype=float)
    
    n_every_col = 4*n_bays
    full_frame_col_length = np.array(n_every_col*h_story, dtype=float)
    full_frame_beam_length = np.array(L_beam * n_bays * n_frames, dtype=float)
    grav_col_length = full_frame_col_length - floor_col_length
    grav_beam_length = full_frame_beam_length - floor_beam_length
    
    # assume W14x120 grav columns, W16x31 beams
    grav_col_wt = np.repeat(120.0, n_stories)*grav_col_length
    grav_beam_wt = np.repeat(31.0, n_stories)*grav_beam_length
    
    floor_col_wt = col_wt*floor_col_length 
    floor_beam_wt = beam_wt*floor_beam_length
    floor_brace_wt = brace_wt*floor_brace_length
    
    bldg_wt = (sum(floor_col_wt) + sum(floor_beam_wt) + sum(floor_brace_wt) +
               sum(grav_col_wt) + sum(grav_beam_wt))
    
    steel_cost = steel_per_unit*bldg_wt
    
    return(steel_cost)

# calc cost of existing db
def calc_steel_cost(df, brace_db, steel_per_unit=1.25):
    superstructure_system = df.superstructure_system
    
    if superstructure_system == 'MF':
        return moment_frame_cost(df, steel_per_unit=steel_per_unit)
    else:
        return braced_frame_cost(df, brace_db, steel_per_unit=steel_per_unit)
    
#%% normalize DVs and prepare all variables

def loss_percentages(df_main, df_loss, df_max):
    df_main['bldg_area'] = df_main['L_bldg']**2 * (df_main['num_stories'] + 1)

    df_main['replacement_cost'] = 600.0*(df_main['bldg_area'])
    df_main['total_cmp_cost'] = df_max['cost_50%']
    df_main['cmp_replace_cost_ratio'] = df_main['total_cmp_cost']/df_main['replacement_cost']
    df_main['median_cost_ratio'] = df_loss['cost_50%']/df_main['replacement_cost']
    df_main['cmp_cost_ratio'] = df_loss['cost_50%']/df_main['total_cmp_cost']

    # but working in parallel (2x faster)
    df_main['replacement_time'] = df_main['bldg_area']/1000*365
    df_main['total_cmp_time'] = df_max['time_l_50%']
    df_main['cmp_replace_time_ratio'] = df_main['total_cmp_time']/df_main['replacement_time']
    df_main['median_time_ratio'] = df_loss['time_l_50%']/df_main['replacement_time']
    df_main['cmp_time_ratio'] = df_loss['time_l_50%']/df_main['total_cmp_time']

    df_main['replacement_freq'] = df_loss['replacement_freq']

    df_main[['B_50%', 'C_50%', 'D_50%', 'E_50%']] = df_loss[['B_50%', 'C_50%', 'D_50%', 'E_50%']]

    df_main['impacted'] = pd.to_numeric(df_main['impacted'])

    mask = df_loss['B_50%'].isnull()

    df_main['B_50%'].loc[mask] = df_max['B_50%'].loc[mask]
    df_main['C_50%'].loc[mask] = df_max['C_50%'].loc[mask]
    df_main['D_50%'].loc[mask] = df_max['D_50%'].loc[mask]
    df_main['E_50%'].loc[mask] = df_max['E_50%'].loc[mask]
    
    return(df_main)
    
df = loss_percentages(df, df_loss, df_loss_max)

cost_var = 'cmp_cost_ratio'
time_var = 'cmp_time_ratio'
repl_var= 'replacement_freq'
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

db_string = '../../resource/'
brace_db = pd.read_csv(db_string+'braceShapes.csv', index_col=None, header=0)  

df['steel_cost'] = df.apply(
       lambda row: calc_steel_cost(
           row, brace_db=brace_db,
           steel_per_unit=1.25),
       axis='columns', result_type='expand')

df['steel_cost_per_sf'] = df['steel_cost'] / df['bldg_area']

df['system'] = df['superstructure_system'] +'-' + df['isolator_system']

#%% subsets

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']

df_cbf = df[df['superstructure_system'] == 'CBF'].reset_index()
df_cbf['dummy_index'] = df_cbf['replacement_freq'] + df_cbf['index']*1e-9
df_mf = df[df['superstructure_system'] == 'MF'].reset_index()
df_mf['dummy_index'] = df_mf['replacement_freq'] + df_mf['index']*1e-9

df_mf_o = df_mf[df_mf['impacted'] == 0]
df_cbf_o = df_cbf[df_cbf['impacted'] == 0]

df_mf_tfp = df_tfp[df_tfp['superstructure_system'] == 'MF']
df_mf_lrb = df_lrb[df_lrb['superstructure_system'] == 'MF']

df_cbf_tfp = df_tfp[df_tfp['superstructure_system'] == 'CBF']
df_cbf_lrb = df_lrb[df_lrb['superstructure_system'] == 'CBF']


df_mf_tfp_i = df_mf_tfp[df_mf_tfp['impacted'] == 1]
df_mf_tfp_o = df_mf_tfp[df_mf_tfp['impacted'] == 0]
df_mf_lrb_i = df_mf_lrb[df_mf_lrb['impacted'] == 1]
df_mf_lrb_o = df_mf_lrb[df_mf_lrb['impacted'] == 0]

df_cbf_tfp_i = df_cbf_tfp[df_cbf_tfp['impacted'] == 1]
df_cbf_tfp_o = df_cbf_tfp[df_cbf_tfp['impacted'] == 0]
df_cbf_lrb_i = df_cbf_lrb[df_cbf_lrb['impacted'] == 1]
df_cbf_lrb_o = df_cbf_lrb[df_cbf_lrb['impacted'] == 0]

#%%

PID_vecs = df['PID']

# #%% breakdown of outcomes by systems

# mf_tfp_color = 'royalblue'
# mf_lrb_color = 'cornflowerblue'
# cbf_tfp_color = 'darksalmon'
# cbf_lrb_color = 'lightsalmon'

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 22
# subt_font = 22
# label_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# import seaborn as sns

# # make grid and plot classification predictions

# fig = plt.figure(figsize=(13, 9))


# ### cost
# ax = fig.add_subplot(3, 4, 1)
# bx = sns.boxplot(y=cost_var, x= "impacted", data=df_mf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sp = sns.stripplot(x='impacted', y=cost_var, data=df_mf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# ax.set_title('MF-TFP', fontsize=subt_font)
# ax.set_ylabel('Repair cost ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 2)
# bx = sns.boxplot(y=cost_var, x= "impacted", data=df_mf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=cost_var, data=df_mf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# ax.set_title('MF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair cost ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 3)
# bx = sns.boxplot(y=cost_var, x= "impacted", data=df_cbf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=cost_var, data=df_cbf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# ax.set_title('CBF-TFP', fontsize=subt_font)
# # ax.set_ylabel('Repair cost ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 4)
# bx = sns.boxplot(y=cost_var, x= "impacted", data=df_cbf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=cost_var, data=df_cbf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# ax.set_title('CBF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair cost ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ## time
# ax = fig.add_subplot(3, 4, 5)
# bx = sns.boxplot(y=time_var, x= "impacted", data=df_mf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sp = sns.stripplot(x='impacted', y=time_var, data=df_mf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('MF-TFP', fontsize=subt_font)
# ax.set_ylabel('Repair time ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 6)
# bx = sns.boxplot(y=time_var, x= "impacted", data=df_mf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=time_var, data=df_mf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('MF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair time ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 7)
# bx = sns.boxplot(y=time_var, x= "impacted", data=df_cbf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=time_var, data=df_cbf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('CBF-TFP', fontsize=subt_font)
# # ax.set_ylabel('Repair time ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 8)
# bx = sns.boxplot(y=time_var, x= "impacted", data=df_cbf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=time_var, data=df_cbf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('CBF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair time ratio', fontsize=axis_font)
# # ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')


# ## time
# ax = fig.add_subplot(3, 4, 9)
# bx = sns.boxplot(y=repl_var, x= "impacted", data=df_mf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sp = sns.stripplot(x='impacted', y=repl_var, data=df_mf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('MF-TFP', fontsize=subt_font)
# ax.set_ylabel('Replacement frequency', fontsize=axis_font)
# ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 10)
# bx = sns.boxplot(y=repl_var, x= "impacted", data=df_mf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=repl_var, data=df_mf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=mf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('MF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair repl ratio', fontsize=axis_font)
# ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 11)
# bx = sns.boxplot(y=repl_var, x= "impacted", data=df_cbf_tfp,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=repl_var, data=df_cbf_tfp, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_tfp_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('CBF-TFP', fontsize=subt_font)
# # ax.set_ylabel('Repair repl ratio', fontsize=axis_font)
# ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# ax = fig.add_subplot(3, 4, 12)
# bx = sns.boxplot(y=repl_var, x= "impacted", data=df_cbf_lrb,  showfliers=False,
#             boxprops={'facecolor': 'none'},
#             width=0.5, ax=ax)
# sns.stripplot(x='impacted', y=repl_var, data=df_cbf_lrb, ax=ax, jitter=True,
#               alpha=0.3, s=5, color=cbf_lrb_color)
# bx.set(xlabel=None)
# bx.set(ylabel=None)
# # ax.set_title('CBF-LRB', fontsize=subt_font)
# # ax.set_ylabel('Repair repl ratio', fontsize=axis_font)
# ax.set_xlabel('Impact', fontsize=axis_font)
# # ax.set_yscale('log')

# fig.tight_layout()
# plt.show()

# # plt.savefig('./eng_struc_figures/impact_dvs.pdf')

# #%% mf drift profile
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 22
# subt_font = 22
# label_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 
# from matplotlib.lines import Line2D

# plt.close('all')
# import seaborn as sns

# # make grid and plot classification predictions

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(1, 2, 1)

# df_plot = df_mf_tfp_i.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     sc =plt.scatter(row['PID'], row['stories'],
#                 s=20, color='red', marker='x')
#     # plt.plot(row['PID'], row['stories']/row['num_stories'],
#     #          color='black', linewidth=0.5)
    
# df_plot = df_mf_tfp_o.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     plt.scatter(row['PID'], row['stories'],
#                 s=20, color='black', marker='x')
    
# # ax.grid(visible=True)
# custom_lines = [Line2D([-1], [-1], color='white', marker='x', markeredgecolor='black'
#                        , markerfacecolor='black', markersize=5),
#                 Line2D([-1], [-1], color='white', marker='x', markeredgecolor='red'
#                                        , markerfacecolor='red', markersize=5)
#                 ]

# ax.legend(custom_lines, ['No impact', 'Impacted'], 
#            fontsize=subt_font)

# ax.set_ylabel(r'Story', fontsize=axis_font)
# ax.set_xlabel(r'Story drift', fontsize=axis_font)
# ax.set_xlim([-0.01, 0.2])

# ax.set_title(r'a) MF-TFP', fontsize=axis_font)

# #### mf-lrb
# ax = fig.add_subplot(1, 2, 2)
# df_plot = df_mf_lrb_i.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     sc =plt.scatter(row['PID'], row['stories'],
#                 s=20, color='red', marker='x')
#     # plt.plot(row['PID'], row['stories']/row['num_stories'],
#     #          color='black', linewidth=0.5)
    
# df_plot = df_mf_lrb_o.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     plt.scatter(row['PID'], row['stories'],
#                 s=20, color='black', marker='x')
    
# # ax.grid(visible=True)
# custom_lines = [Line2D([-1], [-1], color='white', marker='x', markeredgecolor='black'
#                        , markerfacecolor='black', markersize=5),
#                 Line2D([-1], [-1], color='white', marker='x', markeredgecolor='red'
#                                        , markerfacecolor='red', markersize=5)
#                 ]

# # ax.legend(custom_lines, ['No impact', 'Impacted'], 
# #            fontsize=subt_font)

# ax.set_ylabel(r'Story', fontsize=axis_font)
# ax.set_xlabel(r'Story drift', fontsize=axis_font)
# ax.set_title(r'b) MF-LRB', fontsize=axis_font)
# ax.set_xlim([-0.01, 0.2])
# fig.tight_layout()
# plt.show()
# # plt.savefig('./dissertation_figures/mf_profiles.pdf')

# #%% cbf drift profile
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 22
# subt_font = 22
# label_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 
# from matplotlib.lines import Line2D

# plt.close('all')
# import seaborn as sns

# # make grid and plot classification predictions

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(1, 2, 1)

# df_plot = df_cbf_tfp_i.copy()
# print(df_plot['max_drift'].median())

# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     sc =plt.scatter(row['PID'], row['stories'],
#                 s=20, color='red', marker='x')
#     # plt.plot(row['PID'], row['stories']/row['num_stories'],
#     #          color='black', linewidth=0.5)
    
# df_plot = df_cbf_tfp_o.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     plt.scatter(row['PID'], row['stories'],
#                 s=20, color='black', marker='x')
    
# # ax.grid(visible=True)
# custom_lines = [Line2D([-1], [-1], color='white', marker='x', markeredgecolor='black'
#                        , markerfacecolor='black', markersize=5),
#                 Line2D([-1], [-1], color='white', marker='x', markeredgecolor='red'
#                                        , markerfacecolor='red', markersize=5)
#                 ]

# ax.legend(custom_lines, ['No impact', 'Impacted'], 
#            fontsize=subt_font)

# ax.set_ylabel(r'Story', fontsize=axis_font)
# ax.set_xlabel(r'Story drift', fontsize=axis_font)
# ax.set_xlim([-0.01, 0.1])

# ax.set_title(r'a) CBF-TFP', fontsize=axis_font)

# #### cbf-lrb
# ax = fig.add_subplot(1, 2, 2)
# df_plot = df_cbf_lrb_i.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     sc =plt.scatter(row['PID'], row['stories'],
#                 s=20, color='red', marker='x')
#     # plt.plot(row['PID'], row['stories']/row['num_stories'],
#     #          color='black', linewidth=0.5)
    
# df_plot = df_cbf_lrb_o.copy()
# print(df_plot['max_drift'].median())
# df_plot['stories'] = df_plot['num_stories'].map(lambda x: np.arange(1, x+1))
# df_plot['floors'] = df_plot['num_stories'].map(lambda x: np.arange(0, x+1))
# for i, row in df_plot.iterrows():
#     plt.scatter(row['PID'], row['stories'],
#                 s=20, color='black', marker='x')
    
# # ax.grid(visible=True)
# custom_lines = [Line2D([-1], [-1], color='white', marker='x', markeredgecolor='black'
#                        , markerfacecolor='black', markersize=5),
#                 Line2D([-1], [-1], color='white', marker='x', markeredgecolor='red'
#                                        , markerfacecolor='red', markersize=5)
#                 ]

# # ax.legend(custom_lines, ['No impact', 'Impacted'], 
# #            fontsize=subt_font)

# ax.set_ylabel(r'Story', fontsize=axis_font)
# ax.set_xlabel(r'Story drift', fontsize=axis_font)
# ax.set_title(r'b) CBF-LRB', fontsize=axis_font)
# ax.set_xlim([-0.01, 0.1])
# fig.tight_layout()
# plt.show()
# # plt.savefig('./dissertation_figures/cbf_profiles.pdf')


#%% loss histogram
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 22
# subt_font = 22
# label_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 
# from matplotlib.lines import Line2D

# plt.close('all')
# import seaborn as sns

# # make grid and plot classification predictions

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(1, 2, 1)

# df_plot = df_mf_tfp_i.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# df_plot = df_mf_tfp_o.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# # ax.set_xlim([200, 400])
# ax = fig.add_subplot(1, 2, 2)

# df_plot = df_mf_lrb_i.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# df_plot = df_mf_lrb_o.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# # ax.set_xlim([200, 400])
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(1, 2, 1)

# df_plot = df_cbf_tfp_i.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# df_plot = df_cbf_tfp_o.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# # ax.set_xlim([200, 400])
# ax = fig.add_subplot(1, 2, 2)

# df_plot = df_cbf_lrb_i.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)

# df_plot = df_cbf_lrb_o.copy()
# df_plot['normalized_cost'] = df_plot['total_cmp_cost']/df_plot['bldg_area']

# ax.hist(df_plot['cmp_cost_ratio'], bins=20, alpha=0.5, density=True)
# # ax.set_xlim([200, 400])

#%% generalized curve fitting for cost and time

# TODO: should this be real values ($)

def nlls(params, log_x, no_a, no_c):
    from scipy import stats
    import numpy as np
    sigma, beta = params
    theoretical_fragility_function = stats.norm(np.log(sigma), beta).cdf(log_x)
    likelihood = stats.binom.pmf(no_c, no_a, theoretical_fragility_function)
    log_likelihood = np.log(likelihood)
    log_likelihood_sum = np.sum(log_likelihood)

    return -log_likelihood_sum

def mle_fit_general(x_values, probs, x_init=None):
    from functools import partial
    import numpy as np
    from scipy.optimize import basinhopping
    
    log_x = np.log(x_values)
    number_of_analyses = 1000*np.ones(len(x_values))
    number_of_collapses = np.round(1000*probs)
    
    neg_log_likelihood_sum_partial = partial(
        nlls, log_x=log_x, no_a=number_of_analyses, no_c=number_of_collapses)
    
    if x_init is None:
        x0 = (1, 1)
    else:
        x0 = x_init
    
    bnds = ((1e-6, 0.2), (0.5, 1.5))
    
    # use basin hopping to avoid local minima
    minimizer_kwargs={'bounds':bnds}
    res = basinhopping(neg_log_likelihood_sum_partial, x0, minimizer_kwargs=minimizer_kwargs,
                       niter=100, seed=985)
    
    return res.x[0], res.x[1]

from scipy.stats import norm
from scipy.stats import ecdf
f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
plt.close('all')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


fig = plt.figure(figsize=(13, 11))

my_y_var = df_mf_tfp_i[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 1)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_mf_tfp_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('a) MF-TFP', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])



####

my_y_var = df_mf_lrb_i[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 2)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_mf_lrb_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('b) MF-LRB', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####
my_y_var = df_cbf_tfp_i[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 3)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_cbf_tfp_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('c) CBF-TFP', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####

my_y_var = df_cbf_lrb_i[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 4)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_cbf_lrb_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('d) CBF-LRB', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

plt.legend(fontsize=axis_font)
fig.tight_layout()
plt.show()
plt.savefig('./dissertation_figures/cost_ecdf.pdf')

#%%

plt.close('all')


my_y_var = df_mf_tfp_i[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
fig = plt.figure(figsize=(13, 11))

my_y_var = df_mf_tfp_i[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 1)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_mf_tfp_o[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
ax1.set_title('a) MF-TFP', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])



####

my_y_var = df_mf_lrb_i[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 2)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_mf_lrb_o[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
ax1.set_title('b) MF-LRB', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####
my_y_var = df_cbf_tfp_i[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 3)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_cbf_tfp_o[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
ax1.set_title('c) CBF-TFP', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####

my_y_var = df_cbf_lrb_i[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles


# theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_inv, beta_inv)

ax1=fig.add_subplot(2, 2, 4)
ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")

my_y_var = df_cbf_lrb_o[time_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles 

ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# ax1.plot(xx_pr, p)

# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="black")

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
ax1.set_title('d) CBF-LRB', fontsize=title_font)
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])
plt.legend(fontsize=axis_font)
fig.tight_layout()
plt.show()
plt.savefig('./dissertation_figures/time_ecdf.pdf')



#%% generalized curve fitting for cost and time


from scipy.stats import norm
from scipy.stats import ecdf
f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
# plt.close('all')


my_y_var = df_mf_tfp_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(13 , 11))

theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.03,1))

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_onv, beta_onv)

ax1=fig.add_subplot(2, 2, 1)
ax1.plot(xx_pr, p)

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('MF-TFP', fontsize=title_font)
ax1.plot([ecdf_values], [ecdf_prob], 
          marker='x', markersize=5, color="red")
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])



####

my_y_var = df_mf_lrb_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.03,1))

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_onv, beta_onv)

ax1=fig.add_subplot(2, 2, 2)
ax1.plot(xx_pr, p)

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('MF-LRB', fontsize=title_font)
ax1.plot([ecdf_values], [ecdf_prob], 
          marker='x', markersize=5, color="red")
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####

my_y_var = df_cbf_tfp_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.01,1))

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_onv, beta_onv)

ax1=fig.add_subplot(2, 2, 3)
ax1.plot(xx_pr, p)

ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('CBF-TFP', fontsize=title_font)
ax1.plot([ecdf_values], [ecdf_prob], 
          marker='x', markersize=5, color="red")
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

####

my_y_var = df_cbf_lrb_o[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.01,1))

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_onv, beta_onv)

ax1=fig.add_subplot(2, 2, 4)
ax1.plot(xx_pr, p)

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
ax1.set_title('CBF-LRB', fontsize=title_font)
ax1.plot([ecdf_values], [ecdf_prob], 
          marker='x', markersize=5, color="red")
ax1.grid(True)
# ax1.set_xlim([0, 1.0])
# ax1.set_ylim([0, 1.0])

fig.tight_layout()
plt.show()