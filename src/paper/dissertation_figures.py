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

#%% main predictor

def predict_DV(X, impact_pred_mdl, hit_loss_mdl, miss_loss_mdl,
               outcome='cost_50%', return_var=False):
    """Returns the expected value of the decision variable based on the total
    probability law (law of iterated expectation).
    
    E[cost] = sum_i sum_j E[cost|impact_j] Pr(impact_j) 
    
    Currently, this assumes that the models used are all GPC/GPR.
    
    Parameters
    ----------
    X: pd dataframe of design points
    impact_pred_mdl: classification model predicting impact
    hit_loss_mdl: regression model predicting outcome conditioned on yes impact
    miss_loss_mdl: regression model predicting outcome conditioned on no impact
    outcome: desired name for outcome variable
    
    Returns
    -------
    expected_DV_df: DataFrame of expected DV with single column name outcome+'_pred'
    """
        
    # get probability of impact
    if 'log_reg_kernel' in impact_pred_mdl.named_steps.keys():
        probs_imp = impact_pred_mdl.predict_proba(impact_pred_mdl.K_pr)
    else:
        probs_imp = impact_pred_mdl.predict_proba(X)

    miss_prob = probs_imp[:,0]
    hit_prob = probs_imp[:,1]
    
    hit_loss, hit_std = hit_loss_mdl.predict(X, return_std=True)
    miss_loss, miss_std = miss_loss_mdl.predict(X, return_std=True)
    
    # weight with probability of collapse
    # E[Loss] = (impact loss)*Pr(impact) + (no impact loss)*Pr(no impact)
    # run SVR_hit model on this dataset
    outcome_str = outcome+'_pred'
    expected_DV_hit = pd.DataFrame(
            {outcome_str:np.multiply(
                    hit_loss,
                    hit_prob)})
            
    
    # run miss model on this dataset
    expected_DV_miss = pd.DataFrame(
            {outcome_str:np.multiply(
                    miss_loss,
                    miss_prob)})
    
    expected_DV = expected_DV_hit + expected_DV_miss
    
    # tower variance
    # Var[cost] = E[cost^2] - E[cost]^2 = E[E(cost^2|imp)] - expected_DV^2
    # = (hit_cost^2 + var_hit_cost)*p(hit) + (miss_cost^2 + var_miss_cost)*p(miss) - expected_DV^2
    '''
    if return_var:
        expected_loss_sq = (hit_std**2 + hit_loss**2)*hit_prob + (miss_std**2 + miss_loss**2)*miss_prob
        total_var = expected_loss_sq - expected_DV**2
        return(expected_DV, total_var)
    '''
    
    if return_var:
        # get probability of impact
        gpc_obj = impact_pred_mdl._final_estimator
        base_estimator = gpc_obj.base_estimator_
        K_func = base_estimator.kernel_
        W_inv = np.diag(1/base_estimator.W_sr_**2)
        K_a = K_func(base_estimator.X_train_, base_estimator.X_train_)
        R_inv = np.linalg.inv(W_inv + K_a)
        
        # follow Eq. 3.24 to calculate latent variance
        gpc_scaler = impact_pred_mdl[0]
        X_scaled = gpc_scaler.transform(X)
        K_s = K_func(base_estimator.X_train_, X_scaled)
        k_ss = np.diagonal(K_func(X_scaled, X_scaled))
        var_f = k_ss - np.sum((R_inv @ K_s) * K_s, axis=0)
        
        # propagate uncertainty (Wikipedia example for f = ae^(bA)) and f = aA^b
        pi_ = base_estimator.pi_
        y_train_ = base_estimator.y_train_
        f_star = K_s.T.dot(y_train_ - pi_)
        gamma_ = (1 + np.exp(-f_star))
        prob_var = np.exp(-2*f_star)*var_f/(gamma_**4)
        
        # regression model variances
        hit_var = hit_std**2
        miss_var = miss_std**2
        
        # for now, ignore correlation
        # is there correlation? is probability of impact correlated with cost given that the building impacted
        # propagate uncertainty (f = AB)
        
        if miss_loss < 1e-8:
            miss_loss_min = 1e-3
        else:
            miss_loss_min = miss_loss
            
        impact_side_var = np.multiply(hit_loss, hit_prob)**2*(
            (hit_var/hit_loss**2) + (prob_var/hit_prob**2) + 0)
        
        nonimpact_side_var = np.multiply(miss_loss_min, miss_prob)**2*(
            (miss_var/miss_loss_min**2) + (prob_var/miss_prob**2) + 0)
        
        # propagate uncertainty (f = A + B)
        total_var = impact_side_var + nonimpact_side_var + 0
        
        return(expected_DV, total_var)
    else:
        return(expected_DV)
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
    
#%% Calculate upfront cost of data
# calc cost of new point

# TODO: inverse design

def calc_upfront_cost(X, config_dict, steel_cost_dict,
                      land_cost_per_sqft=2837/(3.28**2)):
    
    from scipy.interpolate import interp1d
    zeta_ref = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Bm_ref = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    interp_f = interp1d(zeta_ref, Bm_ref)
    Bm = interp_f(X['zeta_e'])
    
    # estimate Tm
    config_df = pd.DataFrame(config_dict, index=[0])
    from loads import estimate_period, define_gravity_loads
    W_and_loads = config_df.apply(lambda row: define_gravity_loads(row),
                                            axis='columns', result_type='expand')
    
    # order of outputs are below
    # W_seis, W_super, w_on_frame, P_on_leaning_column, all_w_cases, all_plc_cases
    W_seis = W_and_loads.iloc[0][0]
    W_super = W_and_loads.iloc[0][1]
    
    # perform calculation for both MF and CBF
    X_query = X.copy()
    X_query['h_bldg'] = config_dict['num_stories'] * config_dict['h_story']
    
    # estimate periods
    X_mf = X_query.copy()
    X_mf['superstructure_system'] = 'MF'
    X_mf['T_fbe'] = X_mf.apply(lambda row: estimate_period(row),
                                                     axis='columns', result_type='expand')
    
    X_cbf = X_query.copy()
    X_cbf['superstructure_system'] = 'CBF'
    X_cbf['h_bldg'] = config_dict['num_stories'] * config_dict['h_story']
    X_cbf['T_fbe'] = X_cbf.apply(lambda row: estimate_period(row),
                                                     axis='columns', result_type='expand')
    
    
    X_query['T_fbe_mf'] = X_mf['T_fbe']
    X_query['T_fbe_cbf'] = X_cbf['T_fbe']
    
    X_query['T_m_mf'] = X_query['T_fbe_mf'] * X_query['T_ratio']
    X_query['T_m_cbf'] = X_query['T_fbe_cbf'] * X_query['T_ratio']
    
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    SaTm_mf = config_dict['S_1']/X_query['T_m_mf']
    moat_gap_mf = X_query['gap_ratio'] * (g*(SaTm_mf/Bm)*X_query['T_m_mf']**2)/(4*pi**2)
    
    # calculate design base shear
    kM_mf = (1/g)*(2*pi/X_query['T_m_mf'])**2
    Dm_mf = g*config_dict['S_1']*X_query['T_m_mf']/(4*pi**2*Bm)
    Vb_mf = Dm_mf * kM_mf * W_super / 2
    Vst_mf = Vb_mf*(W_super/W_seis)**(1 - 2.5*X_query['zeta_e'])
    Vs_mf = Vst_mf/X_query['RI']
    
    # regression was done for steel cost ~ Vs
    reg_mf = steel_cost_dict['mf']
    try:
        steel_cost_mf = reg_mf.intercept_.item() + reg_mf.coef_.item()*Vs_mf
    except:
        steel_cost_mf = reg_mf.intercept_ + reg_mf.coef_.item()*Vs_mf    
    
    L_bldg = config_dict['L_bay']*config_dict['num_bays']
    land_area_mf = (L_bldg*12.0 + moat_gap_mf)**2
    land_cost_mf = land_cost_per_sqft/144.0 * land_area_mf
    
    # repeat for cbf
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    SaTm_cbf = config_dict['S_1']/X_query['T_m_cbf']
    moat_gap_cbf = X_query['gap_ratio'] * (g*(SaTm_cbf/Bm)*X_query['T_m_cbf']**2)/(4*pi**2)
    
    # calculate design base shear
    kM_cbf = (1/g)*(2*pi/X_query['T_m_cbf'])**2
    Dm_cbf = g*config_dict['S_1']*X_query['T_m_cbf']/(4*pi**2*Bm)
    Vb_cbf = Dm_cbf * kM_cbf * W_super / 2
    Vst_cbf = Vb_cbf*(W_super/W_seis)**(1 - 2.5*X_query['zeta_e'])
    Vs_cbf = Vst_cbf/X_query['RI']
    
    # regression was done for steel cost ~ Vs
    reg_cbf = steel_cost_dict['cbf']
    try:
        steel_cost_cbf = reg_cbf.intercept_.item() + reg_cbf.coef_.item()*Vs_cbf
    except:
        steel_cost_cbf = reg_cbf.intercept_ + reg_cbf.coef_.item()*Vs_cbf
        
    L_bldg = config_dict['L_bay']*config_dict['num_bays']
    land_area_cbf = (L_bldg*12.0 + moat_gap_cbf)**2
    land_cost_cbf = land_cost_per_sqft/144.0 * land_area_cbf
    
    return({'total_mf': steel_cost_mf + land_cost_mf,
            'steel_mf': steel_cost_mf,
            'land_mf': land_cost_mf,
           'total_cbf': steel_cost_cbf + land_cost_cbf,
           'steel_cbf': steel_cost_cbf,
           'land_cbf': land_cost_cbf,
           'Vs_cbf': Vs_cbf,
           'Vs_mf': Vs_mf})




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
df['log_cost'] = np.log(df['cmp_cost_ratio'])
df['log_time'] = np.log(df['cmp_time_ratio'])
df['log_repl'] = np.log(df['replacement_freq'])

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

# linear regress cost as f(base shear)
from sklearn.linear_model import LinearRegression
reg_mf = LinearRegression(fit_intercept=False)
reg_mf.fit(X=df_mf[['Vs']], y=df_mf[['steel_cost']])

reg_cbf = LinearRegression(fit_intercept=False)
reg_cbf.fit(X=df_cbf[['Vs']], y=df_cbf[['steel_cost']])

reg_dict = {
    'mf':reg_mf,
    'cbf':reg_cbf
    }

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


# def nlls(params, log_x, no_a, no_c):
#     from scipy import stats
#     import numpy as np
#     sigma, beta = params
#     theoretical_fragility_function = stats.norm(np.log(sigma), beta).cdf(log_x)
#     likelihood = stats.binom.pmf(no_c, no_a, theoretical_fragility_function)
#     log_likelihood = np.log(likelihood)
#     log_likelihood_sum = np.sum(log_likelihood)

#     return -log_likelihood_sum

# def mle_fit_general(x_values, probs, x_init=None):
#     from functools import partial
#     import numpy as np
#     from scipy.optimize import basinhopping
    
#     log_x = np.log(x_values)
#     number_of_analyses = 1000*np.ones(len(x_values))
#     number_of_collapses = np.round(1000*probs)
    
#     neg_log_likelihood_sum_partial = partial(
#         nlls, log_x=log_x, no_a=number_of_analyses, no_c=number_of_collapses)
    
#     if x_init is None:
#         x0 = (1, 1)
#     else:
#         x0 = x_init
    
#     bnds = ((1e-6, 0.2), (0.5, 1.5))
    
#     # use basin hopping to avoid local minima
#     minimizer_kwargs={'bounds':bnds}
#     res = basinhopping(neg_log_likelihood_sum_partial, x0, minimizer_kwargs=minimizer_kwargs,
#                        niter=100, seed=985)
    
#     return res.x[0], res.x[1]

# from scipy.stats import norm
# from scipy.stats import ecdf
# f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
# plt.close('all')

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 


# fig = plt.figure(figsize=(13, 11))

# my_y_var = df_mf_tfp_i[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 1)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_mf_tfp_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('a) MF-TFP', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])



# ####

# my_y_var = df_mf_lrb_i[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 2)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_mf_lrb_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('b) MF-LRB', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####
# my_y_var = df_cbf_tfp_i[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 3)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_cbf_tfp_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('c) CBF-TFP', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####

# my_y_var = df_cbf_lrb_i[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 4)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_cbf_lrb_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('d) CBF-LRB', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# plt.legend(fontsize=axis_font)
# fig.tight_layout()
# plt.show()
# plt.savefig('./dissertation_figures/cost_ecdf.pdf')

#%%

# plt.close('all')


# my_y_var = df_mf_tfp_i[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 
# fig = plt.figure(figsize=(13, 11))

# my_y_var = df_mf_tfp_i[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 1)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_mf_tfp_o[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
# ax1.set_title('a) MF-TFP', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])



# ####

# my_y_var = df_mf_lrb_i[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 2)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_mf_lrb_o[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
# ax1.set_title('b) MF-LRB', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####
# my_y_var = df_cbf_tfp_i[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 3)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_cbf_tfp_o[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
# ax1.set_title('c) CBF-TFP', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####

# my_y_var = df_cbf_lrb_i[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles


# # theta_inv, beta_inv = mle_fit_general(ecdf_values,ecdf_prob, x_init=(0.03,1))

# # xx_pr = np.linspace(1e-4, 1.0, 400)
# # p = f(xx_pr, theta_inv, beta_inv)

# ax1=fig.add_subplot(2, 2, 4)
# ax1.ecdf(my_y_var, color='red', linewidth=1.5, label='Impacted')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="red")

# my_y_var = df_cbf_lrb_o[time_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles 

# ax1.ecdf(my_y_var, color='black', linewidth=1.5, label='No impact')
# # ax1.plot(xx_pr, p)

# # ax1.plot([ecdf_values], [ecdf_prob], 
# #           marker='x', markersize=5, color="black")

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair time ratio', fontsize=axis_font)
# ax1.set_title('d) CBF-LRB', fontsize=title_font)
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])
# plt.legend(fontsize=axis_font)
# fig.tight_layout()
# plt.show()
# plt.savefig('./dissertation_figures/time_ecdf.pdf')



#%% generalized curve fitting for cost and time


# from scipy.stats import norm
# from scipy.stats import ecdf
# f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
# # plt.close('all')


# my_y_var = df_mf_tfp_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# fig = plt.figure(figsize=(13 , 11))

# theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_onv, beta_onv)

# ax1=fig.add_subplot(2, 2, 1)
# ax1.plot(xx_pr, p)

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('MF-TFP', fontsize=title_font)
# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])



# ####

# my_y_var = df_mf_lrb_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.03,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_onv, beta_onv)

# ax1=fig.add_subplot(2, 2, 2)
# ax1.plot(xx_pr, p)

# # ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# # ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('MF-LRB', fontsize=title_font)
# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####

# my_y_var = df_cbf_tfp_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.01,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_onv, beta_onv)

# ax1=fig.add_subplot(2, 2, 3)
# ax1.plot(xx_pr, p)

# ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('CBF-TFP', fontsize=title_font)
# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# ####

# my_y_var = df_cbf_lrb_o[cost_var]
# res = ecdf(my_y_var)
# ecdf_prob = res.cdf.probabilities
# ecdf_values = res.cdf.quantiles

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# theta_onv, beta_onv = mle_fit_general(ecdf_values,ecdf_prob, x_onit=(0.01,1))

# xx_pr = np.linspace(1e-4, 1.0, 400)
# p = f(xx_pr, theta_onv, beta_onv)

# ax1=fig.add_subplot(2, 2, 4)
# ax1.plot(xx_pr, p)

# # ax1.set_ylabel(r'$P(X \leq x)$', fontsize=axis_font)
# ax1.set_xlabel(r'Repair cost ratio', fontsize=axis_font)
# ax1.set_title('CBF-LRB', fontsize=title_font)
# ax1.plot([ecdf_values], [ecdf_prob], 
#           marker='x', markersize=5, color="red")
# ax1.grid(True)
# # ax1.set_xlim([0, 1.0])
# # ax1.set_ylim([0, 1.0])

# fig.tight_layout()
# plt.show()


#%% check if conditioned regression is better than raw

#%% impact classification model
# for each system, make separate impact classification model

mdl_all = GP(df)
mdl_all.set_covariates(covariate_list)

mdl_impact_cbf_lrb = GP(df_cbf_lrb)
mdl_impact_cbf_lrb.set_covariates(covariate_list)
mdl_impact_cbf_lrb.set_outcome('impacted')
mdl_impact_cbf_lrb.test_train_split(0.2)

mdl_impact_cbf_tfp = GP(df_cbf_tfp)
mdl_impact_cbf_tfp.set_covariates(covariate_list)
mdl_impact_cbf_tfp.set_outcome('impacted')
mdl_impact_cbf_tfp.test_train_split(0.2)

mdl_impact_mf_lrb = GP(df_mf_lrb)
mdl_impact_mf_lrb.set_covariates(covariate_list)
mdl_impact_mf_lrb.set_outcome('impacted')
mdl_impact_mf_lrb.test_train_split(0.2)

mdl_impact_mf_tfp = GP(df_mf_tfp)
mdl_impact_mf_tfp.set_covariates(covariate_list)
mdl_impact_mf_tfp.set_outcome('impacted')
mdl_impact_mf_tfp.test_train_split(0.2)

print('======= impact classification per system ========')
import time
t0 = time.time()

mdl_impact_cbf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_impact_cbf_tfp.fit_gpc(kernel_name='rbf_iso')
mdl_impact_mf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_impact_mf_tfp.fit_gpc(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPC training for impact done for 4 models in %.3f s" % tp)

# density estimation model to enable constructability 
print('======= Density estimation per system ========')

t0 = time.time()

mdl_impact_mf_lrb.fit_kde()
mdl_impact_cbf_lrb.fit_kde()
mdl_impact_mf_tfp.fit_kde()
mdl_impact_cbf_tfp.fit_kde()

tp = time.time() - t0

print("KDE training done for 4 models in %.3f s" % tp)


impact_classification_mdls = {'mdl_impact_cbf_lrb': mdl_impact_cbf_lrb,
                        'mdl_impact_cbf_tfp': mdl_impact_cbf_tfp,
                        'mdl_impact_mf_lrb': mdl_impact_mf_lrb,
                        'mdl_impact_mf_tfp': mdl_impact_mf_tfp}

#%% regression models: cost
# goal: E[cost|sys=sys, impact=impact]

mdl_cost_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_cost_cbf_lrb_i.set_covariates(covariate_list)
mdl_cost_cbf_lrb_i.set_outcome(cost_var)
mdl_cost_cbf_lrb_i.test_train_split(0.2)

mdl_cost_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_cost_cbf_lrb_o.set_covariates(covariate_list)
mdl_cost_cbf_lrb_o.set_outcome(cost_var)
mdl_cost_cbf_lrb_o.test_train_split(0.2)

mdl_cost_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_cost_cbf_tfp_i.set_covariates(covariate_list)
mdl_cost_cbf_tfp_i.set_outcome(cost_var)
mdl_cost_cbf_tfp_i.test_train_split(0.2)

mdl_cost_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_cost_cbf_tfp_o.set_covariates(covariate_list)
mdl_cost_cbf_tfp_o.set_outcome(cost_var)
mdl_cost_cbf_tfp_o.test_train_split(0.2)

mdl_cost_mf_lrb_i = GP(df_mf_lrb_i)
mdl_cost_mf_lrb_i.set_covariates(covariate_list)
mdl_cost_mf_lrb_i.set_outcome(cost_var)
mdl_cost_mf_lrb_i.test_train_split(0.2)

mdl_cost_mf_lrb_o = GP(df_mf_lrb_o)
mdl_cost_mf_lrb_o.set_covariates(covariate_list)
mdl_cost_mf_lrb_o.set_outcome(cost_var)
mdl_cost_mf_lrb_o.test_train_split(0.2)

mdl_cost_mf_tfp_i = GP(df_mf_tfp_i)
mdl_cost_mf_tfp_i.set_covariates(covariate_list)
mdl_cost_mf_tfp_i.set_outcome(cost_var)
mdl_cost_mf_tfp_i.test_train_split(0.2)

mdl_cost_mf_tfp_o = GP(df_mf_tfp_o)
mdl_cost_mf_tfp_o.set_covariates(covariate_list)
mdl_cost_mf_tfp_o.set_outcome(cost_var)
mdl_cost_mf_tfp_o.test_train_split(0.2)

print('======= cost regression per system per impact ========')
import time
t0 = time.time()

mdl_cost_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for cost done for 8 models in %.3f s" % tp)

cost_regression_mdls = {'mdl_cost_cbf_lrb_i': mdl_cost_cbf_lrb_i,
                        'mdl_cost_cbf_lrb_o': mdl_cost_cbf_lrb_o,
                        'mdl_cost_cbf_tfp_i': mdl_cost_cbf_tfp_i,
                        'mdl_cost_cbf_tfp_o': mdl_cost_cbf_tfp_o,
                        'mdl_cost_mf_lrb_i': mdl_cost_mf_lrb_i,
                        'mdl_cost_mf_lrb_o': mdl_cost_mf_lrb_o,
                        'mdl_cost_mf_tfp_i': mdl_cost_mf_tfp_i,
                        'mdl_cost_mf_tfp_o': mdl_cost_mf_tfp_o}

#%% regression models: time
# goal: E[time|sys=sys, impact=impact]

mdl_time_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_time_cbf_lrb_i.set_covariates(covariate_list)
mdl_time_cbf_lrb_i.set_outcome(time_var)
mdl_time_cbf_lrb_i.test_train_split(0.2)

mdl_time_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_time_cbf_lrb_o.set_covariates(covariate_list)
mdl_time_cbf_lrb_o.set_outcome(time_var)
mdl_time_cbf_lrb_o.test_train_split(0.2)

mdl_time_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_time_cbf_tfp_i.set_covariates(covariate_list)
mdl_time_cbf_tfp_i.set_outcome(time_var)
mdl_time_cbf_tfp_i.test_train_split(0.2)

mdl_time_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_time_cbf_tfp_o.set_covariates(covariate_list)
mdl_time_cbf_tfp_o.set_outcome(time_var)
mdl_time_cbf_tfp_o.test_train_split(0.2)

mdl_time_mf_lrb_i = GP(df_mf_lrb_i)
mdl_time_mf_lrb_i.set_covariates(covariate_list)
mdl_time_mf_lrb_i.set_outcome(time_var)
mdl_time_mf_lrb_i.test_train_split(0.2)

mdl_time_mf_lrb_o = GP(df_mf_lrb_o)
mdl_time_mf_lrb_o.set_covariates(covariate_list)
mdl_time_mf_lrb_o.set_outcome(time_var)
mdl_time_mf_lrb_o.test_train_split(0.2)

mdl_time_mf_tfp_i = GP(df_mf_tfp_i)
mdl_time_mf_tfp_i.set_covariates(covariate_list)
mdl_time_mf_tfp_i.set_outcome(time_var)
mdl_time_mf_tfp_i.test_train_split(0.2)

mdl_time_mf_tfp_o = GP(df_mf_tfp_o)
mdl_time_mf_tfp_o.set_covariates(covariate_list)
mdl_time_mf_tfp_o.set_outcome(time_var)
mdl_time_mf_tfp_o.test_train_split(0.2)

print('======= downtime regression per system per impact ========')
import time
t0 = time.time()

mdl_time_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for time done for 8 models in %.3f s" % tp)

time_regression_mdls = {'mdl_time_cbf_lrb_i': mdl_time_cbf_lrb_i,
                        'mdl_time_cbf_lrb_o': mdl_time_cbf_lrb_o,
                        'mdl_time_cbf_tfp_i': mdl_time_cbf_tfp_i,
                        'mdl_time_cbf_tfp_o': mdl_time_cbf_tfp_o,
                        'mdl_time_mf_lrb_i': mdl_time_mf_lrb_i,
                        'mdl_time_mf_lrb_o': mdl_time_mf_lrb_o,
                        'mdl_time_mf_tfp_i': mdl_time_mf_tfp_i,
                        'mdl_time_mf_tfp_o': mdl_time_mf_tfp_o}

#%% regression models: repl
# goal: E[repl|sys=sys, impact=impact]

mdl_repl_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_repl_cbf_lrb_i.set_covariates(covariate_list)
mdl_repl_cbf_lrb_i.set_outcome('replacement_freq')
mdl_repl_cbf_lrb_i.test_train_split(0.2)

mdl_repl_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_repl_cbf_lrb_o.set_covariates(covariate_list)
mdl_repl_cbf_lrb_o.set_outcome('replacement_freq')
mdl_repl_cbf_lrb_o.test_train_split(0.2)

mdl_repl_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_repl_cbf_tfp_i.set_covariates(covariate_list)
mdl_repl_cbf_tfp_i.set_outcome('replacement_freq')
mdl_repl_cbf_tfp_i.test_train_split(0.2)

mdl_repl_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_repl_cbf_tfp_o.set_covariates(covariate_list)
mdl_repl_cbf_tfp_o.set_outcome('replacement_freq')
mdl_repl_cbf_tfp_o.test_train_split(0.2)

mdl_repl_mf_lrb_i = GP(df_mf_lrb_i)
mdl_repl_mf_lrb_i.set_covariates(covariate_list)
mdl_repl_mf_lrb_i.set_outcome('replacement_freq')
mdl_repl_mf_lrb_i.test_train_split(0.2)

mdl_repl_mf_lrb_o = GP(df_mf_lrb_o)
mdl_repl_mf_lrb_o.set_covariates(covariate_list)
mdl_repl_mf_lrb_o.set_outcome('replacement_freq')
mdl_repl_mf_lrb_o.test_train_split(0.2)

mdl_repl_mf_tfp_i = GP(df_mf_tfp_i)
mdl_repl_mf_tfp_i.set_covariates(covariate_list)
mdl_repl_mf_tfp_i.set_outcome('replacement_freq')
mdl_repl_mf_tfp_i.test_train_split(0.2)

mdl_repl_mf_tfp_o = GP(df_mf_tfp_o)
mdl_repl_mf_tfp_o.set_covariates(covariate_list)
mdl_repl_mf_tfp_o.set_outcome('replacement_freq')
mdl_repl_mf_tfp_o.test_train_split(0.2)

t0 = time.time()

print('======= replacement regression per system per impact ========')

mdl_repl_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for replacement done for 8 models in %.3f s" % tp)

repl_regression_mdls = {'mdl_repl_cbf_lrb_i': mdl_repl_cbf_lrb_i,
                        'mdl_repl_cbf_lrb_o': mdl_repl_cbf_lrb_o,
                        'mdl_repl_cbf_tfp_i': mdl_repl_cbf_tfp_i,
                        'mdl_repl_cbf_tfp_o': mdl_repl_cbf_tfp_o,
                        'mdl_repl_mf_lrb_i': mdl_repl_mf_lrb_i,
                        'mdl_repl_mf_lrb_o': mdl_repl_mf_lrb_o,
                        'mdl_repl_mf_tfp_i': mdl_repl_mf_tfp_i,
                        'mdl_repl_mf_tfp_o': mdl_repl_mf_tfp_o}

#%%
mdl_cost_cbf_lrb = GP(df_cbf_lrb)
mdl_cost_cbf_lrb.set_covariates(covariate_list)
mdl_cost_cbf_lrb.set_outcome(cost_var)
mdl_cost_cbf_lrb.test_train_split(0.2)


mdl_cost_cbf_tfp = GP(df_cbf_tfp)
mdl_cost_cbf_tfp.set_covariates(covariate_list)
mdl_cost_cbf_tfp.set_outcome(cost_var)
mdl_cost_cbf_tfp.test_train_split(0.2)

mdl_cost_mf_lrb = GP(df_mf_lrb)
mdl_cost_mf_lrb.set_covariates(covariate_list)
mdl_cost_mf_lrb.set_outcome(cost_var)
mdl_cost_mf_lrb.test_train_split(0.2)

mdl_cost_mf_tfp = GP(df_mf_tfp)
mdl_cost_mf_tfp.set_covariates(covariate_list)
mdl_cost_mf_tfp.set_outcome(cost_var)
mdl_cost_mf_tfp.test_train_split(0.2)

mdl_cost_cbf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp.fit_gpr(kernel_name='rbf_iso')



mdl_time_cbf_lrb = GP(df_cbf_lrb)
mdl_time_cbf_lrb.set_covariates(covariate_list)
mdl_time_cbf_lrb.set_outcome(time_var)
mdl_time_cbf_lrb.test_train_split(0.2)


mdl_time_cbf_tfp = GP(df_cbf_tfp)
mdl_time_cbf_tfp.set_covariates(covariate_list)
mdl_time_cbf_tfp.set_outcome(time_var)
mdl_time_cbf_tfp.test_train_split(0.2)

mdl_time_mf_lrb = GP(df_mf_lrb)
mdl_time_mf_lrb.set_covariates(covariate_list)
mdl_time_mf_lrb.set_outcome(time_var)
mdl_time_mf_lrb.test_train_split(0.2)

mdl_time_mf_tfp = GP(df_mf_tfp)
mdl_time_mf_tfp.set_covariates(covariate_list)
mdl_time_mf_tfp.set_outcome(time_var)
mdl_time_mf_tfp.test_train_split(0.2)

mdl_time_cbf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_tfp.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_tfp.fit_gpr(kernel_name='rbf_iso')


mdl_repl_cbf_lrb = GP(df_cbf_lrb)
mdl_repl_cbf_lrb.set_covariates(covariate_list)
mdl_repl_cbf_lrb.set_outcome(repl_var)
mdl_repl_cbf_lrb.test_train_split(0.2)


mdl_repl_cbf_tfp = GP(df_cbf_tfp)
mdl_repl_cbf_tfp.set_covariates(covariate_list)
mdl_repl_cbf_tfp.set_outcome(repl_var)
mdl_repl_cbf_tfp.test_train_split(0.2)

mdl_repl_mf_lrb = GP(df_mf_lrb)
mdl_repl_mf_lrb.set_covariates(covariate_list)
mdl_repl_mf_lrb.set_outcome(repl_var)
mdl_repl_mf_lrb.test_train_split(0.2)

mdl_repl_mf_tfp = GP(df_mf_tfp)
mdl_repl_mf_tfp.set_covariates(covariate_list)
mdl_repl_mf_tfp.set_outcome(repl_var)
mdl_repl_mf_tfp.test_train_split(0.2)

mdl_repl_cbf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_tfp.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_lrb.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_tfp.fit_gpr(kernel_name='rbf_iso')


#%%
print('============= mean squared error of cost regression, no conditioning =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - cost
y_pred = mdl_cost_mf_tfp.gpr.predict(mdl_cost_mf_tfp.X_test)
mse = mean_squared_error(mdl_cost_mf_tfp.y_test, y_pred)
print('MF-TFP, cost:', mse)

# MF LRB - cost
y_pred = mdl_cost_mf_lrb.gpr.predict(mdl_cost_mf_lrb.X_test)
mse = mean_squared_error(mdl_cost_mf_lrb.y_test, y_pred)
print('MF-LRB, cost', mse)

# CBF TFP - cost
y_pred = mdl_cost_cbf_tfp.gpr.predict(mdl_cost_cbf_tfp.X_test)
mse = mean_squared_error(mdl_cost_cbf_tfp.y_test, y_pred)
print('CBF-TFP, cost', mse)

# CBF LRB - cost
y_pred = mdl_cost_cbf_lrb.gpr.predict(mdl_cost_cbf_lrb.X_test)
mse = mean_squared_error(mdl_cost_cbf_lrb.y_test, y_pred)
print('CBF-LRB, cost', mse)

print('============= mean squared error of cost regression, impact conditioned =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - cost
y_pred = predict_DV(mdl_cost_mf_tfp.X_test, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr, 
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_cost_mf_tfp.y_test, y_pred)
print('MF-TFP, cost:', mse)

# MF LRB - cost
y_pred = predict_DV(mdl_cost_mf_lrb.X_test, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr, 
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_cost_mf_lrb.y_test, y_pred)
print('MF-LRB, cost', mse)

# CBF TFP - cost
y_pred = predict_DV(mdl_cost_cbf_tfp.X_test, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr, 
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_cost_cbf_tfp.y_test, y_pred)
print('CBF-TFP, cost', mse)

# CBF LRB - cost
y_pred = predict_DV(mdl_cost_cbf_lrb.X_test, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr, 
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_cost_cbf_lrb.y_test, y_pred)
print('CBF-LRB, cost', mse)

#%%
print('============= mean squared error of time regression, no conditioning =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - time
y_pred = mdl_time_mf_tfp.gpr.predict(mdl_time_mf_tfp.X_test)
mse = mean_squared_error(mdl_time_mf_tfp.y_test, y_pred)
print('MF-TFP, time:', mse)

# MF LRB - time
y_pred = mdl_time_mf_lrb.gpr.predict(mdl_time_mf_lrb.X_test)
mse = mean_squared_error(mdl_time_mf_lrb.y_test, y_pred)
print('MF-LRB, time', mse)

# CBF TFP - time
y_pred = mdl_time_cbf_tfp.gpr.predict(mdl_time_cbf_tfp.X_test)
mse = mean_squared_error(mdl_time_cbf_tfp.y_test, y_pred)
print('CBF-TFP, time', mse)

# CBF LRB - time
y_pred = mdl_time_cbf_lrb.gpr.predict(mdl_time_cbf_lrb.X_test)
mse = mean_squared_error(mdl_time_cbf_lrb.y_test, y_pred)
print('CBF-LRB, time', mse)

print('============= mean squared error of time regression, impact conditioned =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - time
y_pred = predict_DV(mdl_time_mf_tfp.X_test, mdl_impact_mf_tfp.gpc, mdl_time_mf_tfp_i.gpr, mdl_time_mf_tfp_o.gpr, 
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_time_mf_tfp.y_test, y_pred)
print('MF-TFP, time:', mse)

# MF LRB - time
y_pred = predict_DV(mdl_time_mf_lrb.X_test, mdl_impact_mf_lrb.gpc, mdl_time_mf_lrb_i.gpr, mdl_time_mf_lrb_o.gpr, 
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_time_mf_lrb.y_test, y_pred)
print('MF-LRB, time', mse)

# CBF TFP - time
y_pred = predict_DV(mdl_time_cbf_tfp.X_test, mdl_impact_cbf_tfp.gpc, mdl_time_cbf_tfp_i.gpr, mdl_time_cbf_tfp_o.gpr, 
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_time_cbf_tfp.y_test, y_pred)
print('CBF-TFP, time', mse)

# CBF LRB - time
y_pred = predict_DV(mdl_time_cbf_lrb.X_test, mdl_impact_cbf_lrb.gpc, mdl_time_cbf_lrb_i.gpr, mdl_time_cbf_lrb_o.gpr, 
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_time_cbf_lrb.y_test, y_pred)
print('CBF-LRB, time', mse)

#%%
print('============= mean squared error of repl regression, no conditioning =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - repl
y_pred = mdl_repl_mf_tfp.gpr.predict(mdl_repl_mf_tfp.X_test)
mse = mean_squared_error(mdl_repl_mf_tfp.y_test, y_pred)
print('MF-TFP, repl:', mse)

# MF LRB - repl
y_pred = mdl_repl_mf_lrb.gpr.predict(mdl_repl_mf_lrb.X_test)
mse = mean_squared_error(mdl_repl_mf_lrb.y_test, y_pred)
print('MF-LRB, repl', mse)

# CBF TFP - repl
y_pred = mdl_repl_cbf_tfp.gpr.predict(mdl_repl_cbf_tfp.X_test)
mse = mean_squared_error(mdl_repl_cbf_tfp.y_test, y_pred)
print('CBF-TFP, repl', mse)

# CBF LRB - repl
y_pred = mdl_repl_cbf_lrb.gpr.predict(mdl_repl_cbf_lrb.X_test)
mse = mean_squared_error(mdl_repl_cbf_lrb.y_test, y_pred)
print('CBF-LRB, repl', mse)

print('============= mean squared error of repl regression, impact conditioned =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - repl
y_pred = predict_DV(mdl_repl_mf_tfp.X_test, mdl_impact_mf_tfp.gpc, mdl_repl_mf_tfp_i.gpr, mdl_repl_mf_tfp_o.gpr, 
                    outcome=repl_var, return_var=False)
mse = mean_squared_error(mdl_repl_mf_tfp.y_test, y_pred)
print('MF-TFP, repl:', mse)

# MF LRB - repl
y_pred = predict_DV(mdl_repl_mf_lrb.X_test, mdl_impact_mf_lrb.gpc, mdl_repl_mf_lrb_i.gpr, mdl_repl_mf_lrb_o.gpr, 
                    outcome=repl_var, return_var=False)
mse = mean_squared_error(mdl_repl_mf_lrb.y_test, y_pred)
print('MF-LRB, repl', mse)

# CBF TFP - repl
y_pred = predict_DV(mdl_repl_cbf_tfp.X_test, mdl_impact_cbf_tfp.gpc, mdl_repl_cbf_tfp_i.gpr, mdl_repl_cbf_tfp_o.gpr, 
                    outcome=repl_var, return_var=False)
mse = mean_squared_error(mdl_repl_cbf_tfp.y_test, y_pred)
print('CBF-TFP, repl', mse)

# CBF LRB - repl
y_pred = predict_DV(mdl_repl_cbf_lrb.X_test, mdl_impact_cbf_lrb.gpc, mdl_repl_cbf_lrb_i.gpr, mdl_repl_cbf_lrb_o.gpr, 
                    outcome=repl_var, return_var=False)
mse = mean_squared_error(mdl_repl_cbf_lrb.y_test, y_pred)
print('CBF-LRB, repl', mse)


#%% sample for CBF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
# plt.savefig('./dissertation_figures/cbf_lrb_conditioned.pdf')
#%% sample for CBF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample for MF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample for MF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()


#%% sample unconditioned for CBF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_cost_cbf_lrb.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_cost_cbf_lrb.gpr.predict(X_plot)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
plt.show()
# plt.savefig('./dissertation_figures/cbf_lrb_gpr_only.pdf')
#%% sample unconditioned for CBF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_cost_cbf_tfp.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_cost_cbf_tfp.gpr.predict(X_plot)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample unconditioned for MF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_cost_mf_tfp.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_cost_mf_tfp.gpr.predict(X_plot)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample unconditioned for MF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_mf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_cost_mf_lrb.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_mf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_cost_mf_lrb.gpr.predict(X_plot)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])


ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()


#%% what about predicting log
mdl_log_cost_cbf_lrb = GP(df_cbf_lrb)
mdl_log_cost_cbf_lrb.set_covariates(covariate_list)
mdl_log_cost_cbf_lrb.set_outcome('log_cost')
mdl_log_cost_cbf_lrb.test_train_split(0.2)

mdl_log_cost_cbf_lrb.fit_gpr(kernel_name='rbf_ard')

from sklearn.metrics import mean_squared_error
# CBF-LRB - log cost
y_pred = mdl_log_cost_cbf_lrb.gpr.predict(mdl_log_cost_cbf_lrb.X_test)
mse = mean_squared_error(mdl_cost_cbf_lrb.y_test, np.exp(y_pred))
print('CBF-LRB, log cost:', mse)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_log_cost_cbf_lrb.gpr.predict(X_plot)
Z = np.exp(Z)

xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_cost_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_log_cost_cbf_lrb.gpr.predict(X_plot)
Z = np.exp(Z)
xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
plt.show()

#%% sample for CBF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_cost_cbf_lrb.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('a) GPR only: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('b) Conditioned: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)
fig.tight_layout()
# plt.savefig('./dissertation_figures/cbf_lrb_conditioned_compare.pdf')

#%% sample for MF-TFP

# mdl_collapse_mf_tfp = GP(df_mf_tfp)
# mdl_collapse_mf_tfp.set_covariates(covariate_list)
# mdl_collapse_mf_tfp.set_outcome('collapse_prob')
# mdl_collapse_mf_tfp.test_train_split(0.2)
# mdl_collapse_mf_tfp.fit_gpr(kernel_name='rbf_iso')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 24
subt_font = 24
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = mdl_repl_mf_tfp.gpr.predict(X_plot)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[repl_var], c=df_mf_tfp[repl_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement frequency', fontsize=axis_font)
ax.set_title('a) GPR only: $T_M/T_{fb} = 2.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_mf_tfp.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_mf_tfp.gpc, mdl_repl_mf_tfp_i.gpr, mdl_repl_mf_tfp_o.gpr,
                    outcome=repl_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[repl_var], c=df_mf_tfp[repl_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement frequency', fontsize=axis_font)
ax.set_title('b) Conditioned: $T_M/T_{fb} = 2.0$, $\zeta_M = 0.15$', fontsize=subt_font)
fig.tight_layout()
# plt.savefig('./dissertation_figures/mf_tfp_conditioned_compare.pdf')

#%%

# TODO: impact classification plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 28
title_font = 28
subt_font = 24
import matplotlib as mpl
label_size = 24
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
# make grid and plot classification predictions

fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(1, 2, 1)

xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(df_cbf_lrb[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact_cbf_lrb.gpc.predict_proba(X_plot)[:,1]


x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

lvls = [0.1, 0.3, 0.5, 0.7, 0.9]
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.5, colors='white',
                  levels=lvls)
clabels = plt.clabel(cs, fontsize=subt_font, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='black', pad=0)) for txt in clabels]

ax.scatter(df_cbf_lrb_i[xvar][:plt_density],
            df_cbf_lrb_i[yvar][:plt_density],
            s=80, c='red', marker='X', edgecolors='black', label='Impacted')

ax.scatter(df_cbf_lrb_o[xvar][:plt_density],
            df_cbf_lrb_o[yvar][:plt_density],
            s=50, c='blue', edgecolors='white', label='No impact')
plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.0)
ax.set_title(r'a) CBF-LRB impact', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.grid('on', zorder=0)
####

ax = fig.add_subplot(1, 2, 2)
xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(df_cbf_tfp[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact_cbf_tfp.gpc.predict_proba(X_plot)[:,1]


x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.5, colors='white',
                  levels=lvls)
clabels = plt.clabel(cs, fontsize=subt_font, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='black', pad=0)) for txt in clabels]

ax.scatter(df_cbf_tfp_i[xvar][:plt_density],
            df_cbf_tfp_i[yvar][:plt_density],
            s=80, c='red', marker='X', edgecolors='black', label='Impacted')

ax.scatter(df_cbf_tfp_o[xvar][:plt_density],
            df_cbf_tfp_o[yvar][:plt_density],
            s=50, c='blue', edgecolors='white', label='No impact')
# plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.0)
ax.set_title(r'b) CBF-TFP impact', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)

ax.grid('on', zorder=0)
# plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
fig.tight_layout()
plt.show()

# plt.savefig('./dissertation_figures/cbf_impact_clf.pdf')



#### MF


fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(1, 2, 1)

xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(df_mf_lrb[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact_mf_lrb.gpc.predict_proba(X_plot)[:,1]


x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

lvls = [0.1, 0.3, 0.5, 0.7, 0.9]
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.5, colors='white',
                  levels=lvls)
clabels = plt.clabel(cs, fontsize=subt_font, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='black', pad=0)) for txt in clabels]

ax.scatter(df_mf_lrb_i[xvar][:plt_density],
            df_mf_lrb_i[yvar][:plt_density],
            s=80, c='red', marker='X', edgecolors='black', label='Impacted')

ax.scatter(df_mf_lrb_o[xvar][:plt_density],
            df_mf_lrb_o[yvar][:plt_density],
            s=50, c='blue', edgecolors='white', label='No impact')
plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.0)
ax.set_title(r'a) MF-LRB impact', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.grid('on', zorder=0)
####

ax = fig.add_subplot(1, 2, 2)
xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(df_mf_tfp[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact_mf_tfp.gpc.predict_proba(X_plot)[:,1]


x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_classif = Z.reshape(xx_pl.shape)

plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.5, colors='white',
                  levels=lvls)
clabels = plt.clabel(cs, fontsize=subt_font, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='black', pad=0)) for txt in clabels]

ax.scatter(df_mf_tfp_i[xvar][:plt_density],
            df_mf_tfp_i[yvar][:plt_density],
            s=80, c='red', marker='X', edgecolors='black', label='Impacted')

ax.scatter(df_mf_tfp_o[xvar][:plt_density],
            df_mf_tfp_o[yvar][:plt_density],
            s=50, c='blue', edgecolors='white', label='No impact')
# plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.0)
ax.set_title(r'b) MF-TFP impact', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)

ax.grid('on', zorder=0)
# plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
fig.tight_layout()
plt.show()

# plt.savefig('./dissertation_figures/mf_impact_clf.pdf')

#%% nonimpact prediction


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 24
subt_font = 24
label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16,13))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(df_mf_lrb_i[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.2)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_mf_lrb_i.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6,
                        vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_i[xvar], df_mf_lrb_i[yvar], df_mf_lrb_i[cost_var], c=df_mf_lrb_i[cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.08])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$GR$', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title(r'a) Moat wall impact, $T_M/T_{fb}=3.0, \zeta_M=0.2$', fontsize=axis_font)
ax.grid()



res = 75
X_plot = make_2D_plotting_space(df_mf_lrb_o[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.2)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6,
                        vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.08])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$GR$', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title(r'b) No moat wall impact, $T_M/T_{fb}=3.0, \zeta_M=0.2$', fontsize=axis_font)
ax.grid()



xvar = 'T_ratio'
yvar = 'zeta_e'
res = 75
X_plot = make_2D_plotting_space(df_mf_lrb_i[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_mf_lrb_i.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 3, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6,
                        vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_i[xvar], df_mf_lrb_i[yvar], df_mf_lrb_i[cost_var], c=df_mf_lrb_i[cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.08])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title(r'c) Moat wall impact, $GR=1.0, R_y=2.0$', fontsize=axis_font)
ax.grid()


xvar = 'T_ratio'
yvar = 'zeta_e'
res = 75
X_plot = make_2D_plotting_space(df_mf_lrb_o[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 4, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6,
                        vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.08])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title(r'd) No moat wall impact, $GR=1.0, R_y=2.0$', fontsize=axis_font)
ax.grid()

fig.tight_layout()

plt.savefig('./dissertation_figures/mf_lrb_cost_separated.pdf')




#%% 3d surf for replacement risk

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 32
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(7, 6))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(df_mf_lrb[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = predict_DV(X_plot,
                       mdl_impact_cbf_lrb.gpc,
                       mdl_cost_cbf_lrb_i.gpr,
                       mdl_cost_cbf_lrb_o.gpr,
                       outcome=repl_var)



x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

df_sc = df_mf_lrb.copy()

ax.scatter(df_sc[xvar], df_sc[yvar], df_sc['replacement_freq'], c=df_sc['replacement_freq'],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$GR$', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair loss', fontsize=axis_font)
ax.grid()
plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
# plt.box(False) #remove box
# ax.set_title(r'GP regression', fontsize=subt_font)
fig.tight_layout()
plt.show()

#%% system selector

# TODO: system selection

# consider: replacement freq, num_stories, num_bays, repair cost
covariate_list_sys = ['cmp_cost_ratio', 'cmp_time_ratio', 'replacement_freq', 'steel_cost_per_sf']
clf_struct = GP(df)
clf_struct.set_covariates(covariate_list_sys)
clf_struct.set_outcome('superstructure_system', use_ravel=False)
clf_struct.test_train_split(0.2)
clf_struct.fit_ensemble()
# clf_struct.fit_svc(neg_wt=False)
clf_struct.fit_gpc(kernel_name='rbf_iso')
# clf_struct.fit_kernel_logistic(kernel_name='rbf', neg_wt=False)
# clf_struct.fit_dt()

clf_isol = GP(df)
clf_isol.set_covariates(covariate_list_sys)
clf_isol.set_outcome('isolator_system', use_ravel=False)
clf_isol.test_train_split(0.2)
clf_isol.fit_ensemble()
# clf_isol.fit_svc(neg_wt=False)
clf_isol.fit_gpc(kernel_name='rbf_iso')
# clf_isol.fit_kernel_logistic(kernel_name='rbf', neg_wt=False)

#%%
plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 30
subt_font = 24
label_size = 24
title_font = 24
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


#################################
xvar = 'cmp_cost_ratio'
yvar = 'replacement_freq'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list_sys,
                                third_var_set = 0.1, fourth_var_set = 6.0)


fig = plt.figure(figsize=(16, 13))

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(2, 2, 1)



xx = X_plot[xvar]
yy = X_plot[yvar]
Z = clf_struct.gpc.predict(X_plot)

lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
x_pl = np.unique(xx)
y_pl = np.unique(yy)

Z_numbered = clf_struct.gpc.predict_proba(X_plot)[:,1]
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_classif = Z_numbered.reshape(xx_pl.shape)

# plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm_r,
    )

ax.scatter(df_cbf[xvar], df_cbf[yvar], color=color[0],
           edgecolors='k', alpha = 0.6, label='CBF', marker='^')
ax.scatter(df_mf[xvar], df_mf[yvar], color=color[1],
           edgecolors='k', alpha = 0.6, label='MF', marker='^')
plt.legend(fontsize=axis_font)

ax.set_title(r'a) Superstructure: repair cost-replacement', fontsize=title_font)
ax.set_ylabel(r'Replacement \%', fontsize=axis_font)
ax.set_xlabel(r'Repair cost ratio', fontsize=axis_font)

#################################
xvar = 'cmp_cost_ratio'
yvar = 'replacement_freq'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list_sys,
                                third_var_set = 0.1, fourth_var_set = 6.0)


color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(2, 2, 2)



xx = X_plot[xvar]
yy = X_plot[yvar]
Z = clf_isol.gpc.predict(X_plot)

lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
x_pl = np.unique(xx)
y_pl = np.unique(yy)

Z_numbered = clf_isol.gpc.predict_proba(X_plot)[:,1]
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_classif = Z_numbered.reshape(xx_pl.shape)

# plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm_r,
    )

ax.scatter(df_lrb[xvar], df_lrb[yvar], color=color[0],
           edgecolors='k', alpha = 0.6, label='LRB')
ax.scatter(df_tfp[xvar], df_tfp[yvar], color=color[1],
           edgecolors='k', alpha = 0.6, label='TFP')
plt.legend(fontsize=axis_font)

ax.set_title(r'b) Isolators: repair cost-replacement', fontsize=title_font)
# ax.set_ylabel(r'Replacement probability', fontsize=axis_font)
ax.set_xlabel(r'Repair cost ratio', fontsize=axis_font)

yvar = 'cmp_time_ratio'
xvar = 'steel_cost_per_sf'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list_sys,
                                third_var_set = 0.2, fourth_var_set = 0.05)



color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(2, 2, 3)



xx = X_plot[xvar]
yy = X_plot[yvar]
Z = clf_struct.gpc.predict(X_plot)

lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
x_pl = np.unique(xx)
y_pl = np.unique(yy)

Z_numbered = clf_struct.gpc.predict_proba(X_plot)[:,1]
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_classif = Z_numbered.reshape(xx_pl.shape)

# plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm_r,
    )

ax.scatter(df_cbf[xvar], df_cbf[yvar], color=color[0],
           edgecolors='k', alpha = 0.6, label='CBF', marker='^')
ax.scatter(df_mf[xvar], df_mf[yvar], color=color[1],
           edgecolors='k', alpha = 0.6, label='MF', marker='^')
plt.legend(fontsize=axis_font)

ax.set_title(r'c) Superstructure: steel cost-downtime', fontsize=title_font)
ax.set_xlabel(r'Steel cost per ft$^2$', fontsize=axis_font)
ax.set_ylabel(r'Downtime ratio', fontsize=axis_font)

#################################
yvar = 'cmp_time_ratio'
xvar = 'steel_cost_per_sf'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list_sys,
                                third_var_set = 0.2, fourth_var_set = 0.05)


color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(2, 2, 4)



xx = X_plot[xvar]
yy = X_plot[yvar]
Z = clf_isol.gpc.predict(X_plot)

lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
x_pl = np.unique(xx)
y_pl = np.unique(yy)

Z_numbered = clf_isol.gpc.predict_proba(X_plot)[:,1]
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_classif = Z_numbered.reshape(xx_pl.shape)

# plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
plt.imshow(
        Z_classif,
        interpolation="nearest",
        extent=(xx.min(), xx.max(),
                yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.coolwarm_r,
    )

ax.scatter(df_lrb[xvar], df_lrb[yvar], color=color[0],
           edgecolors='k', alpha = 0.6, label='LRB')
ax.scatter(df_tfp[xvar], df_tfp[yvar], color=color[1],
           edgecolors='k', alpha = 0.6, label='TFP')
plt.legend(fontsize=axis_font)

ax.set_title(r'd) Isolators: steel cost-downtime', fontsize=title_font)
ax.set_xlabel(r'Steel cost per ft$^2$', fontsize=axis_font)
# ax.set_ylabel(r'Downtime ratio', fontsize=axis_font)
fig.tight_layout()

# plt.savefig('./dissertation_figures/system_selection.pdf')

#%% dummy design space

### regular
ns = 4
hs = 13.
nb = 6
Lb = 30.

config_dict_moderate = {
    'num_stories': ns,
    'h_story': hs,
    'num_bays': nb,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': Lb,
    'S_1': 1.017,
    'h_bldg': hs*ns,
    'L_bldg': Lb*nb
    }

my_targets = {
    cost_var: 0.2,
    time_var: 0.2,
    'replacement_freq': 0.1,
    'constructability': -6.0}


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16, 13))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

# lvls = np.array([0.2])
lvls = np.arange(0.00, .25, 0.05)


####### MFs
res = 100
X_plot = make_2D_plotting_space(df_mf[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

X_sc = make_2D_plotting_space(df_mf[covariate_list], 20, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

## mf-TFP: cost
ax = fig.add_subplot(2, 2, 1)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)

qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)
X_sc_qual_cost = X_sc[qual_cost[cost_var+'_pred'] < 0.2]
sc = ax.scatter(X_sc_qual_cost[xvar], X_sc_qual_cost[yvar], c='white', edgecolors='black', s=10)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: repair cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

## mf-TFP: replacement
ax = fig.add_subplot(2, 2, 2)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)


qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)
X_sc_qual_repl = X_sc[qual_cost[repl_var+'_pred'] < 0.1]
sc = ax.scatter(X_sc_qual_repl[xvar], X_sc_qual_repl[yvar], c='white', edgecolors='black', s=10)


Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: Replacement', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: constructability
ax = fig.add_subplot(2, 2, 3)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))
lvl_kde = np.arange(-9, -2, 1)
kde_scr = mdl_impact_mf_lrb.kde.score_samples(X_plot)

qual_cost = mdl_impact_mf_lrb.kde.score_samples(X_sc)
X_sc_qual_kde = X_sc[qual_cost > -6.1]
sc = ax.scatter(X_sc_qual_kde[xvar], X_sc_qual_kde[yvar], c='white', edgecolors='black', s=10)

Z = np.array(kde_scr)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', levels=lvl_kde)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: Constructability', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: cost
ax = fig.add_subplot(2, 2, 4)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))


all_upfront_costs  = calc_upfront_cost(
    X_plot, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

mf_upfront_cost = all_upfront_costs['total_mf']


X_sc_qual = X_sc[np.logical_and.reduce((
        X_sc.index.isin(X_sc_qual_cost.index), 
        X_sc.index.isin(X_sc_qual_repl.index),
        X_sc.index.isin(X_sc_qual_kde.index)))]
sc = ax.scatter(X_sc_qual[xvar], X_sc_qual[yvar], c='white', edgecolors='black', s=10)

qual_upfront_cost  = calc_upfront_cost(
    X_sc_qual, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

cheapest_idx = qual_upfront_cost['total_mf'].idxmin()

# least upfront cost of the viable designs
the_design = X_sc_qual.loc[cheapest_idx]

ax.scatter(the_design[xvar], the_design[yvar], marker='x', c='red', s=100)

Z = np.array(mf_upfront_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: upfront cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

fig.tight_layout()


#%%
# #################################
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 30
# subt_font = 30
# label_size = 24
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# yvar = 'replacement_freq'
# xvar = 'steel_cost_per_sf'

# res = 75
# X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
#                                 all_vars=covariate_list_sys,
#                                 third_var_set = 0.2, fourth_var_set = 0.05)


# fig = plt.figure(figsize=(16, 7))

# color = plt.cm.Set1(np.linspace(0, 1, 10))

# ax=fig.add_subplot(1, 2, 1)



# xx = X_plot[xvar]
# yy = X_plot[yvar]
# Z = clf_struct.gpc.predict(X_plot)

# lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
# x_pl = np.unique(xx)
# y_pl = np.unique(yy)

# Z_numbered = clf_struct.gpc.predict_proba(X_plot)[:,1]
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# Z_classif = Z_numbered.reshape(xx_pl.shape)

# # plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
# plt.imshow(
#         Z_classif,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.coolwarm_r,
#     )

# ax.scatter(df_cbf[xvar], df_cbf[yvar], color=color[0],
#            edgecolors='k', alpha = 1.0, label='CBF')
# ax.scatter(df_mf[xvar], df_mf[yvar], color=color[1],
#            edgecolors='k', alpha = 1.0, label='MF')
# plt.legend(fontsize=axis_font)

# ax.set_title(r'Superstructures', fontsize=title_font)
# ax.set_xlabel(r'Steel cost per ft$^2$', fontsize=axis_font)
# ax.set_ylabel(r'Replacement probability', fontsize=axis_font)

# #################################
# yvar = 'replacement_freq'
# xvar = 'steel_cost_per_sf'


# res = 75
# X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
#                                 all_vars=covariate_list_sys,
#                                 third_var_set = 0.2, fourth_var_set = 0.05)


# color = plt.cm.Set1(np.linspace(0, 1, 10))

# ax=fig.add_subplot(1, 2, 2)



# xx = X_plot[xvar]
# yy = X_plot[yvar]
# Z = clf_isol.gpc.predict(X_plot)

# lookup_table, Z_numbered = np.unique(Z, return_inverse=True)
# x_pl = np.unique(xx)
# y_pl = np.unique(yy)

# Z_numbered = clf_isol.gpc.predict_proba(X_plot)[:,1]
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# Z_classif = Z_numbered.reshape(xx_pl.shape)

# # plt.contourf(xx_pl, yy_pl, Z_classif, cmap=plt.cm.coolwarm_r)
# plt.imshow(
#         Z_classif,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.coolwarm_r,
#     )

# ax.scatter(df_lrb[xvar], df_lrb[yvar], color=color[0],
#            edgecolors='k', alpha = 1.0, label='LRB')
# ax.scatter(df_tfp[xvar], df_tfp[yvar], color=color[1],
#            edgecolors='k', alpha = 1.0, label='TFP')
# plt.legend(fontsize=axis_font)

# ax.set_title(r'Isolators', fontsize=title_font)
# ax.set_xlabel(r'Steel cost per ft$^2$', fontsize=axis_font)
# # ax.set_ylabel(r'Downtime ratio', fontsize=axis_font)
# fig.tight_layout()

#%% dummy design space

### regular
ns = 4
hs = 13.
nb = 6
Lb = 30.

config_dict_moderate = {
    'num_stories': ns,
    'h_story': hs,
    'num_bays': nb,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': Lb,
    'S_1': 1.017,
    'h_bldg': hs*ns,
    'L_bldg': Lb*nb
    }

my_targets = {
    cost_var: 0.2,
    time_var: 0.2,
    'replacement_freq': 0.1,
    'constructability': -6.0}


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16, 13))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

# lvls = np.array([0.2])
lvls = np.arange(0.00, .25, 0.05)


####### MFs
res = 100
X_plot = make_2D_plotting_space(df_mf[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

X_sc = make_2D_plotting_space(df_mf[covariate_list], 20, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

## mf-TFP: cost
ax = fig.add_subplot(2, 2, 1)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)

qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)
X_sc_qual_cost = X_sc[qual_cost[cost_var+'_pred'] < 0.2]
sc = ax.scatter(X_sc_qual_cost[xvar], X_sc_qual_cost[yvar], c='white', edgecolors='black', s=10)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'a) MF-LRB: repair cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

## mf-TFP: replacement
ax = fig.add_subplot(2, 2, 2)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)


qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)
X_sc_qual_repl = X_sc[qual_cost[repl_var+'_pred'] < 0.1]
sc = ax.scatter(X_sc_qual_repl[xvar], X_sc_qual_repl[yvar], c='white', edgecolors='black', s=10)


Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'b) MF-LRB: Replacement', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: constructability
ax = fig.add_subplot(2, 2, 3)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))
lvl_kde = np.arange(-9, -2, 1)
kde_scr = mdl_impact_mf_lrb.kde.score_samples(X_plot)

qual_cost = mdl_impact_mf_lrb.kde.score_samples(X_sc)
X_sc_qual_kde = X_sc[qual_cost > -6.1]
sc = ax.scatter(X_sc_qual_kde[xvar], X_sc_qual_kde[yvar], c='white', edgecolors='black', s=10)

Z = np.array(kde_scr)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', levels=lvl_kde)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'c) MF-LRB: Constructability', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: cost
ax = fig.add_subplot(2, 2, 4)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))


all_upfront_costs  = calc_upfront_cost(
    X_plot, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

mf_upfront_cost = all_upfront_costs['total_mf']


X_sc_qual = X_sc[np.logical_and.reduce((
        X_sc.index.isin(X_sc_qual_cost.index), 
        X_sc.index.isin(X_sc_qual_repl.index),
        X_sc.index.isin(X_sc_qual_kde.index)))]
sc = ax.scatter(X_sc_qual[xvar], X_sc_qual[yvar], c='white', edgecolors='black', s=10)

qual_upfront_cost  = calc_upfront_cost(
    X_sc_qual, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

cheapest_idx = qual_upfront_cost['total_mf'].idxmin()

# least upfront cost of the viable designs
the_design = X_sc_qual.loc[cheapest_idx]

ax.scatter(the_design[xvar], the_design[yvar], marker='x', c='red', s=100)

Z = np.array(mf_upfront_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'd) MF-LRB: upfront cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

fig.tight_layout()

# plt.savefig('./dissertation_figures/mf_lrb_design_dots.pdf')

#%% dummy design space

### regular
ns = 4
hs = 13.
nb = 6
Lb = 30.

config_dict_moderate = {
    'num_stories': ns,
    'h_story': hs,
    'num_bays': nb,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': Lb,
    'S_1': 1.017,
    'h_bldg': hs*ns,
    'L_bldg': Lb*nb
    }

my_targets = {
    cost_var: 0.2,
    time_var: 0.2,
    'replacement_freq': 0.1,
    'constructability': -6.0}


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16, 13))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

# lvls = np.array([0.2])
lvls = np.arange(0.00, .25, 0.05)


####### MFs
res = 100
X_plot = make_2D_plotting_space(df_mf[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

X_sc = make_2D_plotting_space(df_mf[covariate_list], 20, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.87, fourth_var_set = 0.23)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

## mf-TFP: cost
ax = fig.add_subplot(2, 2, 1)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)

qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)
X_sc_qual_cost = X_sc[qual_cost[cost_var+'_pred'] < 0.2]
sc = ax.scatter(X_sc_qual_cost[xvar], X_sc_qual_cost[yvar], c='white', edgecolors='black', s=10)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'a) MF-LRB: repair cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

## mf-TFP: replacement
ax = fig.add_subplot(2, 2, 2)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)


qual_cost = predict_DV(X_sc,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)
X_sc_qual_repl = X_sc[qual_cost[repl_var+'_pred'] < 0.1]
sc = ax.scatter(X_sc_qual_repl[xvar], X_sc_qual_repl[yvar], c='white', edgecolors='black', s=10)


Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'b) MF-LRB: Replacement', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: constructability
ax = fig.add_subplot(2, 2, 3)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))
lvl_kde = np.arange(-9, -2, 1)
kde_scr = mdl_impact_mf_lrb.kde.score_samples(X_plot)

qual_cost = mdl_impact_mf_lrb.kde.score_samples(X_sc)
X_sc_qual_kde = X_sc[qual_cost > -6.1]
sc = ax.scatter(X_sc_qual_kde[xvar], X_sc_qual_kde[yvar], c='white', edgecolors='black', s=10)

Z = np.array(kde_scr)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', levels=lvl_kde)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'c) MF-LRB: Constructability', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: cost
ax = fig.add_subplot(2, 2, 4)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))


all_upfront_costs  = calc_upfront_cost(
    X_plot, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

mf_upfront_cost = all_upfront_costs['total_mf']


X_sc_qual = X_sc[np.logical_and.reduce((
        X_sc.index.isin(X_sc_qual_cost.index), 
        X_sc.index.isin(X_sc_qual_repl.index),
        X_sc.index.isin(X_sc_qual_kde.index)))]
sc = ax.scatter(X_sc_qual[xvar], X_sc_qual[yvar], c='white', edgecolors='black', s=10)

qual_upfront_cost  = calc_upfront_cost(
    X_sc_qual, config_dict=config_dict_moderate, steel_cost_dict=reg_dict)

cheapest_idx = qual_upfront_cost['total_mf'].idxmin()

# least upfront cost of the viable designs
the_design = X_sc_qual.loc[cheapest_idx]

ax.scatter(the_design[xvar], the_design[yvar], marker='x', c='red', s=100)

Z = np.array(mf_upfront_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5)

clabels = ax.clabel(cs, fontsize=clabel_size)
ax.set_xlim([0.5, 2.0])
ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'd) MF-LRB: upfront cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

fig.tight_layout()

# plt.savefig('./dissertation_figures/mf_lrb_design_dots.pdf')