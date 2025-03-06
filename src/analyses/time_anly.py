############################################################################
#               Time analysis

############################################################################

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

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

main_obj = pd.read_pickle("../../data/loss/structural_db_complete_distributions.pickle")
old_obj = pd.read_pickle("../../data/loss/structural_db_complete_normloss.pickle")

old_loss = old_obj.loss_data
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

df['k2'] = (df['k_e']*df['D_m'] - df['Q'])/df['D_m']
df['k1'] = df['k_ratio'] * df['k2']

df_loss = main_obj.loss_data

max_obj = pd.read_pickle("../../data/loss/structural_db_complete_max_loss.pickle")
df_loss_max = max_obj.max_loss

#%% readjust Tm of TFPs for W live

df['bldg_area'] = (df['num_bays']*df['L_bay'])**2 * (df['num_stories'] + 1)

def wL_calc(num_stories, bldg_area):
    
    # in kip/ft^2
    LL_avg = (num_stories*50.0/1000 + 20.0/1000)/(num_stories + 1)
    W_L = LL_avg*bldg_area
    return W_L

def TFP_period_shift(W_D, W_L, bearing):
    # shift assumes that the load case is 1.0D + 0.5L
    if bearing == 'TFP':
        return (W_D/(W_D+0.5*W_L))**0.5
    else:
        return 1.0

df['W_L'] = df.apply(lambda x: wL_calc(x['num_stories'], x['bldg_area']), axis=1)
df['Tshift_coef'] = df.apply(
    lambda x: TFP_period_shift(x['W'], x['W_L'], x['isolator_system']), axis=1)

df['T_M_adj'] = pd.to_numeric(df['T_m'] * df['Tshift_coef'])

df['T_ratio'] = pd.to_numeric(df['T_M_adj']/df['T_fb'])

from gms import get_ST


df['sa_tm_adj'] = df.apply(
    lambda x: get_ST(x, x['T_M_adj'],
                      db_dir='../../resource/ground_motions/gm_db.csv',
                      spec_dir='../../resource/ground_motions/gm_spectra.csv'), 
    axis=1)

df['GR_OG'] = pd.to_numeric((df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2))
    
df['gap_ratio'] = pd.to_numeric((df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm_adj']/df['Bm'])*df['T_M_adj']**2))
    
df['GR_shift_coef'] = pd.to_numeric(df['gap_ratio'])/pd.to_numeric(df['GR_OG'])
df['GR-Ad_coef'] = pd.to_numeric(df['GR_OG'])/pd.to_numeric(df['moat_ampli'])

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
    
#%% normalize DVs and prepare all variables

def loss_percentages(df_main, df_loss, df_max):
    df_main['bldg_area'] = df_main['L_bldg']**2 * (df_main['num_stories'] + 1)

    df_main['replacement_cost'] = 600.0*(df_main['bldg_area'])
    df_main['total_cmp_cost'] = df_max['cost_50%']
    df_main['total_cmp_cost_ub'] = df_max['cost_90%']
    df_main['cmp_replace_cost_ratio'] = df_main['total_cmp_cost']/df_main['replacement_cost']
    df_main['median_cost_ratio'] = df_loss['cost_50%']/df_main['replacement_cost']
    df_main['cmp_cost_ratio'] = df_loss['cost_50%']/df_main['total_cmp_cost']

    # but working in parallel (2x faster)
    df_main['replacement_time'] = df_main['bldg_area']/1000*365
    df_main['total_cmp_time'] = df_max['time_l_50%']
    df_main['total_cmp_time_ub'] = df_max['time_l_90%']
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
    
    copied_vars = ['cost_theta', 'cost_beta', 'time_l_theta', 'time_l_beta',
                   'cost_lam', 'cost_k', 'time_l_k', 'time_l_lam',
                   'cost_weibull_ks_pvalue', 'cost_lognormal_ks_pvalue',
                   'time_l_weibull_ks_pvalue', 'time_l_lognormal_ks_pvalue',
                   'cost_weibull_aic', 'cost_lognormal_aic',
                   'time_l_weibull_aic', 'time_l_lognormal_aic']
    
    df_main[copied_vars] = df_loss[copied_vars]
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


df['theta_ratio_cost'] = df['cost_theta']/df['total_cmp_cost']
df['theta_ratio_time'] = df['time_l_theta']/df['total_cmp_time']

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


#%% load scenario

scn_meta = pd.read_csv('../../resource/hazard_curves/sa_1.csv', index_col=None, header=0)  

#%% define earthquake hazard
# reference: P-58 Implementation, Ch. 3.5

import json
import bisect
from scipy.interpolate import interp2d
with open('../../resource/hazard_curves/dwight_7th.json') as f:
    site_hazard_curves = json.load(f)['response']

np.seterr(divide='ignore')


def get_hazard_bins(T, hazard_curves, sa_max=1.016):
    T_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    idx_between = bisect.bisect(T_list, T)
    
    # 0 is total
    below_lambda = hazard_curves[idx_between-1]['data'][0]['yvalues']
    below_sa = hazard_curves[idx_between-1]['metadata']['xvalues']
    
    above_lambda = hazard_curves[idx_between]['data'][0]['yvalues']
    above_sa = hazard_curves[idx_between]['metadata']['xvalues']
    
    x2 = T_list[idx_between]
    x1 = T_list[idx_between-1]
    
    # assume that both series have the same length
    sa_T = [(g + h) / 2 for g, h in zip(below_sa, above_sa)]
    lambda_T = [y1+(T-x1)*(y2-y1)/(x2-x1) for y1, y2 in zip(below_lambda, above_lambda)]
    
    # sa max is max sa_avg from the dataset
    if T > 1.0:
        sa_min = 0.05/T
    else:
        sa_min = 0.05
        
    # use 8 bins
    sa_ends = np.linspace(sa_min, sa_max, 9)
    sa_bins = (sa_ends[1:] + sa_ends[:-1]) / 2
    
    # interpolate in logspace
    log_lambda = np.log(lambda_T)
    log_lambda[log_lambda == -np.inf] = -100
    lambda_bins = np.exp(np.interp(np.log(sa_bins), np.log(sa_T), log_lambda))
    
    # from here, methodology is to use sa_bins to scale ground motions and analyze
    return(sa_bins, lambda_bins, sa_T, lambda_T)

sa_bins, lambda_bins, sa_T, lambda_T = get_hazard_bins(3.36, site_hazard_curves,
                                                       sa_max=0.376)


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 22
label_size = 20
clabel_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(8,7))
ax=fig.add_subplot(1, 1, 1)

ax.loglog(sa_bins,lambda_bins, '--^', label='Bins hazard', linewidth=2.0)
ax.loglog(sa_T,lambda_T, '-o', label='Site hazard', linewidth=0.8)
ax.legend(fontsize=axis_font)
ax.set_xlabel(r'$Sa(T_M)$', fontsize=axis_font)
ax.set_ylabel(r'$\lambda$', fontsize=axis_font)
ax.grid()

#%%
def scatter_hist(x, y, c, alpha, ax, ax_histx, ax_histy, label=None):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    cmap = plt.cm.Blues
    ax.scatter(x, y, alpha=alpha, edgecolors='black', s=25, facecolors=c,
               label=label)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    
    if y.name == 'zeta_e':
        binwidth = 0.02
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bin_y = np.arange(-lim, lim + binwidth, binwidth)
    elif y.name == 'RI':
        binwidth = 0.15
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bin_y = np.arange(-lim, lim + binwidth, binwidth)
    else:
        bin_y = bins
    ax_histx.hist(x, bins=bins, alpha = 0.5, weights=np.ones(len(x)) / len(x),
                  facecolor = c, edgecolor='navy', linewidth=0.5)
    ax_histy.hist(y, bins=bin_y, orientation='horizontal', alpha = 0.5, weights=np.ones(len(x)) / len(x),
                  facecolor = c, edgecolor='navy', linewidth=0.5)


plt.close('all')
fig = plt.figure(figsize=(13, 6), layout='constrained')

# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 4,  width_ratios=(5, 1, 5, 1), height_ratios=(1, 5),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0., hspace=0.)
# # Create the Axes.
# fig = plt.figure(figsize=(13, 10))
# ax1=fig.add_subplot(2, 2, 1)
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df[cost_var], df_loss['cost_beta'], 'orange', 0.3, ax, ax_histx, ax_histy)
ax.set_xlabel(r'Cost ratio', fontsize=axis_font)
ax.set_ylabel(r'$\beta$', fontsize=axis_font)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.1, 1.6])
ax.grid()

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df[time_var], df_loss['time_l_beta'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'Time ratio', fontsize=axis_font)
ax.set_ylabel(r'$\beta$', fontsize=axis_font)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.1, 1.6])
ax.grid()



#%% rebuild loss curves from distros

from scipy.stats import norm

# make lambda function for generic lognormal distribution
import numpy as np
lognorm_f = lambda x,theta,beta: norm(np.log(theta), beta**0.5).cdf(np.log(x))

# make lambda function for generic weibull distribution
from scipy.stats import weibull_min
weibull_f = lambda x,k,lam,loc: weibull_min(k, loc=loc, scale=lam).cdf(x)

#%%  rebuilt curves

def plot_loss(row):
    # plot lognormal fits
    
    import matplotlib.pyplot as plt
    plt.close('all')
    fig = plt.figure(figsize=(8, 7))
    ax1=fig.add_subplot(1, 1, 1)
    
    # res = ecdf(row['repair_cost'])
    ecdf_prob = row['cost_quantiles'].index
    ecdf_values = row['cost_quantiles'].values
    median_val = row['cost_50%']
    ax1.plot([ecdf_values], [ecdf_prob], 
              marker='x', markersize=5, color="red")
    
    xx_pr = np.linspace(1e-4, 10*median_val, 400)
    p = lognorm_f(xx_pr, row['cost_theta'], row['cost_beta'])
    ax1.plot(xx_pr, p, label='lognormal fit')
    p = weibull_f(xx_pr, row['cost_k'], row['cost_lam'], 0)
    ax1.plot(xx_pr, p, label='weibull fit')  
    p = weibull_f(xx_pr, row['cost_k_trunc'], row['cost_lam_trunc'], row['cost_min'])
    ax1.plot(xx_pr, p, label='weibull truncated fit') 
    ax1.set_xlim([-median_val, 10*median_val])
    ax1.set_ylabel('CDF(cost)', fontsize=axis_font)
    ax1.set_xlabel('repair cost (USD)', fontsize=axis_font)
    ax1.legend(fontsize=axis_font)
    
    
    fig = plt.figure(figsize=(8, 7))
    ax1=fig.add_subplot(1, 1, 1)
    # res = ecdf(row['repair_cost'])
    ecdf_prob = row['time_l_quantiles'].index
    ecdf_values = row['time_l_quantiles'].values
    median_val = row['time_l_50%']
    ax1.plot([ecdf_values], [ecdf_prob], 
              marker='x', markersize=5, color="red")
    
    xx_pr = np.linspace(1e-4, 10*median_val, 400)
    p = lognorm_f(xx_pr, row['time_l_theta'], row['time_l_beta'])
    ax1.plot(xx_pr, p, label='lognormal fit')
    p = weibull_f(xx_pr, row['time_l_k'], row['time_l_lam'], 0)
    ax1.plot(xx_pr, p, label='weibull fit')  
    p = weibull_f(xx_pr, row['time_l_k_trunc'], row['time_l_lam_trunc'], row['time_l_min'])
    ax1.plot(xx_pr, p, label='weibull truncated fit') 
    ax1.set_xlim([-median_val, 10*median_val])
    ax1.set_ylabel('CDF(time)', fontsize=axis_font)
    ax1.set_xlabel('repair time (worker-day)', fontsize=axis_font)
    ax1.legend(fontsize=axis_font)

    return

current_idx = 299
current_row = df_loss.iloc[current_idx]
plot_loss(current_row)

#%%

# df_test = df_loss[df_loss['replacement_freq'] != 1]

df_test = df_loss[(df_loss['cost_lognormal_ks_pvalue'] < df_loss['cost_weibull_trunc_ks_pvalue']) &
                  (df_loss['replacement_freq'] == 0)]

# df_test = df_loss[(df_loss['cost_lognormal_aic'] > df_loss['cost_weibull_trunc_aic']) &
#                   (df_loss['replacement_freq'] != 1)]
#%% regression models: beta
# goal: E[beta|theta]

# remove outlier may help fit quality
df_test = df_mf_lrb.copy()
# df_test = df_cbf_tfp[np.abs(stats.zscore(df_cbf_tfp['time_l_beta'])) < 5].copy()

### cost
beta_covariates = [cost_var]
mdl_beta_cost_mf_tfp = GP(df_mf_tfp)
mdl_beta_cost_mf_tfp.set_covariates(beta_covariates)
mdl_beta_cost_mf_tfp.set_outcome('cost_beta')
mdl_beta_cost_mf_tfp.test_train_split(0.2)

mdl_beta_cost_mf_lrb = GP(df_mf_lrb)
mdl_beta_cost_mf_lrb.set_covariates(beta_covariates)
mdl_beta_cost_mf_lrb.set_outcome('cost_beta')
mdl_beta_cost_mf_lrb.test_train_split(0.2)

mdl_beta_cost_cbf_tfp = GP(df_cbf_tfp)
mdl_beta_cost_cbf_tfp.set_covariates(beta_covariates)
mdl_beta_cost_cbf_tfp.set_outcome('cost_beta')
mdl_beta_cost_cbf_tfp.test_train_split(0.2)

mdl_beta_cost_cbf_lrb = GP(df_cbf_lrb)
mdl_beta_cost_cbf_lrb.set_covariates(beta_covariates)
mdl_beta_cost_cbf_lrb.set_outcome('cost_beta')
mdl_beta_cost_cbf_lrb.test_train_split(0.2)

### time
beta_covariates = [time_var]
mdl_beta_time_mf_tfp = GP(df_mf_tfp)
mdl_beta_time_mf_tfp.set_covariates(beta_covariates)
mdl_beta_time_mf_tfp.set_outcome('time_l_beta')
mdl_beta_time_mf_tfp.test_train_split(0.2)

mdl_beta_time_mf_lrb = GP(df_mf_lrb)
mdl_beta_time_mf_lrb.set_covariates(beta_covariates)
mdl_beta_time_mf_lrb.set_outcome('time_l_beta')
mdl_beta_time_mf_lrb.test_train_split(0.2)

mdl_beta_time_cbf_tfp = GP(df_cbf_tfp)
mdl_beta_time_cbf_tfp.set_covariates(beta_covariates)
mdl_beta_time_cbf_tfp.set_outcome('time_l_beta')
mdl_beta_time_cbf_tfp.test_train_split(0.2)

mdl_beta_time_cbf_lrb = GP(df_cbf_lrb)
mdl_beta_time_cbf_lrb.set_covariates(beta_covariates)
mdl_beta_time_cbf_lrb.set_outcome('time_l_beta')
mdl_beta_time_cbf_lrb.test_train_split(0.2)

print('======= beta regression per system  ========')
t0 = time.time()

# note, fitting kernel ridge is just kernel regression here bc only one feature
mdl_beta_cost_mf_tfp.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_cost_mf_lrb.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_cost_cbf_tfp.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_cost_cbf_lrb.fit_kernel_ridge(kernel_name='laplacian')

mdl_beta_time_mf_tfp.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_time_mf_lrb.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_time_cbf_tfp.fit_kernel_ridge(kernel_name='laplacian')
mdl_beta_time_cbf_lrb.fit_kernel_ridge(kernel_name='laplacian')

tp = time.time() - t0

print("KR training for beta done for 8 models in %.3f s" % tp)

beta_regression_mdls = {'mdl_beta_cost_mf_tfp': mdl_beta_cost_mf_tfp,
                        'mdl_beta_cost_mf_lrb': mdl_beta_cost_mf_lrb,
                        'mdl_beta_cost_cbf_tfp': mdl_beta_cost_cbf_tfp,
                        'mdl_beta_cost_cbf_lrb': mdl_beta_cost_cbf_lrb,
                        'mdl_beta_time_mf_tfp': mdl_beta_time_mf_tfp,
                        'mdl_beta_time_mf_lrb': mdl_beta_time_mf_lrb,
                        'mdl_beta_time_cbf_tfp': mdl_beta_time_cbf_tfp,
                        'mdl_beta_time_cbf_lrb': mdl_beta_time_cbf_lrb}

mdl_beta_cost_cbf_lrb.fit_poly(degree=5)

#%%

# low loss & extremely high loss = less dispersion
# extremely low loss: variance high bc of scaling/relative
# moderate loss: greater dispersion
import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(figsize=(8, 7))
ax1=fig.add_subplot(1, 1, 1)

ax1.plot([np.array(mdl_beta_cost_cbf_lrb.X_train).ravel()], [mdl_beta_cost_cbf_lrb.y_train], 
          marker='x', markersize=5, color="red")

nplot = 400
xx_pr = np.linspace(1e-4, 1.0, nplot).reshape(-1,1)
yy_pr = mdl_beta_cost_cbf_lrb.kr.predict(xx_pr)
ax1.plot(xx_pr, yy_pr, label='kernel ridge - laplacian')

# TODO: in truth, you should not use filter/smoothing, but rather achieve smoothing
# through kernel hyperparameter bounds

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # cumsum = np.cumsum(np.insert(y, 0, 0)) 
    # return (cumsum[box_pts:] - cumsum[:-box_pts]) / float(box_pts)
    return y_smooth

from scipy.signal import savgol_filter

y_sg = savgol_filter(yy_pr.ravel(), int(.05*nplot), 2)
ax1.plot(xx_pr, y_sg, color='black', label='smoothed with savgol')

y_sm = smooth(yy_pr.ravel(), int(.05*nplot))
ax1.plot(xx_pr, y_sm, color='green', label='moving average smooth')

# yy_pr = mdl_beta_cost_cbf_lrb.o_ridge.predict(xx_pr)
# ax1.plot(xx_pr, yy_pr, label='ordinary ridge')

# yy_pr = mdl_beta_cost_cbf_lrb.gpr.predict(xx_pr)
# ax1.plot(xx_pr, yy_pr, label='gpr-rq')

yy_pr = mdl_beta_cost_cbf_lrb.poly.predict(xx_pr)
ax1.plot(xx_pr, yy_pr, label='5th degree poly')
ax1.set_ylim([-.1, 2])
ax1.set_title(r'CBF-LRB lognormal betas', fontsize=axis_font)
ax1.set_xlabel(r'$\theta$', fontsize=axis_font)
ax1.set_ylabel(r'$\beta$', fontsize=axis_font)
ax1.legend(fontsize=axis_font)


#%% calculate lifetime loss, annualized


from scipy.signal import savgol_filter


def calculate_lifetime_loss(row, impact_clfs, cost_regs, time_regs, beta_regs,
                            cost_var='cmp_cost_ratio', time_var='cmp_time_ratio'):
    
    # here, we have to hack together a representation of the expected loss at different sa
    T = row['T_m']
    
    # use maximum as 1.5* mce level Sa(T_m)
    mce_Sa_Tm = row['S_1']/row['T_m']
    sa_bins, lambda_bins, sa_T, lambda_T = get_hazard_bins(T, site_hazard_curves,
                                                            sa_max=1.5*mce_Sa_Tm)
    
    
    # sa_bins, lambda_bins, sa_T, lambda_T = get_hazard_bins(T, site_hazard_curves)
    
    
    # how are each design variables affected by changing Sa
    # only GR changes
    GR_bins = row['gap_ratio']*row['sa_tm']/sa_bins
    
    # set of new design variables corresponding to the bins' hazards
    X_bins = pd.DataFrame({'gap_ratio':GR_bins,
                         'RI':np.repeat(row['RI'], len(GR_bins)),
                         'T_ratio':np.repeat(row['T_ratio'], len(GR_bins)),
                         'zeta_e':np.repeat(row['zeta_e'], len(GR_bins))
                         })
    
    
    ############################################################################
    # approach 1
    # use GP to find the 
    ############################################################################
    # pick the correct GP models
    # get system name
    system_name = row.system.lower().replace('-','_')
    
    # identify cost models
    mdl_impact_name = 'mdl_impact_' + system_name
    mdl_cost_hit_name = 'mdl_cost_' + system_name + '_i'
    mdl_cost_miss_name = 'mdl_cost_' + system_name + '_o'
    
    mdl_impact = impact_clfs[mdl_impact_name]
    mdl_cost_hit = cost_regs[mdl_cost_hit_name]
    mdl_cost_miss = cost_regs[mdl_cost_miss_name]
    
    # identify time models
    mdl_time_hit_name = 'mdl_time_' + system_name + '_i'
    mdl_time_miss_name = 'mdl_time_' + system_name + '_o'
    
    mdl_time_hit = time_regs[mdl_time_hit_name]
    mdl_time_miss = time_regs[mdl_time_miss_name]
    
    # identify beta models
    mdl_cost_beta_name = 'mdl_beta_cost_' + system_name
    mdl_time_beta_name = 'mdl_beta_time_' + system_name
    
    mdl_cost_beta = beta_regs[mdl_cost_beta_name]
    mdl_time_beta = beta_regs[mdl_time_beta_name]
    
    # for the set of "new" design variables, use GP to calculate loss ratio
    # assumes GPC/GPR, predict the outcome for the design space
    # TODO: change this to simple GP?
    cost_ratio_bins = predict_DV(X_bins, 
                                   mdl_impact.gpc, 
                                   mdl_cost_hit.gpr, 
                                   mdl_cost_miss.gpr, 
                                   outcome=cost_var)
    
    time_ratio_bins = predict_DV(X_bins,
                                mdl_impact.gpc,
                                mdl_time_hit.gpr,
                                mdl_time_miss.gpr,
                                outcome=time_var)
    
    # predict dispersion given theta for each bin
    
    # using true kr to find beta from theta
    
    # cost_beta_bins = mdl_cost_beta.kr.predict(
    #     np.array(cost_ratio_bins[cost_var+'_pred']).reshape(-1,1))
    
    # time_beta_bins = mdl_time_beta.kr.predict(
    #     np.array(time_ratio_bins[time_var+'_pred']).reshape(-1,1))
    
    # using kr + savgol filter smoothing to find beta from theta
    nplot = 400
    xx_pr = np.linspace(1e-4, 1.2, nplot).reshape(-1,1)
    yy_pr = mdl_cost_beta.kr.predict(xx_pr)
    y_sg = savgol_filter(yy_pr.ravel(), int(.05*nplot), 2)
    y_sm = smooth(yy_pr.ravel(), int(.05*nplot))
    # cost_beta_bins = np.interp(cost_ratio_bins[cost_var+'_pred'].values, 
    #                            xx_pr.ravel(), y_sg)
    cost_beta_bins = np.interp(cost_ratio_bins[cost_var+'_pred'].values, 
                                xx_pr.ravel(), y_sm)
    
    yy_pr = mdl_time_beta.kr.predict(xx_pr)
    y_sg = savgol_filter(yy_pr.ravel(), int(.05*nplot), 2)
    y_sm = smooth(yy_pr.ravel(), int(.05*nplot))
    # time_beta_bins = np.interp(time_ratio_bins[time_var+'_pred'].values, 
    #                            xx_pr.ravel(), y_sg)
    time_beta_bins = np.interp(time_ratio_bins[time_var+'_pred'].values, 
                                xx_pr.ravel(), y_sm)
    
    # unnormalize loss ratio back to loss
    cost_bins = cost_ratio_bins.values*row.total_cmp_cost
    time_bins = time_ratio_bins.values*row.total_cmp_time
    
    cost_bins[cost_bins < 0.0] = 1.0
    time_bins[time_bins < 0.0] = 0.04167
    
    # set any <0 cost to a dollar
    # set any <0 time to 30 mins
    
    # make exceedance curve for each scenario
    # use total replacement just to have a bigger number
    cost_loss_values = np.linspace(1e-4, row['replacement_cost'], 1000)
    time_loss_values = np.linspace(1e-4, row['replacement_time'], 1000)
    
    
    # cost_loss_values = np.linspace(1e-4, row['total_cmp_cost'], 1000)
    # time_loss_values = np.linspace(1e-4, row['total_cmp_time'], 1000)
    
    cost_scns = np.zeros([len(cost_loss_values), len(cost_bins)])
    time_scns = np.zeros([len(time_loss_values), len(time_bins)])
    
    
    for scn_idx in range(len(cost_bins)):
        
        cost_scns[:,scn_idx] = lognorm_f(cost_loss_values, cost_bins[scn_idx], cost_beta_bins[scn_idx])
        time_scns[:,scn_idx] = lognorm_f(time_loss_values, time_bins[scn_idx], time_beta_bins[scn_idx])
        
    cost_scns[cost_loss_values > row['total_cmp_cost_ub'], :] = 1.0
    time_scns[time_loss_values > row['total_cmp_time_ub'], :] = 1.0
    
    pr_exceedance_cost = 1 - cost_scns
    pr_exceedance_time = 1 - time_scns
    
    cost_loss_rates = np.multiply(pr_exceedance_cost, lambda_bins)
    time_loss_rates = np.multiply(pr_exceedance_time, lambda_bins)
    
    
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # fig = plt.figure(figsize=(8, 7))
    # ax1=fig.add_subplot(1, 1, 1)
    # ax1.plot(sa_bins, cost_bins.ravel(), '-o')
    # ax1.set_xlabel(r'$Sa(T_M)$', fontsize=axis_font)
    # ax1.set_ylabel(r'GP predicted median repair cost (\$)', fontsize=axis_font)
    # ax1.grid()
    # plt.show()
    # # ax1.set_xlim([0, row['replacement_cost']])
    
    # import matplotlib.pyplot as plt
    # # plt.close('all')
    # fig = plt.figure(figsize=(9, 6))
    # ax1=fig.add_subplot(1, 1, 1)
    
    # for scn_idx in range(len(cost_bins)):
    #     ax1.plot(cost_loss_values, pr_exceedance_cost[:,scn_idx], label='scn_'+str(scn_idx))
    # ax1.legend()
    # ax1.set_xlabel(r'Cost (\$)', fontsize=axis_font)
    # ax1.set_ylabel(r'$Pr[X \geq \$]$', fontsize=axis_font)
    # ax1.grid()
    # ax1.set_xlim([0, row['replacement_cost']])
    
    
    # import matplotlib.pyplot as plt
    # # plt.close('all')
    # fig = plt.figure(figsize=(9, 7))
    # ax1=fig.add_subplot(1, 1, 1)
    
    # for scn_idx in range(len(cost_bins)):
    #     ax1.plot(cost_loss_values, cost_loss_rates[:,:scn_idx+1].sum(axis=1), label='scn_'+str(scn_idx))
    # ax1.legend()
    # ax1.set_xlabel(r'Cost (\$)', fontsize=axis_font)
    # ax1.set_ylabel(r'$Pr[X \geq \$]$', fontsize=axis_font)
    # ax1.grid()
    # ax1.set_xlim([0, row['replacement_cost']])
    
    # fig = plt.figure(figsize=(9, 7))
    # ax1=fig.add_subplot(1, 1, 1)
    
    # for scn_idx in range(len(time_bins)):
    #     ax1.plot(time_loss_values, time_loss_rates[:,:scn_idx+1].sum(axis=1), label='scn_'+str(scn_idx))
    # ax1.legend()
    # ax1.set_xlabel(r'time (man-hour)', fontsize=axis_font)
    # ax1.set_ylabel(r'$Pr[X \geq t]$', fontsize=axis_font)
    # ax1.grid()
    # ax1.set_xlim([0, row['replacement_time']])
    
    # breakpoint()
    
    # multiply scenarios' exceedance curve with corresponding return rate
    # sum across all scenarios
    agg_cost_exceedance_rate = pr_exceedance_cost @ lambda_bins
    agg_time_exceedance_rate = pr_exceedance_time @ lambda_bins
    
    # integrate to attain lifetime dollar, time
    mean_cumulative_annual_cost = np.trapz(agg_cost_exceedance_rate, cost_loss_values)
    mean_cumulative_annual_time = np.trapz(agg_time_exceedance_rate, time_loss_values)
    
    # renormalize
    
    return mean_cumulative_annual_cost, mean_cumulative_annual_time

#%% 

row_no = 0
row = df.iloc[row_no]
mcac, mcat = calculate_lifetime_loss(row,
                                     impact_clfs=impact_classification_mdls, 
                                    cost_regs=cost_regression_mdls, 
                                    time_regs=time_regression_mdls,
                                    beta_regs=beta_regression_mdls)
    
#%% 

t0 = time.time()
df[['mean_cumulative_annual_cost', 
    'mean_cumulative_annual_time']] = df.apply(
        lambda row: calculate_lifetime_loss(row, impact_clfs=impact_classification_mdls, 
                                            cost_regs=cost_regression_mdls, 
                                            time_regs=time_regression_mdls,
                                            beta_regs=beta_regression_mdls),
        axis='columns', result_type='expand')

tp = time.time() - t0
print("Calculated lifetime losses for 1000 points in  %.3f s" % tp)

#%%
df['annual_cost_ratio'] = df['mean_cumulative_annual_cost']/df['total_cmp_cost']
df['annual_time_ratio'] = df['mean_cumulative_annual_time']/df['total_cmp_time']


#%% re subsets

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

#%% GP models for MCACs (done without impact classif)


mdl_mcac_cbf_lrb = GP(df_cbf_lrb)
mdl_mcac_cbf_lrb.set_covariates(covariate_list)
mdl_mcac_cbf_lrb.set_outcome('annual_cost_ratio')
mdl_mcac_cbf_lrb.test_train_split(0.2)

mdl_mcac_cbf_tfp = GP(df_cbf_tfp)
mdl_mcac_cbf_tfp.set_covariates(covariate_list)
mdl_mcac_cbf_tfp.set_outcome('annual_cost_ratio')
mdl_mcac_cbf_tfp.test_train_split(0.2)

mdl_mcac_mf_lrb = GP(df_mf_lrb)
mdl_mcac_mf_lrb.set_covariates(covariate_list)
mdl_mcac_mf_lrb.set_outcome('annual_cost_ratio')
mdl_mcac_mf_lrb.test_train_split(0.2)

mdl_mcac_mf_tfp = GP(df_mf_tfp)
mdl_mcac_mf_tfp.set_covariates(covariate_list)
mdl_mcac_mf_tfp.set_outcome('annual_cost_ratio')
mdl_mcac_mf_tfp.test_train_split(0.2)



mdl_mcat_cbf_lrb = GP(df_cbf_lrb)
mdl_mcat_cbf_lrb.set_covariates(covariate_list)
mdl_mcat_cbf_lrb.set_outcome('annual_time_ratio')
mdl_mcat_cbf_lrb.test_train_split(0.2)

mdl_mcat_cbf_tfp = GP(df_cbf_tfp)
mdl_mcat_cbf_tfp.set_covariates(covariate_list)
mdl_mcat_cbf_tfp.set_outcome('annual_time_ratio')
mdl_mcat_cbf_tfp.test_train_split(0.2)

mdl_mcat_mf_lrb = GP(df_mf_lrb)
mdl_mcat_mf_lrb.set_covariates(covariate_list)
mdl_mcat_mf_lrb.set_outcome('annual_time_ratio')
mdl_mcat_mf_lrb.test_train_split(0.2)

mdl_mcat_mf_tfp = GP(df_mf_tfp)
mdl_mcat_mf_tfp.set_covariates(covariate_list)
mdl_mcat_mf_tfp.set_outcome('annual_time_ratio')
mdl_mcat_mf_tfp.test_train_split(0.2)

print('======= MCAC/MCAT regression per system ========')
import time
t0 = time.time()

mdl_mcac_cbf_lrb.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcac_cbf_tfp.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcac_mf_lrb.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcac_mf_tfp.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))

mdl_mcat_cbf_lrb.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcat_cbf_tfp.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcat_mf_lrb.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))
mdl_mcat_mf_tfp.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-13, 1e2))

tp = time.time() - t0

print("GPR MCAC/MCAT training for cost done for 4 models in %.3f s" % tp)

mcac_regression_mdls = {'mdl_mcac_cbf_lrb': mdl_mcac_cbf_lrb,
                        'mdl_mcac_cbf_tfp': mdl_mcac_cbf_tfp,
                        'mdl_mcac_mf_lrb': mdl_mcac_mf_lrb,
                        'mdl_mcac_mf_tfp': mdl_mcac_mf_tfp}

mcat_regression_mdls = {'mdl_mcat_cbf_lrb': mdl_mcat_cbf_lrb,
                        'mdl_mcat_cbf_tfp': mdl_mcat_cbf_tfp,
                        'mdl_mcat_mf_lrb': mdl_mcat_mf_lrb,
                        'mdl_mcat_mf_tfp': mdl_mcat_mf_tfp}

#%% sample for CBF-LRB

annual_cost_var = 'annual_cost_ratio'
mdl_mcat = mcat_regression_mdls['mdl_mcat_cbf_lrb']

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
                            third_var_set = 5.0, fourth_var_set = 0.15)

Z = mdl_mcat.gpr.predict(X_plot)


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

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[annual_cost_var], c=df_cbf_lrb[annual_cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
# ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Annual repair cost (\%)', fontsize=axis_font)
ax.set_title('CBF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact_cbf_lrb.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = mdl_mcat.gpr.predict(X_plot)

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

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[annual_cost_var], c=df_cbf_lrb[annual_cost_var],
            edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')
# ax.set_zlim([-0.1, 1.1])

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Annual repair cost (\%)', fontsize=axis_font)
ax.set_title('CBF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()



#%% Calculate upfront cost of data
# calc cost of new point

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


#%% Testing the design space

def grid_search_inverse_design(res, system_name, targets_dict, config_dict, 
                               impact_clfs, mcac_regs, mcat_regs,
                               cost_var='annual_cost_ratio', time_var='annual_time_ratio'):
    import time
    
    # isolator_system = system_name.split('_')[1]
    # system_X = impact_clfs['mdl_impact_'+system_name].X
    structural_system = system_name.split('_')[0]
    
    if structural_system == 'mf':
        bound_dict = {
            'gap_ratio': (0.6, 2.0),
            'RI': (0.5, 2.25),
            'T_ratio': (2.0, 5.8),
            'zeta_e': (0.1, 0.25),
            'k_ratio': (5.0, 12.0)}
    else:
       bound_dict = {
           'gap_ratio': (0.6, 2.0),
           'RI': (0.5, 2.25),
           'T_ratio': (2.0, 11.0),
           'zeta_e': (0.1, 0.25),
           'k_ratio': (5.0, 12.0)} 
    
    X_space = make_design_space(res, bound_dict=bound_dict)
    
    # identify impact models (which has constructable kde)
    mdl_impact_name = 'mdl_impact_' + system_name
    mdl_impact = impact_clfs[mdl_impact_name]
    
    # identify cost models
    
    mdl_mcac_name = 'mdl_mcac_' + system_name
    mdl_mcac = mcac_regs[mdl_mcac_name]
    
    # identify time models
    mdl_mcat_name = 'mdl_mcat_' + system_name
    mdl_mcat = mcat_regs[mdl_mcat_name]
    
    # first, scan whole range for constructable bounds
    # constructable
    space_constr = mdl_impact.kde.score_samples(X_space)
    constr_thresh = targets_dict['constructability']
    ok_constr = X_space.loc[space_constr >= constr_thresh]
    constr_bounds = ok_constr.agg(['min', 'max'])
    variable_names = list(constr_bounds.columns)
    temp_dict = constr_bounds.to_dict()
    ranges = [tuple(temp_dict[key].values()) for key in variable_names]
    bounds = {k:v for (k,v) in zip(variable_names, ranges)}
    
    
    # then recreate a finer design space within constructable range
    X_space = make_design_space(res, bound_dict=bounds)
    
    t0 = time.time()
    space_mcac = mdl_mcac.gpr.predict(X_space)
    tp = time.time() - t0
    print("GPC-GPR MCAC prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))
    
    t0 = time.time()
    space_mcat = mdl_mcat.gpr.predict(X_space)
    tp = time.time() - t0
    print("GPC-GPR MCAT prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                                   tp))
    
    space_constr = mdl_impact.kde.score_samples(X_space)
    tp = time.time() - t0
    print("KDE constructability prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                                   tp))
    
    # filter cost threshold
    mcac_thresh = targets_dict[cost_var]
    ok_cost = X_space.loc[space_mcac<=mcac_thresh]

    # downtime threshold
    mcat_thresh = targets_dict[time_var]
    ok_time = X_space.loc[space_mcat<=mcat_thresh]

    
    # constructable
    constr_thresh = targets_dict['constructability']
    ok_constr = X_space.loc[space_constr >= constr_thresh]

    X_design = X_space[np.logical_and.reduce((
            X_space.index.isin(ok_cost.index), 
            X_space.index.isin(ok_time.index),
            X_space.index.isin(ok_constr.index)))]
    
    
    space_ok_mcac = space_mcac[X_space.index.isin(X_design.index)]
    space_ok_mcat = space_mcat[X_space.index.isin(X_design.index)]


    if X_design.shape[0] < 1:
        print('No suitable design found for system', system_name)
        return None, None, None
    
    # NPV analysis
    # compare cost of designs against baseline of that system
    
    
    if structural_system.upper() == 'MF':
        X_baseline =  pd.DataFrame(np.array([[1.0, 2.0, 2.6, 0.2]]),
                                   columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
        upfront_costs = calc_upfront_cost(
            X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
        baseline_cost = calc_upfront_cost(
            X_baseline, config_dict=config_dict, steel_cost_dict=reg_dict)
        
        mcac_baseline = mdl_mcac.gpr.predict(X_baseline)[0]
        mcat_baseline = mdl_mcat.gpr.predict(X_baseline)[0]
    else:
        
        X_baseline =  pd.DataFrame(np.array([[1.0, 2.0, 5.2, 0.2]]),
                                   columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
        upfront_costs = calc_upfront_cost(
            X_design, config_dict=config_dict, steel_cost_dict=reg_dict,
            land_cost_per_sqft=1978/(3.28**2))
        baseline_cost = calc_upfront_cost(
            X_baseline, config_dict=config_dict, steel_cost_dict=reg_dict,
            land_cost_per_sqft=1978/(3.28**2))
        
        mcac_baseline = mdl_mcac.gpr.predict(X_baseline)[0]
        mcat_baseline = mdl_mcat.gpr.predict(X_baseline)[0]
    
    upgrade_cost = (upfront_costs['total_'+structural_system] - 
                    baseline_cost['total_'+structural_system].item())
    
    avoided_cost = (mcac_baseline - space_ok_mcac)*config_dict['comparable_cost_'+structural_system]
    avoided_time = (mcat_baseline - space_ok_mcat)*config_dict['comparable_time_'+structural_system]
    
    # profit loss and repair cost per worker-day
    # assume 40% of replacement cost is labor, $680/worker-day for SF Bay Area
    bldg_area = (config_dict['num_bays']*config_dict['L_bay'])**2 * (config_dict['num_stories'] + 1)

    # assume $600/sf replacement
    n_worker_series = bldg_area/1000
    n_worker_parallel = n_worker_series/2
    
    # labor rate per worker-day
    avoided_worker_cost = 680.0*avoided_time
    
    # lost profits
    # worker-day (/workers) (*$ per day)
    # based on $25/sq-ft/yr of business rent in Oakland for 3 story
    avoided_business_cost = avoided_time/n_worker_parallel*4440*1.35 * config_dict['num_stories']/3
    
    avoided_time_cost = avoided_worker_cost + avoided_business_cost
    avoided_consequence = avoided_cost + avoided_time_cost
    
    i_rate = config_dict['interest_rate']
    t_yrs = config_dict['timeframe']
    
    # upgrade is worth it if NPV of avoided consequence > upgrade cost over baseline
    NPV = avoided_consequence*((1 - 1/(1 + i_rate)**t_yrs) / i_rate)
    upgrade_value = NPV - upgrade_cost
    upgrade_decision = upgrade_value > 0
    # upgrade_decision = np.repeat(True, X_design.shape[0])
    X_worth = X_design[upgrade_decision]
    worth_costs = upfront_costs['total_'+structural_system][upgrade_decision]
    
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # fig = plt.figure(figsize=(8, 7))
    # ax1=fig.add_subplot(1, 1, 1)
    # ax1.plot(sa_bins, cost_bins.ravel(), '-o')
    # ax1.set_xlabel(r'$Sa(T_M)$', fontsize=axis_font)
    # ax1.set_ylabel(r'GP predicted median repair cost (\$)', fontsize=axis_font)
    # ax1.grid()
    # plt.show()
    # # ax1.set_xlim([0, row['replacement_cost']])
    
    
    design_idx = upgrade_value.idxmax()
    inv_upfront_cost = worth_costs[upgrade_value.idxmax()]
    
    # cheapest_idx = worth_costs.idxmin()
    # inv_upfront_cost = worth_costs.min()
    
    # least upfront cost of the viable designs
    inv_design = X_worth.loc[design_idx]
    inv_mcat = space_mcat[design_idx]
    inv_mcac = space_mcac[design_idx]
    inv_upgrade_cost = upgrade_cost[design_idx]
    inv_avoided_cost = (mcac_baseline - inv_mcac)*config_dict['comparable_cost_'+structural_system]
    inv_avoided_time = (mcat_baseline - inv_mcat)*config_dict['comparable_time_'+structural_system]
    inv_avoided_consequence = inv_avoided_cost + inv_avoided_time * (
        680 + 4440*1.35/n_worker_parallel * config_dict['num_stories']/3)
    inv_NPV = inv_avoided_consequence*((1 - 1/(1 + i_rate)**t_yrs) / i_rate)
    
    inv_performance = {
        'mcat': inv_mcat,
        'mcac': inv_mcac,
        'upfront_cost': inv_upfront_cost,
        'upgrade_cost': inv_upgrade_cost,
        'avoided_cost': inv_avoided_cost,
        'avoided_time': inv_avoided_time,
        'NPV': inv_NPV}
    
    # TODO: if NPV is unifying loss function, use optimizer

    # read out predictions
    print('==================================')
    print('            Predictions           ')
    print('==================================')
    print('======= Targets =======')
    print('System:', system_name)
    print('MCAC fraction:', f'{mcac_thresh*100:,.2f}%')
    print('MCAT fraction:', f'{mcat_thresh*100:,.2f}%')


    print('======= Overall inverse design =======')
    print(inv_design)
    print('Upfront cost of selected design: ',
          f'${inv_upfront_cost:,.2f}')
    print('Upgrade cost over baseline design: '
          f'${inv_upgrade_cost:,.2f}')
    print('Predicted MCAC ratio: ',
          f'{inv_mcac*100:,.2f}%')
    print('Predicted comparable MCAC: '
          f'${inv_avoided_cost:,.2f}')
    print('Predicted MCAT ratio: ',
          f'{inv_mcat*100:,.2f}%')
    print('Predicted comparable MCAT: '
          f'{inv_avoided_time:,.2f} worker-days')
    print('Predicted NPV: '
          f'${inv_NPV:,.2f}')
    
    return(inv_design, inv_performance, X_worth)

### test
ns = 4
hs = 13.
nb = 6
Lb = 30.

example_bldg_area = (nb*Lb)**2 * (ns + 1)
workers_parallel = example_bldg_area/1000/2


similar_mfs = df_mf[(df_mf['num_stories'] == ns) & (df_mf['num_bays'] == nb)]
similar_mf_cost = similar_mfs['total_cmp_cost'].median()
similar_mf_time = similar_mfs['total_cmp_time'].median()

similar_cbfs = df_cbf[(df_cbf['num_stories'] == ns) & (df_cbf['num_bays'] == nb)]
similar_cbf_cost = similar_cbfs['total_cmp_cost'].median()
similar_cbf_time = similar_cbfs['total_cmp_time'].median()


annual_cost_target_mf = 50e3/similar_mf_cost
annual_cost_target_cbf = 50e3/similar_cbf_cost

# 5 days target annually, with only 10% of the workers
annual_dt_target_mf = 3*0.1*workers_parallel/similar_mf_time
annual_dt_target_cbf = 3*0.1*workers_parallel/similar_cbf_time

config_dict_test = {
    'num_stories': ns,
    'h_story': hs,
    'num_bays': nb,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': Lb,
    'S_1': 1.017,
    'h_bldg': hs*ns,
    'L_bldg': Lb*nb,
    'comparable_cost_mf': similar_mf_cost,
    'comparable_cost_cbf': similar_cbf_cost,
    'comparable_time_mf': similar_mf_time,
    'comparable_time_cbf': similar_cbf_time,
    'interest_rate': 0.07,
    'timeframe': 40.0
    }

mcac_var = 'annual_cost_ratio'
mcat_var = 'annual_time_ratio'
my_targets = {
    mcac_var: annual_cost_target_mf,
    mcat_var: annual_dt_target_mf,
    'constructability': -6.0}

test_inv_design, test_inv_performance, test_space = grid_search_inverse_design(
    20, 'mf_tfp', my_targets, config_dict_test, 
    impact_classification_mdls, mcac_regression_mdls, 
    mcat_regression_mdls)

#%% inverse design filters
### regular
ns = 4
hs = 13.
nb = 6
Lb = 30.

similar_mfs = df_mf[(df_mf['num_stories'] == ns) & (df_mf['num_bays'] == nb)]
similar_mf_cost = similar_mfs['total_cmp_cost'].median()
similar_mf_time = similar_mfs['total_cmp_time'].median()

similar_cbfs = df_cbf[(df_cbf['num_stories'] == ns) & (df_cbf['num_bays'] == nb)]
similar_cbf_cost = similar_cbfs['total_cmp_cost'].median()
similar_cbf_time = similar_cbfs['total_cmp_time'].median()

config_dict_annual = {
    'num_stories': ns,
    'h_story': hs,
    'num_bays': nb,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': Lb,
    'S_1': 1.017,
    'h_bldg': hs*ns,
    'L_bldg': Lb*nb,
    'comparable_cost_mf': similar_mf_cost,
    'comparable_cost_cbf': similar_cbf_cost,
    'comparable_time_mf': similar_mf_time,
    'comparable_time_cbf': similar_cbf_time,
    'interest_rate': 0.07,
    'timeframe': 40.0
    }

mcac_var = 'annual_cost_ratio'
mcat_var = 'annual_time_ratio'

mf_targets= {
    mcac_var: annual_cost_target_mf,
    mcat_var: annual_dt_target_mf,
    'constructability': -5.0}

cbf_targets= {
    mcac_var: annual_cost_target_cbf,
    mcat_var: annual_dt_target_cbf,
    'constructability': -6.0}


mf_tfp_inv_design, mf_tfp_inv_performance, mf_tfp_space = grid_search_inverse_design(
    20, 'mf_tfp', mf_targets, config_dict_annual, 
    impact_classification_mdls, mcac_regression_mdls, 
    mcat_regression_mdls)

mf_lrb_inv_design, mf_lrb_inv_performance, mf_lrb_space = grid_search_inverse_design(
    20, 'mf_lrb', mf_targets, config_dict_annual, 
    impact_classification_mdls, mcac_regression_mdls, 
    mcat_regression_mdls)

cbf_tfp_inv_design, cbf_tfp_inv_performance, cbf_tfp_space = grid_search_inverse_design(
    20, 'cbf_tfp', cbf_targets, config_dict_annual, 
    impact_classification_mdls, mcac_regression_mdls, 
    mcat_regression_mdls)

cbf_lrb_inv_design, cbf_lrb_inv_performance, cbf_lrb_space = grid_search_inverse_design(
    20, 'cbf_lrb', cbf_targets, config_dict_annual, 
    impact_classification_mdls, mcac_regression_mdls, 
    mcat_regression_mdls)

#%% design the systems
from loads import estimate_period

from db import prepare_ida_util
import json
### regular

print ('======== designing structures ==========')
my_design = mf_tfp_inv_design.copy()
my_design['superstructure_system'] = 'MF'
my_design['isolator_system'] = 'TFP'
my_design['k_ratio'] = 12

mf_tfp_dict = my_design.to_dict()

# estimate period to find bins
test_dict = dict(mf_tfp_dict)
test_dict.update(config_dict_annual)
tf_est = estimate_period(test_dict)
tm_est = test_dict['T_ratio']/0.9*tf_est
sa_collapse_est = 1.5*test_dict['S_1']/tm_est
sa_mce_est = test_dict['S_1']/tm_est
sa_mf_tfp_bins, lambda_mf_tfp_bins, sa_T, lambda_T = get_hazard_bins(
    tm_est, site_hazard_curves, sa_max=sa_collapse_est)
levels_mf_tfp = sa_mf_tfp_bins/sa_mce_est

config_mf_tfp = dict(config_dict_annual)
config_mf_tfp['ida_levels'] = list(levels_mf_tfp)

# for TFPs, we'll readjust T_ratio by x1/0.9, since the design -> analysis process will
# change T_m by x0.9
# since in the design script, GR is applied on the unadjusted T, we'll need to adjust
# the specified GR by x0.9
mf_tfp_dict['T_ratio'] = mf_tfp_dict['T_ratio']/0.9
mf_tfp_dict['gap_ratio'] = mf_tfp_dict['gap_ratio']*0.9

ida_mf_tfp_df = prepare_ida_util(mf_tfp_dict, levels=levels_mf_tfp,
                                 db_string='../../resource/',
                                 config_dict=config_dict_annual)

print('Length of MF-TFP IDA:', len(ida_mf_tfp_df))


# with open('../inputs/mf_tfp_annual_func_hazard.in', 'w') as file:
#     file.write(json.dumps(mf_tfp_dict))
#     file.close()
    
# with open('../inputs/mf_tfp_annual_func_hazard.cfg', 'w') as file:
#     file.write(json.dumps(config_mf_tfp))
#     file.close()

my_design = cbf_tfp_inv_design.copy()
my_design['superstructure_system'] = 'CBF'
my_design['isolator_system'] = 'TFP'
my_design['k_ratio'] = 7


cbf_tfp_dict = my_design.to_dict()

# estimate period to find bins
test_dict = dict(cbf_tfp_dict)
test_dict.update(config_dict_annual)
tf_est = estimate_period(test_dict)
tm_est = test_dict['T_ratio']/0.9*tf_est
sa_collapse_est = 1.5*test_dict['S_1']/tm_est
sa_mce_est = test_dict['S_1']/tm_est
sa_cbf_tfp_bins, lambda_cbf_tfp_bins, sa_T, lambda_T = get_hazard_bins(
    tm_est, site_hazard_curves, sa_max=sa_collapse_est)
levels_cbf_tfp = sa_cbf_tfp_bins/sa_mce_est

config_cbf_tfp = dict(config_dict_annual)
config_cbf_tfp['ida_levels'] = list(levels_cbf_tfp)

# for TFPs, we'll readjust T_ratio by x1/0.9, since the design -> analysis process will
# change T_m by x0.9
# since in the design script, GR is applied on the unadjusted T, we'll need to adjust
# the specified GR by x0.9
cbf_tfp_dict['T_ratio'] = cbf_tfp_dict['T_ratio']/0.9
cbf_tfp_dict['gap_ratio'] = cbf_tfp_dict['gap_ratio']*0.9
    
ida_cbf_tfp_df = prepare_ida_util(cbf_tfp_dict, levels=levels_cbf_tfp,
                                  db_string='../../resource/',
                                 config_dict=config_dict_annual)

print('Length of CBF-TFP IDA:', len(ida_cbf_tfp_df))

# with open('../inputs/cbf_tfp_annual_func_hazard.in', 'w') as file:
#     file.write(json.dumps(cbf_tfp_dict))
#     file.close()
    
# with open('../inputs/cbf_tfp_annual_func_hazard.cfg', 'w') as file:
#     file.write(json.dumps(config_cbf_tfp))
#     file.close()


my_design = mf_lrb_inv_design.copy()
my_design['superstructure_system'] = 'MF'
my_design['isolator_system'] = 'LRB'
my_design['k_ratio'] = 10

mf_lrb_dict = my_design.to_dict()

# estimate period to find bins
test_dict = dict(mf_lrb_dict)
test_dict.update(config_dict_annual)
tf_est = estimate_period(test_dict)
tm_est = test_dict['T_ratio']*tf_est
sa_collapse_est = 1.5*test_dict['S_1']/tm_est
sa_mce_est = test_dict['S_1']/tm_est
sa_mf_lrb_bins, lambda_mf_lrb_bins, sa_T, lambda_T = get_hazard_bins(
    tm_est, site_hazard_curves, sa_max=sa_collapse_est)
levels_mf_lrb = sa_mf_lrb_bins/sa_mce_est

config_mf_lrb = dict(config_dict_annual)
config_mf_lrb['ida_levels'] = list(levels_mf_lrb)

ida_mf_lrb_df = prepare_ida_util(mf_lrb_dict, levels=levels_mf_lrb,
                                 db_string='../../resource/',
                                 config_dict=config_dict_annual)

print('Length of MF-LRB IDA:', len(ida_mf_lrb_df))

# with open('../inputs/mf_lrb_annual_func_hazard.in', 'w') as file:
#     file.write(json.dumps(mf_lrb_dict))
#     file.close()
    
# with open('../inputs/mf_lrb_annual_func_hazard.cfg', 'w') as file:
#     file.write(json.dumps(config_mf_lrb))
#     file.close()


my_design = cbf_lrb_inv_design.copy()
my_design['superstructure_system'] = 'CBF'
my_design['isolator_system'] = 'LRB'
my_design['k_ratio'] = 10

cbf_lrb_dict = my_design.to_dict()

# estimate period to find bins
test_dict = dict(cbf_lrb_dict)
test_dict.update(config_dict_annual)
tf_est = estimate_period(test_dict)
tm_est = test_dict['T_ratio']*tf_est
sa_collapse_est = 1.5*test_dict['S_1']/tm_est
sa_mce_est = test_dict['S_1']/tm_est
sa_cbf_lrb_bins, lambda_cbf_lrb_bins, sa_T, lambda_T = get_hazard_bins(
    tm_est, site_hazard_curves, sa_max=sa_collapse_est)
levels_cbf_lrb = sa_cbf_lrb_bins/sa_mce_est

config_cbf_lrb = dict(config_dict_annual)
config_cbf_lrb['ida_levels'] = list(levels_cbf_lrb)

ida_cbf_lrb_df = prepare_ida_util(cbf_lrb_dict, levels=levels_cbf_lrb,
                                  db_string='../../resource/',
                                 config_dict=config_dict_annual)



# with open('../inputs/cbf_lrb_annual_func_hazard.in', 'w') as file:
#     file.write(json.dumps(cbf_lrb_dict))
#     file.close()
    
# with open('../inputs/cbf_lrb_annual_func_hazard.cfg', 'w') as file:
#     file.write(json.dumps(config_cbf_lrb))
#     file.close()
    
print('Length of CBF-LRB IDA:', len(ida_cbf_lrb_df))

#%%

def print_latex_design_table(sys_name, val_results):

    typ_design = val_results.iloc[0]
    
    
    # moat in cm
    moat = typ_design['D_m']*typ_design['moat_ampli']*2.54
    largest_beam = typ_design['beam'][0]
    largest_column = typ_design['column'][0]
    try:
        largest_brace = typ_design['brace'][0]
    except:
        largest_brace = 'n/a'
    
    # d bearings in cm, R curvature in mm
    try:
        bearing_param_1 = typ_design['mu_1']
    except:
        bearing_param_1 = typ_design['d_lead']*2.54
        
    try:
        bearing_param_2 = typ_design['mu_2']
    except:
        bearing_param_2 = typ_design['d_bearing'] *2.54
        
    try:
        bearing_param_3 = typ_design['R_1']*25.4
    except:
        bearing_param_3 = typ_design['t_r']* 2.54
        
    # print as either TFP or LRB
    try:
        bearing_param_4 = typ_design['R_2']*25.4
        latex_string = f"& {sys_name} & {moat:.1f} cm & {largest_beam} & {largest_column} & {largest_brace} \
            & {bearing_param_1:.3f} & {bearing_param_2:.3f} & {bearing_param_3:.0f} mm &  {bearing_param_4:.0f} mm \\\\"
    except:
        bearing_param_4 = typ_design['n_layers'] 
        latex_string = f"& {sys_name} & {moat:.1f} cm & {largest_beam} & {largest_column} & {largest_brace} \
            & {bearing_param_1:.1f} cm & {bearing_param_2:.1f} cm & {bearing_param_3:.1f} cm &  {bearing_param_4:.0f}  \\\\"
    
    # print('Average median repair cost: ',
    #       f'${val_cost[0]:,.2f}')
    # print('Repair cost ratio: ', 
    #       f'{val_cost_ratio[0]:,.3f}')
    # print('Repair time ratio: ',
    #       f'{val_downtime_ratio[0]:,.3f}')
    # print('Estimated replacement frequency: ',
    #       f'{val_replacement[0]:.2%}')
    
    
    print(latex_string)
    return

print_latex_design_table('MF-TFP', ida_mf_tfp_df)
print_latex_design_table('CBF-TFP', ida_cbf_tfp_df)
print_latex_design_table('MF-LRB', ida_mf_lrb_df)
print_latex_design_table('CBF-LRB', ida_cbf_lrb_df)

#%%

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
fig = plt.figure(figsize=(9, 7))

ax=fig.add_subplot(1, 1, 1, projection='3d')
sc = ax.scatter(mf_lrb_space['gap_ratio'], mf_lrb_space['RI'], mf_lrb_space['T_ratio'],
                alpha = 1, cmap=plt.cm.Spectral_r)
ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax.set_title(r'$\zeta_M$ not shown', fontsize=title_font)

#%%

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
yvar = 'T_ratio'

# lvls = np.array([0.2])
lvls = np.arange(0.00, .002, 0.0002)

X_baseline =  pd.DataFrame(np.array([[1.0, 2.0, 2.6, 0.2]]),
                           columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
baseline_cost = calc_upfront_cost(
    X_baseline, config_dict=config_dict_annual, steel_cost_dict=reg_dict)

mcac_baseline = mcac_regression_mdls['mdl_mcac_mf_lrb'].gpr.predict(X_baseline)[0]
mcat_baseline = mcat_regression_mdls['mdl_mcat_mf_lrb'].gpr.predict(X_baseline)[0]


####### MFs
res = 100
X_plot = make_2D_plotting_space(df_mf[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.23)

X_sc = make_2D_plotting_space(df_mf[covariate_list], 20, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.23)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

## mf-TFP: cost
ax = fig.add_subplot(2, 2, 1)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_cost =  mcac_regression_mdls['mdl_mcac_mf_lrb'].gpr.predict(X_plot)
qual_cost = mcac_regression_mdls['mdl_mcac_mf_lrb'].gpr.predict(X_sc)

X_sc_qual_cost = X_sc[qual_cost < 0.001]
sc = ax.scatter(X_sc_qual_cost[xvar], X_sc_qual_cost[yvar], c='white', edgecolors='black', s=10)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
# ax.set_xlim([0.5, 2.0])
# ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: annual repair cost ratio', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)

## mf-TFP: time
ax = fig.add_subplot(2, 2, 2)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

grid_time =  mcat_regression_mdls['mdl_mcat_mf_lrb'].gpr.predict(X_plot)
qual_time = mcat_regression_mdls['mdl_mcat_mf_lrb'].gpr.predict(X_sc)

X_sc_qual_time = X_sc[qual_time < 0.001]
sc = ax.scatter(X_sc_qual_time[xvar], X_sc_qual_time[yvar], c='white', edgecolors='black', s=10)

Z = np.array(grid_time)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5, levels=lvls)

clabels = ax.clabel(cs, fontsize=clabel_size)
# ax.set_xlim([0.5, 2.0])
# ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: annual repair time ratio', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)


## mf-TFP: cost
ax = fig.add_subplot(2, 2, 3)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))


all_upfront_costs  = calc_upfront_cost(
    X_plot, config_dict=config_dict_annual, steel_cost_dict=reg_dict)

mf_upfront_cost = all_upfront_costs['total_mf']


X_sc_qual = X_sc[np.logical_and.reduce((
        X_sc.index.isin(X_sc_qual_cost.index), 
        X_sc.index.isin(X_sc_qual_time.index)))]

sc = ax.scatter(X_sc_qual[xvar], X_sc_qual[yvar], c='white', edgecolors='black', s=10)

qual_upfront_cost  = calc_upfront_cost(
    X_sc_qual, config_dict=config_dict_annual, steel_cost_dict=reg_dict)

cheapest_idx = qual_upfront_cost['total_mf'].idxmin()

# least upfront cost of the viable designs
the_design = X_sc_qual.loc[cheapest_idx]

ax.scatter(the_design[xvar], the_design[yvar], marker='x', c='red', s=100)

Z = np.array(mf_upfront_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-0.5)

clabels = ax.clabel(cs, fontsize=clabel_size)
# ax.set_xlim([0.5, 2.0])
# ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: upfront cost', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)



##### NPV

upgrade_cost = (mf_upfront_cost- 
                baseline_cost['total_mf'].item())

avoided_cost = (mcac_baseline - grid_cost)*config_dict_annual['comparable_cost_mf']
avoided_time = (mcat_baseline - grid_time)*config_dict_annual['comparable_time_mf']

# profit loss and repair cost per worker-hour
# assume 40% of replacement cost is labor, $680/worker-day for SF Bay Area
profit_loss_per_worker_day = 680.0
avoided_time_cost = avoided_time * profit_loss_per_worker_day
avoided_consequence = avoided_cost + avoided_time_cost

i_rate = config_dict_annual['interest_rate']
t_yrs = config_dict_annual['timeframe']

# upgrade is worth it if NPV of avoided consequence > upgrade cost over baseline
NPV = avoided_consequence*((1 - 1/(1 + i_rate)**t_yrs) / i_rate)
upgrade_decision = (NPV - upgrade_cost) > 0

ax = fig.add_subplot(2, 2, 4)
# plt.setp(ax, xticks=np.arange(2.0, 11.0, step=1.0))

# grid_time =  mcat_regression_mdls['mdl_mcat_mf_lrb'].gpr.predict(X_plot)
# qual_time = mcat_regression_mdls['mdl_mcat_mf_lrb'].gpr.predict(X_sc)

# X_sc_qual_time = X_sc[qual_time < 0.001]
# sc = ax.scatter(X_sc_qual_time[xvar], X_sc_qual_time[yvar], c='white', edgecolors='black', s=10)

Z = np.array(NPV - upgrade_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues')

# ax.imshow(
#         Z_cont,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.coolwarm_r,
    # )

clabels = ax.clabel(cs, fontsize=clabel_size)
# ax.set_xlim([0.5, 2.0])
# ax.set_ylim([0.5, 2.3])


ax.grid(visible=True)
ax.set_title(r'MF-LRB: NPV', fontsize=title_font)
# ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.20$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

fig.tight_layout()


#%% 

# tail-favored metrics

# TODO: report as ROI

# TODO: should we validate at a non-MCE earthquake?


#%% generalized results of inverse design
# TODO: validation

def process_results(run_case):
    
    import numpy as np
    # load in validation and max run
    val_dir = '../../data/validation/'+run_case+'/'
    
    loss_file = run_case+'_loss.pickle'
    max_loss_file = run_case+'_max_loss.pickle'
    
    val_obj = pd.read_pickle(val_dir+loss_file)
    ida_results_df = val_obj.ida_results.reset_index(drop=True)
    loss_results_df = val_obj.loss_data.reset_index(drop=True)
    
    val_max_obj = pd.read_pickle(val_dir+max_loss_file)
    max_loss_results_df = val_max_obj.max_loss.reset_index(drop=True)
    
    # calculate loss ratios
    ida_results_df = loss_percentages(
        ida_results_df, loss_results_df, max_loss_results_df)
    
    # print out the results
    ida_levels = np.array(ida_results_df['ida_level'].unique())
    # ida_levels = [1.0, 1.5, 2.0]
    n = len(ida_levels)

    val_cost  = np.zeros((n,))
    val_replacement = np.zeros((n,))
    val_cost_ratio = np.zeros((n,))
    val_downtime_ratio = np.zeros((n,))
    val_downtime = np.zeros((n,))
    impact_freq = np.zeros((n,))
    struct_cost = np.zeros((n,))
    nsc_cost = np.zeros((n,))
    gap_ratios = np.zeros((n,))
    T_ratios = np.zeros((n,))
    
    GR_adjs = np.zeros((n,))
    
    isolator_system = run_case.split('_')[1]
    
    # collect variable: currently working with means of medians
    cost_var_ida = 'cost_50%'
    time_var_ida = 'time_l_50%'
    
    cost_var = 'cmp_cost_ratio'
    time_var = 'cmp_time_ratio'
    
    for i, lvl in enumerate(ida_levels):
        val_ida = ida_results_df[ida_results_df['ida_level']==lvl]
        loss_ida = loss_results_df[ida_results_df['ida_level']==lvl]
        
        val_replacement[i] = val_ida['replacement_freq'].mean()
        val_cost[i] = loss_ida[cost_var_ida].mean()
        val_cost_ratio[i] = val_ida[cost_var].mean()
        val_downtime[i] = loss_ida[time_var_ida].mean()
        val_downtime_ratio[i] = val_ida[time_var].mean()
        impact_freq[i] = val_ida['impacted'].mean()
        struct_cost[i] = val_ida['B_50%'].mean()
        nsc_cost[i] = val_ida['C_50%'].mean() + val_ida['D_50%'].mean() + val_ida['E_50%'].mean() 
            
        
        zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
        Bm = np.interp(val_ida['zeta_e'], zetaRef, BmRef)
        
        if isolator_system == 'tfp':
            T_shifted = np.mean(val_ida['T_m']*0.9)
        else:
            T_shifted = np.mean(val_ida['T_m'])
            
        sa_tm_adj = val_ida.apply(
            lambda x: get_ST(x, T_shifted,
                              db_dir='../../resource/ground_motions/gm_db.csv',
                              spec_dir='../../resource/ground_motions/gm_spectra.csv'), 
            axis=1)
        
        gap_ratios_all = (val_ida['constructed_moat']*4*pi**2)/ \
            (g*(val_ida['sa_tm']/Bm)*val_ida['T_m']**2)
        gap_ratios[i] = gap_ratios_all.mean()
        
        # breakpoint()
        GR_adj = (val_ida['constructed_moat']*4*pi**2)/ \
            (g*(sa_tm_adj/Bm)*T_shifted**2)
        GR_adjs[i] = GR_adj.mean()
            
        T_ratio_adj = T_shifted / val_ida['T_fbe'].mean()
        
        T_ratios[i] = T_shifted / val_ida['T_fb'].mean()
        
    # print(T_shifted)
    # print(GR_adjs)
    # print(T_ratio_adj)
    # print(T_ratios)
    
    
    design_list = []
    ss_sys = ida_results_df['superstructure_system'].iloc[0]
    iso_sys = ida_results_df['isolator_system'].iloc[0]
    if ss_sys == 'CBF':
        design_list.extend(['beam', 'column', 'brace'])
    else:
        design_list.extend(['beam', 'column'])
    if iso_sys == 'LRB':
        design_list.extend(['d_bearing', 'd_lead', 't_r', 'n_layers'])
    else:
        design_list.extend(['mu_1', 'mu_2', 'R_1', 'R_2'])
        
    
    sys_name = ss_sys+'-'+iso_sys
    
    # design_tested = ida_results_df[['moat_ampli', 'RI', 'T_ratio' , 'zeta_e']].iloc[0]
    # design_specifics = ida_results_df[design_list].iloc[0]
    # print('==================================')
    # print('   Validation results  (1.0 MCE)  ')
    # print('==================================')
    
    # print('System:', ss_sys+'-'+iso_sys)
    # print('Average median repair cost: ',
    #       f'${val_cost[0]:,.2f}')
    # print('Repair cost ratio: ', 
    #       f'{val_cost_ratio[0]:,.3f}')
    # print('Repair time ratio: ',
    #       f'{val_downtime_ratio[0]:,.3f}')
    # print('Estimated replacement frequency: ',
    #       f'{val_replacement[0]:.2%}')
    # print(design_tested)
    # print(design_specifics)
    
    latex_string = f"& {sys_name} & {val_cost_ratio[0]:.3f} & {val_cost_ratio[1]:.3f} & {val_cost_ratio[2]:.3f} \
        & {val_downtime_ratio[0]:.3f} & {val_downtime_ratio[1]:.3f} & {val_downtime_ratio[2]:.3f} \
            & {val_replacement[0]:.3f} & {val_replacement[1]:.3f} & {val_replacement[2]:.3f} \\\\"
    
    print(latex_string)  
    
    
    # n_workers = (ida_results_df['bldg_area']/1000).mean()

    # print('Cost total:', ida_results_df['total_cmp_cost'].mean()/1e6)
    # print('Time total:', ida_results_df['total_cmp_time'].mean()/n_workers)
    
    # print('GR:', gap_ratios)
    # print('TR:', T_ratios)
    # print('Impact:', impact_freq)
    # print('Structural cost:', struct_cost/1e6)
    # print('Non-structural cost:', nsc_cost/1e6)
    
    
    # latex_string = f"& {sys_name} & {mce_cost_ratio:.3f} & {mce_time_ratio:.3f} & {mce_repl_ratio:.3f} \
    #     & {val_cost_ratio[0]:.2f} & {GP_time_ratio:.2f} & {GP_repl_risk:.2f} &  \${upfront_cost/1e6:.2f} M \\\\"
    
    return(ida_results_df, val_replacement, val_cost, 
           val_cost_ratio, val_downtime, val_downtime_ratio)

(mf_tfp_val_results, mf_tfp_val_repl, mf_tfp_val_cost, mf_tfp_val_cost_ratio, 
 mf_tfp_val_downtime, mf_tfp_val_downtime_ratio) = process_results('mf_tfp_annual_func_hazard')
(mf_lrb_val_results, mf_lrb_val_repl, mf_lrb_val_cost, mf_lrb_val_cost_ratio, 
 mf_lrb_val_downtime, mf_lrb_val_downtime_ratio) = process_results('mf_lrb_annual_func_hazard')
(cbf_tfp_val_results, cbf_tfp_val_repl, cbf_tfp_val_cost, cbf_tfp_val_cost_ratio, 
 cbf_tfp_val_downtime, cbf_tfp_val_downtime_ratio) = process_results('cbf_tfp_annual_func_hazard')
(cbf_lrb_val_results, cbf_lrb_val_repl, cbf_lrb_val_cost, cbf_lrb_val_cost_ratio, 
 cbf_lrb_val_downtime, cbf_lrb_val_downtime_ratio) = process_results('cbf_lrb_annual_func_hazard')

#%%
def print_latex_inverse_table(sys_name, design_dict, performance_dict):

    
    GR = design_dict['gap_ratio']
    Ry = design_dict['RI']
    T_ratio = design_dict['T_ratio'] # this is the "designed" value
    zeta = design_dict['zeta_e']
    GP_cost_ratio = performance_dict['mcac']
    GP_time_ratio = performance_dict['mcat']
    GP_repl_risk = performance_dict['replacement_freq']
    upfront_cost = performance_dict['upfront_cost']
    
    
    latex_string = f"& {sys_name} & {GR:.2f} & {Ry:.2f} & {T_ratio:.2f} & {zeta:.2f} \
        & {GP_cost_ratio:.3f} & {GP_time_ratio:.3f} & {GP_repl_risk:.3f} &  \${upfront_cost/1e6:.2f} M \\\\"
    print(latex_string)
    return

# print_latex_inverse_table('MF-TFP', mf_tfp_inv_design, mf_tfp_inv_performance)   
# print_latex_inverse_table('MF-LRB', mf_lrb_inv_design, mf_lrb_inv_performance)   
# print_latex_inverse_table('CBF-TFP', cbf_tfp_inv_design, cbf_tfp_inv_performance)   
# print_latex_inverse_table('CBF-LRB', cbf_lrb_inv_design, cbf_lrb_inv_performance)   

print()

def print_latex_design_table(sys_name, val_results):

    typ_design = val_results.iloc[0]
    
    # moat in cm
    moat = typ_design['constructed_moat']*2.54
    largest_beam = typ_design['beam'][0]
    largest_column = typ_design['column'][0]
    try:
        largest_brace = typ_design['brace'][0]
    except:
        largest_brace = 'n/a'
    
    # d bearings in cm, R curvature in mm
    try:
        bearing_param_1 = typ_design['mu_1']
    except:
        bearing_param_1 = typ_design['d_lead']*2.54
        
    try:
        bearing_param_2 = typ_design['mu_2']
    except:
        bearing_param_2 = typ_design['d_bearing'] *2.54
        
    try:
        bearing_param_3 = typ_design['R_1']*25.4
    except:
        bearing_param_3 = typ_design['t_r']* 2.54
        
    # print as either TFP or LRB
    try:
        bearing_param_4 = typ_design['R_2']*25.4
        latex_string = f"& {sys_name} & {moat:.1f} cm & {largest_beam} & {largest_column} & {largest_brace} \
            & {bearing_param_1:.3f} & {bearing_param_2:.3f} & {bearing_param_3:.0f} mm &  {bearing_param_4:.0f} mm \\\\"
    except:
        bearing_param_4 = typ_design['n_layers'] 
        latex_string = f"& {sys_name} & {moat:.1f} cm & {largest_beam} & {largest_column} & {largest_brace} \
            & {bearing_param_1:.1f} cm & {bearing_param_2:.1f} cm & {bearing_param_3:.1f} cm &  {bearing_param_4:.0f}  \\\\"
    
    # print('Average median repair cost: ',
    #       f'${val_cost[0]:,.2f}')
    # print('Repair cost ratio: ', 
    #       f'{val_cost_ratio[0]:,.3f}')
    # print('Repair time ratio: ',
    #       f'{val_downtime_ratio[0]:,.3f}')
    # print('Estimated replacement frequency: ',
    #       f'{val_replacement[0]:.2%}')
    
    
    print(latex_string)
    return

print_latex_design_table('MF-TFP', mf_tfp_val_results)
print_latex_design_table('CBF-TFP', cbf_tfp_val_results)
print_latex_design_table('MF-LRB', mf_lrb_val_results)
print_latex_design_table('CBF-LRB', cbf_lrb_val_results)

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
                       niter=10, seed=985)
    
    return res.x[0], res.x[1]

from scipy.stats import ecdf
f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
# plt.close('all')

# moderate designs
cbf_tfp_ida = cbf_tfp_val_results[cbf_tfp_val_results['ida_level']==1.0]
mf_tfp_ida = mf_tfp_val_results[mf_tfp_val_results['ida_level']==1.0]
cbf_lrb_ida = cbf_lrb_val_results[cbf_lrb_val_results['ida_level']==1.0]
mf_lrb_ida = mf_lrb_val_results[mf_lrb_val_results['ida_level']==1.0]

my_y_var = mf_tfp_ida[cost_var]
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

fig = plt.figure(figsize=(13, 9))

# theta_inv, beta_inv = mle_fit_general(
#     ecdf_values,ecdf_prob, x_init=(np.median(ecdf_values),0.5))

theta_inv = np.exp(np.log(my_y_var).mean())
beta_inv = np.log(my_y_var).var()

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_inv, beta_inv)

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

my_y_var = mf_lrb_ida[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

# theta_inv, beta_inv = mle_fit_general(
#     ecdf_values,ecdf_prob, x_init=(np.median(ecdf_values),0.25))

theta_inv = np.exp(np.log(my_y_var).mean())
beta_inv = np.log(my_y_var).var()

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_inv, beta_inv)

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

my_y_var = cbf_tfp_ida[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

# theta_inv, beta_inv = mle_fit_general(
#     ecdf_values,ecdf_prob, x_init=(np.median(ecdf_values),1))

theta_inv = np.exp(np.log(my_y_var).mean())
beta_inv = np.log(my_y_var).var()

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_inv, beta_inv)

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

my_y_var = cbf_lrb_ida[cost_var]
res = ecdf(my_y_var)
ecdf_prob = res.cdf.probabilities
ecdf_values = res.cdf.quantiles

# theta_inv, beta_inv = mle_fit_general(
#     ecdf_values,ecdf_prob, x_init=(np.median(ecdf_values),1))

theta_inv = np.exp(np.log(my_y_var).mean())
beta_inv = np.log(my_y_var).var()

xx_pr = np.linspace(1e-4, 1.0, 400)
p = f(xx_pr, theta_inv, beta_inv)

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

#%%
# make lambda function for generic lognormal distribution
import numpy as np
lognorm_f = lambda x,theta,beta: norm(np.log(theta), beta**0.5).cdf(np.log(x))

def validate_lifetime_loss(val_results, hazard_curves,
                           cost_var='cmp_cost_ratio', time_var='cmp_time_ratio'):
    
    
    # get return rate of the three Sa_avg of the IDA
    T_m = val_results['T_m'].unique().item()
    T_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
    idx_between = bisect.bisect(T_list, T_m)
    
    # 0 is total
    below_lambda = hazard_curves[idx_between-1]['data'][0]['yvalues']
    below_sa = hazard_curves[idx_between-1]['metadata']['xvalues']
    
    above_lambda = hazard_curves[idx_between]['data'][0]['yvalues']
    above_sa = hazard_curves[idx_between]['metadata']['xvalues']
    
    x2 = T_list[idx_between]
    x1 = T_list[idx_between-1]
    # assume that both series have the same length
    sa_T = [(g + h) / 2 for g, h in zip(below_sa, above_sa)]
    lambda_T = [y1+(T_m-x1)*(y2-y1)/(x2-x1) for y1, y2 in zip(below_lambda, above_lambda)]
    
    # # grab Sa_avg of IDA
    # sa_bins = np.array([float(sa) for sa in val_results['sa_avg'].unique()])
    
    # grab Sa_Tm average of IDA level
    ida_levels = np.array([float(lvl) for lvl in val_results['ida_level'].unique()])
    sa_bins = np.zeros(len(ida_levels))
    for scn_idx, ida_lvl in enumerate(ida_levels):
        ida_df = val_results[val_results['ida_level']==ida_lvl]
        sa_bins[scn_idx] = ida_df['sa_tm'].mean()
    
    # breakpoint()
    # interpolate in logspace
    log_lambda = np.log(lambda_T)
    log_lambda[log_lambda == -np.inf] = -100
    lambda_bins = np.exp(np.interp(np.log(sa_bins), np.log(sa_T), log_lambda))
    
    # lambda_bins = np.array([1e-2, 2e-3, 2e-4])
    
    
    # get the loss "curve" at the three IDA levels
    from scipy.stats import ecdf
    
    # two choice: fit lognormal OR use ECDF
    f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))
    # plt.close('all')
    
    # value to renormalize ratio into consequences
    total_cmp_cost = val_results['total_cmp_cost'].median()
    total_cmp_time = val_results['total_cmp_time'].median()
    
    # value to set max of linspace array of consequences of exceedance curve
    repl_cost_max = val_results['replacement_cost'].median()
    repl_time_max = val_results['replacement_time'].median()
    cost_loss_values = np.linspace(1e-4, repl_cost_max, 1000)
    time_loss_values = np.linspace(1e-4, repl_time_max, 1000)
    
    
    
    cost_scns = np.zeros([len(cost_loss_values), len(sa_bins)])
    time_scns = np.zeros([len(time_loss_values), len(sa_bins)])
    
    for scn_idx, ida_lvl in enumerate(ida_levels):
        ida_df = val_results[val_results['ida_level']==ida_lvl]

        # from individual IDA result, calculate distribution
        cost_ratio_ida = ida_df[cost_var]
        res = ecdf(my_y_var)
        ecdf_prob_cost_ratio = res.cdf.probabilities
        ecdf_values_cost_ratio = res.cdf.quantiles
        theta_cost = np.exp(np.log(cost_ratio_ida).mean())
        beta_cost = np.log(cost_ratio_ida).var()
        
        
        time_ratio_ida = ida_df[time_var]
        res = ecdf(my_y_var)
        ecdf_prob_time_ratio = res.cdf.probabilities
        ecdf_values_time_ratio = res.cdf.quantiles
        theta_time = np.exp(np.log(time_ratio_ida).mean())
        beta_time = np.log(time_ratio_ida).var()
        
        # unnormalize loss ratio back to loss
        cost_bins = cost_ratio_ida*total_cmp_cost
        time_bins = time_ratio_ida*total_cmp_time
        
        # make exceedance curve for each scenario
        # use total replacement just to have a bigger number
        cost_scns[:,scn_idx] = lognorm_f(cost_loss_values, theta_cost*total_cmp_cost, beta_cost)
        time_scns[:,scn_idx] = lognorm_f(time_loss_values, theta_time*total_cmp_time, beta_time)
        
        
    # upper bound of considered replacement
    total_cmp_cost_ub = val_results['total_cmp_cost_ub'].median()
    total_cmp_time_ub = val_results['total_cmp_time_ub'].median()
    cost_scns[cost_loss_values > total_cmp_cost_ub, :] = 1.0
    time_scns[time_loss_values > total_cmp_time_ub, :] = 1.0
    
    
    pr_exceedance_cost = 1 - cost_scns
    pr_exceedance_time = 1 - time_scns
    
    cost_loss_rates = np.multiply(pr_exceedance_cost, lambda_bins)
    time_loss_rates = np.multiply(pr_exceedance_time, lambda_bins)
    
    
    
    import matplotlib.pyplot as plt
    # plt.close('all')
    fig = plt.figure(figsize=(9, 6))
    ax1=fig.add_subplot(1, 1, 1)
    
    for scn_idx in range(len(sa_bins)):
        ax1.plot(cost_loss_values, pr_exceedance_cost[:,scn_idx], label='scn_'+str(scn_idx))
    ax1.legend()
    ax1.set_xlabel(r'Cost (\$)', fontsize=axis_font)
    ax1.set_ylabel(r'$Pr[X \geq \$]$', fontsize=axis_font)
    ax1.grid()
    ax1.set_xlim([0, repl_cost_max])
    
    
    import matplotlib.pyplot as plt
    # plt.close('all')
    fig = plt.figure(figsize=(9, 7))
    ax1=fig.add_subplot(1, 1, 1)
    
    for scn_idx in range(len(sa_bins)):
        ax1.plot(cost_loss_values, cost_loss_rates[:,:scn_idx+1].sum(axis=1), label='scn_'+str(scn_idx))
    ax1.legend()
    ax1.set_xlabel(r'Cost (\$)', fontsize=axis_font)
    ax1.set_ylabel(r'$Pr[X \geq \$]$', fontsize=axis_font)
    ax1.grid()
    ax1.set_xlim([0, 0.1*repl_cost_max])
    
    fig = plt.figure(figsize=(9, 7))
    ax1=fig.add_subplot(1, 1, 1)
    
    for scn_idx in range(len(sa_bins)):
        ax1.plot(time_loss_values, time_loss_rates[:,:scn_idx+1].sum(axis=1), label='scn_'+str(scn_idx))
    ax1.legend()
    ax1.set_xlabel(r'time (worker-day)', fontsize=axis_font)
    ax1.set_ylabel(r'$Pr[X \geq t]$', fontsize=axis_font)
    ax1.grid()
    ax1.set_xlim([0, 0.1*repl_time_max])
    
    # breakpoint()
    
    # multiply scenarios' exceedance curve with corresponding return rate
    # sum across all scenarios
    agg_cost_exceedance_rate = pr_exceedance_cost @ lambda_bins
    agg_time_exceedance_rate = pr_exceedance_time @ lambda_bins
    
    # integrate to attain lifetime dollar, time
    mean_cumulative_annual_cost = np.trapz(agg_cost_exceedance_rate, cost_loss_values)
    mean_cumulative_annual_time = np.trapz(agg_time_exceedance_rate, time_loss_values)
    
    # renormalize
    mcac_ratio = mean_cumulative_annual_cost/total_cmp_cost
    mcat_ratio = mean_cumulative_annual_time/total_cmp_time
    
    return mean_cumulative_annual_cost, mean_cumulative_annual_time, mcac_ratio, mcat_ratio

plt.close('all')
mcac_mf_tfp, mcat_mf_tfp, mcac_ratio_mf_tfp, mcat_ratio_mf_tfp = validate_lifetime_loss(
    mf_tfp_val_results, site_hazard_curves)

mcac_mf_lrb, mcat_mf_lrb, mcac_ratio_mf_lrb, mcat_ratio_mf_lrb = validate_lifetime_loss(
    mf_lrb_val_results, site_hazard_curves)

mcac_cbf_tfp, mcat_cbf_tfp, mcac_ratio_cbf_tfp, mcat_ratio_cbf_tfp = validate_lifetime_loss(
    cbf_tfp_val_results, site_hazard_curves)

mcac_cbf_lrb, mcat_cbf_lrb, mcac_ratio_cbf_lrb, mcat_ratio_cbf_lrb = validate_lifetime_loss(
    cbf_lrb_val_results, site_hazard_curves)