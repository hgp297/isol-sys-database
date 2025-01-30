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





#%% Testing the design space

def grid_search_inverse_design(res, system_name, targets_dict, config_dict,
                               impact_clfs, cost_regs, time_regs, repl_regs,
                               cost_var='cmp_cost_ratio', time_var='cmp_time_ratio'):
    import time
    
    # isolator_system = system_name.split('_')[1]
    # system_X = impact_clfs['mdl_impact_'+system_name].X
    X_space = make_design_space(res)
    
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
    
    mdl_impact = impact_clfs[mdl_impact_name]
    mdl_time_hit = time_regs[mdl_time_hit_name]
    mdl_time_miss = time_regs[mdl_time_miss_name]
    
    # identify replacement models
    mdl_repl_hit_name = 'mdl_repl_' + system_name + '_i'
    mdl_repl_miss_name = 'mdl_repl_' + system_name + '_o'
    
    mdl_impact = impact_clfs[mdl_impact_name]
    mdl_repl_hit = repl_regs[mdl_repl_hit_name]
    mdl_repl_miss = repl_regs[mdl_repl_miss_name]
    
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
    
    # assumes GPC/GPR, predict the outcome for the design space
    space_repair_cost = predict_DV(X_space, 
                                   mdl_impact.gpc, 
                                   mdl_cost_hit.gpr, 
                                   mdl_cost_miss.gpr, 
                                   outcome=cost_var)
    tp = time.time() - t0
    print("GPC-GPR repair cost prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))
    
    t0 = time.time()
    space_downtime = predict_DV(X_space,
                                mdl_impact.gpc,
                                mdl_time_hit.gpr,
                                mdl_time_miss.gpr,
                                outcome=time_var)
    tp = time.time() - t0
    print("GPC-GPR downtime prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                                   tp))

    t0 = time.time()
    space_repl = predict_DV(X_space,
                            mdl_impact.gpc,
                            mdl_repl_hit.gpr,
                            mdl_repl_miss.gpr,
                            outcome='replacement_freq')
    tp = time.time() - t0
    print("GPC-GPR replacement prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                                   tp))
    
    t0 = time.time()
    space_constr = mdl_impact.kde.score_samples(X_space)
    tp = time.time() - t0
    print("KDE constructability prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                                   tp))
    
    # filter cost threshold
    cost_thresh = targets_dict[cost_var]
    ok_cost = X_space.loc[space_repair_cost[cost_var+'_pred']<=cost_thresh]

    # downtime threshold
    dt_thresh = targets_dict[time_var]
    ok_time = X_space.loc[space_downtime[time_var+'_pred']<=dt_thresh]

    # acceptable replacement risk
    repl_thresh = targets_dict['replacement_freq']
    ok_repl = X_space.loc[space_repl['replacement_freq_pred']<=
                          repl_thresh]
    
    # constructable
    constr_thresh = targets_dict['constructability']
    ok_constr = X_space.loc[space_constr >= constr_thresh]

    X_design = X_space[np.logical_and.reduce((
            X_space.index.isin(ok_cost.index), 
            X_space.index.isin(ok_time.index),
            X_space.index.isin(ok_repl.index),
            X_space.index.isin(ok_constr.index)))]

    if X_design.shape[0] < 1:
        print('No suitable design found for system', system_name)
        return None, None
    
    
    # select best viable design
    upfront_costs = calc_upfront_cost(
        X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
    structural_system = system_name.split('_')[0]
    cheapest_idx = upfront_costs['total_'+structural_system].idxmin()
    inv_upfront_cost = upfront_costs['total_'+structural_system].min()

    # least upfront cost of the viable designs
    inv_design = X_design.loc[cheapest_idx]
    inv_downtime = space_downtime.iloc[cheapest_idx].item()
    inv_repair_cost = space_repair_cost.iloc[cheapest_idx].item()
    inv_repl_risk = space_repl.iloc[cheapest_idx].item()
    
    inv_performance = {
        'time': inv_downtime,
        'cost': inv_repair_cost,
        'replacement_freq': inv_repl_risk,
        'upfront_cost': inv_upfront_cost}
    
    bldg_area = (config_dict['num_bays']*config_dict['L_bay'])**2 * (config_dict['num_stories'] + 1)

    # # assume $600/sf replacement
    # n_worker_series = bldg_area/1000
    # n_worker_parallel = n_worker_series/2

    # read out predictions
    print('==================================')
    print('            Predictions           ')
    print('==================================')
    print('======= Targets =======')
    print('System:', system_name)
    print('Repair cost fraction:', f'{cost_thresh*100:,.2f}%')
    print('Repair time fraction:', f'{dt_thresh*100:,.2f}%')
    print('Replacement risk:', f'{repl_thresh*100:,.2f}%')


    print('======= Overall inverse design =======')
    print(inv_design)
    print('Upfront cost of selected design: ',
          f'${inv_upfront_cost:,.2f}')
    print('Predicted median repair cost ratio: ',
          f'{inv_repair_cost*100:,.2f}%')
    print('Predicted repair time ratio: ',
          f'{inv_downtime*100:,.2f}%')
    print('Predicted replacement risk: ',
          f'{inv_repl_risk:.2%}')
    
    return(inv_design, inv_performance, X_design)
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

#%% cost

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

#%% inverse design filters
### regular
config_dict = {
    'num_stories': 4,
    'h_story': 13.0,
    'num_bays': 6,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': 30.0,
    'S_1': 1.017
    }

my_targets = {
    cost_var: 0.2,
    time_var: 0.2,
    'replacement_freq': 0.1,
    'constructability': -6.0}

mf_tfp_inv_design, mf_tfp_inv_performance, mf_tfp_space = grid_search_inverse_design(
    20, 'mf_tfp', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

mf_lrb_inv_design, mf_lrb_inv_performance, mf_lrb_space = grid_search_inverse_design(
    20, 'mf_lrb', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

cbf_tfp_inv_design, cbf_tfp_inv_performance, cbf_tfp_space = grid_search_inverse_design(
    20, 'cbf_tfp', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

cbf_lrb_inv_design, cbf_lrb_inv_performance, cbf_lrb_space = grid_search_inverse_design(
    20, 'cbf_lrb', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

### strict
config_dict = {
    'num_stories': 4,
    'h_story': 13.0,
    'num_bays': 4,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': 30.0,
    'S_1': 1.017
    }

my_targets = {
    cost_var: 0.1,
    time_var: 0.1,
    'replacement_freq': 0.05,
    'constructability': -6.0}

mf_tfp_strict_design, mf_tfp_strict_performance, mf_tfp_strict_space = grid_search_inverse_design(
    20, 'mf_tfp', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

mf_lrb_strict_design, mf_lrb_strict_performance, mf_lrb_strict_space = grid_search_inverse_design(
    20, 'mf_lrb', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

cbf_tfp_strict_design, cbf_tfp_strict_performance, cbf_tfp_strict_space = grid_search_inverse_design(
    20, 'cbf_tfp', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

cbf_lrb_strict_design, cbf_lrb_strict_performance, cbf_lrb_strict_space = grid_search_inverse_design(
    20, 'cbf_lrb', my_targets, config_dict, 
    impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls,
    cost_var='cmp_cost_ratio', time_var='cmp_time_ratio')

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
    ida_levels = [1.0, 1.5, 2.0]

    val_cost  = np.zeros((3,))
    val_replacement = np.zeros((3,))
    val_cost_ratio = np.zeros((3,))
    val_downtime_ratio = np.zeros((3,))
    val_downtime = np.zeros((3,))
    impact_freq = np.zeros((3,))
    struct_cost = np.zeros((3,))
    nsc_cost = np.zeros((3,))
    gap_ratios = np.zeros((3,))
    T_ratios = np.zeros((3,))
    
    
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
        
        gap_ratios_all = (val_ida['constructed_moat']*4*pi**2)/ \
            (g*(val_ida['sa_tm']/Bm)*val_ida['T_m']**2)
        gap_ratios[i] = gap_ratios_all.mean()
            
        T_ratios[i] = val_ida['T_m'].mean() / val_ida['T_fb'].mean()
    
    
    
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
 mf_tfp_val_downtime, mf_tfp_val_downtime_ratio) = process_results('mf_tfp_moderate')
(mf_lrb_val_results, mf_lrb_val_repl, mf_lrb_val_cost, mf_lrb_val_cost_ratio, 
 mf_lrb_val_downtime, mf_lrb_val_downtime_ratio) = process_results('mf_lrb_moderate')
(cbf_tfp_val_results, cbf_tfp_val_repl, cbf_tfp_val_cost, cbf_tfp_val_cost_ratio, 
 cbf_tfp_val_downtime, cbf_tfp_val_downtime_ratio) = process_results('cbf_tfp_moderate')
(cbf_lrb_val_results, cbf_lrb_val_repl, cbf_lrb_val_cost, cbf_lrb_val_cost_ratio, 
 cbf_lrb_val_downtime, cbf_lrb_val_downtime_ratio) = process_results('cbf_lrb_moderate')

(mf_tfp_strict_results, mf_tfp_strict_repl, mf_tfp_strict_cost, mf_tfp_strict_cost_ratio, 
 mf_tfp_strict_downtime, mf_tfp_strict_downtime_ratio) = process_results('mf_tfp_enhanced')
(mf_lrb_strict_results, mf_lrb_strict_repl, mf_lrb_strict_cost, mf_lrb_strict_cost_ratio, 
 mf_lrb_strict_downtime, mf_lrb_strict_downtime_ratio) = process_results('mf_lrb_enhanced')
(cbf_tfp_strict_results, cbf_tfp_strict_repl, cbf_tfp_strict_cost, cbf_tfp_strict_cost_ratio, 
 cbf_tfp_strict_downtime, cbf_tfp_strict_downtime_ratio) = process_results('cbf_tfp_enhanced')
(cbf_lrb_strict_results, cbf_lrb_strict_repl, cbf_lrb_strict_cost, cbf_lrb_strict_cost_ratio, 
 cbf_lrb_strict_downtime, cbf_lrb_strict_downtime_ratio) = process_results('cbf_lrb_enhanced')

#%% debrief predictions: try to get predictions of actual ran building
# two values differ
# GR should be different because real ground motion suite has different Sa than design spectrum
# T ratio should be different because actual T_fb is not perfectly equal to Tfbe

def one_time_pred(X_built, system_name, 
                        impact_clfs, cost_regs, time_regs, repl_regs):
    
    cost_var = 'cmp_cost_ratio'
    time_var = 'cmp_time_ratio'
    
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
    
    mdl_impact = impact_clfs[mdl_impact_name]
    mdl_time_hit = time_regs[mdl_time_hit_name]
    mdl_time_miss = time_regs[mdl_time_miss_name]
    
    # identify replacement models
    mdl_repl_hit_name = 'mdl_repl_' + system_name + '_i'
    mdl_repl_miss_name = 'mdl_repl_' + system_name + '_o'
    
    mdl_impact = impact_clfs[mdl_impact_name]
    mdl_repl_hit = repl_regs[mdl_repl_hit_name]
    mdl_repl_miss = repl_regs[mdl_repl_miss_name]
    
    # assumes GPC/GPR, predict the outcome for the design space
    ab_repair_cost, ab_repair_cost_var = predict_DV(
        X_built, mdl_impact.gpc, mdl_cost_hit.gpr, mdl_cost_miss.gpr, 
        outcome=cost_var, return_var=True)
    
    ab_downtime, ab_downtime_var = predict_DV(
        X_built, mdl_impact.gpc, mdl_time_hit.gpr, mdl_time_miss.gpr, 
        outcome=time_var, return_var=True)
    
    ab_repl, ab_repl_var = predict_DV(
        X_built, mdl_impact.gpc, mdl_repl_hit.gpr, mdl_repl_miss.gpr,
        outcome='replacement_freq', return_var=True)
    try:
        ab_performance = {
            'time': ab_downtime.iloc[0][time_var+'_pred'],
            'cost': ab_repair_cost.iloc[0][cost_var+'_pred'],
            'replacement_freq': ab_repl.iloc[0]['replacement_freq'+'_pred'],
            'time_var': ab_downtime_var[0],
            'cost_var': ab_repair_cost_var[0],
            'replacement_freq_var': ab_repl_var[0]}
    except:
        ab_performance = {
            'time': ab_downtime.iloc[0][time_var+'_pred'],
            'cost': ab_repair_cost.iloc[0][cost_var+'_pred'],
            'replacement_freq': ab_repl.iloc[0]['replacement_freq'+'_pred'],
            'time_var': ab_downtime_var.iloc[0][time_var+'_pred'],
            'cost_var': ab_repair_cost_var.iloc[0][cost_var+'_pred'],
            'replacement_freq_var': ab_repl_var.iloc[0]['replacement_freq'+'_pred']}

    return(ab_performance)


mf_tfp_inv_pred = one_time_pred(
    pd.DataFrame(mf_tfp_inv_design).T, 'mf_tfp', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)

mf_lrb_inv_pred = one_time_pred(
    pd.DataFrame(mf_lrb_inv_design).T, 'mf_lrb', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)

cbf_tfp_inv_pred = one_time_pred(
    pd.DataFrame(cbf_tfp_inv_design).T, 'cbf_tfp', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)
 
cbf_lrb_inv_pred = one_time_pred(
    pd.DataFrame(cbf_lrb_inv_design).T, 'cbf_lrb', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)

mf_tfp_strict_pred = one_time_pred(
    pd.DataFrame(mf_tfp_strict_design).T, 'mf_tfp', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)

mf_lrb_strict_pred = one_time_pred(
    pd.DataFrame(mf_lrb_strict_design).T, 'mf_lrb', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)

cbf_tfp_strict_pred = one_time_pred(
    pd.DataFrame(cbf_tfp_strict_design).T, 'cbf_tfp', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)
 
cbf_lrb_strict_pred = one_time_pred(
    pd.DataFrame(cbf_lrb_strict_design).T, 'cbf_lrb', impact_classification_mdls, cost_regression_mdls, 
    time_regression_mdls, repl_regression_mdls)
#%% MLE fragility curves

def neg_log_likelihood_sum(params, im_l, no_a, no_c):
    from scipy import stats
    import numpy as np
    sigma, beta = params
    theoretical_fragility_function = stats.norm(np.log(sigma), beta).cdf(im_l)
    likelihood = stats.binom.pmf(no_c, no_a, theoretical_fragility_function)
    log_likelihood = np.log(likelihood)
    log_likelihood_sum = np.sum(log_likelihood)

    return -log_likelihood_sum

def mle_fit_collapse(ida_levels, pr_collapse):
    from functools import partial
    import numpy as np
    from scipy import optimize
    
    im_log = np.log(ida_levels)
    number_of_analyses = np.array([1000, 1000, 1000 ])
    number_of_collapses = np.round(1000*pr_collapse)
    
    neg_log_likelihood_sum_partial = partial(
        neg_log_likelihood_sum, im_l=im_log, no_a=number_of_analyses, no_c=number_of_collapses)
    
    
    res = optimize.minimize(neg_log_likelihood_sum_partial, (1, 1), method="Nelder-Mead")
    return res.x[0], res.x[1]

mf_tfp_color = '#004AAE'
mf_lrb_color = '#FC9313'
cbf_tfp_color = '#018943'
cbf_lrb_color = 'deepskyblue'


# print out the results
ida_levels = [1.0, 1.5, 2.0]

from scipy.stats import norm
f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 24
subt_font = 21
label_size = 22
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
from matplotlib.lines import Line2D
plt.close('all')

fig = plt.figure(figsize=(7,6))
ax1=fig.add_subplot(1, 1, 1)

######
# MF-TFP
theta_inv, beta_inv = mle_fit_collapse(ida_levels,mf_tfp_val_repl)
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_inv, beta_inv)

mf_tfp_repl_risk = mf_tfp_inv_pred['replacement_freq']
mf_tfp_repl_risk_var = mf_tfp_inv_pred['replacement_freq_var']
upper = mf_tfp_repl_risk+mf_tfp_repl_risk_var**0.5

MCE_level = float(p[xx_pr==1.0])
ax1.plot(xx_pr, p, linewidth=2.0, color=mf_tfp_color)
ax1.axhline(mf_tfp_repl_risk, linestyle='-.', color=mf_tfp_color)
# ax1.axhline(upper, linestyle=':', color=mf_tfp_color)
# ax1.text(1.5,mf_tfp_repl_risk+0.02, r'Predicted replacement risk',
#           fontsize=subt_font, color=mf_tfp_color)
# ax1.text(1.5,upper+0.02, r'$+1\sigma$',
#           fontsize=subt_font, color=mf_tfp_color)
ax1.text(0.6, MCE_level+.08, f'{MCE_level:,.4f}',
          fontsize=subt_font, color=mf_tfp_color, bbox=dict(facecolor='white', edgecolor=mf_tfp_color))

ax1.set_ylabel('Replacement probability', fontsize=axis_font)
ax1.set_xlabel(r'$\times$MCE intensity', fontsize=axis_font)
# ax1.set_title('MF-TFP', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [mf_tfp_val_repl[i]], 
              marker='x', markersize=15, color=mf_tfp_color)
    
######
# MF-LRB
theta_inv, beta_inv = mle_fit_collapse(ida_levels,mf_lrb_val_repl)
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_inv, beta_inv)

mf_lrb_repl_risk = mf_lrb_inv_pred['replacement_freq']
mf_lrb_repl_risk_var = mf_lrb_inv_pred['replacement_freq_var']
upper = mf_lrb_repl_risk+mf_lrb_repl_risk_var**0.5

MCE_level = float(p[xx_pr==1.0])
ax1.plot(xx_pr, p, linewidth=2.0, color=mf_lrb_color)
ax1.axhline(mf_lrb_repl_risk, linestyle='-.', color=mf_lrb_color)
# ax1.axhline(upper, linestyle=':', color=mf_lrb_color)
# ax1.text(1.5,mf_lrb_repl_risk+0.02, r'Predicted replacement risk',
#           fontsize=subt_font, color=mf_lrb_color)
# ax1.text(1.5,upper+0.02, r'$+1\sigma$',
#           fontsize=subt_font, color=mf_lrb_color)
ax1.text(0.6, MCE_level+0.0, f'{MCE_level:,.4f}',
          fontsize=subt_font, color=mf_lrb_color, bbox=dict(facecolor='white', edgecolor=mf_lrb_color))


# ax1.set_title('MF-TFP', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [mf_lrb_val_repl[i]], 
              marker='x', markersize=15, color=mf_lrb_color)
    
######
# CBF-TFP
theta_inv, beta_inv = mle_fit_collapse(ida_levels,cbf_tfp_val_repl)
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_inv, beta_inv)

cbf_tfp_repl_risk = cbf_tfp_inv_pred['replacement_freq']
cbf_tfp_repl_risk_var = cbf_tfp_inv_pred['replacement_freq_var']
upper = cbf_tfp_repl_risk+cbf_tfp_repl_risk_var**0.5

MCE_level = float(p[xx_pr==1.0])
ax1.plot(xx_pr, p, linewidth=2.0, color=cbf_tfp_color)
ax1.axhline(cbf_tfp_repl_risk, linestyle='-.', color=cbf_tfp_color)
# ax1.axhline(upper, linestyle=':', color=cbf_tfp_color)
# ax1.text(1.5,cbf_tfp_repl_risk+0.02, r'Predicted replacement risk',
#           fontsize=subt_font, color=cbf_tfp_color)
# ax1.text(1.5,upper+0.02, r'$+1\sigma$',
#           fontsize=subt_font, color=cbf_tfp_color)
ax1.text(0.6, MCE_level+0.045, f'{MCE_level:,.4f}',
          fontsize=subt_font, color=cbf_tfp_color, bbox=dict(facecolor='white', edgecolor=cbf_tfp_color))


# ax1.set_title('MF-TFP', fontsize=title_font)
    
######
# CBF-LRB
theta_inv, beta_inv = mle_fit_collapse(ida_levels,cbf_lrb_val_repl)
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_inv, beta_inv)

cbf_lrb_repl_risk = cbf_lrb_inv_pred['replacement_freq']
cbf_lrb_repl_risk_var = cbf_lrb_inv_pred['replacement_freq_var']
upper = cbf_lrb_repl_risk+cbf_lrb_repl_risk_var**0.5

MCE_level = float(p[xx_pr==1.0])
ax1.plot(xx_pr, p, linewidth=2.0, color=cbf_lrb_color)
ax1.axhline(cbf_lrb_repl_risk, linestyle='-.', color=cbf_lrb_color)
# ax1.axhline(upper, linestyle=':', color=cbf_lrb_color)
# ax1.text(1.5,cbf_lrb_repl_risk+0.02, r'Predicted replacement risk',
#           fontsize=subt_font, color=cbf_lrb_color)
# ax1.text(1.5,upper+0.02, r'$+1\sigma$',
#           fontsize=subt_font, color=cbf_lrb_color)
ax1.text(0.6, MCE_level-0.005, f'{MCE_level:,.4f}',
          fontsize=subt_font, color=cbf_lrb_color, bbox=dict(facecolor='white', edgecolor=cbf_lrb_color))


# ax1.set_title('MF-TFP', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [cbf_lrb_val_repl[i]], 
              marker='x', markersize=15, color=cbf_lrb_color)

    
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [cbf_tfp_val_repl[i]], 
              marker='x', markersize=15, color=cbf_tfp_color)
    
ax1.axhline(0.1, linestyle=':', color='black', linewidth=1.5)
ax1.axvline(1.0, linestyle=':', color='black')
ax1.text(0.9, 0.3, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')
ax1.text(1.4, 0.12, r'Performance target',
          fontsize=subt_font, color='black')
ax1.grid(True)
ax1.set_xlim([0, 2.5])
ax1.set_ylim([0, 1.0])
fig.tight_layout()
plt.show()

    
custom_lines = [Line2D([-1], [-1], color=mf_tfp_color),
                Line2D([-1], [-1], color=mf_lrb_color),
                Line2D([-1], [-1], color=cbf_tfp_color),
                Line2D([-1], [-1], color=cbf_lrb_color),
                ]

ax1.legend(custom_lines, ['MF-TFP', 'MF-LRB', 'CBF-TFP', 'CBF-LRB'], 
            fontsize=subt_font)

plt.savefig('./poster_figure/inverse_repl_frag.svg')

#%% cost validation distr

def plot_predictions(mean, var, y_middle, y_bar, color):
    middle = mean
    upper = mean + var**0.5
    lower = mean - var**0.5
    
    ax.plot(middle, y_bar, 'v', ms=10, markeredgecolor='black', markerfacecolor=color, zorder=5)
    ax.plot([lower, upper], 
            [y_bar,y_bar], color=color)
    
    ax.axvline(middle, ymin=y_middle-0.075, ymax=y_middle+0.075, linestyle='--', color=color)
    ax.axvline(lower, ymin=y_middle-0.025, ymax=y_middle+0.025, color=color)
    ax.axvline(upper, ymin=y_middle-0.025, ymax=y_middle+0.025, color=color)
    
# plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 22
label_size = 20
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


# fig, axes = plt.subplots(1, 1, 
#                          figsize=(10, 6))

fig = plt.figure(figsize=(8,6))

mf_tfp_color = 'royalblue'
mf_lrb_color = 'cornflowerblue'
cbf_tfp_color = 'lightsalmon'
cbf_lrb_color = 'darksalmon'

### regular
cbf_tfp_ida = cbf_tfp_val_results[cbf_tfp_val_results['ida_level']==1.0]
mf_tfp_ida = mf_tfp_val_results[mf_tfp_val_results['ida_level']==1.0]
cbf_lrb_ida = cbf_lrb_val_results[cbf_lrb_val_results['ida_level']==1.0]
mf_lrb_ida = mf_lrb_val_results[mf_lrb_val_results['ida_level']==1.0]

mf_tfp_repl_cases = mf_tfp_ida[mf_tfp_ida['replacement_freq'] >= 0.99].shape[0]
cbf_tfp_repl_cases = cbf_tfp_ida[cbf_tfp_ida['replacement_freq'] >= 0.99].shape[0]
mf_lrb_repl_cases = mf_lrb_ida[mf_lrb_ida['replacement_freq'] >= 0.99].shape[0]
cbf_lrb_repl_cases = cbf_lrb_ida[cbf_lrb_ida['replacement_freq'] >= 0.99].shape[0]

print('MF-TFP runs requiring replacement:', mf_tfp_repl_cases)
print('MF-LRB runs requiring replacement:', mf_lrb_repl_cases)
print('CBF-TFP runs requiring replacement:', cbf_tfp_repl_cases)
print('CBF-LRB runs requiring replacement:', cbf_lrb_repl_cases)


df_dt = pd.DataFrame.from_dict(
    data=dict([('MF-TFP', mf_tfp_ida[cost_var]),
               ('MF-LRB', mf_lrb_ida[cost_var]),
               ('CBF-TFP', cbf_tfp_ida[cost_var]),
               ('CBF-LRB', cbf_lrb_ida[cost_var]),]),
    orient='index',
).T

import seaborn as sns

cbf_tfp_repair_cost = cbf_tfp_inv_pred['cost']
mf_tfp_repair_cost = mf_tfp_inv_pred['cost']
cbf_lrb_repair_cost = cbf_lrb_inv_pred['cost']
mf_lrb_repair_cost = mf_lrb_inv_pred['cost']

cbf_tfp_repair_cost_var = cbf_tfp_inv_pred['cost_var']
mf_tfp_repair_cost_var = mf_tfp_inv_pred['cost_var']
cbf_lrb_repair_cost_var = cbf_lrb_inv_pred['cost_var']
mf_lrb_repair_cost_var = mf_lrb_inv_pred['cost_var']



###### time

### regular
cbf_tfp_ida = cbf_tfp_val_results[cbf_tfp_val_results['ida_level']==1.0]
mf_tfp_ida = mf_tfp_val_results[mf_tfp_val_results['ida_level']==1.0]
cbf_lrb_ida = cbf_lrb_val_results[cbf_lrb_val_results['ida_level']==1.0]
mf_lrb_ida = mf_lrb_val_results[mf_lrb_val_results['ida_level']==1.0]

mf_tfp_repl_cases = mf_tfp_ida[mf_tfp_ida['replacement_freq'] >= 0.99].shape[0]
cbf_tfp_repl_cases = cbf_tfp_ida[cbf_tfp_ida['replacement_freq'] >= 0.99].shape[0]
mf_lrb_repl_cases = mf_lrb_ida[mf_lrb_ida['replacement_freq'] >= 0.99].shape[0]
cbf_lrb_repl_cases = cbf_lrb_ida[cbf_lrb_ida['replacement_freq'] >= 0.99].shape[0]

print('MF-TFP runs requiring replacement:', mf_tfp_repl_cases)
print('MF-LRB runs requiring replacement:', mf_lrb_repl_cases)
print('CBF-TFP runs requiring replacement:', cbf_tfp_repl_cases)
print('CBF-LRB runs requiring replacement:', cbf_lrb_repl_cases)


df_dt = pd.DataFrame.from_dict(
    data=dict([('MF-TFP', mf_tfp_ida[time_var]),
               ('MF-LRB', mf_lrb_ida[time_var]),
               ('CBF-TFP', cbf_tfp_ida[time_var]),
               ('CBF-LRB', cbf_lrb_ida[time_var]),]),
    orient='index',
).T

import seaborn as sns

cbf_tfp_repair_time = cbf_tfp_inv_pred['time']
mf_tfp_repair_time = mf_tfp_inv_pred['time']
cbf_lrb_repair_time = cbf_lrb_inv_pred['time']
mf_lrb_repair_time = mf_lrb_inv_pred['time']

cbf_tfp_repair_time_var = cbf_tfp_inv_pred['time_var']
mf_tfp_repair_time_var = mf_tfp_inv_pred['time_var']
cbf_lrb_repair_time_var = cbf_lrb_inv_pred['time_var']
mf_lrb_repair_time_var = mf_lrb_inv_pred['time_var']


ax=fig.add_subplot(1, 1, 1)
plt.setp(ax, xticks=np.arange(0.0, 1.1, step=0.1))
ax = sns.stripplot(data=df_dt, orient='h', palette='coolwarm', alpha=0.8, size=6,
                   linewidth=1.0)
meanpointprops = dict(marker='D', markeredgecolor='black', markersize=10,
                      markerfacecolor='gray')

sns.boxplot(data=df_dt, saturation=0.5, ax=ax, orient='h', palette='coolwarm',
            width=0.4, showmeans=True, meanprops=meanpointprops, meanline=False, showfliers=False)
ax.set_xlabel(r'Repair time ratio', fontsize=axis_font)
ax.axvline(0.2, color='black', linestyle=':', linewidth=2.0)

plot_predictions(mf_tfp_repair_time, mf_tfp_repair_time_var, 0.875, 0, mf_tfp_color)
plot_predictions(mf_lrb_repair_time, mf_lrb_repair_time_var, 0.625, 1, mf_lrb_color)
plot_predictions(cbf_tfp_repair_time, cbf_tfp_repair_time_var, 0.375, 2, cbf_tfp_color)
plot_predictions(cbf_lrb_repair_time, cbf_lrb_repair_time_var, 0.125, 3, cbf_lrb_color)


ax.text(0.3, 2.75, 'Targeted performance',
          fontsize=axis_font+4, color='black', bbox=dict(facecolor='white', edgecolor='black'))

ax.grid(visible=True)


import matplotlib.patches as mpatches
custom_lines = [mpatches.Patch(facecolor='dimgray', edgecolor='black'),
                Line2D([-1], [-1], color='white', marker='o', markeredgecolor='black',
                       alpha=0.3, markerfacecolor='lightgray', markersize=6),
                Line2D([-1], [-1], color='white', marker='D', markeredgecolor='black'
                       , markerfacecolor='gray', markersize=10),
                Line2D([-1], [-1], color='lightgray', marker='v', markeredgecolor='black',
                       linestyle='--' , markerfacecolor='lightgray', markersize=10)
                ]

ax.legend(custom_lines, ['IDA boxplot', 'IDA run median consequence', 'Mean of IDA runs', 
                         'Predicted consequence'], 
           fontsize=subt_font, loc='upper right')

fig.tight_layout()
plt.show()
plt.savefig('./poster_figure/time_distro.svg')

#%% filter design graphic

# TODO: filter

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

from matplotlib.lines import Line2D
fig = plt.figure(figsize=(7,6))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls_cost = np.array([0.2])
lvls_time = np.array([0.2])
lvls_repl = np.array([0.1])

lvls_cost_enhanced = np.array([0.1])
lvls_time_enhanced = np.array([0.1])
lvls_repl_enhanced = np.array([0.05])

res = 100
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.2, fourth_var_set = 0.24)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)



#### MF-LRB
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.35, fourth_var_set = 0.20)

ax = fig.add_subplot(1, 1, 1)
# plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_cost = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_cost_mf_lrb_i.gpr,
                       mdl_cost_mf_lrb_o.gpr,
                       outcome=cost_var)

Z = np.array(grid_cost)
Z_cost = Z.reshape(xx_pl.shape)

# clabels = ax.clabel(cs, fontsize=clabel_size)

grid_time = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_time_mf_lrb_i.gpr,
                       mdl_time_mf_lrb_o.gpr,
                       outcome=time_var)

Z = np.array(grid_time)
Z_time = Z.reshape(xx_pl.shape)

lvls = np.array([0.1])
grid_repl = predict_DV(X_plot,
                       mdl_impact_mf_lrb.gpc,
                       mdl_repl_mf_lrb_i.gpr,
                       mdl_repl_mf_lrb_o.gpr,
                       outcome=repl_var)

Z = np.array(grid_repl)
Z_repl = Z.reshape(xx_pl.shape)





cs = ax.contour(xx_pl, yy_pl, Z_repl, linewidths=2.0, 
                colors='black', linestyles='dotted', levels=lvls_repl)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_repl, colors='mistyrose', alpha=0.5, levels=[lvls_repl, 1.0])

cs = ax.contour(xx_pl, yy_pl, Z_time, linewidths=2.0, 
                colors='black', linestyles='dashed', levels=lvls_time)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_time, colors='lightpink', alpha=0.5, levels=[lvls_time, 1.0])

cs = ax.contour(xx_pl, yy_pl, Z_cost, linewidths=2.0, colors='black', levels=lvls_cost)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_cost, colors='orangered', alpha=0.5, levels=[lvls_cost, 1.0])

cs = ax.contour(xx_pl, yy_pl, Z_repl, linewidths=2.0, 
                colors='black', linestyles='dotted', levels=lvls_repl_enhanced)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_repl, colors='lightcyan', alpha=0.2, levels=[lvls_repl_enhanced, 1.0])

cs = ax.contour(xx_pl, yy_pl, Z_time, linewidths=2.0, 
                colors='black', linestyles='dashed', levels=lvls_time_enhanced)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_time, colors='powderblue', alpha=0.2, levels=[lvls_time_enhanced, 1.0])

cs = ax.contour(xx_pl, yy_pl, Z_cost, linewidths=2.0, colors='black', levels=lvls_cost_enhanced)
ax.clabel(cs, fontsize=clabel_size)
cs = ax.contourf(xx_pl, yy_pl, Z_cost, colors='lightskyblue', alpha=0.2, levels=[lvls_cost_enhanced, 1.0])

ax.grid(visible=True)



ax.axvline(1.0, linewidth=1.0, linestyle='--', color='red')
ax.axhline(2.0, linewidth=1.0, linestyle='--', color='red')
ax.text(0.55, 1.7, 'Stronger\n than code',
          fontsize=axis_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
ax.arrow(0.45, 2.0, 0.0, -0.15, width=0.02, facecolor='red', edgecolor='black')
ax.text(1.3, 1.1, 'More capacity \n than code',
          fontsize=axis_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
ax.arrow(1.0, 1.15, 0.15, 0.0, width=0.02, facecolor='red', edgecolor='black')
ax.set_xlim([0.35, 2.0])
ax.set_ylim([0.5, 2.25])

ax.grid(visible=True)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)



custom_lines = [Line2D([-1], [-1], color='black', 
                        linestyle='-' ),
                Line2D([-1], [-1], color='black', 
                                        linestyle='--' ),
                Line2D([-1], [-1], color='black', 
                                        linestyle=':' ),
                ]

ax.legend(custom_lines, ['Repair cost ratio', 'Repair time ratio', 'Replacement probability'], 
            fontsize=subt_font, loc='lower right')



fig.tight_layout()
plt.savefig('./poster_figure/filter_design.svg')
# ax.text(1.2, 1.00, 'OK space',
#           fontsize=axis_font, color='green')