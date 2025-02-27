############################################################################
#               Inverse design functions

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2025

# Description:  Includes many functions for inverse design, like cost calculations
#               and preprocessing

############################################################################
import pandas as pd
import numpy as np

def preprocess_data(main_obj, max_obj):
    
        
    main_obj.calculate_collapse()

    df_raw = main_obj.ops_analysis
    df = df_raw.reset_index(drop=True)


    df['max_isol_disp'] = pd.to_numeric(df['max_isol_disp'])
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

    df['bldg_area'] = (df['num_bays']*df['L_bay'])**2 * (df['num_stories'] + 1)

    df_loss = main_obj.loss_data
    df_loss_max = max_obj.max_loss
    
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
    df['sa_tm_shift'] = pd.to_numeric(df['sa_tm_adj']/df['sa_tm'])
    
    df = loss_percentages(df, df_loss, df_loss_max)

    db_string = '../../resource/'
    brace_db = pd.read_csv(db_string+'braceShapes.csv', index_col=None, header=0)  

    df['steel_cost'] = df.apply(
           lambda row: calc_steel_cost(
               row, brace_db=brace_db,
               steel_per_unit=1.25),
           axis='columns', result_type='expand')

    df['steel_cost_per_sf'] = df['steel_cost'] / df['bldg_area']

    df['system'] = df['superstructure_system'] +'-' + df['isolator_system']
    
    return(df)
 
#%% helper functions
# live load calculation   
def wL_calc(num_stories, bldg_area):
    
    # in kip/ft^2
    LL_avg = (num_stories*50.0/1000 + 20.0/1000)/(num_stories + 1)
    W_L = LL_avg*bldg_area
    return W_L

# period shift of TFP due to additional live loading
def TFP_period_shift(W_D, W_L, bearing):
    # shift assumes that the load case is 1.0D + 0.5L
    if bearing == 'TFP':
        return (W_D/(W_D+0.5*W_L))**0.5
    else:
        return 1.0
 
#%% 
# frame costs
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
    
#%% loss DVs
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

#%% main DV predictor

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

#%% inverse designer

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
        
        X_space = pd.DataFrame({var_list[0]:xx.ravel(),
                             var_list[1]:yy.ravel(),
                             var_list[2]:uu.ravel(),
                             fixed_var:vv.ravel()})
    return(X_space)

def grid_search_inverse_design(res, system_name, targets_dict, config_dict, reg_dict,
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
    structural_system = system_name.split('_')[0]
    
    if structural_system.upper() == 'MF':
        upfront_costs = calc_upfront_cost(
            X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
    else:
        upfront_costs = calc_upfront_cost(
            X_design, config_dict=config_dict, steel_cost_dict=reg_dict,
            land_cost_per_sqft=1978/(3.28**2))
        
    # upfront_costs = calc_upfront_cost(
    #     X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
    
    # compare = X_design.copy()
    # compare['cost'] = upfront_costs['total_'+structural_system]
    # compare['repair'] = space_repair_cost[space_repair_cost.index.isin(X_design.index)]
    
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