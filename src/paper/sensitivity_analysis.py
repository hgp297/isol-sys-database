############################################################################
#               Sensitivity analysis

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: May 2024

# Description:  Specific plotting for sensitivity analysis

# Open issues:  (1) 

############################################################################

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from doe import GP

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

# hard-coded
def make_design_space(res):
    xx, yy, uu, vv = np.meshgrid(np.linspace(0.6, 1.5,
                                             res),
                                 np.linspace(0.5, 2.25,
                                             res),
                                 np.linspace(2.0, 5.0,
                                             res),
                                 0.15)
                                 
    X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                         'RI':yy.ravel(),
                         'T_ratio':uu.ravel(),
                         'zeta_e':vv.ravel()})

    return(X_space)

#%% Calculate upfront cost of data

def get_steel_coefs(df, steel_per_unit=1.25):
    n_bays = df.num_bays
    n_stories = df.num_stories
    # ft
    L_bldg = df.L_bldg
    L_beam = df.L_bay
    h_story = df.h_story
    
    # weights
    W = df.W
    Ws = df.W_s
    
    
    all_beams = df.beam
    all_cols = df.column
    
    # sum of per-length-weight of all floors
    col_wt = [[float(member.split('X',1)[1]) for member in col_list] 
                       for col_list in all_cols]
    beam_wt = [[float(member.split('X',1)[1]) for member in beam_list] 
                       for beam_list in all_beams]
    col_all_wt = np.array(list(map(sum, col_wt)))
    beam_all_wt = np.array(list(map(sum, beam_wt)))
    
    # find true steel costs
    n_frames = 4
    n_cols = 4*n_bays
    
    total_floor_col_length = np.array(n_cols*h_story, dtype=float)
    total_floor_beam_length = np.array(L_beam * n_bays * n_frames, dtype=float)
        
    total_col_wt = col_all_wt*total_floor_col_length 
    total_beam_wt = beam_all_wt*total_floor_beam_length
    
    bldg_wt = total_col_wt + total_beam_wt
    
    steel_cost = steel_per_unit*bldg_wt
    bldg_sf = np.array(n_stories * L_bldg**2, dtype=float)
    steel_cost_per_sf = steel_cost/bldg_sf
    
    # find design base shear as a feature
    pi = 3.14159
    g = 386.4
    kM = (1/g)*(2*pi/df['T_m'])**2
    S1 = 1.017
    Dm = g*S1*df['T_m']/(4*pi**2*df['Bm'])
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*df['zeta_e'])
    Vs = np.array(Vst/df['RI']).reshape(-1,1)
    
    # linear regress cost as f(base shear)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X=Vs, y=steel_cost_per_sf)
    return({'coef':reg.coef_, 'intercept':reg.intercept_})

def calc_upfront_cost(X_test, steel_coefs,
                      land_cost_per_sqft=2837/(3.28**2),
                      W=3037.5, Ws=2227.5):
    
    from scipy.interpolate import interp1d
    zeta_ref = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Bm_ref = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    interp_f = interp1d(zeta_ref, Bm_ref)
    Bm = interp_f(X_test['zeta_e'])
    
    # estimate Tm
    
    from loads import estimate_period
    
    # current dummy structure: 4 bays, 4 stories
    # 13 ft stories, 30 ft bays
    X_query = X_test.copy()
    X_query['superstructure_system'] = 'MF'
    X_query['h_bldg'] = 4*13.0
    X_query['T_fbe'] = X_query.apply(lambda row: estimate_period(row),
                                                     axis='columns', result_type='expand')
    
    X_query['T_m'] = X_query['T_fbe'] * X_query['T_ratio']
    
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    S1 = 1.017
    SaTm = S1/X_query['T_m']
    moat_gap = X_query['gap_ratio'] * (g*(SaTm/Bm)*X_query['T_m']**2)/(4*pi**2)
    
    # calculate design base shear
    kM = (1/g)*(2*pi/X_query['T_m'])**2
    Dm = g*S1*X_query['T_m']/(4*pi**2*Bm)
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*X_query['zeta_e'])
    Vs = Vst/X_query['RI']
    
    # steel coefs now represent cost/sf as a function of Vs
    steel_cost_per_sf = steel_coefs['intercept'] + steel_coefs['coef']*Vs
    # land_area = 2*(90.0*12.0)*moat_gap - moat_gap**2
    
    bldg_area = 4 * (30*4)**2
    steel_cost = steel_cost_per_sf * bldg_area
    land_area = (4*30*12.0 + moat_gap)**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return({'total': steel_cost + land_cost,
            'steel': steel_cost,
            'land': land_cost})

#%% pareto
def simple_cull_df(input_df, dominates, *args):
    pareto_df = pd.DataFrame(columns=input_df.columns)
    candidateRowNr = 0
    dominated_df = pd.DataFrame(columns=input_df.columns)
    while True:
        candidateRow = input_df.iloc[[candidateRowNr]]
        input_df = input_df.drop(index=candidateRow.index)
        rowNr = 0
        nonDominated = True
        while input_df.shape[0] != 0 and rowNr < input_df.shape[0]:
            row = input_df.iloc[[rowNr]]
            if dominates(candidateRow, row, *args):
                # If it is worse on all features remove the row from the array
                input_df = input_df.drop(index=row.index)
                dominated_df = dominated_df.append(row)
            elif dominates(row, candidateRow, *args):
                nonDominated = False
                dominated_df = dominated_df.append(candidateRow)
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            pareto_df = pareto_df.append(candidateRow)

        if input_df.shape[0] == 0:
            break
    return pareto_df, dominated_df

def dominates_pd(row, candidate_row, mdl, cost, coefs):
    row_pr = mdl.predict(row).item()
    cand_pr = mdl.predict(candidate_row).item()
    
    row_cost = cost(row, coefs).item()
    cand_cost = cost(candidate_row, coefs).item()
    
    return ((row_pr < cand_pr) and (row_cost < cand_cost))

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
#%% doe data set GP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 20
import matplotlib as mpl
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
clabel_size = 16

main_obj_doe = pd.read_pickle('../../data/tfp_mf_db_doe_prestrat.pickle')

# with open("../../data/tfp_mf_db_doe.pickle", 'rb') as picklefile:
#     main_obj_doe = pickle.load(picklefile)



collapse_drift_def_mu_std = 0.1
df_doe = main_obj_doe.doe_analysis

#%%

df_doe['max_drift'] = df_doe.PID.apply(max)
df_doe['log_drift'] = np.log(df_doe['max_drift'])

df_doe['max_velo'] = df_doe.PFV.apply(max)
df_doe['max_accel'] = df_doe.PFA.apply(max)

df_doe['T_ratio'] = df_doe['T_m'] / df_doe['T_fb']
df_doe['T_ratio_e'] = df_doe['T_m'] / df_doe['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_doe['Bm'] = np.interp(df_doe['zeta_e'], zetaRef, BmRef)

df_doe['gap_ratio'] = (df_doe['constructed_moat']*4*pi**2)/ \
    (g*(df_doe['sa_tm']/df_doe['Bm'])*df_doe['T_m']**2)

from loads import define_gravity_loads

config_dict = {
    'S_1' : 1.017,
    'k_ratio' : 10,
    'Q': 0.06,
    'num_frames' : 2,
    'num_bays' : 4,
    'num_stories' : 4,
    'L_bay': 30.0,
    'h_story': 13.0,
    'isolator_system' : 'TFP',
    'superstructure_system' : 'MF',
    'S_s' : 2.2815
}
(W_seis, W_super, w_on_frame, P_on_leaning_column,
       all_w_cases, all_plc_cases) = define_gravity_loads(config_dict)

#%% sensitivity to collapse definitions
theta_mu_stds = np.arange(0.03, 0.1, 0.01)


covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

kernel_name = 'rbf_iso'

steel_price = 4.0
coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)

design_res = 20
X_design_cand = make_design_space(design_res)

collapse_sens_10 = np.zeros((len(theta_mu_stds), len(covariate_list)))
collapse_sens_5 = np.zeros((len(theta_mu_stds), len(covariate_list)))
collapse_sens_2_5 = np.zeros((len(theta_mu_stds), len(covariate_list)))

for sens_idx, theta in enumerate(theta_mu_stds):
    
    print('Sensitivity analysis for reference drift:', theta)
    from experiment import collapse_fragility
    df_doe[['max_drift',
        'collapse_prob']] = df_doe.apply(
            lambda row: collapse_fragility(row, drift_at_mu_plus_std=theta), 
            axis='columns', result_type='expand')
           
    mdl_doe = GP(df_doe)
    mdl_doe.set_covariates(covariate_list)
    mdl_doe.set_outcome('collapse_prob')
    mdl_doe.fit_gpr(kernel_name=kernel_name)
    
    fmu_design = mdl_doe.gpr.predict(X_design_cand, return_std=False)
    
    risk_thresh = 0.1
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]


    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_10 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.05
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_5 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.025
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_2_5 = ok_risk.loc[cheapest_design_idx]
    
    collapse_sens_10[sens_idx,:] = np.array(design_10).T
    collapse_sens_5[sens_idx,:] = np.array(design_5).T
    collapse_sens_2_5[sens_idx,:] = np.array(design_2_5).T
    
    print('Done!')
    
#%% sensitivity to steel cost

drift_mu_std = 0.06
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
kernel_name = 'rbf_iso'

df_doe[['max_drift',
    'collapse_prob']] = df_doe.apply(
        lambda row: collapse_fragility(row, drift_at_mu_plus_std=drift_mu_std), 
        axis='columns', result_type='expand')
       
mdl_doe = GP(df_doe)
mdl_doe.set_covariates(covariate_list)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name=kernel_name)

fmu_design = mdl_doe.gpr.predict(X_design_cand, return_std=False)

steel_prices = np.arange(1.0, 10.0, 1.0)

steel_sens_10 = np.zeros((len(steel_prices), len(covariate_list)))
steel_sens_5 = np.zeros((len(steel_prices), len(covariate_list)))
steel_sens_2_5 = np.zeros((len(steel_prices), len(covariate_list)))

for steel_idx, steel_price in enumerate(steel_prices):
    
    print('Sensitivity analysis for steel price:', steel_price)
    coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)
    
    risk_thresh = 0.1
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]


    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_10 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.05
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_5 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.025
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_2_5 = ok_risk.loc[cheapest_design_idx]
    
    steel_sens_10[steel_idx,:] = np.array(design_10).T
    steel_sens_5[steel_idx,:] = np.array(design_5).T
    steel_sens_2_5[steel_idx,:] = np.array(design_2_5).T
    
    print('Done!')
    
#%% sensitivity to land cost

drift_mu_std = 0.06
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
kernel_name = 'rbf_iso'

df_doe[['max_drift',
    'collapse_prob']] = df_doe.apply(
        lambda row: collapse_fragility(row, drift_at_mu_plus_std=drift_mu_std), 
        axis='columns', result_type='expand')
       
mdl_doe = GP(df_doe)
mdl_doe.set_covariates(covariate_list)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name=kernel_name)

fmu_design = mdl_doe.gpr.predict(X_design_cand, return_std=False)



land_prices = np.arange(50.0, 400.0, 50.0)

land_sens_10 = np.zeros((len(land_prices), len(covariate_list)))
land_sens_5 = np.zeros((len(land_prices), len(covariate_list)))
land_sens_2_5 = np.zeros((len(land_prices), len(covariate_list)))

coef_dict = get_steel_coefs(df_doe, steel_per_unit=4.0)
# baseline is $250/sqft
for land_idx, land_price in enumerate(land_prices):
    
    print('Sensitivity analysis for land price:', land_price)
    
    risk_thresh = 0.1
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]


    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, land_cost_per_sqft=land_price,
                                      W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_10 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.05
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, land_cost_per_sqft=land_price,
                                      W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_5 = ok_risk.loc[cheapest_design_idx]

    risk_thresh = 0.025
    space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
    ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                          risk_thresh]

    upfront_costs = calc_upfront_cost(ok_risk, coef_dict, land_cost_per_sqft=land_price,
                                      W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    # least upfront cost of the viable designs
    design_2_5 = ok_risk.loc[cheapest_design_idx]
    
    land_sens_10[land_idx,:] = np.array(design_10).T
    land_sens_5[land_idx,:] = np.array(design_5).T
    land_sens_2_5[land_idx,:] = np.array(design_2_5).T
    
    print('Done!')
    
#%%
plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(9, 14))
ax=fig.add_subplot(3, 3, 1)

from numpy import log, exp
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(np.array(theta_mu_stds)) - beta_drift*inv_norm)

color = plt.cm.Set1(np.linspace(0, 1, 10))

baseline_ref = np.array([1.0, 2.0, 3.0, 0.15])

plt_array = collapse_sens_10/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(mean_log_drift, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])

ax.set_xlabel(r'$\theta$', fontsize=axis_font)
ax.set_ylabel(r'\% change', fontsize=axis_font)
ax.set_title('10\% collapse', fontsize=title_font)
ax.set_ylim([25, 200])
ax.legend(fontsize=label_size)
ax.grid()

ax=fig.add_subplot(3, 3, 2)
plt_array = collapse_sens_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(mean_log_drift, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'$\theta$', fontsize=axis_font)
ax.set_title('5\% collapse', fontsize=title_font)
ax.set_ylim([25, 200])
ax.grid()

ax=fig.add_subplot(3, 3, 3)
plt_array = collapse_sens_2_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(mean_log_drift, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'$\theta$', fontsize=axis_font)
ax.set_title('2.5\% collapse', fontsize=title_font)
ax.set_ylim([25, 200])
ax.grid()



##

ax=fig.add_subplot(3, 3, 4)
plt_array = land_sens_10/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(land_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_ylabel(r'\% change', fontsize=axis_font)
ax.set_xlabel(r'Land cost per ft$^2$ (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.legend(fontsize=label_size)
ax.grid()

ax=fig.add_subplot(3, 3, 5)
plt_array = land_sens_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(land_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Land cost per ft$^2$ (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.grid()

ax=fig.add_subplot(3, 3, 6)
plt_array = land_sens_2_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(land_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Land cost per ft$^2$ (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.grid()

##

ax=fig.add_subplot(3, 3, 7)
plt_array = steel_sens_10/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(steel_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_ylabel(r'\% change', fontsize=axis_font)
ax.set_xlabel(r'Steel cost per lb (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.legend(fontsize=label_size)
ax.grid()

ax=fig.add_subplot(3, 3, 8)
plt_array = steel_sens_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(steel_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Steel cost per lb (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.grid()

ax=fig.add_subplot(3, 3, 9)
plt_array = steel_sens_2_5/baseline_ref*100
for plt_idx in range(3):
    ax.plot(steel_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Steel cost per lb (\$)', fontsize=axis_font)
ax.set_ylim([25, 200])
ax.grid()

fig.tight_layout()
# plt.savefig('./figures/sensitivity.pdf')