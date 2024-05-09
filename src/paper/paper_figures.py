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
from doe import GP

plt.close('all')

main_obj = pd.read_pickle("../../data/tfp_mf_db.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df_raw = main_obj.ops_analysis

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

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
                                 np.linspace(0.1, 0.25,
                                             res))
                                 
    X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                         'RI':yy.ravel(),
                         'T_ratio':uu.ravel(),
                         'zeta_e':vv.ravel()})

    return(X_space)

#%% collapse fragility def

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

collapse_drift_def_mu_std = 0.1


from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(collapse_drift_def_mu_std) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
# mean_log_drift = 0.05
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

label_size = 16
clabel_size = 12
x = np.linspace(0, 0.15, 200)

mu = log(mean_log_drift)

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
p = ln_dist.cdf(np.array(x))


fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(x, p, label='Collapse (peak)', color='blue')

mu_irr = log(0.01)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

# ax.plot(x, p_irr, color='red', label='Irreparable (residual)')

axis_font = 20
subt_font = 18
xright = 0.0
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, xleft])
ax.set_ylabel('Collapse probability', fontsize=axis_font)
ax.set_xlabel('Peak drift ratio', fontsize=axis_font)

ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
ax.hlines(y=0.5, xmin=xright, xmax=exp(mu), color='blue', linestyle=":")
ax.text(0.01, 0.52, r'$\theta = %.3f$'% mean_log_drift , fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

upper = ln_dist.ppf(0.84)
ax.vlines(x=upper, ymin=0, ymax=0.84, color='blue', linestyle=":")
ax.hlines(y=0.84, xmin=xright, xmax=upper, color='blue', linestyle=":")
ax.text(0.01, 0.87, r'$\theta = %.3f$' % upper, fontsize=axis_font, color='blue')
ax.plot([upper], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.16)
ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
ax.hlines(y=0.16, xmin=xright, xmax=lower, color='blue', linestyle=":")
ax.text(0.01, 0.19, r'$\theta = %.3f$' % lower, fontsize=axis_font, color='blue')
ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


# ax.set_title('Replacement fragility definition', fontsize=axis_font)
ax.grid()
# ax.legend(fontsize=label_size, loc='upper center')
# plt.show()

#%% base-set data
'''
kernel_name = 'rbf_ard'

mdl = GP(df)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl.set_covariates(covariate_list)
mdl.set_outcome('collapse_prob')
mdl.fit_gpr(kernel_name=kernel_name)


res = 75

X_plot = make_2D_plotting_space(mdl.X, res, x_var='T_ratio', y_var='zeta_e', 
                           all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                           third_var_set = 1.0, fourth_var_set=2.0)

import time
t0 = time.time()

fmu_base, fs1_base = mdl.gpr.predict(X_plot, return_std=True)
fs2_base = fs1_base**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_plot.shape[0],
                                                               tp))

#%% base-set, Tm_zeta plot
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_var = 'T_ratio'
xx = X_plot[x_var]
y_var = 'zeta_e'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z = fmu_base.reshape(xx_pl.shape)

plt.figure(figsize=(8,6))
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df['T_ratio'], df['zeta_e'], 
            c=df['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
# plt.xlim([0.5,2.0])
# plt.ylim([0.5, 2.25])
plt.xlabel('T ratio', fontsize=axis_font)
plt.ylabel(r'$\zeta_e$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using full 400 points', fontsize=axis_font)
plt.show()

#%%
X_plot = make_2D_plotting_space(mdl.X, res, 
                                third_var_set=3.0, fourth_var_set=0.2)

import time
t0 = time.time()

fmu_base, fs1_base = mdl.gpr.predict(X_plot, return_std=True)
fs2_base = fs1_base**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_plot.shape[0],
                                                               tp))

#%% base-set, gap_Ry plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z = fmu_base.reshape(xx_pl.shape)

plt.figure(figsize=(8,6))
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df['gap_ratio'], df['RI'], 
            c=df['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.5,2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using full 400 points', fontsize=axis_font)
plt.show()

#%% base-set, gap_Ry plot, kernel ridge

mdl.fit_kernel_ridge()
fmu_base = mdl.kr.predict(X_plot).ravel()


#%% 10% design

X_design_cand = make_design_space(25)

X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
                          columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
baseline_risk, baseline_fs1 = mdl.gpr.predict(X_baseline, return_std=True)
baseline_risk = baseline_risk.item()
baseline_fs2 = baseline_fs1**2
baseline_fs1 = baseline_fs1.item()
baseline_fs2 = baseline_fs2.item()

t0 = time.time()
fmu_design, fs1_design = mdl.gpr.predict(X_design_cand, return_std=True)
fs2_design = fs1_design**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_design_cand.shape[0],
                                                                tp))


'''
#%% Calculate upfront cost of data

# TODO: normalize cost for building size

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
    compare = pd.DataFrame([steel_cost_per_sf, df['RI']])
    
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

# TODO: add economy of scale for land
# TODO: investigate upfront cost's influence by Tm
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
    
    # TODO: formalize this for all structures
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
    
    # TODO: change if generalized building
    bldg_area = 4 * (30*4)**2
    steel_cost = steel_cost_per_sf * bldg_area
    land_area = (4*30*12.0 + moat_gap)**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return({'total': steel_cost + land_cost,
            'steel': steel_cost,
            'land': land_cost})
   
''' 
#%% baseline & prediction from 400-base-set


risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

# get_structural_cmp_MF(df, metadata)
steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

baseline_costs = calc_upfront_cost(X_baseline, coef_dict)
baseline_total = baseline_costs['total'].item()
baseline_steel = baseline_costs['steel'].item()
baseline_land = baseline_costs['land'].item()

# least upfront cost of the viable designs

print('========== Baseline design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${baseline_total:,.2f}')
print('Predicted collapse risk: ',
      f'{baseline_risk:.2%}')
print(X_baseline)

import warnings
warnings.filterwarnings('ignore')

upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()
design_steel_cost = upfront_costs['steel'][cheapest_design_idx]
design_land_cost = upfront_costs['land'][cheapest_design_idx]
# least upfront cost of the viable designs
best_design = ok_risk.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']
warnings.resetwarnings()

print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Steel cost of selected design, ',
      f'${design_steel_cost:,.2f}')
print('Land cost of selected design, ',
      f'${design_land_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)

# some issues that may lead to poor design
# data is lacking in T_ratio and zeta realm
# be careful if the considered design space falls outside of the 
# available data space (model reverts to 0)

'''
#%%
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
    
'''
all_costs = calc_upfront_cost(X_design_cand, coef_dict)
constr_costs = all_costs['total']
predicted_risk = space_collapse_pred['collapse probability']
pareto_array = np.array([constr_costs, predicted_risk]).transpose()

t0 = time.time()
pareto_mask = is_pareto_efficient(pareto_array)
tp = time.time() - t0

print("Culled %d points in %.3f s" % (X_design_cand.shape[0], tp))

X_pareto = X_design_cand.iloc[pareto_mask]
risk_pareto = predicted_risk.iloc[pareto_mask]
cost_pareto = constr_costs.iloc[pareto_mask]

plt.figure(figsize=(8,6))
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
plt.scatter(risk_pareto, cost_pareto, 
            edgecolors='k', s=20.0)
plt.xlabel('Collapse risk', fontsize=axis_font)
plt.ylabel('Construction cost', fontsize=axis_font)
plt.grid(True)
plt.title('Pareto front', fontsize=axis_font)
plt.show()


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fig = plt.figure(figsize=(14,6))
ax1=fig.add_subplot(1, 2, 1)
ax1.scatter(X_pareto['gap_ratio'], X_pareto['RI'], 
            c=risk_pareto,
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
ax1.set_xlim([0.5,2.0])
ax1.set_ylim([0.45, 2.3])
ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.grid(True)
ax1.set_title('Pareto efficient designs', fontsize=axis_font)

ax2=fig.add_subplot(1, 2, 2)
ax2.scatter(X_pareto['T_ratio'], X_pareto['zeta_e'], 
            c=risk_pareto,
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
ax2.set_xlabel('T ratio', fontsize=axis_font)
ax2.set_ylabel(r'$\zeta_e$', fontsize=axis_font)
ax2.grid(True)
ax2.set_title('Pareto efficient designs', fontsize=axis_font)
fig.tight_layout()
'''

###############################################################################
# DOE
###############################################################################

#%% doe data set GP
import time

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 20
import matplotlib as mpl
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
clabel_size = 16

main_obj_doe = pd.read_pickle('../../data/tfp_mf_db_doe.pickle')

# with open("../../data/tfp_mf_db_doe.pickle", 'rb') as picklefile:
#     main_obj_doe = pickle.load(picklefile)

kernel_name = 'rbf_iso'

collapse_drift_def_mu_std = 0.1
df_doe = main_obj_doe.doe_analysis

#%%
from experiment import collapse_fragility
df_doe[['max_drift',
   'collapse_prob']] = df_doe.apply(
       lambda row: collapse_fragility(row, drift_at_mu_plus_std=collapse_drift_def_mu_std), 
       axis='columns', result_type='expand')



# df_doe = df_doe.drop(columns=['index'])

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

    
print('% impact of DoE set:', sum(df_doe['impacted'])/df_doe.shape[0])
print('average drift:', df_doe['max_drift'].mean())

df_init = df_doe.head(50)

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

#%% doe convergence plots

rmse_hist = main_obj_doe.rmse_hist
mae_hist = main_obj_doe.mae_hist
nrmse_hist = main_obj_doe.nrmse_hist
theta = main_obj_doe.hyperparam_list


if kernel_name == 'rbf_iso':
    theta_vars = [r'$\kappa$', r'$\ell$', r'$\sigma_n^2$']
else:
    theta_vars = [r'$\kappa$', r'$\ell_{GR}$', r'$\ell_{R_y}$', 
                  r'$\ell_{T}$', r'$\ell_{\zeta}$' , r'$\sigma_n^2$']

plt.close('all')
fig = plt.figure(figsize=(16, 6))
batch_size = 5

ax1=fig.add_subplot(1, 3, 1)
ax1.plot(np.arange(0, (len(rmse_hist))*batch_size, batch_size), rmse_hist)
ax1.set_title(r'RMSE on test set', fontsize=axis_font)
ax1.set_xlabel(r'Points added', fontsize=axis_font)
ax1.set_ylabel(r'Error metric', fontsize=axis_font)
# ax1.set_xlim([0, 140])
# ax1.set_ylim([0.19, 0.28])
ax1.grid(True)


ax2=fig.add_subplot(1, 3, 2)
ax2.plot(np.arange(0, (len(rmse_hist))*batch_size, batch_size), nrmse_hist)
ax2.set_title('NRMSE-LOOCV of training set', fontsize=axis_font)
ax2.set_xlabel('Points added', fontsize=axis_font)
ax2.grid(True)

from numpy import exp
all_hyperparams = exp(theta)
hyperparam_norm = all_hyperparams / all_hyperparams.max(axis=0)
ax3=fig.add_subplot(1, 3, 3)
for param_idx in range(all_hyperparams.shape[1]):
    ax3.plot(np.arange(0, (len(rmse_hist))*batch_size, batch_size), 
             hyperparam_norm[:,param_idx], label=theta_vars[param_idx])
ax3.legend()
ax3.set_title('Normalized hyperparameter convergence', fontsize=axis_font)
ax3.set_xlabel('Points added', fontsize=axis_font)
ax3.grid(True)
fig.tight_layout()
plt.savefig('./figures/convergence.eps')

#%%  dumb scatters

# plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

y_var = 'max_drift'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)

cmap = plt.cm.coolwarm
sc = ax1.scatter(df_doe['gap_ratio'], df_doe[y_var], alpha=0.2, c=df_doe['impacted'], cmap=cmap)
ax1.set_ylabel('Peak story drift', fontsize=axis_font)
ax1.set_xlabel(r'GR', fontsize=axis_font)
ax1.set_title('a) Gap ratio', fontsize=title_font)
ax1.set_ylim([0, 0.15])

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', label='No impact',
                          markerfacecolor=cmap(0.), alpha=0.4, markersize=15),
                Line2D([0], [0], marker='o', color='w', label='Wall impact',
                       markerfacecolor=cmap(1.), alpha=0.4, markersize=15)]
ax1.legend(custom_lines, ['No impact', 'Impact'], fontsize=subt_font)
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df_doe['RI'], df_doe[y_var], alpha=0.3, c=df_doe['impacted'], cmap=cmap)
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('b) Superstructure strength', fontsize=title_font)
ax2.set_ylim([0, 0.15])
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df_doe['T_ratio'], df_doe[y_var], alpha=0.3, c=df_doe['impacted'], cmap=cmap)
# ax3.scatter(df_doe['T_ratio_e'], df_doe[y_var])
ax3.set_ylabel('Peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('c) Bearing period ratio', fontsize=title_font)
ax3.set_ylim([0, 0.15])
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df_doe['zeta_e'], df_doe[y_var], alpha=0.3, c=df_doe['impacted'], cmap=cmap)
ax4.set_xlabel(r'$\zeta_e$', fontsize=axis_font)
ax4.set_title('d) Bearing damping', fontsize=title_font)
ax4.set_ylim([0, 0.15])
ax4.grid(True)

fig.tight_layout()
plt.show()
plt.savefig('./figures/scatter.pdf')

#%%

fig = plt.figure(figsize=(13, 6))

ax1=fig.add_subplot(1, 2, 1)

cmap = plt.cm.coolwarm
sc = ax1.scatter(df_doe['moat_ampli'], df_doe['gap_ratio'], alpha=0.2)
ax1.plot([0, 5], [0, 5])
ax1.set_ylabel(r'True gap ratio', fontsize=axis_font)
ax1.set_xlabel(r'Constructed moat gap / $D_M$', fontsize=axis_font)
ax1.set_title('a) Gap ratio', fontsize=title_font)
ax1.set_xlim([0, 5])
ax1.grid(True)

ax2=fig.add_subplot(1, 2, 2)

ax2.scatter(df_doe['T_ratio_e'], df_doe['T_ratio'], alpha=0.3)
ax2.plot([0, 5], [0, 5])
ax2.set_xlabel(r'$T_M / T_{fbe}$', fontsize=axis_font)
ax2.set_ylabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax2.set_title('b) Period ratio', fontsize=title_font)
ax2.set_xlim([1, 6])
ax2.grid(True)
plt.show()

#%% seaborn scatter with histogram: DoE data
def scatter_hist(x, y, c, alpha, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    cmap = plt.cm.Blues
    ax.scatter(x, y, alpha=alpha, edgecolors='black', s=25, facecolors=c)

    # now determine nice limits by hand:
    binwidth = 0.25
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
    
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# Start with a square Figure.
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
scatter_hist(df_init['gap_ratio'], df_init['RI'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df_doe['gap_ratio'], df_doe['RI'], 'orange', 0.3, ax, ax_histx, ax_histy)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_xlim([0.0, 4.0])
ax.set_ylim([0.5, 2.25])

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_init['T_ratio'], df_init['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df_doe['T_ratio'], df_doe['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlim([1.25, 5.0])
ax.set_ylim([0.1, 0.25])


# ax = fig.add_subplot(gs[3, 0])
# ax_histx = fig.add_subplot(gs[2, 0], sharex=ax)
# ax_histy = fig.add_subplot(gs[3, 1], sharey=ax)
# # Draw the scatter plot and marginals.
# scatter_hist(df_doe['gap_ratio'], df_doe['RI'], df_doe['collapse_prob'], ax, ax_histx, ax_histy)
# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# # ax.set_xlim([0.4, 4.0])
# # ax.set_ylim([0.45, 2.4])

# ax = fig.add_subplot(gs[3, 2])
# ax_histx = fig.add_subplot(gs[2, 2], sharex=ax)
# ax_histy = fig.add_subplot(gs[3, 3], sharey=ax)
# # Draw the scatter plot and marginals.
# scatter_hist(df_doe['T_ratio'], df_doe['zeta_e'], df_doe['collapse_prob'], ax, ax_histx, ax_histy)
# ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
# ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

plt.savefig('./figures/doe_hist.pdf')

# ax.set_xlim([0.5, 4.0])
# ax.set_ylim([0.5, 2.25])
#%% DoE GP


mdl_doe = GP(df_doe)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_doe.set_covariates(covariate_list)
mdl_doe.set_outcome('collapse_prob')

mdl_doe.fit_gpr(kernel_name=kernel_name)

X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 2.0, 0.15]]),
                          columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])


baseline_risk, baseline_fs1 = mdl_doe.gpr.predict(X_baseline, return_std=True)
baseline_risk = baseline_risk.item()
baseline_fs2 = baseline_fs1**2
baseline_fs1 = baseline_fs1.item()
baseline_fs2 = baseline_fs2.item()

steel_price = 4.0
coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)
baseline_costs = calc_upfront_cost(X_baseline, coef_dict, W=W_seis, Ws=W_super)

baseline_total = baseline_costs['total'].item()
baseline_steel = baseline_costs['steel'].item()
baseline_land = baseline_costs['land'].item()

risk_thresh = 0.1

print('========== Baseline design (DoE) ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${baseline_total:,.2f}')
print('Predicted collapse risk: ',
      f'{baseline_risk:.2%}')
print(X_baseline)

design_res = 20
X_design_cand = make_design_space(design_res)


t0 = time.time()
fmu_design, fs1_design = mdl_doe.gpr.predict(X_design_cand, return_std=True)
fs2_design = fs1_design**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_design_cand.shape[0],
                                                                tp))


risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

import warnings
warnings.filterwarnings('ignore')

upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()
design_steel_cost = upfront_costs['steel'][cheapest_design_idx]
design_land_cost = upfront_costs['land'][cheapest_design_idx]
# least upfront cost of the viable designs
best_design = ok_risk.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']
warnings.resetwarnings()

print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Steel cost of selected design, ',
      f'${design_steel_cost:,.2f}')
print('Land cost of selected design, ',
      f'${design_land_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)


risk_thresh = 0.05
space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

import warnings
warnings.filterwarnings('ignore')

upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()
design_steel_cost = upfront_costs['steel'][cheapest_design_idx]
design_land_cost = upfront_costs['land'][cheapest_design_idx]
# least upfront cost of the viable designs
best_design = ok_risk.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']
warnings.resetwarnings()

print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Steel cost of selected design, ',
      f'${design_steel_cost:,.2f}')
print('Land cost of selected design, ',
      f'${design_land_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)


risk_thresh = 0.025
space_collapse_pred = pd.DataFrame(fmu_design, columns=['collapse probability'])
ok_risk = X_design_cand.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

import warnings
warnings.filterwarnings('ignore')

upfront_costs = calc_upfront_cost(ok_risk, coef_dict, W=W_seis, Ws=W_super)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()
design_steel_cost = upfront_costs['steel'][cheapest_design_idx]
design_land_cost = upfront_costs['land'][cheapest_design_idx]
# least upfront cost of the viable designs
best_design = ok_risk.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']
warnings.resetwarnings()

print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Steel cost of selected design, ',
      f'${design_steel_cost:,.2f}')
print('Land cost of selected design, ',
      f'${design_land_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)



#%% doe-set, Tm_zeta plot

res = 75
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var='T_ratio', y_var='zeta_e', 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

import time
t0 = time.time()

fmu_doe, fs1_doe = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_doe = fs1_doe**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_plot.shape[0],
                                                                tp))


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_var = 'T_ratio'
xx = X_plot[x_var]
y_var = 'zeta_e'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_Tze = fmu_doe.reshape(xx_pl.shape)

plt.figure(figsize=(8,6))
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z_Tze, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_doe['T_ratio'], df_doe['zeta_e'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
# plt.xlim([2.0,5.0])
plt.ylim([0.1, 0.25])
plt.xlabel('T ratio', fontsize=axis_font)
plt.ylabel(r'$\zeta_e$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using post DoE points', fontsize=axis_font)
plt.show()

#%% doe-set, gap_Ry plot
X_plot = make_2D_plotting_space(mdl_doe.X, res)

import time
t0 = time.time()

fmu_doe, fs1_doe = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_doe = fs1_doe**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_plot.shape[0],
                                                                tp))

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_GRy = fmu_doe.reshape(xx_pl.shape)

plt.figure(figsize=(8,6))
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z_GRy, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_doe['gap_ratio'], df_doe['RI'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.5,2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using post-DOE points', fontsize=axis_font)
plt.show()


#%% doe plot (put on gap Ry dimensions)

# remake X_plot in gap Ry
X_plot = make_2D_plotting_space(mdl_doe.X, res)

gp_obj = mdl_doe.gpr._final_estimator
X_train = gp_obj.X_train_

if kernel_name == 'rbf_iso':
    lengthscale = gp_obj.kernel_.theta[1]
else:
    lengthscale = gp_obj.kernel_.theta[1:5]

def loo_error_approx(X_cand, X_data, lengthscale, e_cv_sq):
    point = np.array([np.asarray(X_cand)])
    from scipy.spatial.distance import cdist
    from numpy import exp
    dist_list = cdist(point/lengthscale, X_data/lengthscale).flatten()
    gamma = exp(-dist_list**2)
    numerator = np.sum(np.multiply(gamma, e_cv_sq))
    denominator = np.sum(gamma)
    return(numerator/denominator)

L = gp_obj.L_
K_mat = L @ L.T
alpha_ = gp_obj.alpha_.flatten()
K_inv_diag = np.linalg.inv(K_mat).diagonal()

e_cv_sq = np.divide(alpha_, K_inv_diag)**2

loo_error = X_plot.apply(lambda row: loo_error_approx(row, X_train, lengthscale, e_cv_sq),
                           axis='columns', result_type='expand')

x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)

# loocv
mse_w = np.array(loo_error*fs2_doe)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z = mse_w.reshape(xx_pl.shape)

batch_size = 5

df_doe = df_doe.reset_index(drop=True)
df_doe['batch_id'] = df_doe.index.values//batch_size + 1

# plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font = 20

fig= plt.figure(figsize=(8,6))
cmap = plt.cm.plasma
sc = plt.scatter(df_doe['gap_ratio'], df_doe['RI'], c=df_doe['batch_id'], alpha=0.5, cmap=cmap)
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)

fig.colorbar(sc, label='Batch added')
plt.clabel(cs, fontsize=clabel_size)
plt.xlabel(r'GR', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title(r'$MSE_w$ selection criterion', fontsize=axis_font)
plt.grid(True)
plt.savefig('./figures/doe.pdf')

#%% DoE effect plot

X_plot = make_2D_plotting_space(mdl_doe.X, res)

x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)



# plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

# first we show initial dataset


mdl_init = GP(df_init)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_init.set_covariates(covariate_list)
mdl_init.set_outcome('collapse_prob')

mdl_init.fit_gpr(kernel_name=kernel_name)
fmu_init, fs1_init = mdl_init.gpr.predict(X_plot, return_std=True)
fs2_init = fs1_init**2

# kernel stuff for loocv
gp_obj = mdl_init.gpr._final_estimator
X_train = gp_obj.X_train_
if kernel_name == 'rbf_iso':
    lengthscale = gp_obj.kernel_.theta[1]
else:
    lengthscale = gp_obj.kernel_.theta[1:5]
L = gp_obj.L_
K_mat = L @ L.T
alpha_ = gp_obj.alpha_.flatten()
K_inv_diag = np.linalg.inv(K_mat).diagonal()

e_cv_sq = np.divide(alpha_, K_inv_diag)**2
  
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_init = fmu_init.reshape(xx_pl.shape)

fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)

cmap = plt.cm.Blues
sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
                 alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax1.contour(xx_pl, yy_pl, Z_init, linewidths=1.1, cmap=cmap, vmin=-1)
ax1.clabel(cs, fontsize=clabel_size)
ax1.set_xlabel(r'$GR$', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.set_title('a) Collapse risk, pre-DoE', fontsize=title_font)
ax1.set_xlim([0.5, 2.5])
ax1.set_ylim([0.5, 2.25])
ax1.grid()

loo_error = X_plot.apply(lambda row: loo_error_approx(row, X_train, lengthscale, e_cv_sq),
                           axis='columns', result_type='expand')
# loocv
mse_w = np.array(loo_error*fs2_init)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z_mse = mse_w.reshape(xx_pl.shape)

# then we show selection criterion

ax2=fig.add_subplot(2, 2, 2)

sc = ax2.scatter(df_doe['gap_ratio'][50:55,], df_doe['RI'][50:55,] , edgecolors='black',
                 alpha=0.6, c=df_doe['collapse_prob'][50:55,], cmap=cmap)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax2.contour(xx_pl, yy_pl, Z_mse, linewidths=1.1, cmap=cmap, vmin=-1)
ax2.clabel(cs, fontsize=clabel_size)
ax2.set_xlabel(r'$GR$', fontsize=axis_font)
ax2.set_ylabel(r'$R_y$', fontsize=axis_font)
ax2.set_title(r'b) $MSE_w$ selection criterion, first batch', fontsize=title_font)
ax2.set_xlim([0.5, 2.5])
ax2.set_ylim([0.5, 2.25])
ax2.grid()

# then we show prediction of the full set

ax3=fig.add_subplot(2, 2, 3)

sc = ax3.scatter(df_doe['gap_ratio'], df_doe['RI'], alpha=0.6, edgecolors='black', s=15,
                 c=df_doe['collapse_prob'], cmap=cmap)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax3.contour(xx_pl, yy_pl, Z_GRy, linewidths=1.1, cmap=cmap, vmin=-0.5)
ax3.clabel(cs, fontsize=clabel_size)
ax3.set_xlabel(r'GR', fontsize=axis_font)
ax3.set_ylabel(r'$R_y$', fontsize=axis_font)
ax3.set_title('c) Collapse risk, post-DoE', fontsize=title_font)
handles, labels = sc.legend_elements(prop="colors")
legend2 = ax2.legend(handles, labels, loc="lower right", title=r"Pr(collapse)",
                      fontsize=16, title_fontsize=16, edgecolor='black')
ax3.set_xlim([0.5, 2.5])
ax3.set_ylim([0.5, 2.25])
ax3.grid()

# then we show prediction of the full set

ax4=fig.add_subplot(2, 2, 4)
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var='T_ratio', y_var='zeta_e', 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
x_var = 'T_ratio'
xx = X_plot[x_var]
y_var = 'zeta_e'
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

sc = ax4.scatter(df_doe['T_ratio'], df_doe['zeta_e'], alpha=0.6, edgecolors='black', s=15,
                 c=df_doe['collapse_prob'], cmap=cmap)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax4.contour(xx_pl, yy_pl, Z_Tze, linewidths=1.1, cmap=cmap, vmin=-0.5)
ax4.clabel(cs, fontsize=clabel_size)
ax4.set_xlabel(r'$T_M$ / $T_{fb}$', fontsize=axis_font)
ax4.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax4.set_xlim([2.0, 5.0])
ax4.set_title('d) Collapse risk, post-DoE', fontsize=title_font)
handles, labels = sc.legend_elements(prop="colors")
legend2 = ax2.legend(handles, labels, loc="lower right", title=r"Pr(collapse)",
                      fontsize=16, title_fontsize=16, edgecolor='black')
ax4.grid()

fig.tight_layout()
plt.savefig('./figures/doe_full.pdf')
#%% pareto - doe

# remake X_plot in gap Ry
X_plot = make_2D_plotting_space(mdl_doe.X, res)


risk_thresh = 0.1

all_costs = calc_upfront_cost(X_design_cand, coef_dict, W=W_seis, Ws=W_super)
constr_costs = all_costs['total']
predicted_risk = space_collapse_pred['collapse probability']
# predicted_risk[predicted_risk < 0] = 0


# acceptable_mask = predicted_risk < risk_thresh
# X_acceptable = X_design_cand[acceptable_mask]
# acceptable_cost = constr_costs[acceptable_mask]
# acceptable_risk = predicted_risk[acceptable_mask]

pareto_array = np.array([constr_costs, predicted_risk]).transpose()
# pareto_array = np.array([acceptable_cost, acceptable_risk]).transpose()

t0 = time.time()
pareto_mask = is_pareto_efficient(pareto_array)
tp = time.time() - t0

print("Culled %d points in %.3f s" % (pareto_array.shape[0], tp))

X_pareto = X_design_cand.iloc[pareto_mask].copy()
risk_pareto = predicted_risk.iloc[pareto_mask]
cost_pareto = constr_costs.iloc[pareto_mask]

# X_pareto = X_acceptable.iloc[pareto_mask].copy()
# risk_pareto = acceptable_risk.iloc[pareto_mask]
# cost_pareto = acceptable_cost.iloc[pareto_mask]

# -1 if predicted risk > allowable

X_pareto['acceptable_risk'] = np.sign(risk_thresh - risk_pareto)
X_pareto['predicted_risk'] = risk_pareto

dom_idx = np.random.choice(len(pareto_array), 1000, replace = False)
dominated_sample = np.array([pareto_array[i] for i in dom_idx])

# plt.close('all')
fig = plt.figure(figsize=(13, 10))

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
ax = fig.add_subplot(2, 2, 1)
ax.scatter(risk_pareto, cost_pareto, marker='s', facecolors='none',
            edgecolors='green', s=20.0, label='Pareto optimal designs')
ax.scatter(risk_pareto, cost_pareto, s=1, color='black')
ax.scatter(dominated_sample[:,1], dominated_sample[:,0], s=1, color='black',
           label='Dominated designs')
ax.set_xlabel('Collapse risk', fontsize=axis_font)
ax.set_ylabel('Construction cost', fontsize=axis_font)
# ax.set_ylim([4.64e6, 4.75e6])
ax.grid(True)
ax.legend()
plt.title('a) Pareto front', fontsize=axis_font)
plt.show()


x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

cmap = plt.cm.Spectral_r
ax1=fig.add_subplot(2, 2, 2)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
sc = ax1.scatter(X_pareto['gap_ratio'], X_pareto['RI'], 
            c=X_pareto['predicted_risk'],
            edgecolors='k', s=20.0, cmap=cmap)

# cbar = plt.colorbar(sc, ticks=[0, 0.2, 0.4])
cs = plt.contour(xx_pl, yy_pl, Z_GRy, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
ax1.set_xlim([0.5, 2.0])
ax1.set_ylim([0.5, 2.25])
ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.grid(True)
ax1.set_title(r'b) $T_M / T_{fb} = 3.0$, $\zeta_e = 0.15$', fontsize=axis_font)



from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(ax1,  width="3%", height="20%", loc='lower right',
                    bbox_to_anchor=(-0.05,0.05,1,1), bbox_transform=ax1.transAxes) 
plt.colorbar(sc, cax=cbaxes, orientation='vertical')
cbaxes.set_ylabel('Collapse risk', fontsize=axis_font)
cbaxes.yaxis.set_ticks_position('left')


'''
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var='T_ratio', y_var='zeta_e', 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

x_var = 'T_ratio'
xx = X_plot[x_var]
y_var = 'zeta_e'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax2=fig.add_subplot(1, 2, 2)
plt.clabel(cs, fontsize=clabel_size)
cs = plt.contour(xx_pl, yy_pl, Z_Tze, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
ax2.scatter(X_pareto['T_ratio'], X_pareto['zeta_e'], 
            c=X_pareto['predicted_risk'],
            edgecolors='k', s=20.0, cmap=plt.cm.Spectral_r)
ax2.set_xlabel('T ratio', fontsize=axis_font)
ax2.set_ylabel(r'$\zeta_e$', fontsize=axis_font)
ax2.grid(True)
ax2.set_title(r'b) $GR = 1.0$, $R_y = 2.0$', fontsize=axis_font)


plt.savefig('./figures/pareto.eps')
'''

# 3D

ax=fig.add_subplot(2, 2, 3, projection='3d')

sc = ax.scatter(X_pareto['gap_ratio'], X_pareto['RI'], X_pareto['T_ratio'], 
           c=X_pareto['predicted_risk'],
           edgecolors='k', alpha = 1, cmap=plt.cm.Spectral_r)
ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax.set_title(r'c) $\zeta_e$ not shown', fontsize=axis_font)

ax=fig.add_subplot(2, 2, 4, projection='3d')

ax.scatter(X_pareto['gap_ratio'], X_pareto['RI'], X_pareto['zeta_e'], 
           c=X_pareto['predicted_risk'],
           edgecolors='k', alpha = 1, cmap=plt.cm.Spectral_r)
ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$\zeta_e$', fontsize=axis_font)
ax.set_title(r'd) $T_M/T_{fb}$ not shown', fontsize=axis_font)
fig.colorbar(sc, ax=ax)
fig.tight_layout(w_pad=0.0)
plt.savefig('./figures/pareto_full.pdf')
plt.show()


#%% histogram
'''
# plt.close('all')
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)

ax1.hist(df_doe['gap_ratio'][:50,], 
         alpha = 0.5, edgecolor='black', lw=3, color= 'r', label='Original set')
ax1.hist(df_doe['gap_ratio'][50:,], 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b', label='DoE set')
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_ylabel('Frequency of points', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.legend()
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.hist(df_doe['RI'][:50,], bins='auto', 
         alpha = 1, edgecolor='black', lw=3, color= 'r')
ax2.hist(df_doe['RI'][50:,], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.hist(df_doe['T_ratio'][:50,], bins='auto', 
         alpha = 1, edgecolor='black', lw=3, color= 'r')
ax3.hist(df_doe['T_ratio'][50:,], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax3.set_ylabel('Frequency of points', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.hist(df_doe['zeta_e'][:50,], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'r')
ax4.hist(df_doe['zeta_e'][50:,], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax4.set_xlabel(r'$\zeta_e$', fontsize=axis_font)
ax4.set_title('Bearing damping', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()
plt.show()

'''
#%% Prediction 3ds

# as an example, let's do T-ratio vs zeta as gap evolves
# plt.close('all')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

res = 75
plt_density = 200
x_var = 'T_ratio'
y_var = 'zeta_e'
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [2.0, 5.0],
                            third_var_set = 0.5, fourth_var_set = 2.0)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_3d, fs1_3d = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_3d = fs1_3d**2

Z_3d = fmu_3d.reshape(xx_pl.shape)

fig = plt.figure(figsize=(11, 9))
ax=fig.add_subplot(2, 2, 1, projection='3d')

# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_3d, cmap=plt.cm.Spectral_r,
                        linewidth=0, antialiased=False,
                        alpha=0.7, vmin=0, vmax=0.075)

df = df_doe[df_doe['gap_ratio'] < 0.75]
ax.scatter(df[x_var][:plt_density], df[y_var][:plt_density], 
            df['collapse_prob'][:plt_density],
            edgecolors='k')

ax.set_xlabel(r'$T_M/ T_{fb}$', fontsize=axis_font, linespacing=0.5)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font, linespacing=1.0)
ax.set_zlabel(r'Collapse risk', fontsize=axis_font, linespacing=3.0)
ax.set_title(r'$GR = 0.5$', fontsize=title_font)
ax.set_xlim([2, 5])
ax.set_zlim([0, 1])
# plt.show()

# #################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [2.0, 5.0],
                            third_var_set = 0.75, fourth_var_set = 2.0)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_3d, fs1_3d = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_3d = fs1_3d**2

Z_3d = fmu_3d.reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 2, projection='3d')

# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_3d, cmap=plt.cm.Spectral_r,
                        linewidth=0, antialiased=False,
                        alpha=0.7, vmin=0, vmax=0.075)

df = df_doe[(df_doe['gap_ratio'] < 1.0) & (df_doe['gap_ratio'] > 0.75)]
ax.scatter(df[x_var][:plt_density], df[y_var][:plt_density], 
            df['collapse_prob'][:plt_density],
            edgecolors='k')

ax.set_xlabel(r'$T_M/ T_{fb}$', fontsize=axis_font, linespacing=0.5)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font, linespacing=1.0)
ax.set_zlabel(r'Collapse risk', fontsize=axis_font, linespacing=3.0)
ax.set_title(r'$GR = 0.75$', fontsize=title_font)
ax.set_xlim([2, 5])
ax.set_zlim([0, 1])

# #################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [2.0, 5.0],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_3d, fs1_3d = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_3d = fs1_3d**2

Z_3d = fmu_3d.reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 3, projection='3d')

# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_3d, cmap=plt.cm.Spectral_r,
                        linewidth=0, antialiased=False,
                        alpha=0.7, vmin=0, vmax=0.075)

df = df_doe[(df_doe['gap_ratio'] < 1.5) & (df_doe['gap_ratio'] > 1.0)]
ax.scatter(df[x_var][:plt_density], df[y_var][:plt_density], 
            df['collapse_prob'][:plt_density],
            edgecolors='k')

ax.set_xlabel(r'$T_M/ T_{fb}$', fontsize=axis_font, linespacing=0.5)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font, linespacing=1.0)
ax.set_zlabel(r'Collapse risk', fontsize=axis_font, linespacing=3.0)
ax.set_title(r'$GR = 1.0$', fontsize=title_font)
ax.set_xlim([2, 5])
ax.set_zlim([0, 1])


# #################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [2.0, 5.0],
                            third_var_set = 2.0, fourth_var_set = 2.0)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_3d, fs1_3d = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_3d = fs1_3d**2

Z_3d = fmu_3d.reshape(xx_pl.shape)

ax=fig.add_subplot(2, 2, 4, projection='3d')

# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_3d, cmap=plt.cm.Spectral_r,
                        linewidth=0, antialiased=False,
                        alpha=0.7, vmin=0, vmax=0.075)

df = df_doe[(df_doe['gap_ratio'] > 1.5) ]
ax.scatter(df[x_var][:plt_density], df[y_var][:plt_density], 
            df['collapse_prob'][:plt_density],
            edgecolors='k')

ax.set_xlabel(r'$T_M/ T_{fb}$', fontsize=axis_font, linespacing=0.5)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font, linespacing=1.0)
ax.set_zlabel(r'Collapse risk', fontsize=axis_font, linespacing=3.0)
ax.set_title(r'$GR = 2.0$', fontsize=title_font)
ax.set_xlim([2, 5])
ax.set_zlim([0, 1])

fig.tight_layout(w_pad=0.0)
plt.savefig('./figures/surf.pdf')
# plt.show()

#%% GP plots, highlight design targets

# plt.close('all')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

res = 75
plt_density = 200
x_var = 'gap_ratio'
y_var = 'RI'
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 2.0, fourth_var_set = 0.1)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

fig = plt.figure(figsize=(9, 14))
ax=fig.add_subplot(3, 2, 1)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
lvls = [0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.2, 0.3]
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')
    

    

# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'a) $T_M/T_{fb} = 2.0$, $\zeta_e = 0.10$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()


##########################################################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 3.5, fourth_var_set = 0.1)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

ax=fig.add_subplot(3, 2, 2)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')

# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'b) $T_M/T_{fb} = 3.5$, $\zeta_e = 0.10$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()


##########################################################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

ax=fig.add_subplot(3, 2, 3)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')
    

    

# ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'c) $T_M/T_{fb} = 2.0$, $\zeta_e = 0.15$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()

##########################################################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 3.5, fourth_var_set = 0.15)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

ax=fig.add_subplot(3, 2, 4)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')
    

    

# ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'd) $T_M/T_{fb} = 3.5$, $\zeta_e = 0.15$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()

##########################################################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 2.0, fourth_var_set = 0.20)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

ax=fig.add_subplot(3, 2, 5)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')
    

    

ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'e) $T_M/T_{fb} = 2.0$, $\zeta_e = 0.20$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()

##########################################################################

X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var=x_var, y_var=y_var, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            x_bounds = [0.5, 1.5], y_bounds = [0.5, 2.25],
                            third_var_set = 3.5, fourth_var_set = 0.2)
xx = X_plot[x_var]
yy = X_plot[y_var]
x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fmu_highlight = mdl_doe.gpr.predict(X_plot, return_std=False)

Z_highlight = fmu_highlight.reshape(xx_pl.shape)

ax=fig.add_subplot(3, 2, 6)

# Plot the surface.
cmap = plt.cm.magma
# sc = ax1.scatter(df_init['gap_ratio'], df_init['RI'], edgecolors='black',
#                  alpha=0.6, c=df_init['collapse_prob'], cmap=cmap)
cs = ax.contour(xx_pl, yy_pl, Z_highlight, levels=lvls,  linewidths=1.1, cmap=cmap)
ax.clabel(cs, fontsize=clabel_size)

# draw lines for design targets

prob_list = [0.025, 0.05, 0.1]
offset_list = [0.65, 0.5, 0.5]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.5, 1.5, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_highlight)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='maroon',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='maroon', linewidth=2.0)
    ax.text(theGap, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='maroon')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='maroon')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='steelblue',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='steelblue', linewidth=2.0)
    ax.text(theGap, 1.25, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='steelblue')
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='steelblue')
    

    

ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'f) $T_M/T_{fb} = 3.5$, $\zeta_e = 0.20$', fontsize=title_font)
ax.set_xlim([0.5, 1.5])
ax.set_ylim([0.5, 2.25])
ax.grid()

fig.tight_layout()
plt.savefig('./figures/inverse_design_contours.eps')
# plt.show()

#%% impact histogram
'''
# plt.close('all')
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)
df_impact = df_doe[df_doe['impacted'] == 1]
df_no_impact = df_doe[df_doe['impacted'] == 0]

ax1.hist(df_impact['gap_ratio'], 
         alpha = 0.5, edgecolor='black', lw=3, color= 'r', label='Impact')
ax1.hist(df_no_impact['gap_ratio'], 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b', label='No impact')
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_ylabel('Frequency of points', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.legend()
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.hist(df_impact['RI'], bins='auto', 
         alpha = 1, edgecolor='black', lw=3, color= 'r')
ax2.hist(df_no_impact['RI'], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.hist(df_impact['T_ratio'], bins='auto', 
         alpha = 1, edgecolor='black', lw=3, color= 'r')
ax3.hist(df_no_impact['T_ratio'], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax3.set_ylabel('Frequency of points', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.hist(df_impact['zeta_e'], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'r')
ax4.hist(df_no_impact['zeta_e'], bins='auto', 
         alpha = 0.5, edgecolor='black', lw=3, color= 'b')
ax4.set_xlabel(r'$\zeta_e$', fontsize=axis_font)
ax4.set_title('Bearing damping', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()
plt.show()
'''
#%%
# TODO: validation curves

###############################################################################
# VALIDATION
###############################################################################

def df_collapse(df, drift_mu_plus_std=0.1):
    
    from experiment import collapse_fragility
    df[['max_drift',
       'collapse_prob']] = df.apply(lambda row: collapse_fragility(row, drift_at_mu_plus_std=drift_mu_plus_std),
                                            axis='columns', result_type='expand')
                           
    from numpy import log
    df['log_collapse_prob'] = log(df['collapse_prob'])
    
    return df
    
#%% 10% validation

# with open("../../data/validation/tfp_mf_db_ida_10.pickle", 'rb') as picklefile:
#     main_obj_val = pickle.load(picklefile)
    
# main_obj_val.calculate_collapse()

main_obj_val = pd.read_pickle("../../data/validation/tfp_mf_db_ida_10_iso.pickle")
df_val_10 = df_collapse(main_obj_val.ida_results)

# with open("../../data/validation/tfp_mf_db_ida_baseline.pickle", 'rb') as picklefile:
#     main_obj_val = pickle.load(picklefile)

main_obj_val = pd.read_pickle("../../data/validation/tfp_mf_db_ida_baseline.pickle")
df_val_baseline = df_collapse(main_obj_val.ida_results)

df_val_10['max_drift'] = df_val_10.PID.apply(max)
df_val_10['log_drift'] = np.log(df_val_10['max_drift'])
df_val_10['max_velo'] = df_val_10.PFV.apply(max)
df_val_10['max_accel'] = df_val_10.PFA.apply(max)
df_val_10['T_ratio'] = df_val_10['T_m'] / df_val_10['T_fb']
pi = 3.14159
g = 386.4
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_val_10['Bm'] = np.interp(df_val_10['zeta_e'], zetaRef, BmRef)
df_val_10['gap_ratio'] = (df_val_10['constructed_moat']*4*pi**2)/ \
    (g*(df_val_10['sa_tm']/df_val_10['Bm'])*df_val_10['T_m']**2)
    
df_val_baseline['max_drift'] = df_val_baseline.PID.apply(max)
df_val_baseline['log_drift'] = np.log(df_val_baseline['max_drift'])
df_val_baseline['max_velo'] = df_val_baseline.PFV.apply(max)
df_val_baseline['max_accel'] = df_val_baseline.PFA.apply(max)
df_val_baseline['T_ratio'] = df_val_baseline['T_m'] / df_val_baseline['T_fb']
pi = 3.14159
g = 386.4
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_val_baseline['Bm'] = np.interp(df_val_baseline['zeta_e'], zetaRef, BmRef)
df_val_baseline['gap_ratio'] = (df_val_baseline['constructed_moat']*4*pi**2)/ \
    (g*(df_val_baseline['sa_tm']/df_val_baseline['Bm'])*df_val_baseline['T_m']**2)
    

ida_levels = [1.0, 1.5, 2.0]
val_10_collapse = np.zeros((3,))
baseline_collapse = np.zeros((3,))

for i, lvl in enumerate(ida_levels):
    val_10_ida = df_val_10[df_val_10['ida_level']==lvl]
    base_ida = df_val_baseline[df_val_baseline['ida_level']==lvl]
    
    val_10_collapse[i] = val_10_ida['collapse_prob'].mean()
    baseline_collapse[i] = base_ida['collapse_prob'].mean()
    
print('==================================')
print('   Validation results  (1.0 MCE)  ')
print('==================================')

inverse_collapse = val_10_collapse[0]
baseline_collapse_mce = baseline_collapse[0]

print('====== INVERSE DESIGN (10%) ======')
print('MCE collapse frequency: ',
      f'{inverse_collapse:.2%}')

print('====== BASELINE DESIGN ======')
print('MCE collapse frequency: ',
      f'{baseline_collapse_mce:.2%}')