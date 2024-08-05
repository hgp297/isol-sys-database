############################################################################
#               Sample inverse design

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

main_obj = pd.read_pickle("../../data/loss/structural_db_complete_loss.pickle")
    
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

max_obj = pd.read_pickle("../../data/loss/structural_db_complete_max_loss.pickle")
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

def make_design_space(res, var_list=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                      fixed_var=None):
    
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

db_string = '../../resource/'
brace_db = pd.read_csv(db_string+'braceShapes.csv', index_col=None, header=0)  

df['steel_cost'] = df.apply(
       lambda row: calc_steel_cost(
           row, brace_db=brace_db,
           steel_per_unit=1.25),
       axis='columns', result_type='expand')

df['steel_cost_per_sf'] = df['steel_cost'] / df['bldg_area']

#%% subsets

df_hit = df[df['impacted'] == 1]
df_miss = df[df['impacted'] == 0]

# remove the singular outlier point
from scipy import stats
df_no_impact = df_miss[np.abs(stats.zscore(df_miss['cmp_cost_ratio'])) < 5].copy()

df_tfp = df_no_impact[df_no_impact['isolator_system'] == 'TFP']
df_lrb = df_no_impact[df_no_impact['isolator_system'] == 'LRB']
df_cbf = df_no_impact[df_no_impact['superstructure_system'] == 'CBF']
df_mf = df_no_impact[df_no_impact['superstructure_system'] == 'MF']

#%%

def predict_DV(X, impact_pred_mdl, hit_loss_mdl, miss_loss_mdl,
               outcome='cost_50%', return_var=False):
        
    # get probability of impact
    if 'log_reg_kernel' in impact_pred_mdl.named_steps.keys():
        probs_imp = impact_pred_mdl.predict_proba(impact_pred_mdl.K_pr)
    else:
        probs_imp = impact_pred_mdl.predict_proba(X)

    miss_prob = probs_imp[:,0]
    hit_prob = probs_imp[:,1]
    
    # weight with probability of collapse
    # E[Loss] = (impact loss)*Pr(impact) + (no impact loss)*Pr(no impact)
    # run SVR_hit model on this dataset
    outcome_str = outcome+'_pred'
    expected_DV_hit = pd.DataFrame(
            {outcome_str:np.multiply(
                    hit_loss_mdl.predict(X).ravel(),
                    hit_prob)})
            
    
    # run miss model on this dataset
    expected_DV_miss = pd.DataFrame(
            {outcome_str:np.multiply(
                    miss_loss_mdl.predict(X).ravel(),
                    miss_prob)})
    
    expected_DV = expected_DV_hit + expected_DV_miss
    
    if return_var:
        pass
    else:
        return(expected_DV)
    
    
#%% impact prediction

print('========== Fitting impact classification (GPC or KLR) ============')

# prepare the problem
mdl_impact = GP(df)
mdl_impact.set_covariates(covariate_list)
mdl_impact.set_outcome('impacted', use_ravel=True)
mdl_impact.test_train_split(0.2)

mdl_impact.fit_gpc(kernel_name='rbf_iso')
mdl_impact.fit_svc(kernel_name='rbf')
mdl_impact.fit_kernel_logistic(kernel_name='rbf')

#%% regression on impact-conditioned data

mdl_cost_hit = GP(df_hit)
mdl_cost_hit.set_covariates(covariate_list)
mdl_cost_hit.set_outcome(cost_var)
mdl_cost_hit.test_train_split(0.2)

mdl_cost_miss = GP(df_no_impact)
mdl_cost_miss.set_covariates(covariate_list)
mdl_cost_miss.set_outcome(cost_var)
mdl_cost_miss.test_train_split(0.2)

mdl_time_hit = GP(df_hit)
mdl_time_hit.set_covariates(covariate_list)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = GP(df_no_impact)
mdl_time_miss.set_covariates(covariate_list)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_repl_hit = GP(df_hit)
mdl_repl_hit.set_covariates(covariate_list)
mdl_repl_hit.set_outcome('replacement_freq')
mdl_repl_hit.test_train_split(0.2)

mdl_repl_miss = GP(df_no_impact)
mdl_repl_miss.set_covariates(covariate_list)
mdl_repl_miss.set_outcome('replacement_freq')
mdl_repl_miss.test_train_split(0.2)

# Fit conditioned DVs using kernel ridge
print('========== Fitting regressions (kernel ridge) ============')

# fit impacted set
mdl_cost_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')

# fit no impact set
mdl_cost_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')


mdl_repl_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_repl_miss.fit_kernel_ridge(kernel_name='rbf')


print('========== Fitting regressions (GPR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_gpr(kernel_name='rbf_iso')
mdl_time_hit.fit_gpr(kernel_name='rbf_iso')

# fit no impact set
mdl_cost_miss.fit_gpr(kernel_name='rbf_iso')
mdl_time_miss.fit_gpr(kernel_name='rbf_iso')


mdl_repl_hit.fit_gpr(kernel_name='rbf_iso')
mdl_repl_miss.fit_gpr(kernel_name='rbf_iso')

#%% Classification plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
title_font = 22
subt_font = 18
import matplotlib as mpl
label_size = 18
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# make grid and plot classification predictions

fig, ax = plt.subplots(1, 1, figsize=(9,7))

xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact.gpc.predict_proba(X_plot)[:,1]


# # kernel logistic impact prediction
# K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)
# probs_imp = mdl_impact.log_reg_kernel.predict_proba(K_space)
# Z = probs_imp[:,1]

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
        cmap=plt.cm.Blues,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=np.linspace(0.1,1.0,num=10))
clabels = plt.clabel(cs, fontsize=clabel_size, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

ax.scatter(df_hit[xvar][:plt_density],
            df_hit[yvar][:plt_density],
            s=40, c='darkblue', marker='v', edgecolors='crimson', label='Impacted')

ax.scatter(df_miss[xvar][:plt_density],
            df_miss[yvar][:plt_density],
            s=40, c='azure', edgecolors='k', label='No impact')
plt.legend(fontsize=axis_font)

ax.set_xlim(0.3, 2.0)
ax.set_title(r'Impact likelihood: $R_y = 2.0$, $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$T_M/T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()


####

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
title_font = 22
subt_font = 18
import matplotlib as mpl
label_size = 18
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
# make grid and plot classification predictions

fig, ax = plt.subplots(1, 1, figsize=(9,7))

xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 5.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

# GPC impact prediction
Z = mdl_impact.gpc.predict_proba(X_plot)[:,1]


# # kernel logistic impact prediction
# K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)
# probs_imp = mdl_impact.log_reg_kernel.predict_proba(K_space)
# Z = probs_imp[:,1]

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
        cmap=plt.cm.Blues,
    )
plt_density = 200
cs = plt.contour(xx_pl, yy_pl, Z_classif, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=np.linspace(0.1,1.0,num=10))
clabels = plt.clabel(cs, fontsize=clabel_size, colors='black')
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

ax.scatter(df_hit[xvar][:plt_density],
            df_hit[yvar][:plt_density],
            s=40, c='darkblue', marker='v', edgecolors='crimson', label='Impacted')

ax.scatter(df_miss[xvar][:plt_density],
            df_miss[yvar][:plt_density],
            s=40, c='azure', edgecolors='k', label='No impact')
plt.legend(fontsize=axis_font)

# ax.set_xlim(0.3, 2.0)
ax.set_title(r'Impact likelihood: $R_y = 2.0$, $GR = 1.0$', fontsize=title_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()

#%% dirty contours repair time and cost


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax = fig.add_subplot(1, 2, 1)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_cost = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_cost_hit.gpr,
                       mdl_cost_miss.gpr,
                       outcome=cost_var)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.3, 0.2, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 2.0, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$c_r$",
                      fontsize=subt_font, title_fontsize=subt_font)

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax = fig.add_subplot(1, 2, 2)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_time = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_time_hit.gpr,
                       mdl_time_miss.gpr,
                       outcome=time_var)

Z = np.array(grid_time)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.3, 0.2, 0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 2.0, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower left", title="$t_r$",
                      fontsize=subt_font, title_fontsize=subt_font)


fig.tight_layout()
plt.show()

#### replacement

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=22
axis_font = 22
subt_font = 20
label_size = 20
clabel_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

fig, ax = plt.subplots(1, 1, figsize=(9,7))
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

prob_target = 0.1
grid_repl = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_repl_hit.gpr,
                       mdl_repl_miss.gpr,
                       outcome='replacement_freq')

Z = np.array(grid_repl)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
clabels = ax.clabel(cs, fontsize=clabel_size)
# [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in clabels]

prob_list = [0.1]
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    xq = np.linspace(0.4, 2.0, 200)
    
    Ry_target = 1.0
    
    interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = Ry_target
    
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
    # Ry = 2.0
    Ry_target = 2.0
    pts[:,0] = Ry_target
    lq = interp(pts)
    
    the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
                linewidth=2.0)
    ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5) & 
           (df['zeta_e']<=0.2) & (df['zeta_e']>=0.13)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc['replacement_freq'], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$T_M/T_{fb}= 3.0$ , $\zeta_M = 0.15$', fontsize=title_font)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$p_r$",
                      fontsize=subt_font, title_fontsize=subt_font)

fig.tight_layout()
plt.show()



#%% Testing the design space
import time

res_des = 20
X_space = make_design_space(res_des)
#K_space = mdl.get_kernel(X_space, kernel_name='rbf', gamma=gam)

# choice GPC
# consider using GP if computational resources allow, and GP looks good

# choice GPR for all prediction model for richness. HOWEVER must keep resolution low
t0 = time.time()
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

# TODO: baseline is... MF-TFP?
### baseline
X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
                          columns=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'])
baseline_repair_cost_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_cost_hit.kr,
                                      mdl_cost_miss.kr,
                                      outcome=cost_var)[cost_var+'_pred'].item()
baseline_downtime_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_time_hit.kr,
                                      mdl_time_miss.kr,
                                      outcome=time_var)[time_var+'_pred'].item()


baseline_repl_risk_pred = predict_DV(X_baseline,
                                      mdl_impact.gpc,
                                      mdl_repl_hit.kr,
                                      mdl_repl_miss.kr,
                                      outcome='replacement_freq')['replacement_freq_pred'].item()

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

#%% calculate cost of inverse design building
config_dict = {
    'num_stories': 4,
    'h_story': 13.0,
    'num_bays': 4,
    'num_frames': 2,
    'S_s': 2.2815,
    'L_bay': 30.0,
    'S_1': 1.017
    }


#%% one single inverse design

# TODO: thought is that you might want two separate designs
# otherwise, you are predicting one X that will have the same outcome regardless of the system
# or come up with a unifier

# sample building is 4 bay, 4 stories
og_df = main_obj.ops_analysis.reset_index(drop=True)

# take all 4bay/4stories building from db, calculate average max cost of cmps
my_type = df_loss_max[(og_df['num_bays'] == 4) & (og_df['num_stories'] == 4)]

nb = 4
ns = 4
bldg_area = (nb*30)**2 * (ns + 1)

# assume $600/sf replacement
n_worker_series = bldg_area/1000
n_worker_parallel = n_worker_series/2
replacement_cost = 600*bldg_area
cmp_cost = my_type['cost_50%'].mean()

# assume 2 years replacement, 1 worker per 1000 sf, but working in parallel (2x faster)
# = (n_worker_parallel) * (365 * 2) = (n_worker_series / 2) * (365 * 2)
# = (n_worker_series) * (365)
replacement_time = bldg_area/1000*365
cmp_time = my_type['time_l_50%'].mean()


# < 40% of cost of replacing all components
percent_of_replacement = 0.2
ok_cost = X_space.loc[space_repair_cost[cost_var+'_pred']<=percent_of_replacement]

# <4 weeks for a team of 36
dt_thresh = n_worker_parallel*28

dt_thresh_ratio = dt_thresh / cmp_time
ok_time = X_space.loc[space_downtime[time_var+'_pred']<=dt_thresh_ratio]

repl_thresh = 0.1
ok_repl = X_space.loc[space_repl['replacement_freq_pred']<=
                      repl_thresh]

X_design = X_space[np.logical_and.reduce((
        X_space.index.isin(ok_cost.index), 
        X_space.index.isin(ok_time.index),
        X_space.index.isin(ok_repl.index)))]

# select best viable design
upfront_costs = calc_upfront_cost(
    X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
cheapest_mf_idx = upfront_costs['total_mf'].idxmin()
mf_upfront_cost = upfront_costs['total_mf'].min()

# least upfront cost of the viable designs
mf_design = X_design.loc[cheapest_mf_idx]
mf_downtime = space_downtime.iloc[cheapest_mf_idx].item()
mf_repair_cost = space_repair_cost.iloc[cheapest_mf_idx].item()
mf_repl_risk = space_repl.iloc[cheapest_mf_idx].item()

# read out predictions
print('==================================')
print('            Predictions           ')
print('==================================')
print('======= Targets =======')
print('Repair cost fraction:', f'{percent_of_replacement*100:,.2f}%')
print('Repair time (days):', dt_thresh/n_worker_parallel)
print('Replacement risk:', f'{repl_thresh*100:,.2f}%')


print('======= Overall MF inverse design =======')
print(mf_design)
print('Upfront cost of selected design: ',
      f'${mf_upfront_cost:,.2f}')
print('Predicted median repair cost ratio: ',
      f'{mf_repair_cost*100:,.2f}%')
print('Predicted median repair cost: ',
      f'${mf_repair_cost*cmp_cost:,.2f}')
print('Predicted repair time (parallel): ',
      f'{mf_downtime*cmp_time/n_worker_parallel:,.2f}', 'days')
print('Predicted repair time (parallel): ',
      f'{mf_downtime*cmp_time:,.2f}', 'worker-days')
print('Predicted repair time ratio: ',
      f'{mf_downtime*100:,.2f}%')
print('Predicted replacement risk: ',
      f'{mf_repl_risk:.2%}')


# select best viable design
upfront_costs = calc_upfront_cost(
    X_design, config_dict=config_dict, steel_cost_dict=reg_dict)
cheapest_cbf_idx = upfront_costs['total_cbf'].idxmin()
cbf_upfront_cost = upfront_costs['total_cbf'].min()

# least upfront cost of the viable designs
cbf_design = X_design.loc[cheapest_cbf_idx]
cbf_downtime = space_downtime.iloc[cheapest_cbf_idx].item()
cbf_repair_cost = space_repair_cost.iloc[cheapest_cbf_idx].item()
cbf_repl_risk = space_repl.iloc[cheapest_cbf_idx].item()

print('======= Overall CBF inverse design =======')
print(cbf_design)
print('Upfront cost of selected design: ',
      f'${cbf_upfront_cost:,.2f}')
print('Predicted median repair cost ratio: ',
      f'{cbf_repair_cost*100:,.2f}%')
print('Predicted median repair cost: ',
      f'${cbf_repair_cost*cmp_cost:,.2f}')
print('Predicted repair time (parallel): ',
      f'{cbf_downtime*cmp_time/n_worker_parallel:,.2f}', 'days')
print('Predicted repair time (parallel): ',
      f'{cbf_downtime*cmp_time:,.2f}', 'worker-days')
print('Predicted repair time ratio: ',
      f'{cbf_downtime*100:,.2f}%')
print('Predicted replacement risk: ',
      f'{cbf_repl_risk:.2%}')

baseline_upfront_cost_all = calc_upfront_cost(
    X_baseline, config_dict=config_dict, steel_cost_dict=reg_dict)
baseline_upfront_cost = baseline_upfront_cost_all['total_cbf'].item()

print('======= Predicted baseline performance =======')
print('Upfront cost of baseline design: ',
      f'${baseline_upfront_cost:,.2f}')
print('Baseline median repair cost ratio: ',
      f'{baseline_repair_cost_pred*100:,.2f}%')
print('Baseline median repair cost: ',
      f'${baseline_repair_cost_pred*cmp_cost:,.2f}')
print('Baseline repair time (parallel): ',
      f'{baseline_downtime_pred*cmp_time/n_worker_parallel:,.2f}', 'days')
print('Baseline repair time (parallel): ',
      f'{baseline_downtime_pred*cmp_time:,.2f}', 'worker-days')
print('Baseline repair time ratio: ',
      f'{baseline_downtime_pred*100:,.2f}%')
print('Baseline replacement risk: ',
      f'{baseline_repl_risk_pred:.2%}')

#%% design the systems

# TODO: pass the length of the df to run controllers
import pandas as pd
from db import prepare_ida_util
import json

mf_design['superstructure_system'] = 'MF'
mf_design['isolator_system'] = 'TFP'
mf_design['k_ratio'] = 10

mf_dict = mf_design.to_dict()
ida_mf_df = prepare_ida_util(mf_dict, db_string='../../resource/')

print('Length of MF-TFP IDA:', len(ida_mf_df))

with open('../inputs/mf_tfp_inverse.in', 'w') as file:
    file.write(json.dumps(mf_dict))
    file.close()

cbf_design['superstructure_system'] = 'CBF'
cbf_design['isolator_system'] = 'TFP'
cbf_design['k_ratio'] = 10

cbf_dict = cbf_design.to_dict()
ida_cbf_df = prepare_ida_util(cbf_dict, db_string='../../resource/')

with open('../inputs/cbf_tfp_inverse.in', 'w') as file:
    file.write(json.dumps(cbf_dict))
    file.close()
print('Length of CBF-TFP IDA:', len(ida_cbf_df))