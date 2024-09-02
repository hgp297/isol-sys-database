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

main_obj = pd.read_pickle("../../data/loss/structural_db_complete_loss.pickle")

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

max_obj = pd.read_pickle("../../data/loss/structural_db_complete_max_loss.pickle")
df_loss_max = max_obj.max_loss

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

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']
df_cbf = df[df['superstructure_system'] == 'CBF']
df_mf = df[df['superstructure_system'] == 'MF']

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
def make_design_space(res, zeta_fix=None):
    if zeta_fix is None:
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.6, 1.5,
                                                 res),
                                     np.linspace(0.5, 2.25,
                                                 res),
                                     np.linspace(2.0, 5.0,
                                                 res),
                                     np.linspace(0.1, 0.25,
                                                 res))
    else:
        xx, yy, uu, vv = np.meshgrid(np.linspace(0.6, 1.5,
                                                 res),
                                     np.linspace(0.5, 2.25,
                                                 res),
                                     np.linspace(2.0, 5.0,
                                                 res),
                                     zeta_fix)    
                                 
    X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                         'RI':yy.ravel(),
                         'T_ratio':uu.ravel(),
                         'zeta_e':vv.ravel()})

    return(X_space)

#%% ml training

# covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e', 'k_ratio', 'Q']
# clf_struct = GP(df)
# clf_struct.set_covariates(covariate_list)
# clf_struct.set_outcome('superstructure_system', use_ravel=True)
# clf_struct.test_train_split(0.2)
# clf_struct.fit_svc(neg_wt=False)
# clf_struct.fit_gpc(kernel_name='rbf_iso')
# clf_struct.fit_kernel_logistic(kernel_name='rbf', neg_wt=False)

#%% system selector
# consider: replacement freq, num_stories, num_bays, repair cost
covariate_list = ['gap_ratio', 'k_ratio', 'T_ratio', 'RI']
clf_struct = GP(df)
clf_struct.set_covariates(covariate_list)
clf_struct.set_outcome('superstructure_system', use_ravel=False)
clf_struct.test_train_split(0.2)
clf_struct.fit_ensemble()
clf_struct.fit_svc(neg_wt=False)
clf_struct.fit_gpc(kernel_name='rbf_iso')
clf_struct.fit_kernel_logistic(kernel_name='rbf', neg_wt=False)
# clf_struct.fit_dt()

clf_isol = GP(df)
clf_isol.set_covariates(covariate_list)
clf_isol.set_outcome('isolator_system', use_ravel=False)
clf_isol.test_train_split(0.2)
clf_isol.fit_ensemble()
clf_isol.fit_svc(neg_wt=False)
clf_isol.fit_gpc(kernel_name='rbf_iso')
clf_isol.fit_kernel_logistic(kernel_name='rbf', neg_wt=False)
# clf_isol.fit_dt()

#%%
plt.close('all')
#################################
xvar = 'T_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list,
                                third_var_set = 1.0, fourth_var_set = 10.0)


fig = plt.figure(figsize=(16, 7))

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1)



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
           edgecolors='k', alpha = 1.0, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], color=color[1],
           edgecolors='k', alpha = 1.0, label='MF')
plt.legend(fontsize=axis_font)

ax.set_title(r'Superstructures', fontsize=title_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)

#################################
xvar = 'T_ratio'
yvar = 'k_ratio'

res = 75
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar,
                                all_vars=covariate_list,
                                third_var_set = 1.0, fourth_var_set = 2.0)


color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 2)



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
           edgecolors='k', alpha = 1.0, label='LRB')
ax.scatter(df_tfp[xvar], df_tfp[yvar], color=color[1],
           edgecolors='k', alpha = 1.0, label='TFP')
plt.legend(fontsize=axis_font)

ax.set_title(r'Isolators', fontsize=title_font)
ax.set_ylabel(r'$k_1/k_2$', fontsize=axis_font)
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)