import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from doe import GP

plt.close('all')

main_obj = pd.read_pickle("../data/tfp_mf_db.pickle")

# with open("../data/tfp_mf_db.pickle", 'rb') as picklefile:
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

main_obj_doe = pd.read_pickle('../data/tfp_mf_db_doe.pickle')

# with open("../data/tfp_mf_db_doe.pickle", 'rb') as picklefile:
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

#%%

df_wip = df_doe.head(700)

mdl_doe = GP(df_wip)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_doe.set_covariates(covariate_list)
mdl_doe.set_outcome('collapse_prob')

mdl_doe.fit_gpr(kernel_name=kernel_name)

res = 75
# remake X_plot in gap Ry
X_plot = make_2D_plotting_space(mdl_doe.X, res)

fmu_doe, fs1_doe = mdl_doe.gpr.predict(X_plot, return_std=True)
fs2_doe = fs1_doe**2
#%% doe plot (put on gap Ry dimensions)

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
rho = 5.0
mse_w = np.array(loo_error**rho*fs2_doe)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
Z = mse_w.reshape(xx_pl.shape)

batch_size = 5

df_doe = df_doe.reset_index(drop=True)
df_doe['batch_id'] = df_doe.index.values//batch_size + 1

plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font = 20

fig= plt.figure(figsize=(8,6))
cmap = plt.cm.plasma
sc = plt.scatter(df_wip['gap_ratio'], df_wip['RI'], c=df_wip['collapse_prob'], 
                 s=10.0, alpha=0.5, cmap=cmap)
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)

fig.colorbar(sc, label='Batch added')
plt.clabel(cs, fontsize=clabel_size)
plt.xlabel(r'$GR$', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title(r'$MSE_w$ selection criterion', fontsize=axis_font)
plt.grid(True)

fig = plt.figure(figsize=(11, 9))
ax=fig.add_subplot(projection='3d')

# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z, cmap=plt.cm.viridis,
                        linewidth=0, antialiased=False,
                        alpha=0.7)
plt.xlabel(r'$GR$', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title(r'$MSE_w$ selection criterion', fontsize=axis_font)
plt.grid(True)

