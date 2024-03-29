############################################################################
#               ML prediction models for collapse

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) Many models require cross validation of negative weight
#               as well as gamma value for rbf kernels
#               (2) note that KLR works better when there are extremities than
#               SVC in terms of drift-related risks

############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pred import Prediction, predict_DV
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import warnings
warnings.filterwarnings('ignore')

## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

#%% concat with other data
# database_path = './data/tfp_mf/'
# database_file = 'run_data.csv'

database_path = '../tfp_mf/data/doe/old/'
database_file = 'mik_smrf_doe.csv'

# results_path = './results/tfp_mf/'
# results_file = 'loss_estimate_data.csv'
# loss_data = pd.read_csv(results_path+results_file, 
#                         index_col=None)
full_isolation_data = pd.read_csv(database_path+database_file, 
                                  index_col=None)

# df = pd.concat([full_isolation_data, loss_data], axis=1)
df = full_isolation_data
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df.loc[df['collapsed'] == -1, 'collapsed'] = 0


# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

df['collapse_prob'] = ln_dist.cdf(df['max_drift'])
df_miss = df[df['impacted'] == 0]
df_hit = df[df['impacted'] == 1]

df["collapse_binary"] = 0
df['collapse_binary'] = np.where(df["max_drift"] > mean_log_drift, 1,
                                 df["collapse_binary"])

#%% Fit collapse probability (GP regression)

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('impacted')
mdl.test_train_split(0.2)

mdl.fit_gpc(kernel_name='rbf_iso')

mdl_drift_hit = Prediction(df_hit)
mdl_drift_hit.set_outcome('max_drift')
mdl_drift_hit.test_train_split(0.2)

mdl_drift_miss = Prediction(df_miss)
mdl_drift_miss.set_outcome('max_drift')
mdl_drift_miss.test_train_split(0.2)

mdl_drift_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_hit.fit_ols_ridge()

mdl_drift_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_miss.fit_ols_ridge()

mdl_drift = Prediction(df)
mdl_drift.set_outcome('max_drift')
mdl_drift.test_train_split(0.2)
mdl_drift.fit_gpr(kernel_name='rbf_iso')

mdl_collapse = Prediction(df)
mdl_collapse.set_outcome('collapse_prob')
mdl_collapse.test_train_split(0.2)
mdl_collapse.fit_gpr(kernel_name='rbf_iso')

#%% Prediction 3ds

# plt.close('all')
import matplotlib as mpl
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

X_plot = mdl.make_2D_plotting_space(100)
# X_plot['Tm']=3.0
# X_plot['zetaM'] = 0.15
plt_density = 200
grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.o_ridge,
                        mdl_drift_miss.o_ridge,
                                  outcome='max_drift')

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_drift)
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(11, 9))
ax1=fig.add_subplot(1, 2, 1, projection='3d')

# Plot the surface.
surf = ax1.plot_surface(xx, yy, Z, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=0, vmax=0.075)

ax1.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['max_drift'][:plt_density],
           edgecolors='k')

ax1.set_xlabel('\nGap ratio', fontsize=axis_font, linespacing=0.5)
ax1.set_ylabel('\n$R_y$', fontsize=axis_font, linespacing=1.0)
ax1.set_zlabel('\nPID (%)', fontsize=axis_font, linespacing=3.0)
ax1.set_title('PID, impact conditioned (GPC-OR)', fontsize=title_font)


# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

ax2=fig.add_subplot(1, 2, 2, projection='3d')
# Plot the surface.
surf = ax2.plot_surface(xx, yy, Z*100, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=-10, vmax=70)

ax2.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['collapse_prob'][:plt_density]*100,
           edgecolors='k')

ax2.set_xlabel('\nGap ratio', fontsize=axis_font, linespacing=0.5)
ax2.set_ylabel('\n$R_y$', fontsize=axis_font, linespacing=1.0)
ax2.set_zlabel('\nCollapse risk (%)', fontsize=axis_font, linespacing=3.0)
ax2.set_title('Collapse risk', fontsize=title_font)
fig.tight_layout()
plt.show()

grid_drift_nc = mdl_drift.gpr.predict(X_plot)
Z = ln_dist.cdf(np.array(grid_drift_nc))
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(11, 9))
ax1=fig.add_subplot(1, 2, 1, projection='3d')

# Plot the surface.
surf = ax1.plot_surface(xx, yy, Z, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=0, vmax=0.075)

ax1.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['collapse_prob'][:plt_density],
           edgecolors='k')

ax1.set_xlabel('\nGap ratio', fontsize=axis_font, linespacing=0.5)
ax1.set_ylabel('\n$R_y$', fontsize=axis_font, linespacing=1.0)
ax1.set_zlabel('Collapse risk', fontsize=axis_font, linespacing=3.0)
ax1.set_title('Collapse via PID, no conditioning (GPR)', fontsize=title_font)

grid_probs = mdl_collapse.gpr.predict(X_plot)
Z = np.array(grid_probs)
Z = Z.reshape(xx.shape)

ax2=fig.add_subplot(1, 2, 2, projection='3d')

# Plot the surface.
surf = ax2.plot_surface(xx, yy, Z, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=0, vmax=0.075)

ax2.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['collapse_prob'][:plt_density],
           edgecolors='k')

ax2.set_xlabel('\nGap ratio', fontsize=axis_font, linespacing=0.5)
ax2.set_ylabel('\n$R_y$', fontsize=axis_font, linespacing=1.0)
ax2.set_zlabel('Collapse risk', fontsize=axis_font, linespacing=3.0)
ax2.set_title('Direct collapse, no conditioning (GPR)', fontsize=title_font)
plt.show()

#%% fit collapse (gp classification)

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('collapsed')
mdl.test_train_split(0.2)
mdl.fit_gpc(kernel_name='rbf_iso', noisy=False)

# predict the entire dataset
preds_col = mdl.gpc.predict(mdl.X)
probs_col = mdl.gpc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_col).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.gpc, contour_pr=0.5)

# X_plot = mdl.make_2D_plotting_space(100, y_var='Tm')
# mdl.plot_classification(mdl.gpc, yvar='Tm', contour_pr=0.5)

# X_plot = mdl.make_2D_plotting_space(100, x_var='gapRatio', y_var='zetaM')
# mdl.plot_classification(mdl.gpc, xvar='gapRatio', yvar='zetaM', contour_pr=0.5)

#%% fit collapse (kernel logistic classification)

# currently only rbf is working
# TODO: gamma cross validation
krn = 'rbf'
gam = None # if None, defaults to 1/n_features = 0.25
mdl.fit_kernel_logistic(neg_wt=0.3, kernel_name=krn, gamma=gam)

# predict the entire dataset
K_data = mdl.get_kernel(mdl.X, kernel_name=krn, gamma=gam)
pReds_r_imp = mdl.log_reg_kernel.predict(K_data)
probs_imp = mdl.log_reg_kernel.predict_proba(K_data)

cmpr = np.array([mdl.y.values.flatten(), pReds_r_imp]).transpose()

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, pReds_r_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
mdl.plot_classification(mdl.log_reg_kernel)


#%% make design space and predict collapse

import time

# res_des = 30
# X_space = mdl.make_design_space(res_des)

res = 75

# xx, yy, uu = np.meshgrid(np.linspace(0.5, 2.0,
#                                           res),
#                               np.linspace(0.1, 0.2,
#                                           res),
#                               np.linspace(2.5, 4.0,
#                                           res))
                             
# X_space = pd.DataFrame({'gapRatio':xx.ravel(),
#                       'RI':np.repeat(2.0,res**3),
#                       'Tm':uu.ravel(),
#                       'zetaM':yy.ravel()})


xx, yy, uu = np.meshgrid(np.linspace(0.5, 2.0,
                                      res),
                          np.linspace(0.5, 2.0,
                                      res),
                          np.linspace(2.5, 4.0,
                                      res))
                             
X_space = pd.DataFrame({'gapRatio':xx.ravel(),
                      'RI':yy.ravel(),
                      'Tm':uu.ravel(),
                      'zetaM':np.repeat(0.2,res**3)})



t0 = time.time()
space_collapse = mdl.gpc.predict_proba(X_space)

tp = time.time() - t0
print("GPC collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))


#%% predictive variance and weighted variance

# latent variance
# fmu, fs2 = mdl.predict_gpc_latent(X_space)
fmu, fs1 = mdl_collapse.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

#%% plot gpc functions

# plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_pl = np.unique(xx)
y_pl = np.unique(yy)

xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]

Z = fs2_subset.reshape(xx_pl.shape)

plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Reds_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Latent variance', fontsize=axis_font)
plt.colorbar()
plt.show()

Z = fmu_subset.reshape(xx_pl.shape)

plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Reds_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Latent mean', fontsize=axis_font)
plt.colorbar()
plt.show()

# TODO: transition from latent to predictive mean is in the __gpc code (line 7
# of Algorithm 3.2, GPML) (it's the integral of sigmoid(x)*normpdf(x | fmu, fsigma))

from scipy.stats import logistic

# Z = logistic.cdf(fmu_subset.reshape(xx_pl.shape))
# plt.figure()
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Reds_r,
# ) 
# plt.xlabel('Gap ratio', fontsize=axis_font)
# plt.ylabel(r'$R_y$', fontsize=axis_font)
# plt.title('Predictive mean', fontsize=axis_font)
# plt.colorbar()
# plt.show()


# TODO: reexamine DoE weight
from numpy import exp
T = logistic.ppf(0.5)
pi = 3.14159
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - T)**2/(fs2_subset)))
# Wx = exp((-1/2)*((fmu_subset - T)**2/(fs2_subset)))

Z = Wx.reshape(xx_pl.shape)
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Reds_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weight', fontsize=axis_font)
plt.colorbar()
plt.show()

criterion = np.multiply(Wx, fs2_subset)
idx = np.argmax(criterion)


Z = criterion.reshape(xx_pl.shape)
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Reds_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance', fontsize=axis_font)
plt.colorbar()
plt.show()

#%% cost efficiency

from pred import get_steel_coefs, calc_upfront_cost
# plt.close('all')
steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(space_collapse,
                                    columns=['safe probability', 'collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control
    
# TODO: more clever selection criteria (not necessarily the cheapest)

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']

print(best_design)

print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')