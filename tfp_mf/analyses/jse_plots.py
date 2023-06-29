############################################################################
#               ML prediction models for collapse

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1)

############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from doe import GP
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import warnings
warnings.filterwarnings('ignore')

## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

#%% collapse fragility def
label_size = 16
clabel_size = 12
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
x = np.linspace(0, 0.15, 200)
mu = log(0.1)- 0.25*inv_norm
sigma = 0.25;

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(x, p, label='Collapse', color='blue')

mu_irr = log(0.01)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

ax.plot(x, p_irr, color='red', label='Irreparable')

axis_font = 20
subt_font = 18
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, xleft])
ax.set_ylabel('Limit state probability', fontsize=axis_font)
ax.set_xlabel('Drift ratio', fontsize=axis_font)

ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
ax.text(0.105, 0.52, r'PID = 0.078', fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
ax.hlines(y=0.84, xmin=0.1, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.87, r'PID = 0.10', fontsize=axis_font, color='blue')
ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.16)
ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
ax.hlines(y=0.16, xmin=lower, xmax=xleft, color='blue', linestyle=":")
ax.text(0.105, 0.19, r'PID = 0.061', fontsize=axis_font, color='blue')
ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


ax.hlines(y=0.5, xmin=0.0, xmax=exp(mu_irr), color='red', linestyle=":")
lower = ln_dist_irr.ppf(0.16)
ax.hlines(y=0.16, xmin=0.0, xmax=lower, color='red', linestyle=":")
upper = ln_dist_irr.ppf(0.84)
ax.hlines(y=0.84, xmin=0.0, xmax=upper, color='red', linestyle=":")
ax.plot([lower], [0.16], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([0.01], [0.5], marker='*', markersize=15, color="red", linestyle=":")
ax.plot([upper], [0.84], marker='*', markersize=15, color="red", linestyle=":")
ax.vlines(x=upper, ymin=0, ymax=0.84, color='red', linestyle=":")
ax.vlines(x=0.01, ymin=0, ymax=0.5, color='red', linestyle=":")
ax.vlines(x=lower, ymin=0, ymax=0.16, color='red', linestyle=":")

ax.text(0.005, 0.19, r'RID = 0.007', fontsize=axis_font, color='red')
ax.text(0.005, 0.87, r'RID = 0.013', fontsize=axis_font, color='red')
ax.text(0.005, 0.53, r'RID = 0.010', fontsize=axis_font, color='red')

ax.set_title('Replacement fragility definition', fontsize=axis_font)
ax.grid()
ax.legend(fontsize=label_size, loc='upper center')
plt.show()

#%% pre-doe data

database_path = '../data/'
database_file = 'training_set.csv'

df_train = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_train['max_drift'] = df_train[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_train['log_drift'] = np.log(df_train['max_drift'])
df_train['collapse_prob'] = ln_dist.cdf(df_train['max_drift'])

mdl_init = GP(df_train)
mdl_init.set_outcome('collapse_prob')
mdl_init.fit_gpr(kernel_name='rbf_ard')

#%% predict the plotting space
import time
res = 75
xx, yy, uu = np.meshgrid(np.linspace(0.3, 2.0,
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

fmu_train, fs1_train = mdl_init.gpr.predict(X_space, return_std=True)
fs2_train = fs1_train**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% plots

plt.close('all')
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

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2_train[X_space['Tm']==3.25]
fmu_subset = fmu_train[X_space['Tm']==3.25]
Z = fmu_subset.reshape(xx_pl.shape)

plt.figure()
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
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_train['gapRatio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
plt.title('Collapse risk (direct), pre-DoE', fontsize=axis_font)
plt.show()

#%% doe convergence plots

doe_path = '../data/doe/'

import matplotlib.pyplot as plt
import pandas as pd

rmse_df = pd.read_csv(doe_path+'rmse.csv', header=None)
mae_df = pd.read_csv(doe_path+'mae.csv', header=None)
mae_df = mae_df.transpose()
mae_df.columns = ['mae']
rmse_df = rmse_df.transpose()
rmse_df.columns = ['rmse']

plt.close('all')

fig = plt.figure(figsize=(13, 6))
ax1=fig.add_subplot(1, 2, 1)

ax1.plot(rmse_df.index, rmse_df['rmse'])
ax1.set_title(r'Root mean squared error (collapse %)', fontsize=axis_font)
ax1.set_xlabel(r'Batches', fontsize=axis_font)
ax1.set_ylabel(r'Metric', fontsize=axis_font)
plt.grid(True)


ax2=fig.add_subplot(1, 2, 2)
ax2.plot(rmse_df.index, mae_df['mae'])
ax2.set_title('Mean absolute error (collapse %)', fontsize=axis_font)
ax2.set_xlabel('Batches', fontsize=axis_font)
plt.grid(True)

fig.tight_layout()
plt.show()

#%% post-doe data

# database_path = '../data/doe/old/rmse_1_percent/'
database_path = '../data/doe/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

mdl_drift = GP(df_doe)
mdl_drift.set_outcome('max_drift')
mdl_drift.fit_gpr(kernel_name='rbf_ard')


#%% predict the plotting space

# ###############################################################################
# # collapse predictions via drifts
# ###############################################################################

# t0 = time.time()

# fmu_dr, fs1_dr = mdl_drift.gpr.predict(X_space, return_std=True)
# fs2_dr = fs1_dr**2

# tp = time.time() - t0
# print("GPR collapse prediction (from drift) for %d inputs in %.3f s" % (X_space.shape[0],
#                                                                         tp))

# #%% predicting baseline

# X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
#                           columns=['gapRatio', 'RI', 'Tm', 'zetaM'])
# baseline_drift = mdl_drift.gpr.predict(X_baseline).item()
# baseline_risk = ln_dist.cdf(baseline_drift)
# #%% plots

# plt.close('all')
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 20
# subt_font = 18
# import matplotlib as mpl
# label_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 

# x_pl = np.unique(xx)
# y_pl = np.unique(yy)

# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# X_subset = X_space[X_space['Tm']==3.25]
# fs2_subset = fs2_dr[X_space['Tm']==3.25]
# fmu_subset = fmu_dr[X_space['Tm']==3.25]
# Z = ln_dist.cdf(fmu_subset)
# Z = Z.reshape(xx_pl.shape)

# plt.figure()
# # plt.imshow(
# #     Z,
# #     interpolation="nearest",
# #     extent=(xx_pl.min(), xx_pl.max(),
# #             yy_pl.min(), yy_pl.max()),
# #     aspect="auto",
# #     origin="lower",
# #     cmap=plt.cm.Blues,
# # ) 
# lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
# cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)
# plt.clabel(cs, fontsize=clabel_size)
# plt.scatter(df_doe['gapRatio'], df_doe['RI'], 
#             c=df_doe['collapse_prob'],
#             edgecolors='k', s=20.0, cmap=plt.cm.Blues)
# plt.xlim([0.3, 2.0])
# plt.xlim([0.3, 2.0])
# plt.grid()
# # plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Blues)
# plt.xlabel('Gap ratio', fontsize=axis_font)
# plt.ylabel(r'$R_y$', fontsize=axis_font)
# plt.title('Collapse risk (from drift)', fontsize=axis_font)

# plt.show()

# #%% designing 

# from pred import get_steel_coefs, calc_upfront_cost
# plt.close('all')
# steel_price = 2.00
# coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)

# risk_thresh = 0.1
# space_collapse_pred = pd.DataFrame(ln_dist.cdf(fmu_dr), 
#                                    columns=['collapse probability'])
# ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# # in the filter-design process, only one of cost/dt is likely to control

# baseline_costs = calc_upfront_cost(X_baseline, coef_dict).item()

# # least upfront cost of the viable designs



# print('========== Baseline design ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${baseline_costs:,.2f}')
# print('Predicted collapse risk: ',
#       f'{baseline_risk:.4%}')
# print(X_baseline)


# # select best viable design
# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



# print('========== Inverse design ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.2%}')
# print(best_design)

# risk_thresh = 0.05
# space_collapse_pred = pd.DataFrame(ln_dist.cdf(fmu_dr), 
#                                    columns=['collapse probability'])
# ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# # in the filter-design process, only one of cost/dt is likely to control

# # select best viable design
# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']


# print('========== Inverse design ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.2%}')
# print(best_design)

# risk_thresh = 0.025
# space_collapse_pred = pd.DataFrame(ln_dist.cdf(fmu_dr), 
#                                    columns=['collapse probability'])
# ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# # in the filter-design process, only one of cost/dt is likely to control

# # select best viable design
# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



# print('========== Inverse design ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.2%}')
# print(best_design)


#%% predict the plotting space

###############################################################################
# direct collapse predictions
###############################################################################

t0 = time.time()

fmu, fs1 = mdl_doe.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% predicting baseline

X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
                          columns=['gapRatio', 'RI', 'Tm', 'zetaM'])
baseline_risk, baseline_fs1 = mdl_doe.gpr.predict(X_baseline, return_std=True)
baseline_risk = baseline_risk.item()
baseline_fs2 = baseline_fs1**2
baseline_fs2 = baseline_fs2.item()

#%% doe effect plots
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]

n_new = df_doe.shape[0] - df_train.shape[0]

fig = plt.figure(figsize=(13, 10))

# first we show training model

ax1=fig.add_subplot(2, 2, 1)
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2_train[X_space['Tm']==3.25]
fmu_subset = fmu_train[X_space['Tm']==3.25]
Z = fmu_subset.reshape(xx_pl.shape)


lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax1.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax1.clabel(cs, fontsize=clabel_size)
ax1.scatter(df_train['gapRatio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=40.0, cmap=plt.cm.Blues)
ax1.set_xlim([0.3, 2.0])
ax1.set_ylim([0.5, 2.0])
ax1.set_title('Collapse risk (direct), pre-DoE', fontsize=axis_font)
# ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.grid()

# then we show first iteration of weighted variance

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
fs2_subset = fs2_train[X_space['Tm']==3.25]
fmu_subset = fmu_train[X_space['Tm']==3.25]
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
criterion = np.multiply(Wx, fs2_subset)

ax2=fig.add_subplot(2, 2, 2)
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2_train[X_space['Tm']==3.25]
fmu_subset = fmu_train[X_space['Tm']==3.25]
Z = criterion.reshape(xx_pl.shape)

new_pts = df_doe.tail(n_new)
new_pts_first = new_pts.head(20)

lvls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
cs = ax2.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax2.clabel(cs, fontsize=clabel_size)

ax2.scatter(new_pts_first['gapRatio'], new_pts_first['RI'], 
            c=new_pts_first.index,
            edgecolors='k', s=40.0)
ax2.set_xlim([0.3, 2.0])
ax2.set_ylim([0.5, 2.0])
ax2.set_title('Weighted variance, first iteration', fontsize=axis_font)
# ax2.set_xlabel(r'Gap ratio', fontsize=axis_font)
# ax2.set_ylabel(r'$R_y$', fontsize=axis_font)
ax2.grid()

# then we show all added points

fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
criterion = np.multiply(Wx, fs2_subset)

ax3=fig.add_subplot(2, 2, 3)
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
Z = criterion.reshape(xx_pl.shape)

lvls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
cs = ax3.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax3.clabel(cs, fontsize=clabel_size)

ax3.scatter(new_pts['gapRatio'], new_pts['RI'], 
            c=new_pts.index,
            edgecolors='k', s=40.0)
ax3.set_xlim([0.3, 2.0])
ax3.set_ylim([0.5, 2.0])
ax3.set_title('Weighted variance, last iteration', fontsize=axis_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax3.set_ylabel(r'$R_y$', fontsize=axis_font)
ax3.grid()

# then show final results

ax4=fig.add_subplot(2, 2, 4)
# collapse predictions
Z = fmu_subset.reshape(xx_pl.shape)
new_pts = df_doe.tail(n_new)
new_pts_first = new_pts.head(10)

lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax4.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax4.clabel(cs, fontsize=clabel_size)

ax4.scatter(df_doe['gapRatio'], df_doe['RI'], 
            c=df_doe['collapse_prob'], cmap='Blues',
            edgecolors='k', s=40.0)
ax4.set_xlim([0.3, 2.0])
ax4.set_ylim([0.5, 2.0])
ax4.set_title('Collapse risk, post-DoE', fontsize=axis_font)
ax4.set_xlabel(r'Gap ratio', fontsize=axis_font)
# ax4.set_ylabel(r'$R_y$', fontsize=axis_font)
ax4.grid()

fig.tight_layout()
plt.show()

#%% plots

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
new_pts = df_doe.tail(70)

criterion = np.multiply(Wx, fs2_subset)
Z = criterion.reshape(xx_pl.shape)
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Blues,
) 
plt.scatter(new_pts['gapRatio'][:10], new_pts['RI'][:10], 
            c=new_pts['collapse_prob'][:10],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance, first iteration', fontsize=axis_font)

plt.show()

# collapse probabilities (mean)
plt.close('all')
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
Z = fmu_subset.reshape(xx_pl.shape)

plt.figure()
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
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_doe['gapRatio'], df_doe['RI'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
# plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Blues)
plt.grid()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Collapse risk (direct)', fontsize=axis_font)
plt.show()

# collapse probabilities (95% conf int)
plt.close('all')
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
Z = fmu_subset + 1.96*fs2_subset
Z = Z.reshape(xx_pl.shape)

plt.figure()
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
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_doe['gapRatio'], df_doe['RI'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
# plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Blues)
plt.grid()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Collapse risk (direct), 95% confidence interval', fontsize=axis_font)
plt.show()

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
new_pts = df_doe.tail(70)

criterion = np.multiply(Wx, fs2_subset)
Z = criterion.reshape(xx_pl.shape)
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Blues,
) 
plt.scatter(new_pts['gapRatio'][-10:], new_pts['RI'][-10:], 
            c=new_pts['collapse_prob'][-10:],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance, last iteration', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
plt.show()

#%% prediction accuracy

import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
y_hat = mdl_doe.gpr.predict(mdl_doe.X)
y_true = mdl_doe.y

plt.figure()
plt.scatter(y_hat, y_true)
plt.plot([0, 1.0], [0, 1.0], linestyle='-',color='black')
plt.plot([0, 1.0], [0, 1.1], linestyle='--',color='black')
plt.plot([0, 1.0], [0, 0.9], linestyle='--',color='black')
plt.title('Prediction accuracy')
plt.xlabel('Predicted collapse %')
plt.ylabel('True collapse %')
plt.xlim([0, 0.3])
plt.ylim([0, 0.3])
plt.grid(True)
plt.show()

#%% cost efficiency

from pred import get_steel_coefs, calc_upfront_cost
plt.close('all')
steel_price = 2.00
coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)

risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(fmu, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

baseline_costs = calc_upfront_cost(X_baseline, coef_dict).item()

# least upfront cost of the viable designs



print('========== Baseline design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${baseline_costs:,.2f}')
print('Predicted collapse risk: ',
      f'{baseline_risk:.2%}')
print(X_baseline)


# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)

risk_thresh = 0.05
space_collapse_pred = pd.DataFrame(fmu, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']


print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print(best_design)

risk_thresh = 0.025
space_collapse_pred = pd.DataFrame(fmu, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



print('========== Inverse design ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.4%}')
print(best_design)

#%% full validation (IDA data)

val_dir = '../data/val/'
val_10_file = 'ida_jse_10.csv'
val_5_file = 'ida_jse_5.csv'
val_2_file = 'ida_jse_2_5.csv'

baseline_dir = '../data/val/'
baseline_file = 'ida_jse_baseline.csv'

df_val_10 = pd.read_csv(val_dir+val_10_file, index_col=None)
df_val_5 = pd.read_csv(val_dir+val_5_file, index_col=None)
df_val_2 = pd.read_csv(val_dir+val_2_file, index_col=None)
df_base = pd.read_csv(baseline_dir+baseline_file, index_col=None)
cost_var = 'cost_50%'
time_var = 'time_u_50%'

from scipy.stats import lognorm
from math import log, exp
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

df_val_10['max_drift'] = df_val_10[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_val_10['collapse_probs'] = ln_dist.cdf(np.array(df_val_10['max_drift']))

df_val_5['max_drift'] = df_val_5[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_val_5['collapse_probs'] = ln_dist.cdf(np.array(df_val_5['max_drift']))

df_val_2['max_drift'] = df_val_2[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_val_2['collapse_probs'] = ln_dist.cdf(np.array(df_val_2['max_drift']))

df_base['max_drift'] = df_base[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_base['collapse_probs'] = ln_dist.cdf(np.array(df_base['max_drift']))

ida_levels = [1.0, 1.5, 2.0]
val_10_collapse = np.zeros((3,))
val_5_collapse = np.zeros((3,))
val_2_collapse = np.zeros((3,))
baseline_collapse = np.zeros((3,))

for i, lvl in enumerate(ida_levels):
    val_10_ida = df_val_10[df_val_10['IDALevel']==lvl]
    val_5_ida = df_val_5[df_val_5['IDALevel']==lvl]
    val_2_ida = df_val_2[df_val_2['IDALevel']==lvl]
    base_ida = df_base[df_base['IDALevel']==lvl]
    
    val_10_collapse[i] = val_10_ida['collapse_probs'].mean()
    val_5_collapse[i] = val_5_ida['collapse_probs'].mean()
    val_2_collapse[i] = val_2_ida['collapse_probs'].mean()
    
    baseline_collapse[i] = base_ida['collapse_probs'].mean()
    
print('==================================')
print('   Validation results  (1.0 MCE)  ')
print('==================================')

inverse_collapse = val_10_collapse[0]

print('====== INVERSE DESIGN (10%) ======')
print('MCE collapse frequency: ',
      f'{inverse_collapse:.2%}')

inverse_collapse = val_5_collapse[0]

print('====== INVERSE DESIGN (5%) ======')
print('MCE collapse frequency: ',
      f'{inverse_collapse:.2%}')

inverse_collapse = val_2_collapse[0]

print('====== INVERSE DESIGN (2.5%) ======')
print('MCE collapse frequency: ',
      f'{inverse_collapse:.2%}')

baseline_collapse_mce = baseline_collapse[0]

print('====== BASELINE DESIGN ======')
print('MCE collapse frequency: ',
      f'{baseline_collapse_mce:.2%}')

val_10_mce = df_val_10[df_val_10['IDALevel']==1.0]
val_5_mce = df_val_5[df_val_5['IDALevel']==1.0]
val_2_mce = df_val_2[df_val_2['IDALevel']==1.0]
base_mce = df_base[df_base['IDALevel']==1.0]

#%% validation collapse distribution at mce

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 18
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


fig, axes = plt.subplots(1, 1, 
                         figsize=(10, 6))

mce_dict = {'10%': val_10_mce['collapse_probs'],
            '5%': val_5_mce['collapse_probs'],
            '2.5%': val_2_mce['collapse_probs'],
            'Baseline': base_mce['collapse_probs']}
df_mce = pd.DataFrame.from_dict(
    data=mce_dict,
    orient='index',
).T

base_repl_cases = base_mce[base_mce['collapse_probs'] >= 0.1].count()['collapse_probs']
repl_cases_10 = val_10_mce[val_10_mce['collapse_probs'] >= 0.1].count()['collapse_probs']
repl_cases_5 = val_5_mce[val_5_mce['collapse_probs'] >= 0.1].count()['collapse_probs']
repl_cases_2 = val_2_mce[val_2_mce['collapse_probs'] >= 0.1].count()['collapse_probs']

# print('Inverse runs requiring replacement:', repl_cases_10)
# print('Baseline runs requiring replacement:', base_repl_cases)

import seaborn as sns
ax = sns.stripplot(data=df_mce, orient='h', palette='coolwarm', 
                   edgecolor='black', linewidth=1.0)

sns.boxplot(data=df_mce, saturation=0.8, ax=ax, orient='h', palette='coolwarm',
            width=0.4)
ax.set_ylabel('Design case', fontsize=axis_font)
ax.set_xlabel('Collapse probability', fontsize=axis_font)
ax.axvline(0.10, linestyle='--', color='black')
ax.grid(visible=True)

ax.text(0.095, 0, u'\u2192', fontsize=axis_font, color='red')
ax.text(0.095, 1, u'\u2192', fontsize=axis_font, color='red')
ax.text(0.095, 2, u'\u2192', fontsize=axis_font, color='red')
ax.text(0.095, 3, u'\u2192', fontsize=axis_font, color='red')

ax.text(0.084, 0, f'{repl_cases_10} runs', fontsize=axis_font, color='red')
ax.text(0.084, 1, f'{repl_cases_5} runs', fontsize=axis_font, color='red')
ax.text(0.084, 2, f'{repl_cases_2} runs', fontsize=axis_font, color='red')
ax.text(0.084, 3, f'{base_repl_cases} runs', fontsize=axis_font, color='red')

ax.text(0.075, 3.4, r'10% threshold', fontsize=axis_font, color='black')
ax.set_xlim(0, 0.1)

# ax.set_xscale("log")
plt.show()

#%% validation drift distribution at mce

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 18
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

mce_dr_dict = {'10%': val_10_mce['max_drift'],
            '5%': val_5_mce['max_drift'],
            '2.5%': val_2_mce['max_drift'],
            'Baseline': base_mce['max_drift']}

fig, axes = plt.subplots(1, 1, 
                         figsize=(10, 6))
df_mce = pd.DataFrame.from_dict(
    data=mce_dr_dict,
    orient='index',
).T


import seaborn as sns
ax = sns.stripplot(data=df_mce, orient='h', palette='coolwarm', 
                   edgecolor='black', linewidth=1.0)
ax.set_xlim(0, 0.2)
sns.boxplot(data=df_mce, saturation=0.8, ax=ax, orient='h', palette='coolwarm',
            width=0.4)
ax.set_ylabel('Design case', fontsize=axis_font)
ax.set_xlabel('Max drift', fontsize=axis_font)
ax.axvline(0.078, linestyle='--', color='black')
ax.grid(visible=True)

ax.text(0.08, 3.45, r'50% collapse threshold, PID=0.078', fontsize=axis_font, color='black')
# ax.set_xscale("log")
plt.show()


#%% validation collapse histogram at mce

# # plot histogram in log space
# ax = plt.subplot(111)
# ax.hist(base_mce['collapse_probs'], bins=np.logspace(-21, 0, 200), density=True)
# ax.set_xscale("log")

# shape,loc,scale = lognorm.fit(base_mce['collapse_probs'], loc=0)

# x = np.logspace(1e-21, 0.1, 200)
# pdf = lognorm.pdf(x, shape, loc, scale)
# ax.plot(x, pdf, 'r')
# test = lognorm.expect(lambda x:1, args=(shape,), loc=loc, scale=scale)

# y = base_mce['collapse_probs']
# # ax.set_xlim([1e-3, 1e-1])
# plt.show()


#%% fit validation curve (curve fit, not MLE)

# TODO: percentage format, legend, uncertainty

from scipy.stats import lognorm
from scipy.optimize import curve_fit
f = lambda x,mu,sigma: lognorm(mu,sigma).cdf(x)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(13, 10))


theta, beta = curve_fit(f,ida_levels,val_10_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta, beta)

MCE_level = float(p[xx_pr==1.0])
ax1=fig.add_subplot(2, 2, 1)
ax1.plot(xx_pr, p)
ax1.axhline(0.1, linestyle='--', color='black')
ax1.axvline(1.0, linestyle='--', color='black')
ax1.text(2.0, 0.12, r'10% collapse risk',
          fontsize=subt_font, color='black')
ax1.text(0.25, 0.12, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='blue')
ax1.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')

ax1.set_ylabel('Collapse probability', fontsize=axis_font)
# ax1.set_xlabel(r'Scale factor', fontsize=axis_font)
ax1.set_title('10% design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [val_10_collapse[i]], 
              marker='x', markersize=15, color="red")
ax1.grid()
ax1.set_xlim([0, 4.0])
ax1.set_ylim([0, 1.0])

####
theta, beta = curve_fit(f,ida_levels,val_5_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta, beta)

MCE_level = float(p[xx_pr==1.0])
ax2=fig.add_subplot(2, 2, 2)
ax2.plot(xx_pr, p)
ax2.axhline(0.05, linestyle='--', color='black')
ax2.axvline(1.0, linestyle='--', color='black')
ax2.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')
ax2.text(2.0, 0.07, r'5% collapse risk',
          fontsize=subt_font, color='black')
ax2.text(0.25, 0.1, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='blue')

# ax2.set_ylabel('Collapse probability', fontsize=axis_font)
# ax2.set_xlabel(r'Scale factor', fontsize=axis_font)
ax2.set_title('5% design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax2.plot([lvl], [val_5_collapse[i]], 
              marker='x', markersize=15, color="red")
ax2.grid()
ax2.set_xlim([0, 4.0])
ax2.set_ylim([0, 1.0])

####
theta, beta = curve_fit(f,ida_levels,val_2_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta, beta)

MCE_level = float(p[xx_pr==1.0])
ax3=fig.add_subplot(2, 2, 3)
ax3.plot(xx_pr, p)
ax3.axhline(0.025, linestyle='--', color='black')
ax3.axvline(1.0, linestyle='--', color='black')
ax3.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')
ax3.text(2.0, 0.04, r'2.5% collapse risk',
          fontsize=subt_font, color='black')
ax3.text(0.25, 0.04, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='blue')

ax3.set_ylabel('Collapse probability', fontsize=axis_font)
ax3.set_xlabel(r'Scale factor', fontsize=axis_font)
ax3.set_title('2.5% design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax3.plot([lvl], [val_2_collapse[i]], 
              marker='x', markersize=15, color="red")
ax3.grid()
ax3.set_xlim([0, 4.0])
ax3.set_ylim([0, 1.0])

####
theta, beta = curve_fit(f,ida_levels,baseline_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta, beta)

MCE_level = float(p[xx_pr==1.0])
ax4=fig.add_subplot(2, 2, 4)
ax4.plot(xx_pr, p)
ax4.axhline(0.1, linestyle='--', color='black')
ax4.axvline(1.0, linestyle='--', color='black')
ax4.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')
ax4.text(2.0, 0.12, r'10% collapse risk',
          fontsize=subt_font, color='black')
ax4.text(0.25, 0.12, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='blue')

# ax4.set_ylabel('Collapse probability', fontsize=axis_font)
ax4.set_xlabel(r'Scale factor', fontsize=axis_font)
ax4.set_title('Baseline design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax4.plot([lvl], [baseline_collapse[i]], 
              marker='x', markersize=15, color="red")
ax4.grid()
ax4.set_xlim([0, 4.0])
ax4.set_ylim([0, 1.0])

fig.tight_layout()
plt.show()
