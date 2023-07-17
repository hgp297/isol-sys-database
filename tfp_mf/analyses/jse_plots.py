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

ax.plot(x, p, label='Collapse (peak)', color='blue')

mu_irr = log(0.01)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

ax.plot(x, p_irr, color='red', label='Irreparable (residual)')

axis_font = 20
subt_font = 18
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, xleft])
ax.set_ylabel('Limit state probability', fontsize=axis_font)
ax.set_xlabel('Drift ratio', fontsize=axis_font)

ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
ax.hlines(y=0.5, xmin=exp(mu), xmax=0.15, color='blue', linestyle=":")
ax.text(0.115, 0.52, r'$\theta = 0.078$', fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
ax.hlines(y=0.84, xmin=0.1, xmax=xleft, color='blue', linestyle=":")
ax.text(0.115, 0.87, r'$\theta = 0.10$', fontsize=axis_font, color='blue')
ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.16)
ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
ax.hlines(y=0.16, xmin=lower, xmax=xleft, color='blue', linestyle=":")
ax.text(0.115, 0.19, r'$\theta = 0.061$', fontsize=axis_font, color='blue')
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

ax.text(0.002, 0.19, r'$\theta = 0.007$', fontsize=axis_font, color='red')
ax.text(0.002, 0.87, r'$\theta = 0.013$', fontsize=axis_font, color='red')
ax.text(0.002, 0.53, r'$\theta = 0.010$', fontsize=axis_font, color='red')

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
plt.title('Collapse risk, pre-DoE', fontsize=axis_font)
plt.show()

#%% inverse design: mean

from pred import get_steel_coefs, calc_upfront_cost
plt.close('all')
steel_price = 2.00
coef_dict = get_steel_coefs(df_train, steel_per_unit=steel_price)

risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(fmu_train, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

X_baseline = pd.DataFrame(np.array([[1.0, 2.0, 3.0, 0.15]]),
                          columns=['gapRatio', 'RI', 'Tm', 'zetaM'])
baseline_risk = mdl_init.gpr.predict(X_baseline)
baseline_risk = baseline_risk.item()
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


# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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
space_collapse_pred = pd.DataFrame(fmu_train, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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
space_collapse_pred = pd.DataFrame(fmu_train, columns=['collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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
#%% doe convergence plots
database_path = '../data/doe/'

import matplotlib.pyplot as plt
import pandas as pd
plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

rmse_df = pd.read_csv(database_path+'rmse.csv', header=None)
mae_df = pd.read_csv(database_path+'mae.csv', header=None)
mae_df = mae_df.transpose()
mae_df.columns = ['mae']
rmse_df = rmse_df.transpose()
rmse_df.columns = ['rmse']

rmse_df_naive = pd.read_csv(database_path+'rmse_naive.csv', header=None)
mae_df_naive = pd.read_csv(database_path+'mae_naive.csv', header=None)
mae_df_naive = mae_df_naive.transpose()
mae_df_naive.columns = ['mae']
rmse_df_naive = rmse_df_naive.transpose()
rmse_df_naive.columns = ['rmse']

plt.close('all')

fig = plt.figure(figsize=(13, 6))

ax1=fig.add_subplot(1, 2, 1)
ax1.plot(rmse_df.index*10, rmse_df['rmse'], label='Adaptive')
ax1.plot(rmse_df_naive.index*10, rmse_df_naive['rmse'], label='Naive')
ax1.set_title(r'Root mean squared error', fontsize=axis_font)
ax1.set_xlabel(r'Points added', fontsize=axis_font)
ax1.set_ylabel(r'Metric', fontsize=axis_font)
ax1.set_xlim([0, 140])
ax1.set_ylim([0.19, 0.28])
plt.grid(True)


ax2=fig.add_subplot(1, 2, 2)
ax2.plot(rmse_df.index*10, mae_df['mae'], label='Adaptive')
ax2.plot(rmse_df_naive.index*10, mae_df_naive['mae'], label='Naive')
ax2.set_title('Mean absolute error', fontsize=axis_font)
ax2.set_xlabel('Points added', fontsize=axis_font)
# ax2.set_ylabel('Metric', fontsize=axis_font)
ax2.set_xlim([0, 140])
ax2.set_ylim([0.05, 0.14])
plt.grid(True)
plt.legend(fontsize=axis_font)

fig.tight_layout()
plt.show()

#%% naive-doe data

# database_path = '../data/doe/'
database_file = 'rmse_naive_set.csv'

df_naive = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_naive['max_drift'] = df_naive[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_naive['collapse_prob'] = ln_dist.cdf(df_naive['max_drift'])

mdl_naive = GP(df_naive)
mdl_naive.set_outcome('collapse_prob')
mdl_naive.fit_gpr(kernel_name='rbf_ard')

t0 = time.time()

fmu_naive, fs1_naive = mdl_naive.gpr.predict(X_space, return_std=True)
fs2_naive = fs1_naive**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% post-doe data


# database_path = '../data/doe/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

# mdl_drift = GP(df_doe)
# mdl_drift.set_outcome('max_drift')
# mdl_drift.fit_gpr(kernel_name='rbf_ard')


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
baseline_fs1 = baseline_fs1.item()
baseline_fs2 = baseline_fs2.item()

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
lvls = [0.025, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5]
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
plt.title('Collapse risk', fontsize=axis_font)
plt.show()

# collapse probabilities (+1 std)
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
fs1_subset = fs1[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]
Z = fmu_subset + fs1_subset
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
plt.title('Collapse risk, mean + 1 std', fontsize=axis_font)
plt.show()


# just variance

x_pl = np.unique(xx)
y_pl = np.unique(yy)

xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2[X_space['Tm']==3.25]
fs1_subset = fs1[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]
Z = fs1_subset
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
# lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)
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
plt.title('Variance', fontsize=axis_font)
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

#%% inverse design: mean

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


# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

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

#%% inverse design (+1std prediction interval)

import time
res = 75
xx, yy, uu = np.meshgrid(np.linspace(1.0, 4.0,
                                      res),
                          np.linspace(0.25, 1.0,
                                      res),
                          np.linspace(2.5, 4.0,
                                      res))
                             
X_space_pred_int = pd.DataFrame({'gapRatio':xx.ravel(),
                      'RI':yy.ravel(),
                      'Tm':uu.ravel(),
                      'zetaM':np.repeat(0.2,res**3)})

t0 = time.time()

fmu_pred_int, fs1_pred_int = mdl_init.gpr.predict(X_space_pred_int, return_std=True)
fs2_pred_int = fs1_pred_int**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space_pred_int.shape[0],
                                                                tp))

    
# in the filter-design process, only one of cost/dt is likely to control

baseline_costs = calc_upfront_cost(X_baseline, coef_dict)
baseline_total = baseline_costs['total'].item()
baseline_steel = baseline_costs['steel'].item()
baseline_land = baseline_costs['land'].item()

# least upfront cost of the viable designs

baseline_risk_95 = baseline_risk + baseline_fs1

print('========== Baseline design (+1std pred int) ============')
print('Design target', f'{risk_thresh:.2%}')
print('Upfront cost of selected design: ',
      f'${baseline_total:,.2f}')
print('Predicted collapse risk: ',
      f'{baseline_risk_95:.2%}')
print(X_baseline)


# # select best viable design
# risk_thresh = 0.1
# space_collapse_pred = pd.DataFrame(fmu_pred_int+fs1_pred_int,
#                                     columns=['collapse probability'])
# ok_risk = X_space_pred_int.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space_pred_int[X_space_pred_int.index.isin(ok_risk.index)]

# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



# print('========== Inverse design (+1std pred int) ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.2%}')
# print(best_design)

# risk_thresh = 0.05
# ok_risk = X_space_pred_int.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space_pred_int[X_space_pred_int.index.isin(ok_risk.index)]
    
# # in the filter-design process, only one of cost/dt is likely to control

# # select best viable design
# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']


# print('========== Inverse design (+1std pred int) ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.2%}')
# print(best_design)

# risk_thresh = 0.025
# ok_risk = X_space_pred_int.loc[space_collapse_pred['collapse probability']<=
#                       risk_thresh]

# X_design = X_space_pred_int[X_space_pred_int.index.isin(ok_risk.index)]
    
# # in the filter-design process, only one of cost/dt is likely to control

# # select best viable design
# upfront_costs = calc_upfront_cost(X_design, coef_dict)
# cheapest_design_idx = upfront_costs.idxmin()
# design_upfront_cost = upfront_costs.min()

# # least upfront cost of the viable designs
# best_design = X_design.loc[cheapest_design_idx]
# design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']



# print('========== Inverse design (+1std pred int) ============')
# print('Design target', f'{risk_thresh:.2%}')
# print('Upfront cost of selected design: ',
#       f'${design_upfront_cost:,.2f}')
# print('Predicted collapse risk: ',
#       f'{design_collapse_risk:.4%}')
# print(best_design)

#%% doe effect plots
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]

n_new = df_doe.shape[0] - df_train.shape[0]
n_naive = df_naive.shape[0] - df_train.shape[0]

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
ax1.set_title('a) Collapse risk, pre-DoE', fontsize=axis_font)
# ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.grid()

# then we show results if naive predictions are used


ax2=fig.add_subplot(2, 2, 2)
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2_naive[X_space['Tm']==3.25]
fmu_subset = fmu_naive[X_space['Tm']==3.25]
Z = fmu_subset.reshape(xx_pl.shape)

new_pts = df_naive.tail(n_naive)
# new_pts_first = new_pts.head(20)

lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax2.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax2.clabel(cs, fontsize=clabel_size)

ax2.scatter(new_pts['gapRatio'], new_pts['RI'], 
            c=new_pts.index,
            edgecolors='k', s=40.0, cmap='Blues')
ax2.set_xlim([0.3, 2.0])
ax2.set_ylim([0.5, 2.0])
ax2.set_title('b) Collapse risk, naive DoE', fontsize=axis_font)
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

ax3=fig.add_subplot(2, 2, 4)
# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['Tm']==3.25]
Z = criterion.reshape(xx_pl.shape)

lvls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
cs = ax3.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
ax3.clabel(cs, fontsize=clabel_size)

new_pts = df_doe.tail(n_new)
ax3.scatter(new_pts['gapRatio'], new_pts['RI'], 
            c=new_pts.index,
            edgecolors='k', s=40.0, cmap='Blues')
ax3.set_xlim([0.3, 2.0])
ax3.set_ylim([0.5, 2.0])
ax3.set_title('d) Weighted variance for adaptive DoE', fontsize=axis_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
# ax3.set_ylabel(r'$R_y$', fontsize=axis_font)
ax3.grid()

# then show final results

ax4=fig.add_subplot(2, 2, 3)
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
ax4.set_title('c) Collapse risk, adaptive DoE', fontsize=axis_font)
ax4.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax4.set_ylabel(r'$R_y$', fontsize=axis_font)
ax4.grid()

fig.tight_layout()
plt.show()


#%% doe effect plots
# X_subset = X_space[X_space['Tm']==3.25]
# fs2_subset = fs2[X_space['Tm']==3.25]
# fmu_subset = fmu[X_space['Tm']==3.25]

# n_new = df_doe.shape[0] - df_train.shape[0]

# fig = plt.figure(figsize=(13, 10))

# # first we show training model

# ax1=fig.add_subplot(2, 2, 1)
# # collapse predictions
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# X_subset = X_space[X_space['Tm']==3.25]
# fs2_subset = fs2_train[X_space['Tm']==3.25]
# fmu_subset = fmu_train[X_space['Tm']==3.25]
# Z = fmu_subset.reshape(xx_pl.shape)


# lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
# cs = ax1.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)
# ax1.clabel(cs, fontsize=clabel_size)
# ax1.scatter(df_train['gapRatio'], df_train['RI'], 
#             c=df_train['collapse_prob'],
#             edgecolors='k', s=40.0, cmap=plt.cm.Blues)
# ax1.set_xlim([0.3, 2.0])
# ax1.set_ylim([0.5, 2.0])
# ax1.set_title('Collapse risk, pre-DoE', fontsize=axis_font)
# # ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
# ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax1.grid()

# # then we show first iteration of weighted variance

# # tMSE criterion
# from numpy import exp
# pi = 3.14159
# T = 0.5
# fs2_subset = fs2_train[X_space['Tm']==3.25]
# fmu_subset = fmu_train[X_space['Tm']==3.25]
# Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
# criterion = np.multiply(Wx, fs2_subset)

# ax2=fig.add_subplot(2, 2, 2)
# # collapse predictions
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# X_subset = X_space[X_space['Tm']==3.25]
# fs2_subset = fs2_train[X_space['Tm']==3.25]
# fmu_subset = fmu_train[X_space['Tm']==3.25]
# Z = criterion.reshape(xx_pl.shape)

# new_pts = df_doe.tail(n_new)
# new_pts_first = new_pts.head(20)

# lvls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
# cs = ax2.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)
# ax2.clabel(cs, fontsize=clabel_size)

# ax2.scatter(new_pts_first['gapRatio'], new_pts_first['RI'], 
#             c=new_pts_first.index,
#             edgecolors='k', s=40.0)
# ax2.set_xlim([0.3, 2.0])
# ax2.set_ylim([0.5, 2.0])
# ax2.set_title('Weighted variance, first iteration', fontsize=axis_font)
# # ax2.set_xlabel(r'Gap ratio', fontsize=axis_font)
# # ax2.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax2.grid()

# # then we show all added points

# fs2_subset = fs2[X_space['Tm']==3.25]
# fmu_subset = fmu[X_space['Tm']==3.25]

# # tMSE criterion
# from numpy import exp
# pi = 3.14159
# T = 0.5
# Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
# criterion = np.multiply(Wx, fs2_subset)

# ax3=fig.add_subplot(2, 2, 4)
# # collapse predictions
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
# X_subset = X_space[X_space['Tm']==3.25]
# Z = criterion.reshape(xx_pl.shape)

# lvls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
# cs = ax3.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)
# ax3.clabel(cs, fontsize=clabel_size)

# ax3.scatter(new_pts['gapRatio'], new_pts['RI'], 
#             c=new_pts.index,
#             edgecolors='k', s=40.0)
# ax3.set_xlim([0.3, 2.0])
# ax3.set_ylim([0.5, 2.0])
# ax3.set_title('Weighted variance, last iteration', fontsize=axis_font)
# ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
# # ax3.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax3.grid()

# # then show final results

# ax4=fig.add_subplot(2, 2, 3)
# # collapse predictions
# Z = fmu_subset.reshape(xx_pl.shape)
# new_pts = df_doe.tail(n_new)
# new_pts_first = new_pts.head(10)

# lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
# cs = ax4.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)
# ax4.clabel(cs, fontsize=clabel_size)

# ax4.scatter(df_doe['gapRatio'], df_doe['RI'], 
#             c=df_doe['collapse_prob'], cmap='Blues',
#             edgecolors='k', s=40.0)
# ax4.set_xlim([0.3, 2.0])
# ax4.set_ylim([0.5, 2.0])
# ax4.set_title('Collapse risk, post-DoE', fontsize=axis_font)
# ax4.set_xlabel(r'Gap ratio', fontsize=axis_font)
# ax4.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax4.grid()

# fig.tight_layout()
# plt.show()

#%% contour plots, gap vs Ry, highlight Ry

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


fig, ax = plt.subplots(2, 2, figsize=(13, 10))
plt.setp(ax, xticks=np.arange(0.5, 4.0, step=0.5))


ax1 = ax[0][0]
ax2 = ax[0][1]
ax3 = ax[1][0]
ax4 = ax[1][1]

import numpy as np
# x is gap, y is Ry
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.5
y_min = 0.5
y_max = 2.0

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 200

xx, yy = np.meshgrid(np.linspace(x_min,
                                 x_max,
                                 res),
                     np.linspace(y_min,
                                 y_max,
                                 res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                     y_var:yy.ravel(),
                     third_var:np.repeat(2.5,
                                         res*res),
                     fourth_var:np.repeat(0.15,
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

fmu = mdl_doe.gpr.predict(X_plot)
Z = fmu.reshape(xx.shape)

cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)

probDes = 0.1
from scipy.interpolate import RegularGridInterpolator
RyList = [1.0, 2.0]
for j in range(len(RyList)):
    RyTest = RyList[j]
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = RyTest
    lq = interp(pts)
    
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax1.vlines(x=theGap, ymin=0.49, ymax=RyTest, color='red')
    ax1.hlines(y=RyTest, xmin=0.3, xmax=theGap, color='red')
    ax1.text(theGap+0.05, 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red')
    ax1.plot([theGap], [RyTest], marker='*', markersize=15, color="red")

df_sc = df_doe[(df_doe['Tm']<=2.75) & (df_doe['zetaM']<=0.17) & (df_doe['zetaM']>=0.13)]

ax1.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_prob'], cmap='Blues',
            s=30, edgecolors='k')

ax1.clabel(cs, fontsize=clabel_size)

ax1.contour(xx, yy, Z, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2,))

ax1.set_xlim([0.3, 2.0])
ax1.set_ylim([0.49, 2.01])


ax1.grid(visible=True)
ax1.set_title(r'$T_M = 2.5$ s, $\zeta_M = 0.15$', fontsize=title_font)
# ax1.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)


# #####
# x is gap, y is Ry
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.5
y_min = 0.5
y_max = 2.0

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 200

xx, yy = np.meshgrid(np.linspace(x_min,
                                 x_max,
                                 res),
                     np.linspace(y_min,
                                 y_max,
                                 res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                     y_var:yy.ravel(),
                     third_var:np.repeat(3.0,
                                         res*res),
                     fourth_var:np.repeat(0.15,
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

fmu = mdl_doe.gpr.predict(X_plot)
Z = fmu.reshape(xx.shape)

cs = ax2.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)

probDes = 0.1
from scipy.interpolate import RegularGridInterpolator
RyList = [1.0, 2.0]
for j in range(len(RyList)):
    RyTest = RyList[j]
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = RyTest
    lq = interp(pts)
    
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax2.vlines(x=theGap, ymin=0.49, ymax=RyTest, color='red')
    ax2.hlines(y=RyTest, xmin=0.3, xmax=theGap, color='red')
    ax2.text(theGap+0.05, 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red')
    ax2.plot([theGap], [RyTest], marker='*', markersize=15, color="red")

df_sc = df_doe[(df_doe['Tm'] >= 2.75) & (df_doe['Tm'] <= 3.25) &
               (df_doe['zetaM']<=0.17) & (df_doe['zetaM']>=0.13)]

ax2.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_prob'], cmap='Blues',
            s=30, edgecolors='k')

ax2.clabel(cs, fontsize=clabel_size)

ax2.contour(xx, yy, Z, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2,))

ax2.set_xlim([0.3, 2.0])
ax2.set_ylim([0.49, 2.01])


ax2.grid(visible=True)
ax2.set_title(r'$T_M = 3.0$ s, $\zeta_M = 0.15$', fontsize=title_font)
# ax2.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax2.set_ylabel(r'$R_y$', fontsize=axis_font)

#####
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.5
y_min = 0.5
y_max = 2.0

xx, yy = np.meshgrid(np.linspace(x_min,
                                  x_max,
                                  res),
                      np.linspace(y_min,
                                  y_max,
                                  res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                      y_var:yy.ravel(),
                      third_var:np.repeat(3.5,
                                          res*res),
                      fourth_var:np.repeat(0.15,
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

fmu = mdl_doe.gpr.predict(X_plot)
Z = fmu.reshape(xx.shape)

cs = ax3.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)

from scipy.interpolate import RegularGridInterpolator
RyList = [1.0, 2.0]
adj_list = [-0.12, 0.05, 0.05]
for j in range(len(RyList)):
    RyTest = RyList[j]
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = RyTest
    lq = interp(pts)
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax3.vlines(x=theGap, ymin=0.49, ymax=RyTest, color='red')
    ax3.hlines(y=RyTest, xmin=0.3, xmax=theGap, color='red')
    ax3.text(theGap+adj_list[j], 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red')
    ax3.plot([theGap], [RyTest], marker='*', markersize=15, color="red")

ax3.clabel(cs, fontsize=clabel_size)

ax3.contour(xx, yy, Z, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2,))

ax3.set_xlim([0.3, 2.0])
ax3.set_ylim([0.49, 2.01])

df_sc = df_doe[(df_doe['Tm']>=3.25) & (df_doe['Tm']<3.75) & 
               (df_doe['zetaM']<=0.17) & (df_doe['zetaM']>=0.13)]

sc = ax3.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_prob'], cmap='Blues',
            s=30, edgecolors='k')

ax3.grid(visible=True)
ax3.set_title(r'$T_M = 3.5$ s, $\zeta_M = 0.15$', fontsize=title_font)
ax3.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
# ax3.set_ylabel(r'$R_y$', fontsize=axis_font)


#####
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.5
y_min = 0.5
y_max = 2.0

xx, yy = np.meshgrid(np.linspace(x_min,
                                  x_max,
                                  res),
                      np.linspace(y_min,
                                  y_max,
                                  res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                      y_var:yy.ravel(),
                      third_var:np.repeat(4.0,
                                          res*res),
                      fourth_var:np.repeat(0.15,
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

fmu = mdl_doe.gpr.predict(X_plot)
Z = fmu.reshape(xx.shape)

cs = ax4.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)

from scipy.interpolate import RegularGridInterpolator
RyList = [1.0, 2.0]
adj_list = [-0.12, 0.05, 0.05]
for j in range(len(RyList)):
    RyTest = RyList[j]
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = RyTest
    lq = interp(pts)
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax4.vlines(x=theGap, ymin=0.49, ymax=RyTest, color='red')
    ax4.hlines(y=RyTest, xmin=0.3, xmax=theGap, color='red')
    ax4.text(theGap+adj_list[j], 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color='red')
    ax4.plot([theGap], [RyTest], marker='*', markersize=15, color="red")

ax4.clabel(cs, fontsize=clabel_size)

ax4.contour(xx, yy, Z, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2,))

ax4.set_xlim([0.3, 2.0])
ax4.set_ylim([0.49, 2.01])

df_sc = df_doe[(df_doe['Tm']>=3.75) &
               (df_doe['zetaM']<=0.17) & (df_doe['zetaM']>=0.13)]

sc = ax4.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_prob'], cmap='Blues',
            s=30, edgecolors='k')

ax4.grid(visible=True)
ax4.set_title(r'$T_M = 4.0$ s, $\zeta_M = 0.15$', fontsize=title_font)
ax4.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
# ax4.set_ylabel(r'$R_y$', fontsize=axis_font)

# handles, labels = sc.legend_elements(prop="colors", alpha=0.6)
# legend2 = ax4.legend(handles, labels, loc="lower right", title="% collapse",
#                       fontsize=subt_font, title_fontsize=subt_font)

fig.tight_layout()
plt.show()

#%% contour plots, highlight different targets

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
clabel_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

import numpy as np
# x is gap, y is Ry
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.5
y_min = 0.5
y_max = 2.0

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 200

xx, yy = np.meshgrid(np.linspace(x_min,
                                  x_max,
                                  res),
                      np.linspace(y_min,
                                  y_max,
                                  res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                      y_var:yy.ravel(),
                      third_var:np.repeat(3.0,
                                          res*res),
                      fourth_var:np.repeat(0.15,
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

fmu = mdl_doe.gpr.predict(X_plot)
Z = fmu.reshape(xx.shape)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                  levels=lvls)


prob_list = [0.025, 0.05, 0.1]
offset_list = [0.95, 0.72, 0.55]
color_list = ['red', 'red', 'red']
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = 1.0
    lq = interp(pts)
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    
    ax1.vlines(x=theGap, ymin=0.49, ymax=1.0, color=color_list[j],
                linewidth=2.0)
    ax1.hlines(y=1.0, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
    ax1.text(offset_list[j], 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color=color_list[j])
    ax1.plot([theGap], [1.0], marker='*', markersize=15, color=color_list[j])
    
prob_list = [0.025, 0.05, 0.1]
offset_list = [1.5, 1.02, 0.8]
color_list = ['blue', 'blue', 'blue']
from scipy.interpolate import RegularGridInterpolator
for j, prob_des in enumerate(prob_list):
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = 2.0
    lq = interp(pts)
    
    theGapIdx = np.argmin(abs(lq - prob_des))
    
    theGap = xq[theGapIdx]
    
    ax1.vlines(x=theGap, ymin=0.49, ymax=2.0, color=color_list[j],
                linewidth=2.0)
    ax1.hlines(y=1.995, xmin=0.3, xmax=theGap, color='blue', linewidth=2.0)
    ax1.text(offset_list[j], 1.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
              fontsize=subt_font, color=color_list[j])
    ax1.plot([theGap], [2.0], marker='*', markersize=15, color=color_list[j])


df_sc = df_doe[(df_doe['Tm']>=2.8) & (df_doe['Tm']<=3.2) & 
            (df_doe['zetaM']<=0.17) & (df_doe['zetaM']>=0.13)]

ax1.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_prob'], cmap='Blues',
            s=60, edgecolors='k')

ax1.clabel(cs, fontsize=clabel_size)
ax1.set_xlim([0.3, 2.0])
ax1.set_ylim([0.5, 2.0])


ax1.grid(visible=True)
ax1.set_title(r'$T_M = 3.00$ s, $\zeta_M = 0.15$', fontsize=title_font)
ax1.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)

# handles, labels = sc.legend_elements(prop="colors", alpha=0.6)
# legend2 = ax1.legend(handles, labels, loc="lower right", title="% collapse",
#                       fontsize=subt_font, title_fontsize=subt_font)

# ax1.contour(xx, yy, Z, levels = prob_list, colors=('red', 'brown', 'black'),
#             linestyles=('-'),linewidths=(2,))
plt.show()

#%% contour plots, Tm vs zeta

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# title_font=22
# axis_font = 22
# subt_font = 20
# label_size = 20
# clabel_size = 16
# mpl.rcParams['xtick.labelsize'] = label_size 
# mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')


# fig, ax = plt.subplots(2, 2, figsize=(13, 10))
# plt.setp(ax, xticks=np.arange(0.5, 4.0, step=0.5))


# ax1 = ax[0][0]
# ax2 = ax[0][1]
# ax3 = ax[1][0]
# ax4 = ax[1][1]

# import numpy as np
# # x is gap, y is Ry
# x_var = 'Tm'
# y_var = 'zetaM'
# third_var = 'gapRatio'
# fourth_var = 'RI'
# x_min = 2.5
# x_max = 4.0
# y_min = 0.1
# y_max = 0.2

# lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

# res = 200

# xx, yy = np.meshgrid(np.linspace(x_min,
#                                  x_max,
#                                  res),
#                      np.linspace(y_min,
#                                  y_max,
#                                  res))

# X_pl = pd.DataFrame({x_var:xx.ravel(),
#                      y_var:yy.ravel(),
#                      third_var:np.repeat(1.0,
#                                          res*res),
#                      fourth_var:np.repeat(1.0,
#                                           res*res)})

# X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

# fmu = mdl_doe.gpr.predict(X_plot)
# Z = fmu.reshape(xx.shape)

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                   levels=lvls)

# df_sc = df_doe[(df_doe['gapRatio']<=1.15) & (df_doe['gapRatio']>=0.85) & 
#                (df_doe['RI']<=1.15) & (df_doe['RI']>=0.85)]

# ax1.scatter(df_sc[x_var],
#             df_sc[y_var],
#             c=df_sc['collapse_prob'], cmap='Blues',
#             s=30, edgecolors='k')

# ax1.clabel(cs, fontsize=clabel_size)

# ax1.contour(xx, yy, Z, levels = [0.1], colors=('red'),
#             linestyles=('-'),linewidths=(2,))

# ax1.set_xlim([2.5, 4.0])
# ax1.set_ylim([0.1, 0.2])


# ax1.grid(visible=True)
# ax1.set_title(r'$R_y = 1.0$ s, Gap ratio $= 0.15$', fontsize=title_font)
# ax1.set_xlabel(r'$T_M$', fontsize=axis_font)
# ax1.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

#%% cost sens
# land_costs = [2151., 3227., 4303., 5379.]
land_costs = [200., 300., 400., 500.]
# steel_costs = [1., 2., 3., 4.]
steel_costs = [0.5, 0.75, 1., 2.]

import numpy as np
gap_price_grid = np.zeros([4,4])
Ry_price_grid = np.zeros([4,4])
Tm_price_grid = np.zeros([4,4])
zetaM_price_grid = np.zeros([4,4])
moat_price_grid = np.zeros([4,4])

risk_thresh = 0.1
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

# risk_thresh = 0.025
# ok_risk = X_space.loc[space_drift['max_drift_pred']<=
#                       risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]

for idx_l, land in enumerate(land_costs):
    for idx_s, steel in enumerate(steel_costs):
        steel_price = steel
        coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)
        
        # lcps = land/(3.28**2)
        lcps = land
        upfront_costs = calc_upfront_cost(X_design, coef_dict, 
                                          land_cost_per_sqft=lcps)
        
        cheapest_design_idx = upfront_costs['total'].idxmin()
        design_upfront_cost = upfront_costs['total'].min()

        # least upfront cost of the viable designs
        best_design = X_design.loc[cheapest_design_idx]
        gap_price_grid[idx_l][idx_s] = best_design['gapRatio']
        Ry_price_grid[idx_l][idx_s] = best_design['RI']
        Tm_price_grid[idx_l][idx_s] = best_design['Tm']
        zetaM_price_grid[idx_l][idx_s] = best_design['zetaM']

        from numpy import interp
        # from ASCE Ch. 17, get damping multiplier
        zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
        
        B_m = interp(best_design['zetaM'], zetaRef, BmRef)
        
        # design displacement
        g = 386.4
        pi = 3.14159
        moat_price_grid[idx_l][idx_s] = (g*1.017*best_design['Tm']/
                                         (4*pi**2*B_m)*best_design['gapRatio'])

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

steel_rows = ['$0.50', '$0.75', '$1.00', '$2.00']
land_cols=['$200', '$300', '$400', '$500', ]
# print(gap_price_grid)
# print(Ry_price_grid)
# print(Tm_price_grid)
# print(zetaM_price_grid)

gap_df = pd.DataFrame(data=gap_price_grid,
                      index=land_cols,
                      columns=steel_rows)

Ry_df = pd.DataFrame(data=Ry_price_grid,
                      index=land_cols,
                      columns=steel_rows)

Tm_df = pd.DataFrame(data=Tm_price_grid,
                      index=land_cols,
                      columns=steel_rows)

moat_df = pd.DataFrame(data=moat_price_grid,
                      index=land_cols,
                      columns=steel_rows)

# Draw a heatmap with the numeric values in each cell
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]

sns.heatmap(gap_df, annot=True, fmt='.3g', cmap='Blues', cbar=False,
            linewidths=.5, ax=ax1,  annot_kws={'size': 18})
ax1.set_ylabel('Land cost per sq ft.', fontsize=axis_font)
ax1.set_title('Gap ratio', fontsize=subt_font)

sns.heatmap(Ry_df, annot=True, fmt='.3g', cmap='Blues', cbar=False,
            linewidths=.5, ax=ax2, yticklabels=False,  annot_kws={'size': 18})
ax2.set_title(r'$R_y$', fontsize=subt_font)

sns.heatmap(Tm_df, annot=True, fmt='.3g', cmap='Blues', cbar=False,
            linewidths=.5, ax=ax3,  annot_kws={'size': 18})
ax3.set_xlabel('Steel cost per lb.', fontsize=axis_font)
ax3.set_title(r'$T_M$ (s)', fontsize=subt_font)
ax3.set_ylabel('Land cost per sq ft.', fontsize=axis_font)
fig.tight_layout()

sns.heatmap(moat_df, annot=True, fmt='.3g', cmap='Blues', cbar=False,
            linewidths=.5, ax=ax4, yticklabels=False,  annot_kws={'size': 18})
ax4.set_xlabel('Steel cost per lb.', fontsize=axis_font)
ax4.set_title(r'Moat gap (in)', fontsize=subt_font)
fig.tight_layout()
plt.show()

#%% prediction accuracy

# import matplotlib.pyplot as plt
# import pandas as pd

# plt.close('all')
# y_hat = mdl_doe.gpr.predict(mdl_doe.X)
# y_true = mdl_doe.y

# plt.figure()
# plt.scatter(y_hat, y_true)
# plt.plot([0, 1.0], [0, 1.0], linestyle='-',color='black')
# plt.plot([0, 1.0], [0, 1.1], linestyle='--',color='black')
# plt.plot([0, 1.0], [0, 0.9], linestyle='--',color='black')
# plt.title('Prediction accuracy')
# plt.xlabel('Predicted collapse %')
# plt.ylabel('True collapse %')
# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])
# plt.grid(True)
# plt.show()



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

# ax.text(0.095, 0, u'\u2192', fontsize=axis_font, color='red')
# ax.text(0.095, 1, u'\u2192', fontsize=axis_font, color='red')
# ax.text(0.095, 2, u'\u2192', fontsize=axis_font, color='red')
# ax.text(0.095, 3, u'\u2192', fontsize=axis_font, color='red')

# ax.text(0.084, 0, f'{repl_cases_10} runs', fontsize=axis_font, color='red')
# ax.text(0.084, 1, f'{repl_cases_5} runs', fontsize=axis_font, color='red')
# ax.text(0.084, 2, f'{repl_cases_2} runs', fontsize=axis_font, color='red')
# ax.text(0.084, 3, f'{base_repl_cases} runs', fontsize=axis_font, color='red')

ax.text(0.075, 3.4, r'10% threshold', fontsize=axis_font, color='black')
# ax.set_xlim(0, 0.1)

ax.set_xscale("log")
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


theta_10, beta_10 = curve_fit(f,ida_levels,val_10_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_10, beta_10)

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
theta_5, beta_5 = curve_fit(f,ida_levels,val_5_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_5, beta_5)

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
theta_2, beta_2 = curve_fit(f,ida_levels,val_2_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_2, beta_2)

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
theta_base, beta_base = curve_fit(f,ida_levels,baseline_collapse)[0]
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_base, beta_base)

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

print('10% fit mean:', exp(theta_10))
print('10% fit beta:', beta_10)
print('5% fit mean:', exp(theta_5))
print('5% fit beta:', beta_5)
print('2.5% fit mean:', exp(theta_2))
print('2.5% fit beta:', beta_2)
print('Baseline fit mean:', exp(theta_base))
print('Baseline fit beta:', beta_base)

#%% dumb scatters

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

y_var = 'max_drift'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)


ax1.scatter(df_doe['gapRatio'], df_doe[y_var])
ax1.set_ylabel('Peak story drift', fontsize=axis_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df_doe['RI'], df_doe[y_var])
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df_doe['Tm'], df_doe[y_var])
ax3.set_ylabel('Peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M$', fontsize=axis_font)
ax3.set_title('Bearing period', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df_doe['zetaM'], df_doe[y_var])
ax4.set_xlabel(r'$\zeta_M$', fontsize=axis_font)
ax4.set_title('Bearing damping', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()
plt.show()