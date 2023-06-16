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

#%% pre-doe data

database_path = './data/'
database_file = 'training_set.csv'

df_train = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_train['max_drift'] = df_train[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
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
                      'zetaM':np.repeat(0.15,res**3)})



t0 = time.time()

fmu, fs1 = mdl_init.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

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
fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]
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
plt.colorbar()
plt.scatter(df_train['gapRatio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.contour(xx_pl, yy_pl, Z, levels=[0.1])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.title('Collapse risk', fontsize=axis_font)


#%% post-doe data

# database_path = './data/doe/old/rmse_1_percent/'
database_path = './data/doe/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

#%% predict the plotting space

t0 = time.time()

fmu, fs1 = mdl_doe.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

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
    cmap=plt.cm.Reds_r,
) 
plt.scatter(new_pts['gapRatio'][:10], new_pts['RI'][:10], 
            c=new_pts['collapse_prob'][:10],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance, first iteration', fontsize=axis_font)

plt.show()


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
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.Reds_r,
) 
plt.colorbar()
plt.scatter(df_doe['gapRatio'], df_doe['RI'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Reds_r)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Collapse risk', fontsize=axis_font)

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
    cmap=plt.cm.Reds_r,
) 
plt.scatter(new_pts['gapRatio'][-10:], new_pts['RI'][-10:], 
            c=new_pts['collapse_prob'][-10:],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance, last iteration', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.xlim([0.3, 2.0])
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

