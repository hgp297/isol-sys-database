############################################################################
#               Main TFP analysis file (plotting, ML, inverse design)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2024

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


with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
    main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

#%% collapse fragility def

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

label_size = 16
clabel_size = 12
x = np.linspace(0, 0.15, 200)

mu = log(0.1)- 0.25*inv_norm
sigma = 0.25

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

plt.close('all')
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
ax.text(0.01, 0.52, r'$\theta = 0.078$', fontsize=axis_font, color='blue')
ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
ax.hlines(y=0.84, xmin=xright, xmax=0.1, color='blue', linestyle=":")
ax.text(0.01, 0.87, r'$\theta = 0.10$', fontsize=axis_font, color='blue')
ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

lower= ln_dist.ppf(0.16)
ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
ax.hlines(y=0.16, xmin=xright, xmax=lower, color='blue', linestyle=":")
ax.text(0.01, 0.19, r'$\theta = 0.061$', fontsize=axis_font, color='blue')
ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


# ax.set_title('Replacement fragility definition', fontsize=axis_font)
ax.grid()
# ax.legend(fontsize=label_size, loc='upper center')
plt.show()

#%% prepare covariates and outcomes


main_obj.calculate_collapse()
df_raw = main_obj.ops_analysis

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

# df = df_whole.head(100).copy()

df['max_drift'] = df.PID.apply(max)
df['log_drift'] = np.log(df['max_drift'])

df['T_ratio'] = df['T_m'] / df['T_fb']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df['Bm'] = np.interp(df['zeta_e'], zetaRef, BmRef)

df['gap_ratio'] = (df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2)

#%%  dumb scatters
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

y_var = 'max_drift'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)


ax1.scatter(df['gap_ratio'], df[y_var])
ax1.set_ylabel('Peak story drift', fontsize=axis_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df['RI'], df[y_var])
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df['T_ratio'], df[y_var])
ax3.set_ylabel('Peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df['k_ratio'], df[y_var])
ax4.set_xlabel(r'$k_1/  k_2$', fontsize=axis_font)
ax4.set_title('Bearing initial stiffness', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()

#%%  dumb scatters, log edition
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

y_var = 'log_drift'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)


ax1.scatter(df['gap_ratio'], df[y_var])
ax1.set_ylabel('Log peak story drift', fontsize=axis_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df['RI'], df[y_var])
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df['T_ratio'], df[y_var])
ax3.set_ylabel('Log peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df['k_ratio'], df[y_var])
ax4.set_xlabel(r'$k_1/  k_2$', fontsize=axis_font)
ax4.set_title('Bearing initial stiffness', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()

#%%  a demonstration of Ry - Tfb relationships
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

y_var = 'T_ratio'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(1, 1, 1)


ax1.scatter(df['RI'], df[y_var])
ax1.set_ylabel('$T_M / T_{fb}$', fontsize=axis_font)
ax1.set_xlabel(r'RI', fontsize=axis_font)
ax1.grid(True)

#%%  dumb scatters
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

y_var = 'collapse_prob'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)


ax1.scatter(df['gap_ratio'], df[y_var])
ax1.set_ylabel('Log % collapse', fontsize=axis_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_title('Gap', fontsize=title_font)
ax1.set_yscale('log')
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df['RI'], df[y_var])
ax2.set_yscale('log')
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df['T_ratio'], df[y_var])
ax3.set_yscale('log')
ax3.set_ylabel('Log % collapse', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df['k_ratio'], df[y_var])
ax4.set_yscale('log')
ax4.set_xlabel(r'$k_1/  k_2$', fontsize=axis_font)
ax4.set_title('Bearing initial stiffness', fontsize=title_font)
ax4.grid(True)

fig.tight_layout()

#%% train a model

mdl = GP(df)
covariate_list = ['gap_ratio', 'T_ratio', 'k_ratio', 'RI']
mdl.set_covariates(covariate_list)
mdl.set_outcome('collapse_prob')
mdl.fit_gpr(kernel_name='rbf_iso')

#%%

data_bounds = mdl.X.agg(['min', 'max'])

n_var = data_bounds.shape[1]

min_list = [val for val in data_bounds.loc['min']]
max_list = [val for val in data_bounds.loc['max']]

import time
res = 75

xx, yy, uu = np.meshgrid(np.linspace(0.7, 2.0,
                                      res),
                          np.linspace(0.5, 2.5,
                                      res),
                          np.linspace(0.9, 4.5,
                                      res))
                             
X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                      'T_ratio':uu.ravel(),
                      'k_ratio':np.repeat(10.0,res**3),
                      'RI':yy.ravel()})



t0 = time.time()

fmu_train, fs1_train = mdl.gpr.predict(X_space, return_std=True)
fs2_train = fs1_train**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% plots

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
X_subset = X_space[X_space['T_ratio']==2.7]
fs2_subset = fs2_train[X_space['T_ratio']==2.7]
fmu_subset = fmu_train[X_space['T_ratio']==2.7]
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
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df['gap_ratio'], df['RI'], 
            c=df['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.7, 2.0])
plt.ylim([0.5, 2.5])
plt.title('Collapse risk, pre-DoE', fontsize=axis_font)
plt.show()

#%%  a demonstration of k_ratio - Tm relationships
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

x_var = 'k_ratio'
y_var = 'Q'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(1, 1, 1)


ax1.scatter(df[x_var]*df['T_m'], df[y_var])
ax1.set_ylabel(y_var, fontsize=axis_font)
ax1.set_xlabel(x_var, fontsize=axis_font)
ax1.grid(True)
# %% layout effects

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

x_var = 'h_bldg'
y_var = 'T_fb'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(1, 1, 1)


ax1.scatter(df[x_var], df[y_var])
ax1.set_ylabel(y_var, fontsize=axis_font)
ax1.set_xlabel(x_var, fontsize=axis_font)
ax1.grid(True)

#%% regression for Tfb

from sklearn.linear_model import LinearRegression
X = df_raw[['h_bldg']]**0.75
y = df_raw[['T_fb']]
reg = LinearRegression(fit_intercept=False).fit(X, y)
print('Linear regression for Tfb based on h_bldg^(3/4)')
print('score: ', reg.score(X,y))
print('coef: ', reg.coef_)

X_test = df[['h_bldg']]**0.75
y_test = df[['T_fb']]
y_pred = reg.predict(X_test)

fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(1, 1, 1)


ax1.scatter(X_test, y_test)
ax1.plot(X_test, y_pred, color="blue", linewidth=3)
ax1.set_ylabel(r'T_fb', fontsize=axis_font)
ax1.set_xlabel(r'$h^{0.75}$', fontsize=axis_font)
ax1.grid(True)

# %% T_ratio and k_ratio/Tfb

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20

fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(1, 1, 1)


ax1.scatter(df['T_ratio']*(0.078*df['h_bldg']**0.75), df['k_ratio'] )
ax1.set_xlabel(r'$\alpha_T$', fontsize=axis_font)
ax1.set_ylabel(r'$\alpha_k / T_{fbe}$', fontsize=axis_font)
ax1.grid(True)


#%% regression for Q

from sklearn.linear_model import LinearRegression
X = df_raw[['k_ratio', 'T_m']]
y = df_raw[['Q']]
reg = LinearRegression(fit_intercept=True).fit(X, y)
print('Linear regression for Q based on k_ratio and T_m')
print('score: ', reg.score(X,y))
print('coef: ', reg.coef_)
print('intercept: ', reg.intercept_)