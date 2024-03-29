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
# mean_log_drift = 0.05
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

label_size = 16
clabel_size = 12
x = np.linspace(0, 0.15, 200)

mu = log(mean_log_drift)

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
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
plt.show()

#%% prepare covariates and outcomes


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
ax1.set_ylim([0, 0.1])
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df['RI'], df[y_var])
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('Superstructure strength', fontsize=title_font)
ax2.set_ylim([0, 0.1])
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df['T_ratio'], df[y_var])
ax3.set_ylabel('Peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('Bearing period ratio', fontsize=title_font)
ax3.set_ylim([0, 0.1])
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df['k_ratio'], df[y_var])
ax4.set_xlabel(r'$k_1/  k_2$', fontsize=axis_font)
ax4.set_title('Bearing initial stiffness', fontsize=title_font)
ax4.set_ylim([0, 0.1])
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
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
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

xx, yy, uu = np.meshgrid(np.linspace(0.5,2.0,
                                      res),
                          np.linspace(0.5, 2.25,
                                      res),
                          np.linspace(1.0, 5.0,
                                      res))
                             
X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                      'RI':yy.ravel(),
                      'T_ratio':uu.ravel(),
                      'zeta_e':np.repeat(0.2,res**3)})



t0 = time.time()

fmu_train, fs1_train = mdl.gpr.predict(X_space, return_std=True)
fs2_train = fs1_train**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% loocv stuff

gp_obj = mdl.gpr._final_estimator
X_train = gp_obj.X_train_
iso_lengthscale = gp_obj.kernel_.theta[1]

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



#%% plots, gap ry

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
X_subset = X_space[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fs2_subset = fs2_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fmu_subset = fmu_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
Z = fmu_subset.reshape(xx_pl.shape)

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

#%% LOOCV plots


fig = plt.figure(figsize=(13, 8))

loo_error = X_subset.apply(lambda row: loo_error_approx(row, X_train, iso_lengthscale, e_cv_sq),
                           axis='columns', result_type='expand')

NRMSE_cv = ((np.sum(np.divide(alpha_, K_inv_diag)**2)/len(alpha_))**0.5/
            (max(mdl.y[y_var]) - min(mdl.y[y_var])))

loo_subset = np.asarray(loo_error[X_space['T_ratio']==np.median(X_space['T_ratio'])])
Z = loo_subset.reshape(xx_pl.shape)

ax1=fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(xx_pl, yy_pl, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)


xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()
cset = ax1.contour(xx_pl, yy_pl, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax1.contour(xx_pl, yy_pl, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax1.contour(xx_pl, yy_pl, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax1.set_title('a) LOOCV error approximation', fontsize=subt_font)


####
# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))

criterion = np.multiply(Wx, fs2_subset)
# criterion = fs2_subset
Z = criterion.reshape(xx_pl.shape)

ax2=fig.add_subplot(2, 2, 2, projection='3d')
surf = ax2.plot_surface(xx_pl, yy_pl, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)


xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
zlim = ax2.get_zlim()
cset = ax2.contour(xx_pl, yy_pl, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax2.contour(xx_pl, yy_pl, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax2.contour(xx_pl, yy_pl, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax2.set_xlabel('Gap ratio', fontsize=axis_font)
ax2.set_ylabel('$R_y$', fontsize=axis_font)
#ax2.set_zlabel('Median loss ($)', fontsize=axis_font)
ax2.set_title('b) Targeted MSE', fontsize=subt_font)

####
# just predictive variance 

Z = fs2_subset.reshape(xx_pl.shape)

ax3=fig.add_subplot(2, 2, 3, projection='3d')
surf = ax3.plot_surface(xx_pl, yy_pl, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)

xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
zlim = ax3.get_zlim()
cset = ax3.contour(xx_pl, yy_pl, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax3.contour(xx_pl, yy_pl, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax3.contour(xx_pl, yy_pl, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax3.set_xlabel('Gap ratio', fontsize=axis_font)
ax3.set_ylabel('$R_y$', fontsize=axis_font)
#ax3.set_zlabel('Median loss ($)', fontsize=axis_font)
ax3.set_title('c) Predictive variance', fontsize=subt_font)

####
# just target weight 
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))
criterion = Wx
Z = criterion.reshape(xx_pl.shape)

ax4=fig.add_subplot(2, 2, 4, projection='3d')
surf = ax4.plot_surface(xx_pl, yy_pl, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)


xlim = ax4.get_xlim()
ylim = ax4.get_ylim()
zlim = ax4.get_zlim()
cset = ax4.contour(xx_pl, yy_pl, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax4.contour(xx_pl, yy_pl, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax4.contour(xx_pl, yy_pl, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax4.set_xlabel('Gap ratio', fontsize=axis_font)
ax4.set_ylabel('$R_y$', fontsize=axis_font)
#ax3.set_zlabel('Median loss ($)', fontsize=axis_font)
ax4.set_title('d) Weight targeting 50-50', fontsize=subt_font)
fig.tight_layout()
plt.show()
#%%

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))

criterion = np.multiply(Wx, fs2_subset)
# criterion = fs2_subset
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
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance at 400 pts', fontsize=axis_font)

plt.show()
#%% plots, Ry t ratio

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_pl = np.unique(yy)
y_pl = np.unique(uu)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
fs2_subset = fs2_train[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
fmu_subset = fmu_train[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
Z = fmu_subset.reshape(xx_pl.shape)

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
plt.scatter(df['RI'], df['T_ratio'], 
            c=df['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
# plt.xlim([0.5,2.0])
# plt.ylim([0.5, 2.5])
plt.ylabel('T ratio', fontsize=axis_font)
plt.xlabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using full 400 points', fontsize=axis_font)
plt.show()



#%% doe with smaller set

# n_set is both test_train split
n_set = 200
ml_set = df.sample(n=n_set, replace=False, random_state=985)

# split 50/50 for 
df_train = ml_set.head(int(n_set/2))
df_test = ml_set.tail(int(n_set/2))

#%% train a model

mdl = GP(df_train)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
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

xx, yy, uu = np.meshgrid(np.linspace(0.5,2.0,
                                      res),
                          np.linspace(0.5, 2.25,
                                      res),
                          np.linspace(1.0, 5.0,
                                      res))
                             
X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                      'RI':yy.ravel(),
                      'T_ratio':uu.ravel(),
                      'zeta_e':np.repeat(0.2,res**3)})



t0 = time.time()

fmu_train, fs1_train = mdl.gpr.predict(X_space, return_std=True)
fs2_train = fs1_train**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% loocv stuff

gp_obj = mdl.gpr._final_estimator
X_train = gp_obj.X_train_
iso_lengthscale = gp_obj.kernel_.theta[1]

L = gp_obj.L_
K_mat = L @ L.T
alpha_ = gp_obj.alpha_.flatten()
K_inv_diag = np.linalg.inv(K_mat).diagonal()

e_cv_sq = np.divide(alpha_, K_inv_diag)**2


NRMSE_cv = ((np.sum(np.divide(alpha_, K_inv_diag)**2)/len(alpha_))**0.5/
            (max(mdl.y[y_var]) - min(mdl.y[y_var])))

# for each candidate
# for each point
# calculate distance between candidate and point (lengthscale included)
# calculate loocv of point

#%% plots, gap ry

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
X_subset = X_space[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fs2_subset = fs2_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fmu_subset = fmu_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
Z = fmu_subset.reshape(xx_pl.shape)

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
plt.scatter(df_train['gap_ratio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.5, 2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using just 100 pts', fontsize=axis_font)
plt.show()

# LOOCV plots

loo_error = X_subset.apply(lambda row: loo_error_approx(row, X_train, iso_lengthscale, e_cv_sq),
                           axis='columns', result_type='expand')
loo_subset = np.asarray(loo_error[X_space['T_ratio']==np.median(X_space['T_ratio'])])
Z = loo_subset.reshape(xx_pl.shape)

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
cs = plt.contour(xx_pl, yy_pl, Z, linewidths=1.1, cmap='Blues', vmin=-1)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df_train['gap_ratio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.5, 2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('LOOCV error approximation', fontsize=axis_font)
plt.show()

#%%

# tMSE criterion
from numpy import exp
pi = 3.14159
T = 0.5
Wx = 1/((2*pi*(fs2_subset))**0.5) * exp((-1/2)*((fmu_subset - 0.5)**2/(fs2_subset)))

# criterion = np.multiply(Wx, fs2_subset)
criterion = Wx
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
plt.colorbar()
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Weighted variance at 100 pts', fontsize=axis_font)

plt.show()
#%% plots, Ry t ratio

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

x_pl = np.unique(yy)
y_pl = np.unique(uu)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)
X_subset = X_space[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
fs2_subset = fs2_train[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
fmu_subset = fmu_train[X_space['gap_ratio']==np.median(X_space['gap_ratio'])]
Z = fmu_subset.reshape(xx_pl.shape)

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
plt.scatter(df_train['RI'], df_train['T_ratio'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
# plt.xlim([0.5,2.0])
# plt.ylim([0.5, 2.5])
plt.ylabel('T ratio', fontsize=axis_font)
plt.xlabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk using just 100 pts', fontsize=axis_font)
plt.show()

#%% add in doe points

with open("../../data/tfp_mf_db_doe.pickle", 'rb') as picklefile:
    main_obj_doe = pickle.load(picklefile)
    
main_obj_doe.calculate_collapse()

df_doe = main_obj_doe.doe_analysis
df_doe = df_doe.reset_index(drop=True)
df_doe['max_drift'] = df_doe.PID.apply(max)
df_doe['log_drift'] = np.log(df_doe['max_drift'])

df_doe['max_velo'] = df_doe.PFV.apply(max)
df_doe['max_accel'] = df_doe.PFA.apply(max)

df_doe['T_ratio'] = df_doe['T_m'] / df_doe['T_fb']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_doe['Bm'] = np.interp(df_doe['zeta_e'], zetaRef, BmRef)

df_doe['gap_ratio'] = (df_doe['constructed_moat']*4*pi**2)/ \
    (g*(df_doe['sa_tm']/df_doe['Bm'])*df_doe['T_m']**2)


mdl_doe = GP(df_doe)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_doe.set_covariates(covariate_list)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_iso')

#%%

data_bounds = mdl_doe.X.agg(['min', 'max'])

n_var = data_bounds.shape[1]

min_list = [val for val in data_bounds.loc['min']]
max_list = [val for val in data_bounds.loc['max']]

import time
res = 75

xx, yy, uu = np.meshgrid(np.linspace(0.5,2.0,
                                      res),
                          np.linspace(0.5, 2.25,
                                      res),
                          np.linspace(1.0, 5.0,
                                      res))
                             
X_space = pd.DataFrame({'gap_ratio':xx.ravel(),
                      'RI':yy.ravel(),
                      'T_ratio':uu.ravel(),
                      'zeta_e':np.repeat(0.2,res**3)})



t0 = time.time()

fmu_train, fs1_train = mdl_doe.gpr.predict(X_space, return_std=True)
fs2_train = fs1_train**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% plots, gap ry

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
X_subset = X_space[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fs2_subset = fs2_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
fmu_subset = fmu_train[X_space['T_ratio']==np.median(X_space['T_ratio'])]
Z = fmu_subset.reshape(xx_pl.shape)

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
plt.scatter(df_doe['gap_ratio'], df_doe['RI'], 
            c=df_doe['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.5,2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.grid(True)
plt.title('Collapse risk after DoE (280 pts)', fontsize=axis_font)
plt.show()

#%% doe convergence plots

rmse_hist = main_obj_doe.rmse_hist
mae_hist = main_obj_doe.mae_hist

fig = plt.figure(figsize=(9, 6))

ax1=fig.add_subplot(1, 1, 1)
ax1.plot(np.arange(0, (len(rmse_hist)*10), 10), rmse_hist)
# ax1.set_title(r'Root mean squared error', fontsize=axis_font)
ax1.set_xlabel(r'Points added', fontsize=axis_font)
ax1.set_ylabel(r'Root mean squared error (RMSE)', fontsize=axis_font)
# ax1.set_xlim([0, 140])
# ax1.set_ylim([0.19, 0.28])
plt.grid(True)


# ax2=fig.add_subplot(1, 2, 2)
# ax2.plot(np.arange(0, (len(rmse_hist)*10), 10), mae_hist)
# ax2.set_title('Mean absolute error', fontsize=axis_font)
# ax2.set_xlabel('Points added', fontsize=axis_font)
# plt.grid(True)

#%%  a demonstration of Ry - Tfb relationships
# import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20

# y_var = 'T_ratio'
# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)


# ax1.scatter(df['RI'], df[y_var])
# ax1.set_ylabel('$T_M / T_{fb}$', fontsize=axis_font)
# ax1.set_xlabel(r'RI', fontsize=axis_font)
# ax1.grid(True)

#%%  a demonstration of k_ratio - Tm relationships
# import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20

# x_var = 'k_ratio'
# y_var = 'T_m'
# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)


# ax1.scatter(df[x_var], df[y_var])
# ax1.set_ylabel(y_var, fontsize=axis_font)
# ax1.set_xlabel(x_var, fontsize=axis_font)
# ax1.grid(True)

#%% regression for Tfb

# from sklearn.linear_model import LinearRegression
# X = df_raw[['h_bldg']]**0.75
# y = df_raw[['T_fb']]
# reg = LinearRegression(fit_intercept=False).fit(X, y)
# print('Linear regression for Tfb based on h_bldg^(3/4)')
# print('score: ', reg.score(X,y))
# print('coef: ', reg.coef_)

# X_test = df[['h_bldg']]**0.75
# y_test = df[['T_fb']]
# y_pred = reg.predict(X_test)

# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)


# ax1.scatter(X_test, y_test)
# ax1.plot(X_test, y_pred, color="blue", linewidth=3)
# ax1.set_ylabel(r'T_fb', fontsize=axis_font)
# ax1.set_xlabel(r'$h^{0.75}$', fontsize=axis_font)
# ax1.grid(True)

# %% T_ratio and k_ratio/Tfb

# import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20

# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)


# ax1.scatter(df['T_ratio']*(0.078*df['h_bldg']**0.75), df['k_ratio'] )
# ax1.set_xlabel(r'$\alpha_T$', fontsize=axis_font)
# ax1.set_ylabel(r'$\alpha_k / T_{fbe}$', fontsize=axis_font)
# ax1.grid(True)


#%% regression for Q

# from sklearn.linear_model import LinearRegression
# X = df_raw[['k_ratio', 'T_m']]
# y = df_raw[['Q']]
# reg = LinearRegression(fit_intercept=True).fit(X, y)
# print('Linear regression for Q based on k_ratio and T_m')
# print('score: ', reg.score(X,y))
# print('coef: ', reg.coef_)
# print('intercept: ', reg.intercept_)

#%%  a demonstration of Q - Tm relationships
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20

# x_var = 'T_m'
# y_var = 'Q'
# z_var = 'k_ratio'
# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)

# line_vals = (1/8*1.05/1.48* pi/df['T_m'])

# # alpha = (10*2*(df['mu_2'] - df['mu_1']))*df['Bm']*(df['k_ratio'] - 1)*4*pi**2/g/df['S_1']
# # beta = (10*2*(df['mu_2'] - df['mu_1']))*4*pi**2*(df['k_ratio'] - 1)/g


# alpha = (0.4)*1.48*(10- 1)*4*pi**2/g/1.05
# beta = (0.4)*4*pi**2*(10 - 1)/g


# line_vals_2 = np.maximum(beta/(df['T_m']*(df['T_m'] + alpha)), 
#                          np.repeat(0.05, len(line_vals)))

# sns.scatterplot(data=df,
#                      x=x_var, y=y_var,
#                      ax=ax1, legend='brief')

# # ax1.scatter(df[x_var], df[y_var])
# ax1.scatter(df[x_var], line_vals)
# ax1.scatter(df[x_var], line_vals_2)
# ax1.set_ylabel(y_var, fontsize=axis_font)
# ax1.set_xlabel(x_var, fontsize=axis_font)
# ax1.grid(True)

#%%  a demonstration of k_ratio - Tm relationships
# import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# axis_font = 18
# subt_font = 18
# label_size = 16
# title_font=20

# x_var = 'k_ratio'
# y_var = 'Q'
# fig = plt.figure(figsize=(13, 10))

# ax1=fig.add_subplot(1, 1, 1)


# sns.scatterplot(data=df,
#                      x=x_var, y=y_var,
#                      ax=ax1, legend='brief')

# ax1.set_ylabel(y_var, fontsize=axis_font)
# ax1.set_xlabel(x_var, fontsize=axis_font)
# ax1.grid(True)

#%%  variable testing
# from sklearn import preprocessing

# X = df[['k_ratio', 'T_m', 'zeta_e', 'Q']]
# y = df['max_accel'].ravel()

# scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)

# from sklearn.feature_selection import r_regression,f_regression

# r_results = r_regression(X_scaled,y)
# print("Pearson's R test: k_ratio, T_m, zeta, Q")
# print(r_results)


# f_statistic, p_values = f_regression(X_scaled, y)
# f_results = r_regression(X,y)
# print("F test: k_ratio, T_m, zeta, Q")
# print("F-statistics:", f_statistic)
# print("P-values:", p_values)


# import statsmodels.api as sm
# model = sm.OLS(y, X_scaled)
# results = model.fit()
# print(results.summary())

#%% 3d scatter

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x_var = 'T_m'
# y_var = 'Q'
# z_var = 'zeta_e'
# ax.scatter(df[x_var], df[y_var], df[z_var], c=df['k_ratio'])

# ax.set_xlabel(x_var)
# ax.set_ylabel(y_var)
# ax.set_zlabel(z_var)

# plt.show()