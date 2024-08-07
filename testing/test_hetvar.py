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
sys.path.insert(1, '../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None  

plt.close('all')

main_obj = pd.read_pickle("../data/tfp_mf_db.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df = main_obj.ops_analysis
df = df.reset_index(drop=True)


df = df.drop(columns=['index'])
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
stack_obj = pd.read_pickle("../data/tfp_mf_db_stack.pickle")

stack_obj.calculate_collapse()

df_stack = stack_obj.ops_analysis
df_stack = df_stack.reset_index(drop=True)

df_stack = df_stack.drop(columns=['index'])

df_stack['max_drift'] = df_stack.PID.apply(max)
df_stack['log_drift'] = np.log(df_stack['max_drift'])

df_stack['max_velo'] = df_stack.PFV.apply(max)
df_stack['max_accel'] = df_stack.PFA.apply(max)

df_stack['T_ratio'] = df_stack['T_m'] / df_stack['T_fb']
df_stack['T_ratio_e'] = df_stack['T_m'] / df_stack['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_stack['Bm'] = np.interp(df_stack['zeta_e'], zetaRef, BmRef)

df_stack['gap_ratio'] = (df_stack['constructed_moat']*4*pi**2)/ \
    (g*(df_stack['sa_tm']/df_stack['Bm'])*df_stack['T_m']**2)
    
#%% assign group, calculate log of variance of collapse prob

df_stack['gid'] = (df_stack.groupby(['S_1']).cumcount()==0).astype(int)
df_stack['gid'] = df_stack['gid'].cumsum()

df_stack = df_stack.join(df_stack.groupby(['gid'])['collapse_prob'].var(), on='gid', rsuffix='_r')

df_stack.rename(columns={'collapse_prob_r':'var_collapse_prob'}, inplace=True)
from numpy import log
df_stack['log_var_collapse_prob'] = log(df_stack['var_collapse_prob'])

df_stack['gid'] = (df_stack.groupby(['S_1']).cumcount()==0).astype(int)
df_stack['gid'] = df_stack['gid'].cumsum()
df_stack = df_stack.join(df_stack.groupby(['gid'])['collapse_prob'].mean(), on='gid', rsuffix='_r')
df_stack.rename(columns={'collapse_prob_r':'mean_collapse_prob'}, inplace=True)

df_secondary = df_stack.iloc[::11, :]
df_secondary = df_secondary.drop(columns=['gid'])
#%%
'''
def scatter_hist(x, y, c, alpha, ax, ax_histx, ax_histy, label=None):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    cmap = plt.cm.Blues
    ax.scatter(x, y, alpha=alpha, edgecolors='black', s=25, facecolors=c,
               label=label)

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
scatter_hist(df_secondary['gap_ratio'], df_secondary['RI'], 'navy', 0.9, ax, ax_histx, ax_histy,
             label='Replicated set')
scatter_hist(df['gap_ratio'], df['RI'], 'orange', 0.3, ax, ax_histx, ax_histy,
             label='Main set')
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_xlim([0.0, 4.0])
ax.set_ylim([0.5, 2.25])
ax.legend(fontsize=label_size)

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_secondary['T_ratio'], df_secondary['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df['T_ratio'], df['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlim([1.25, 5.0])
ax.set_ylim([0.1, 0.25])
'''
#%% fit secondary GP

from doe import GP
mdl_var = GP(df_secondary)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_var.set_covariates(covariate_list)
mdl_var.set_outcome('log_var_collapse_prob')
# mdl_var.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-5, 1e3))
mdl_var.fit_kernel_ridge(kernel_name='rbf')

# gp_obj = mdl_var.gpr._final_estimator

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

#%% 3d surf for var
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 15
X_plot = make_2D_plotting_space(mdl_var.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# var_est = mdl_var.gpr.predict(X_plot, return_std=False)
# X_array = X_plot.to_numpy()
# var_array = mdl_var.kr.predict(X_array).ravel()
# var_diag = np.diag(var_array)

var_est = mdl_var.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = var_est.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary['log_var_collapse_prob'],
           edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$c^2(x)$', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 15
X_plot = make_2D_plotting_space(mdl_var.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 0.75)
xx = X_plot[xvar]
yy = X_plot[yvar]


# var_est = mdl_var.gpr.predict(X_plot, return_std=False)
var_est = mdl_var.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = var_est.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary['log_var_collapse_prob'],
           edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$c^2(x)$', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
'''
#%%
xvar = 'gap_ratio'
yvar = 'RI'

res = 15
X_plot = make_2D_plotting_space(mdl_var.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)

kr_obj = mdl_var.kr

scaler_fit = kr_obj[0]
kr_only = kr_obj[-1:]._final_estimator
X_fit_var = kr_only.X_fit_
dual_coef_var = kr_only.dual_coef_
alpha_kr = kr_only.alpha
gamma_kr = kr_only.gamma

# %%
df['mean_collapse_prob'] = df['collapse_prob']
df = pd.concat([df, df_secondary], axis=0)
from doe import GP
mdl_main = GP(df)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_main.set_covariates(covariate_list)
mdl_main.set_outcome('mean_collapse_prob')

mdl_main.fit_het_gpr(kernel_name='rbf_iso', 
                     X_fit=X_fit_var, dual_coef=dual_coef_var, 
                     rbf_gamma=gamma_kr)

#%% 3d surf for main var
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

from sklearn.metrics.pairwise import rbf_kernel
from numpy import diag, exp

#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 25
X_plot = make_2D_plotting_space(mdl_main.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]


main_scaler = mdl_main.gpr_het[0]
kr_only = kr_obj[-1:]._final_estimator
var_scaler = kr_obj[0]

dual_coef_var = kr_only.dual_coef_

gpr_het_only = mdl_main.gpr_het[-1:]._final_estimator
a = gpr_het_only.kernel_.k1(X_fit_var, X_fit_var)
ka = gpr_het_only.kernel_.k1
b = gpr_het_only.kernel_.k2(X_fit_var, X_fit_var)
kb = gpr_het_only.kernel_.k2
X_fit_main = gpr_het_only.X_train_
y_fit_main = gpr_het_only.y_train_

rbf_const = np.exp(ka.k1.theta)
rbf_lengthscale = np.exp(ka.k2.theta)
nugget_const = np.exp(kb.k1.theta)

X_pred = main_scaler.transform(X_plot)

ks = gpr_het_only.kernel_.k1(X_pred, X_fit_main)
kss = gpr_het_only.kernel_.k1(X_pred)
from scipy.linalg import cho_solve, cholesky, solve_triangular
kk = gpr_het_only.kernel_.k1(X_fit_main)
n = X_fit_main.shape[0]

scale_main_X_var = var_scaler.transform(X_fit_main)
K_pr = rbf_kernel(scale_main_X_var, X_fit_var, gamma=kr_only.gamma)
y_pr = K_pr @ dual_coef_var
cs = diag(exp(y_pr.ravel()))

R_approx = kk + nugget_const*np.diag(np.full(n,1/n)) @ cs
L = cholesky(kk + nugget_const*np.diag(np.full(n,1/n)) @ cs, lower=True)
alpha = cho_solve(
    (L, True),
    y_fit_main,
    check_finite=False,
)

y_pred = ks @ alpha

V = solve_triangular(
    L, ks.T, lower=True, check_finite=False
)
y_var = np.diag(kss).copy()
y_var -= np.einsum("ij,ji->i", V.T, V)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)


prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

Z_surf = prob_collapse.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary['log_var_collapse_prob'],
#            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$y(x)$', fontsize=axis_font)
ax.set_title('Home-cooked', fontsize=subt_font)

#################################

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = prob_std.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$y(x)$', fontsize=axis_font)
ax.set_title('Using GPR', fontsize=subt_font)
fig.tight_layout()
#%% 3d surf for main var
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 15
X_plot = make_2D_plotting_space(mdl_main.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

pred = prob_collapse + prob_std
Z_surf = pred.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df['collapse_prob'],
            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$\sigma(x)$', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 15
X_plot = make_2D_plotting_space(mdl_main.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 0.75)
xx = X_plot[xvar]
yy = X_plot[yvar]


prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

pred = prob_collapse + prob_std
Z_surf = pred.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df['collapse_prob'],
            edgecolors='k', alpha = 0.7)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$\sigma(x)$', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
'''
# %% baseline

from doe import GP
mdl_base = GP(df)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_base.set_covariates(covariate_list)
mdl_base.set_outcome('collapse_prob')

mdl_base.fit_gpr(kernel_name='rbf_iso')

#%% 3d surf for main var

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 15
X_plot = make_2D_plotting_space(mdl_base.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_base.gpr.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = prob_collapse.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary['log_var_collapse_prob'],
#            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$y(x)$', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 15
X_plot = make_2D_plotting_space(mdl_base.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_base.gpr.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = prob_collapse.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$y(x)$', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% reproduce main

gpr_only = mdl_base.gpr[-1:]._final_estimator
a = gpr_only.kernel_.k1(X_fit_var, X_fit_var)
ka = gpr_only.kernel_.k1
b = gpr_only.kernel_.k2(X_fit_var)
kb = gpr_only.kernel_.k2
X_fit_main = gpr_only.X_train_
y_fit_main = gpr_only.y_train_
scaler_fit = mdl_base.gpr[0]
rbf_const = np.exp(ka.k1.theta)
rbf_lengthscale = np.exp(ka.k2.theta)
nugget_const = np.exp(kb.theta)

X_pred = scaler_fit.transform(X_plot)

ks = gpr_only.kernel_.k1(X_pred, X_fit_main)
kss = gpr_only.kernel_.k1(X_pred)
from scipy.linalg import cho_solve, cholesky, solve_triangular
kk = gpr_only.kernel_.k1(X_fit_main, X_fit_main)
kn = gpr_only.kernel_.k2(X_fit_main)
n = X_fit_main.shape[0]

R_approx = kk + nugget_const*np.diag(np.full(n,1/n))
L = cholesky(kk + kn, lower=True)
alpha = cho_solve(
    (L, True),
    y_fit_main,
    check_finite=False,
)

y_pred = ks @ alpha

V = solve_triangular(
    L, ks.T, lower=True, check_finite=False
)
y_var = kss.copy() - V.T @ V
y_var = np.diag(y_var)
y_var = y_var**0.5

y_correct, y_var_correct = mdl_base.gpr.predict(X_plot, return_std=True)