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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None  

plt.close('all')

main_obj = pd.read_pickle("../../data/loss/tfp_mf_db_doe_loss_max.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df_raw = main_obj.doe_analysis
df_raw = df_raw.reset_index(drop=True)

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

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

df_loss = main_obj.loss_data
df_loss_max = main_obj.max_loss
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

###############################################################################
    # Full prediction models
###############################################################################

# two ways of doing this
        
        # 1) predict impact first (binary), then fit the impact predictions 
        # with the impact-only SVR and likewise with non-impacts. This creates
        # two tiers of predictions that are relatively flat (impact dominated)
        # 2) using expectations, get probabilities of collapse and weigh the
        # two (cost|impact) regressions with Pr(impact). Creates smooth
        # predictions that are somewhat moderate
        
def predict_DV(X, impact_pred_mdl, hit_loss_mdl, miss_loss_mdl,
               outcome='cost_50%', return_var=False):
        
#        # get points that are predicted impact from full dataset
#        preds_imp = impact_pred_mdl.svc.predict(self.X)
#        df_imp = self.X[preds_imp == 1]

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
            
#        # get points that are predicted no impact from full dataset
#        df_mss = self.X[preds_imp == 0]
    
    # run miss model on this dataset
    expected_DV_miss = pd.DataFrame(
            {outcome_str:np.multiply(
                    miss_loss_mdl.predict(X).ravel(),
                    miss_prob)})
    
    expected_DV = expected_DV_hit + expected_DV_miss
    
    # self.median_loss_pred = pd.concat([loss_pred_hit,loss_pred_miss], 
    #                                   axis=0).sort_index(ascending=True)
    if return_var:
        pass
    else:
        return(expected_DV)

#%% collapse fragility def
'''
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


collapse_drift_def_mu_std = 0.1


from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(collapse_drift_def_mu_std) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
# mean_log_drift = 0.05
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

label_size = 16
clabel_size = 12
x = np.linspace(0, 0.15, 200)

mu = log(mean_log_drift)

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
p = ln_dist.cdf(np.array(x))


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
# plt.show()
# plt.savefig('./figures/collapse_def.eps')
'''

#%% collapse fragility def
import numpy as np
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
# collapse as a probability
from scipy.stats import lognorm
from math import log, exp


x = np.linspace(0, 0.15, 200)
mu = log(0.1)- 0.25*inv_norm
sigma = 0.25;

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

# plt.close('all')
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

# ax.set_title('Replacement fragility definition', fontsize=axis_font)
ax.grid()
label_size = 16
clabel_size = 12

ax.legend(fontsize=label_size, loc='upper center')
plt.show()
#%% normalize DVs and prepare all variables
df['bldg_area'] = df['L_bldg']**2 * (df['num_stories'] + 1)

df['replacement_cost'] = 600.0*(df['bldg_area'])
df['total_cmp_cost'] = df_loss_max['cost_50%']
df['cmp_replace_cost_ratio'] = df['total_cmp_cost']/df['replacement_cost']
df['median_cost_ratio'] = df_loss['cost_50%']/df['replacement_cost']
df['cmp_cost_ratio'] = df_loss['cost_50%']/df['total_cmp_cost']

# , but working in parallel (2x faster)
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
#%% determine when to stop DoE

rmse_hist = main_obj.rmse_hist
mae_hist = main_obj.mae_hist
nrmse_hist = main_obj.nrmse_hist
theta = main_obj.hyperparam_list


change_array = np.diff(nrmse_hist)
conv_array = np.abs(change_array) < 5e-4

# when does conv_array hit 10 Trues in a row
for conv_idx in range(10, len(conv_array)):
    # getting Consecutive elements 
    if all(conv_array[conv_idx-10:conv_idx]):
        break

rmse_hist = rmse_hist[:conv_idx]
mae_hist = mae_hist[:conv_idx]
nrmse_hist = nrmse_hist[:conv_idx]

fig = plt.figure(figsize=(16, 6))
batch_size = 5

ax1=fig.add_subplot(1, 3, 1)
ax1.plot(np.arange(0, (len(rmse_hist))*batch_size, batch_size), rmse_hist)
ax1.set_title(r'RMSE on test set', fontsize=axis_font)
ax1.set_xlabel(r'Points added', fontsize=axis_font)
ax1.set_ylabel(r'Error metric', fontsize=axis_font)
# ax1.set_xlim([0, 140])
# ax1.set_ylim([0.19, 0.28])
ax1.grid(True)


ax2=fig.add_subplot(1, 3, 2)
ax2.plot(np.arange(0, (len(rmse_hist))*batch_size, batch_size), nrmse_hist)
ax2.set_title('NRMSE-LOOCV of training set', fontsize=axis_font)
ax2.set_xlabel('Points added', fontsize=axis_font)
ax2.grid(True)


ax3=fig.add_subplot(1, 3, 3)
ax3.plot(np.arange(0, (len(change_array))*batch_size, batch_size), change_array)
ax3.set_title('Relative change', fontsize=axis_font)
ax3.set_xlabel('Points added', fontsize=axis_font)
ax3.grid(True)

df_raw = df.copy()
df = df.head(conv_idx*batch_size + 50)

#%% seaborn scatter with histogram: DoE data
df_init = df_raw.head(50)

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
scatter_hist(df_init['gap_ratio'], df_init['RI'], 'navy', 0.9, ax, ax_histx, ax_histy,
             label='Initial set')
scatter_hist(df['gap_ratio'], df['RI'], 'orange', 0.3, ax, ax_histx, ax_histy,
             label='DoE added')
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_xlim([0.0, 4.0])
ax.set_ylim([0.5, 2.25])
ax.legend(fontsize=label_size)

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_init['T_ratio'], df_init['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df['T_ratio'], df['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlim([1.25, 5.0])
ax.set_ylim([0.1, 0.25])


#%% ml training

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_cost_hit = GP(df_hit)
mdl_cost_hit.set_covariates(covariate_list)
mdl_cost_hit.set_outcome(cost_var)
mdl_cost_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_cost_miss = GP(df_miss)
mdl_cost_miss.set_covariates(covariate_list)
mdl_cost_miss.set_outcome(cost_var)
mdl_cost_miss.test_train_split(0.2)

mdl_time_hit = GP(df_hit)
mdl_time_hit.set_covariates(covariate_list)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = GP(df_miss)
mdl_time_miss.set_covariates(covariate_list)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_repl_hit = GP(df_hit)
mdl_repl_hit.set_covariates(covariate_list)
mdl_repl_hit.set_outcome('replacement_freq')
mdl_repl_hit.test_train_split(0.2)

mdl_repl_miss = GP(df_miss)
mdl_repl_miss.set_covariates(covariate_list)
mdl_repl_miss.set_outcome('replacement_freq')
mdl_repl_miss.test_train_split(0.2)

mdl_unconditioned = GP(df)
mdl_unconditioned.set_covariates(covariate_list)
mdl_unconditioned.set_outcome(cost_var)
mdl_unconditioned.test_train_split(0.2)

#%% seaborn scatter with histogram: impact data
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
scatter_hist(df_hit['T_ratio'], df_hit['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy,
             label='Impacted set')
scatter_hist(df_miss['T_ratio'], df_miss['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy,
             label='Non-impact set')
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_e$', fontsize=axis_font)
ax.set_xlim([1.0, 4.0])
ax.set_ylim([0.1, 0.25])
ax.legend(fontsize=label_size)

ax = fig.add_subplot(gs[1, 2])
ax_histx = fig.add_subplot(gs[0, 2], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 3], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(df_init['T_ratio'], df_init['zeta_e'], 'navy', 0.9, ax, ax_histx, ax_histy)
scatter_hist(df['T_ratio'], df['zeta_e'], 'orange', 0.3, ax, ax_histx, ax_histy)

ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlim([1.25, 5.0])
ax.set_ylim([0.1, 0.25])
#%%  dumb scatters
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(
    np.array(df['T_ratio']).reshape(-1, 1), np.array(df['impacted']))

# plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
subt_font = 18
label_size = 20
title_font = 24

from scipy.special import expit
T_range = np.linspace(1.0, 5.0, 300)
loss = expit(T_range * clf.coef_ + clf.intercept_).ravel()

mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

y_var = 'impacted'
fig = plt.figure(figsize=(13, 10))

ax=fig.add_subplot(1, 1, 1)

cmap = plt.cm.coolwarm
sc = ax.scatter(df['T_ratio'], df[y_var], alpha=0.2, c=df['impacted'], cmap=cmap)
plt.plot(T_range, loss, label="Logistic Regression Model", color="red", linewidth=3)
ax.set_ylabel('Impacted', fontsize=axis_font)
ax.set_xlabel(r'$T_M/ T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()


# plt.savefig('./figures/scatter.pdf')

#%%  dumb scatters
'''

# plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 22
subt_font = 18
label_size = 20
title_font = 24

mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

y_var = 'max_drift'
fig = plt.figure(figsize=(13, 10))

ax1=fig.add_subplot(2, 2, 1)

cmap = plt.cm.coolwarm
sc = ax1.scatter(df['gap_ratio'], df[y_var], alpha=0.2, c=df['impacted'], cmap=cmap)
ax1.set_ylabel('Peak story drift', fontsize=axis_font)
ax1.set_xlabel(r'$GR$', fontsize=axis_font)
ax1.set_title('a) Gap ratio', fontsize=title_font)
ax1.set_ylim([0, 0.15])

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', label='No impact',
                          markerfacecolor=cmap(0.), alpha=0.4, markersize=15),
                Line2D([0], [0], marker='o', color='w', label='Wall impact',
                       markerfacecolor=cmap(1.), alpha=0.4, markersize=15)]
ax1.legend(custom_lines, ['No impact', 'Impact'], fontsize=subt_font)
ax1.grid(True)

ax2=fig.add_subplot(2, 2, 2)

ax2.scatter(df['RI'], df[y_var], alpha=0.3, c=df['impacted'], cmap=cmap)
ax2.set_xlabel(r'$R_y$', fontsize=axis_font)
ax2.set_title('b) Superstructure strength', fontsize=title_font)
ax2.set_ylim([0, 0.15])
ax2.grid(True)

ax3=fig.add_subplot(2, 2, 3)

ax3.scatter(df['T_ratio'], df[y_var], alpha=0.3, c=df['impacted'], cmap=cmap)
# ax3.scatter(df['T_ratio_e'], df[y_var])
ax3.set_ylabel('Peak story drift', fontsize=axis_font)
ax3.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)
ax3.set_title('c) Bearing period ratio', fontsize=title_font)
ax3.set_ylim([0, 0.15])
ax3.grid(True)

ax4=fig.add_subplot(2, 2, 4)

ax4.scatter(df['zeta_e'], df[y_var], alpha=0.3, c=df['impacted'], cmap=cmap)
ax4.set_xlabel(r'$\zeta_M$', fontsize=axis_font)
ax4.set_title('d) Bearing damping', fontsize=title_font)
ax4.set_ylim([0, 0.15])
ax4.grid(True)

fig.tight_layout()
plt.show()
# plt.savefig('./figures/scatter.pdf')
'''
#%% engineering data
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
fig = plt.figure(figsize=(13, 8))

bins = pd.IntervalIndex.from_tuples([(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.5)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['gap_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 1)
import seaborn as sns
sns.stripplot(data=df, x="max_drift", y="bin", orient="h", alpha=0.8, size=5,
              hue='impacted', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)

ax.set_ylabel('$GR$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
plt.xlim([0.0, 0.15])

#####
bins = pd.IntervalIndex.from_tuples([(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.25)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['RI'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 2)
import seaborn as sns
sns.stripplot(data=df, x="max_drift", y="bin", orient="h", size=5, alpha=0.8,
              hue='impacted', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$R_y$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
plt.xlim([0.0, 0.15])



#####
bins = pd.IntervalIndex.from_tuples([(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['T_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 3)
import seaborn as sns
sns.stripplot(data=df, x="max_accel", y="bin", orient="h", size=5, alpha=0.8,
              hue='impacted', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_accel", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$T_M/T_{fb}$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor acceleration (g)', fontsize=axis_font)
plt.xlim([0.0, 5.0])

#####
bins = pd.IntervalIndex.from_tuples([(0.1, 0.14), (0.14, 0.18), (0.18, 0.22), (0.22, 0.25)])
labels=['tiny', 'small', 'okay', 'large']
df['bin'] = pd.cut(df['zeta_e'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 4)
import seaborn as sns
sns.stripplot(data=df, x="max_velo", y="bin", orient="h", size=5, alpha=0.8,
              hue='impacted', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_velo", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$\zeta_M$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor velocity (in/s)', fontsize=axis_font)
plt.xlim([25, 125.0])
fig.tight_layout(h_pad=2.0)
plt.show()



#%% weird pie plot
title_font=22
def plot_pie(x, ax, r=1): 
    # radius for pieplot size on a scatterplot
    c_list = [plt.cm.Dark2(7), plt.cm.Dark2(5), plt.cm.Dark2(0), plt.cm.Dark2(3)]
    patches, texts = ax.pie(x[['B_50%','C_50%','D_50%','E_50%']], 
           center=(x['gap_ratio'],x['RI']), radius=x['r_bin'], colors=c_list)
    
    return(patches)

    
fig, ax = plt.subplots(1, 1, figsize=(8, 8))


df_mini = df.head(200)
df_mini = df_mini[df_mini['gap_ratio'] < 2.5]
df_mini = df_mini[df_mini['gap_ratio'] >0.5]
df_pie = df_mini.copy()

ax.scatter(x=df_pie['gap_ratio'], y=df_pie['RI'], s=0)
# git min/max values for the axes
# y_init = ax.get_ylim()
# x_init = ax.get_xlim()
# bins = pd.IntervalIndex.from_tuples([(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 10.0)])
bins = [0, 0.1, 0.5, 1.0, 10.0]
labels=['tiny', 'small', 'okay', 'large']
df_pie['cost_bin'] = pd.cut(df['cmp_cost_ratio'], bins=bins, labels=False)
equiv = {0: 0.02, 1:0.03, 2:0.04, 3:0.05}
df_pie["r_bin"] = df_pie["cost_bin"].map(equiv)
df_pie.apply(lambda x : plot_pie(x, ax, r=0.04), axis=1)
patch = plot_pie(df_pie.iloc[-1], ax, r=0.04)

ax.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
ax.set_title(r'Repair cost makeup by component', fontsize=title_font)
ax.set_xlim([0.4, 2.5])
ax.set_ylim([0.5, 2.25])
_ = ax.yaxis.set_ticks(np.arange(0.5, 2.1, 0.5))
_ = ax.xaxis.set_ticks(np.arange(0.5, 2.1, 0.5))
# _ = ax.set_title('My')
ax.set_frame_on(True)
labels = ['Structure \& facade', 'Flooring, ceiling, partitions, \& stairs', 'MEP', 'Storage']
plt.legend(patch, labels, loc='lower right', fontsize=14)
# plt.axis('equal')
# plt.tight_layout()
ax.grid()

#%% stacked bars
# plt.close('all')

labels=['<10\%', '10-90%', '>90\%']
bins = pd.IntervalIndex.from_tuples([(-0.001, 0.1), (0.1, 0.9), (0.9, 1.0)])
df['bin'] = pd.cut(df['replacement_freq'], bins=bins, labels=labels)

df['B_frac'] = df['B_50%'] / df['total_cmp_cost']
df['C_frac'] = df['C_50%'] / df['total_cmp_cost']
df['D_frac'] = df['D_50%'] / df['total_cmp_cost']
df['E_frac'] = df['E_50%'] / df['total_cmp_cost']

df_stack_bars = df.groupby('bin')[[
    'B_frac', 'C_frac', 'D_frac', 'E_frac']].mean()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)

risks = ['$<10$ \%', '$10-90$\%', '$>90$\%']
ax.grid(visible=True, zorder=0)
cmap = plt.cm.Dark2
p1 = ax.bar(risks, df_stack_bars['B_frac'], width=0.35, 
            label='Structure \& fa\c{c}ade', zorder=3, color=cmap(7))
p1 = ax.bar(risks, df_stack_bars['C_frac'], width=0.35, 
            label='Flooring, ceiling, partitions, \& stairs', zorder=3, color=cmap(5))
p1 = ax.bar(risks, df_stack_bars['D_frac'], width=0.35, label='MEP', zorder=3, 
            color=cmap(0))
p1 = ax.bar(risks, df_stack_bars['E_frac'], width=0.35, label='Storage', zorder=3,
            color=cmap(3))
ax.set_ylabel("Percent loss", fontsize=axis_font)
ax.set_xlabel('Replacement risk', fontsize=axis_font)

ax.legend(fontsize=axis_font)
plt.show()

#%% impact effect

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

#plt.close('all')
import seaborn as sns

# make grid and plot classification predictions

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
sns.boxplot(y=cost_var, x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.6, ax=ax1)
sns.stripplot(x='impacted', y=cost_var, data=df, ax=ax1, jitter=True,
              alpha=0.3, color='steelblue')
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Repair cost ratio', fontsize=axis_font)
ax1.set_xlabel('Impact', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y=time_var, x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.6, ax=ax2)
sns.stripplot(x='impacted', y=time_var, data=df, ax=ax2, jitter=True,
              alpha=0.3, color='steelblue')
ax2.set_title('Median sequential repair time', fontsize=subt_font)
ax2.set_ylabel('Repair time ratio', fontsize=axis_font)
ax2.set_xlabel('Impact', fontsize=axis_font)
ax2.set_yscale('log')

sns.boxplot(y="replacement_freq", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.5, ax=ax3)
sns.stripplot(x='impacted', y='replacement_freq', data=df, ax=ax3, jitter=True,
              alpha=0.3, color='steelblue')
ax3.set_title('Replacement frequency', fontsize=subt_font)
ax3.set_ylabel('Replacement frequency', fontsize=axis_font)
ax3.set_xlabel('Impact', fontsize=axis_font)
# ax3.set_yscale('log')
fig.tight_layout()
plt.show()

#%% impact prediction

print('========== Fitting impact classification (GPC or KLR) ============')

# prepare the problem
mdl_impact = GP(df)
mdl_impact.set_covariates(covariate_list)
mdl_impact.set_outcome('impacted', use_ravel=True)
mdl_impact.test_train_split(0.2)

mdl_impact.fit_gpc(kernel_name='rbf_ard')

mdl_impact.fit_kernel_logistic(kernel_name='rbf')

# predict the entire dataset
preds_imp = mdl_impact.gpc.predict(mdl_impact.X)
probs_imp = mdl_impact.gpc.predict_proba(mdl_impact.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl_impact.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

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
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
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

# ax1.imshow(
#         Z,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.Greys,
#     )

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

ax.set_xlim(0.3, 2.5)
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
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
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

# ax1.imshow(
#         Z,
#         interpolation="nearest",
#         extent=(xx.min(), xx.max(),
#                 yy.min(), yy.max()),
#         aspect="auto",
#         origin="lower",
#         cmap=plt.cm.Greys,
#     )

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

# ax.set_xlim(0.3, 2.5)
ax.set_title(r'Impact likelihood: $R_y = 2.0$, $GR = 1.0$', fontsize=title_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_xlabel(r'$T_M/T_{fb}$', fontsize=axis_font)

fig.tight_layout()
plt.show()

#%% fit regressions for impact / non-impact set

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

mdl_unconditioned.fit_gpr(kernel_name='rbf_iso')

print('========== Fitting ordinary ridge (OR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_ols_ridge()
mdl_time_hit.fit_ols_ridge()

# fit no impact set
mdl_cost_miss.fit_ols_ridge()
mdl_time_miss.fit_ols_ridge()


mdl_repl_hit.fit_ols_ridge()
mdl_repl_miss.fit_ols_ridge()
#%% plot no-impact regressions
'''
axis_font = 20
subt_font = 18

xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]
Z = mdl_cost_miss.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_regr = Z.reshape(xx_pl.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_regr, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False,
                       alpha=0.5)

ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var],
           c=df_miss[cost_var], alpha=0.3,
           edgecolors='k')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='z', offset=-1e5, cmap='coolwarm')
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='x', offset=xlim[0], cmap='coolwarm_r')
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='y', offset=ylim[1], cmap='coolwarm')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_zlim([0, 0.2])
ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
# ax.set_zlabel('Median loss ($)', fontsize=axis_font)
# ax.set_title('Median cost predictions given no impact (RBF kernel ridge)')
fig.tight_layout()
plt.show()
'''
#%% plot yes-impact regressions
'''
axis_font = 20
subt_font = 18

xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]
Z = mdl_cost_hit.kr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_regr = Z.reshape(xx_pl.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, Z_regr, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False,
                       alpha=0.5)

ax.scatter(df_hit[xvar], df_hit[yvar], df_hit[cost_var],
           c=df_hit[cost_var], alpha=0.3,
           edgecolors='k')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='z', offset=-1e5, cmap='coolwarm')
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='x', offset=xlim[0], cmap='coolwarm_r')
cset = ax.contour(xx_pl, yy_pl, Z_regr, zdir='y', offset=ylim[1], cmap='coolwarm')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
# ax.set_zlabel('Median loss ($)', fontsize=axis_font)
# ax.set_title('Median cost predictions given no impact (RBF kernel ridge)')
fig.tight_layout()
plt.show()
'''
#%% unlabeled generic regression image
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(9, 6))

Z = mdl_cost_miss.gpr.predict(X_plot)
Z_hit_cond = Z.reshape(xx_pl.shape)

cs = plt.contour(xx_pl, Z_hit_cond, yy_pl, linewidths=1.4, cmap='Spectral',
                 levels=np.arange(0.5, 3.0, step=0.25))
plt.scatter(df_miss[xvar], df_miss[cost_var], c=df_miss[yvar],
            cmap='Spectral', alpha=0.5,
          edgecolors='k')
plt.clabel(cs, fontsize=label_size)
# plt.xlabel(r'$X$', fontsize=axis_font)
# plt.ylabel('Loss ratio', fontsize=axis_font)
plt.axis('off')
plt.ylim([0.0, 0.1])
fig.tight_layout()
'''
#%% unconditioned models

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(14, 7))

#################################
xvar = 'RI'
yvar = 'gap_ratio'

res = 75
X_plot = make_2D_plotting_space(mdl_unconditioned.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)


Z = mdl_unconditioned.gpr.predict(X_plot)
Z_unconditioned = Z.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 3, 1)
cs = ax.contour(xx_pl, Z_unconditioned, yy_pl, linewidths=1.1, cmap='coolwarm',
                 levels=np.arange(0.5, 3.0, step=0.25))
ax.scatter(df[xvar], df[cost_var], color='steelblue', alpha=0.5,
          edgecolors='k')
ax.clabel(cs, fontsize=label_size)
ax.set_xlabel(r'$R_y$', fontsize=axis_font)
ax.grid(visible=True)
ax.plot(0.65, 0.01, color='red', label=r'$GR$')
ax.legend(fontsize=axis_font, loc='center left')
ax.set_ylabel('Median repair cost ratio', fontsize=axis_font)
ax.set_title('a) Fit unconditioned models', fontsize=axis_font)
ax.set_ylim([-.3, 3.5])
ax.set_xlim([0.5, 2.25])
plt.show()

#################################
# show dichotomy of data


ax=fig.add_subplot(1, 3, 2)

cmap = plt.cm.coolwarm
sc = ax.scatter(df[xvar], df[cost_var], alpha=0.4, c=df['impacted'], 
                edgecolors='k', cmap=cmap)
ax.set_xlabel(r'$R_y$', fontsize=axis_font)
# ax.set_title('a) Gap ratio', fontsize=title_font)
# ax.set_xlim([0.3, 2.5])
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', label='No impact',
                          markerfacecolor=cmap(0.), alpha=0.4, markersize=15),
                Line2D([0], [0], marker='o', color='w', label='Wall impact',
                       markerfacecolor=cmap(1.), alpha=0.4, markersize=15)]
ax.legend(custom_lines, ['No impact', 'Impact'], fontsize=subt_font)
ax.grid(True)

ax.set_ylim([-.3, 3.5])
ax.set_xlim([0.5, 2.25])
ax.set_title('b) Two "classes" of response', fontsize=axis_font)
#################################
# plot conditioned fits
cmap=plt.cm.coolwarm
ax=fig.add_subplot(2, 3, 3)

Z = mdl_cost_hit.gpr.predict(X_plot)
Z_hit_cond = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, Z_hit_cond, yy_pl, linewidths=1.1, cmap='coolwarm',
                 levels=np.arange(0.5, 3.0, step=0.25))
ax.scatter(df_hit[xvar], df_hit[cost_var], color=cmap(1.), alpha=0.5,
          edgecolors='k')
ax.clabel(cs, fontsize=label_size)
ax.set_title('c) Conditioned fits', fontsize=axis_font)
# ax.set_xlabel(r'$R_y$', fontsize=axis_font)
ax.grid(visible=True)
ax.set_xlim([0.5, 2.25])
plt.show()


ax=fig.add_subplot(2, 3, 6)

Z = mdl_cost_miss.gpr.predict(X_plot)
Z_hit_cond = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, Z_hit_cond, yy_pl, linewidths=1.1, cmap='coolwarm',
                 levels=np.arange(0.5, 3.0, step=0.25))
ax.scatter(df_miss[xvar], df_miss[cost_var], color=cmap(0.), alpha=0.5,
          edgecolors='k')
ax.clabel(cs, fontsize=label_size)
ax.set_xlabel(r'$R_y$', fontsize=axis_font)
ax.grid(visible=True)
ax.set_ylim([0.0, 0.2])
fig.tight_layout()
ax.set_xlim([0.5, 2.25])
plt.show()

#%% 3d surf for replacement risk
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16, 7))



#################################
xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

df_sc = df[(df['T_ratio']<=3.5) & (df['T_ratio']>=2.5)]

ax.scatter(df_sc[xvar], df_sc[yvar], df_sc['replacement_freq'], c=df_sc['replacement_freq'],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_cost_hit.gpr, mdl_cost_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

df_sc = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8)]

ax.scatter(df_sc[xvar], df_sc[yvar], df_sc['replacement_freq'], c=df_sc['replacement_freq'],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Replacement risk', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

df_t = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8) & 
           (df['T_ratio']>=3.0)]

# #################################
# xvar = 'gap_ratio'
# yvar = 'RI'

# res = 75
# X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
#                             all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
#                             third_var_set = 3.0, fourth_var_set = 0.15)
# xx = X_plot[xvar]
# yy = X_plot[yvar]

# Z = predict_DV(X_plot, mdl_impact.gpc, 
#                mdl_time_hit.gpr, mdl_time_miss.gpr, outcome=time_var)


# x_pl = np.unique(xx)
# y_pl = np.unique(yy)
# xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

# Z_time = np.array(Z).reshape(xx_pl.shape)

# ax=fig.add_subplot(1, 3, 3, projection='3d')
# surf = ax.plot_surface(xx_pl, yy_pl, Z_cost, cmap='Blues',
#                        linewidth=0, antialiased=False, alpha=0.6,
#                        vmin=-0.1)

# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False

# ax.scatter(df[xvar], df[yvar], df[time_var], c=df[time_var],
#            edgecolors='k', alpha = 0.7, cmap='Blues')

# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# zlim = ax.get_zlim()
# cset = ax.contour(xx_pl, yy_pl, Z_cost, zdir='x', offset=xlim[0], cmap='Blues_r')
# cset = ax.contour(xx_pl, yy_pl, Z_cost, zdir='y', offset=ylim[1], cmap='Blues')

# ax.set_xlabel('Gap ratio', fontsize=axis_font)
# ax.set_ylabel('$R_y$', fontsize=axis_font)
# #ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
# ax.set_title('c) Replacement time (GPR)', fontsize=subt_font)

# fig.tight_layout(w_pad=0.0)
plt.show()

#%% 2d contours for replacement

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

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_contour = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1)

plt_density = 200
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = plt.contour(xx_pl, yy_pl, Z_contour, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
plt.scatter(df[xvar][:plt_density], df[yvar][:plt_density], 
            c=df['replacement_freq'][:plt_density],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([0.3, 2.0])
plt.ylim([0.5, 2.25])
plt.xlabel('$GR$', fontsize=axis_font)
plt.ylabel('$R_y$', fontsize=axis_font)
plt.grid(True)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome='replacement_freq')


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_contour = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2)

plt_density = 200
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
cs = ax.contour(xx_pl, yy_pl, Z_contour, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
ax.scatter(df[xvar][:plt_density], df[yvar][:plt_density], 
            c=df['replacement_freq'][:plt_density],
            edgecolors='k', s=20.0, cmap=plt.cm.Blues, vmax=5e-1)
plt.xlim([2.0, 5.0])
plt.ylim([0.1, 0.25])
plt.xlabel('$T_M/T_{fb}$', fontsize=axis_font)
plt.ylabel('$\zeta_M$', fontsize=axis_font)
plt.grid(True)
plt.show()

#%% 3d surf for cost ratio
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

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome=cost_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[cost_var], c=df[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_cost_hit.gpr, mdl_cost_miss.gpr, outcome=cost_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[cost_var], c=df[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% 3d surf for time ratio
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

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_repl_hit.gpr, mdl_repl_miss.gpr, outcome=time_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[time_var], c=df[time_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

K_space = mdl_impact.get_kernel(X_plot, kernel_name='rbf', gamma=0.25)

Z = predict_DV(X_plot, mdl_impact.gpc, 
               mdl_cost_hit.gpr, mdl_cost_miss.gpr, outcome=time_var)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df[xvar], df[yvar], df[time_var], c=df[time_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% dirty contours (presenting replacement risk for EMI)

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
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
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
    xq = np.linspace(0.3, 1.5, 200)
    
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

#%% dirty contours replacement time and cost


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
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
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
    xq = np.linspace(0.3, 1.5, 200)
    
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
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
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
    xq = np.linspace(0.3, 1.5, 200)
    
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
legend2 = ax.legend(handles, labels, loc="lower right", title="$t_r$",
                      fontsize=subt_font, title_fontsize=subt_font)


fig.tight_layout()
plt.show()

#%% dirty contours replacement time and cost

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
xvar = 'T_ratio'
yvar = 'zeta_e'

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 1.0)

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

# prob_list = [0.3, 0.2, 0.1]
# from scipy.interpolate import RegularGridInterpolator
# for j, prob_des in enumerate(prob_list):
#     xq = np.linspace(0.3, 1.5, 200)
    
#     Ry_target = 1.0
    
#     interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
#     pts = np.zeros((200,2))
#     pts[:,1] = xq
#     pts[:,0] = Ry_target
    
#     lq = interp(pts)
    
#     the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
#     theGapIdx = np.argmin(abs(lq - prob_des))
    
#     theGap = xq[theGapIdx]
#     ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
#                 linewidth=2.0)
#     ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
#     ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
#               fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
#     ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
#     # Ry = 2.0
#     Ry_target = 2.0
#     pts[:,0] = Ry_target
#     lq = interp(pts)
    
#     the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
#     theGapIdx = np.argmin(abs(lq - prob_des))
    
#     theGap = xq[theGapIdx]
#     ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
#                 linewidth=2.0)
#     ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
#     ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
#               fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
#     ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

# ax.set_xlim([0.3, 2.0])
# ax.set_ylim([0.5, 2.25])

df_sc = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8) & 
           (df['RI']<=2.2) & (df['RI']>=1.8)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$GR = 1.0$ , $R_y = 2.0$', fontsize=title_font)
ax.set_xlabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$c_r$",
                      fontsize=subt_font, title_fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 1.0)

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

# prob_list = [0.3, 0.2, 0.1]
# from scipy.interpolate import RegularGridInterpolator
# for j, prob_des in enumerate(prob_list):
#     xq = np.linspace(0.3, 1.5, 200)
    
#     Ry_target = 1.0
    
#     interp = RegularGridInterpolator((y_pl, x_pl), Z_cont)
#     pts = np.zeros((200,2))
#     pts[:,1] = xq
#     pts[:,0] = Ry_target
    
#     lq = interp(pts)
    
#     the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
#     theGapIdx = np.argmin(abs(lq - prob_des))
    
#     theGap = xq[theGapIdx]
#     ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
#                 linewidth=2.0)
#     ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
#     ax.text(theGap+0.05, 0.55, r'GR = '+f'{theGap:,.2f}', rotation=90,
#               fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
#     ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    
    
#     # Ry = 2.0
#     Ry_target = 2.0
#     pts[:,0] = Ry_target
#     lq = interp(pts)
    
#     the_points = np.vstack((pts[:,0], pts[:,1], lq))
    
#     theGapIdx = np.argmin(abs(lq - prob_des))
    
#     theGap = xq[theGapIdx]
#     ax.vlines(x=theGap, ymin=0.49, ymax=Ry_target, color='red',
#                 linewidth=2.0)
#     ax.hlines(y=Ry_target, xmin=0.3, xmax=theGap, color='red', linewidth=2.0)
#     ax.text(theGap+0.05, 1.7, r'GR = '+f'{theGap:,.2f}', rotation=90,
#               fontsize=subt_font, color='red', bbox=dict(facecolor='white', edgecolor='red'))
#     ax.plot([theGap], [Ry_target], marker='*', markersize=15, color='red')
    


ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2.5,))

# ax.set_xlim([0.3, 2.0])
# ax.set_ylim([0.5, 2.25])

df_sc = df[(df['gap_ratio']<=1.2) & (df['gap_ratio']>=0.8) & 
           (df['RI']<=2.2) & (df['RI']>=1.8)]

sc = ax.scatter(df_sc[xvar],
            df_sc[yvar],
            c=df_sc[cost_var], cmap='Blues',
            s=20, edgecolors='k', linewidth=0.5)

ax.grid(visible=True)
ax.set_title(r'$GR = 1.0$ , $R_y = 2.0$', fontsize=title_font)
ax.set_xlabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors")
legend2 = ax.legend(handles, labels, loc="lower right", title="$t_r$",
                      fontsize=subt_font, title_fontsize=subt_font)


fig.tight_layout()
plt.show()

#%% Testing the design space
import time

res_des = 20
X_space = make_design_space(res_des)
#K_space = mdl.get_kernel(X_space, kernel_name='rbf', gamma=gam)

# choice GPC
# HOWEVER, SVC is poorly calibrated for probablities
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

#%% Calculate upfront cost of data

def get_steel_coefs(df, steel_per_unit=1.25):
    n_bays = df.num_bays
    n_stories = df.num_stories
    # ft
    L_bldg = df.L_bldg
    L_beam = df.L_bay
    h_story = df.h_story
    
    # weights
    W = df.W
    Ws = df.W_s
    
    
    all_beams = df.beam
    all_cols = df.column
    
    # sum of per-length-weight of all floors
    col_wt = [[float(member.split('X',1)[1]) for member in col_list] 
                       for col_list in all_cols]
    beam_wt = [[float(member.split('X',1)[1]) for member in beam_list] 
                       for beam_list in all_beams]
    col_all_wt = np.array(list(map(sum, col_wt)))
    beam_all_wt = np.array(list(map(sum, beam_wt)))
    
    # find true steel costs
    n_frames = 4
    n_cols = 4*n_bays
    
    total_floor_col_length = np.array(n_cols*h_story, dtype=float)
    total_floor_beam_length = np.array(L_beam * n_bays * n_frames, dtype=float)
        
    total_col_wt = col_all_wt*total_floor_col_length 
    total_beam_wt = beam_all_wt*total_floor_beam_length
    
    bldg_wt = total_col_wt + total_beam_wt
    
    steel_cost = steel_per_unit*bldg_wt
    bldg_sf = np.array(n_stories * L_bldg**2, dtype=float)
    steel_cost_per_sf = steel_cost/bldg_sf
    
    # find design base shear as a feature
    pi = 3.14159
    g = 386.4
    kM = (1/g)*(2*pi/df['T_m'])**2
    S1 = 1.017
    Dm = g*S1*df['T_m']/(4*pi**2*df['Bm'])
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*df['zeta_e'])
    Vs = np.array(Vst/df['RI']).reshape(-1,1)
    
    # linear regress cost as f(base shear)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X=Vs, y=steel_cost_per_sf)
    return({'coef':reg.coef_, 'intercept':reg.intercept_})

def calc_upfront_cost(X_test, steel_coefs,
                      land_cost_per_sqft=2837/(3.28**2),
                      W=3037.5, Ws=2227.5):
    
    from scipy.interpolate import interp1d
    zeta_ref = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Bm_ref = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    interp_f = interp1d(zeta_ref, Bm_ref)
    Bm = interp_f(X_test['zeta_e'])
    
    # estimate Tm
    
    from loads import estimate_period
    
    # current dummy structure: 4 bays, 4 stories
    # 13 ft stories, 30 ft bays
    X_query = X_test.copy()
    X_query['superstructure_system'] = 'MF'
    X_query['h_bldg'] = 4*13.0
    X_query['T_fbe'] = X_query.apply(lambda row: estimate_period(row),
                                                     axis='columns', result_type='expand')
    
    X_query['T_m'] = X_query['T_fbe'] * X_query['T_ratio']
    
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    S1 = 1.017
    SaTm = S1/X_query['T_m']
    moat_gap = X_query['gap_ratio'] * (g*(SaTm/Bm)*X_query['T_m']**2)/(4*pi**2)
    
    # calculate design base shear
    kM = (1/g)*(2*pi/X_query['T_m'])**2
    Dm = g*S1*X_query['T_m']/(4*pi**2*Bm)
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*X_query['zeta_e'])
    Vs = Vst/X_query['RI']
    
    # steel coefs now represent cost/sf as a function of Vs
    steel_cost_per_sf = steel_coefs['intercept'] + steel_coefs['coef']*Vs
    # land_area = 2*(90.0*12.0)*moat_gap - moat_gap**2
    
    bldg_area = 4 * (30*4)**2
    steel_cost = steel_cost_per_sf * bldg_area
    land_area = (4*30*12.0 + moat_gap)**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return({'total': steel_cost + land_cost,
            'steel': steel_cost,
            'land': land_cost})

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

#%% baseline predictions
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

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

uncut_baseline_cost = mdl_unconditioned.gpr.predict(X_baseline)

#%% one single inverse design
# sample building is 4 bay, 4 stories
og_df = main_obj.doe_analysis.reset_index(drop=True)

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


steel_price = 4.0
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

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

# manual filter?
X_design = X_design[X_design['T_ratio']>3.0]


# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs['total'].idxmin()
design_upfront_cost = upfront_costs['total'].min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_downtime = space_downtime.iloc[cheapest_design_idx].item()
design_repair_cost = space_repair_cost.iloc[cheapest_design_idx].item()
design_repl_risk = space_repl.iloc[cheapest_design_idx].item()

# read out predictions
print('==================================')
print('            Predictions           ')
print('==================================')
print('======= Targets =======')
print('Repair cost fraction:', f'{percent_of_replacement*100:,.2f}%')
print('Repair time (days):', dt_thresh/n_worker_parallel)
print('Replacement risk:', f'{repl_thresh*100:,.2f}%')


print('======= Overall inverse design =======')
print(best_design)
print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted median repair cost ratio: ',
      f'{design_repair_cost*100:,.2f}%')
print('Predicted median repair cost: ',
      f'${design_repair_cost*cmp_cost:,.2f}')
print('Predicted repair time (parallel): ',
      f'{design_downtime*cmp_time/n_worker_parallel:,.2f}', 'days')
print('Predicted repair time (parallel): ',
      f'{design_downtime*cmp_time:,.2f}', 'worker-days')
print('Predicted repair time ratio: ',
      f'{design_downtime*100:,.2f}%')
print('Predicted replacement risk: ',
      f'{design_repl_risk:.2%}')

baseline_upfront_cost_all = calc_upfront_cost(X_baseline, coef_dict)
baseline_upfront_cost = baseline_upfront_cost_all['total'].item()
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

#%% sensitivity analysis to steel cost
res_des = 30

X_space = make_design_space(res_des, zeta_fix=0.15)

space_repair_cost = predict_DV(X_space, 
                               mdl_impact.gpc, 
                               mdl_cost_hit.gpr, 
                               mdl_cost_miss.gpr, 
                               outcome=cost_var)

space_downtime = predict_DV(X_space,
                            mdl_impact.gpc,
                            mdl_time_hit.gpr,
                            mdl_time_miss.gpr,
                            outcome=time_var)

space_repl = predict_DV(X_space,
                        mdl_impact.gpc,
                        mdl_repl_hit.gpr,
                        mdl_repl_miss.gpr,
                        outcome='replacement_freq')

covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
kernel_name = 'rbf_iso'
       
#%%
from loads import define_gravity_loads
config_dict = {
    'S_1' : 1.017,
    'k_ratio' : 10,
    'Q': 0.06,
    'num_frames' : 2,
    'num_bays' : 4,
    'num_stories' : 4,
    'L_bay': 30.0,
    'h_story': 13.0,
    'isolator_system' : 'TFP',
    'superstructure_system' : 'MF',
    'S_s' : 2.2815
}
(W_seis, W_super, w_on_frame, P_on_leaning_column,
       all_w_cases, all_plc_cases) = define_gravity_loads(config_dict)


##### steel

land_prices = np.arange(50.0, 400.0, 50.0)

land_sens = np.zeros((len(land_prices), len(covariate_list)))

coef_dict = get_steel_coefs(df, steel_per_unit=4)

for land_idx, land_price in enumerate(land_prices):
    
    # print('Sensitivity analysis for land price:', land_price)
    
    # < 20% of cost of replacing all components
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
    upfront_costs = calc_upfront_cost(X_design, coef_dict, land_cost_per_sqft=land_price,
                                      W=W_seis, Ws=W_super)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    design_upfront_cost = upfront_costs['total'].min()

    # least upfront cost of the viable designs
    best_design = X_design.loc[cheapest_design_idx]
    land_sens[land_idx,:] = np.array(best_design).T
    

##### steel

steel_prices = np.arange(1.0, 11.0, 1.0)

steel_sens = np.zeros((len(steel_prices), len(covariate_list)))

for steel_idx, steel_price in enumerate(steel_prices):
    
    # print('Sensitivity analysis for steel price:', steel_price)
    coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)
    
    # < 20% of cost of replacing all components
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
    upfront_costs = calc_upfront_cost(X_design, coef_dict)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    design_upfront_cost = upfront_costs['total'].min()

    # least upfront cost of the viable designs
    best_design = X_design.loc[cheapest_design_idx]
    steel_sens[steel_idx,:] = np.array(best_design).T
    
##### replacement cost

repair_costs = np.arange(0.05, 0.55, 0.05)

repair_cost_sens = np.zeros((len(repair_costs), len(covariate_list)))
coef_dict = get_steel_coefs(df, steel_per_unit=4)

for cr_index, repair_cost in enumerate(repair_costs):
    
    # print('Sensitivity analysis for cost ratio:', repair_cost)
    
    # < 20% of cost of replacing all components
    percent_of_replacement = repair_cost
    ok_cost = X_space.loc[space_repair_cost[cost_var+'_pred']<=percent_of_replacement]

    # remove threshold for downtime and risk
    X_design = X_space[X_space.index.isin(ok_cost.index)]

    # select best viable design
    upfront_costs = calc_upfront_cost(X_design, coef_dict)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    design_upfront_cost = upfront_costs['total'].min()

    # least upfront cost of the viable designs
    best_design = X_design.loc[cheapest_design_idx]
    repair_cost_sens[cr_index,:] = np.array(best_design).T
    
    
##### replacement downtime

repair_times = np.arange(0.1, 1.1, 0.1)

repair_time_sens = np.zeros((len(repair_times), len(covariate_list)))
coef_dict = get_steel_coefs(df, steel_per_unit=4)

for tr_index, repair_time in enumerate(repair_times):
    
    # print('Sensitivity analysis for time ratio:', repair_time)
    
    ok_time = X_space.loc[space_downtime[time_var+'_pred']<=repair_time]

    # remove threshold for cost and risk
    X_design = X_space[X_space.index.isin(ok_time.index)]

    # select best viable design
    upfront_costs = calc_upfront_cost(X_design, coef_dict)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    design_upfront_cost = upfront_costs['total'].min()

    # least upfront cost of the viable designs
    best_design = X_design.loc[cheapest_design_idx]
    repair_time_sens[tr_index,:] = np.array(best_design).T
    
##### replacement downtime

replacement_risks = np.arange(0.05, 0.55, 0.05)

risk_sens = np.zeros((len(replacement_risks), len(covariate_list)))
coef_dict = get_steel_coefs(df, steel_per_unit=4)

for pr_index, risk in enumerate(replacement_risks):
    
    # print('Sensitivity analysis for replacement risk:', risk)
    
    ok_risk = X_space.loc[space_repl['replacement_freq'+'_pred']<=risk]

    # remove threshold for cost and risk
    X_design = X_space[X_space.index.isin(ok_risk.index)]

    # select best viable design
    upfront_costs = calc_upfront_cost(X_design, coef_dict)
    cheapest_design_idx = upfront_costs['total'].idxmin()
    design_upfront_cost = upfront_costs['total'].min()

    # least upfront cost of the viable designs
    best_design = X_design.loc[cheapest_design_idx]
    risk_sens[pr_index,:] = np.array(best_design).T
    
#%% sens plotting

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
title_font=20
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')
fig = plt.figure(figsize=(14, 8))
ax=fig.add_subplot(2, 3, 2)


color = plt.cm.Set1(np.linspace(0, 1, 10))

baseline_ref = np.array([1.0, 2.0, 3.0, 0.15])

plt_array = steel_sens/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(steel_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])

ax.set_xlabel(r'Steel cost per lb. (\$)', fontsize=axis_font)
ax.set_ylabel(r'\% change', fontsize=axis_font)
# ax.set_ylim([25, 200])
# ax.legend(fontsize=label_size)
ax.grid()

ax=fig.add_subplot(2, 3, 4)
plt_array = repair_cost_sens/baseline_ref*100
for plt_idx in range(3):
    ax.plot(repair_costs, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Repair cost ratio targets', fontsize=axis_font)
# ax.set_ylim([25, 200])
ax.grid()

ax=fig.add_subplot(2, 3, 5)
plt_array = repair_time_sens/baseline_ref*100
for plt_idx in range(3):
    ax.plot(repair_times, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Downtime ratio targets', fontsize=axis_font)

ax.set_ylabel(r'\% change', fontsize=axis_font)
# ax.set_ylim([25, 200])
ax.grid()



##

ax=fig.add_subplot(2, 3, 6)
plt_array = risk_sens/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(replacement_risks, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Replacement risk targets', fontsize=axis_font)
# ax.set_ylim([25, 200])
# ax.legend(fontsize=label_size)
ax.grid()

fig.tight_layout()
plt.show()

###
ax=fig.add_subplot(2, 3, 3)
plt_array = land_sens/baseline_ref*100
label = [r'$GR$', r'$R_y$', r'$T_M/T_{fb}$']
for plt_idx in range(3):
    ax.plot(land_prices, plt_array[:,plt_idx], marker='s', markeredgecolor='black', 
            c=color[plt_idx], label=label[plt_idx])
ax.set_xlabel(r'Land cost per sf. (\$)', fontsize=axis_font)
# ax.set_ylim([25, 200])
ax.legend(fontsize=label_size)
ax.grid()

fig.tight_layout()
plt.show()

#%% pareto - doe

# remake X_plot in gap Ry
X_plot = make_2D_plotting_space(mdl_impact.X, res)


risk_thresh = 0.1

all_costs = calc_upfront_cost(X_space, coef_dict, W=W_seis, Ws=W_super)
constr_costs = all_costs['total']
predicted_risk = space_repl['replacement_freq_pred'] #+ fs1_design
predicted_cost = space_repair_cost[cost_var+'_pred']
# predicted_risk[predicted_risk < 0] = 0


# acceptable_mask = predicted_risk < risk_thresh
# X_acceptable = X_space[acceptable_mask]
# acceptable_cost = constr_costs[acceptable_mask]
# acceptable_risk = predicted_risk[acceptable_mask]

pareto_array = np.array([constr_costs, predicted_risk]).transpose()
# pareto_array = np.array([acceptable_cost, acceptable_risk]).transpose()

t0 = time.time()
pareto_mask = is_pareto_efficient(pareto_array)
tp = time.time() - t0

print("Culled %d points in %.3f s" % (pareto_array.shape[0], tp))

X_pareto = X_space.iloc[pareto_mask].copy()
risk_pareto = predicted_risk.iloc[pareto_mask]
cost_pareto = constr_costs.iloc[pareto_mask]

# X_pareto = X_acceptable.iloc[pareto_mask].copy()
# risk_pareto = acceptable_risk.iloc[pareto_mask]
# cost_pareto = acceptable_cost.iloc[pareto_mask]

# -1 if predicted risk > allowable

X_pareto['acceptable_risk'] = np.sign(risk_thresh - risk_pareto)
X_pareto['predicted_risk'] = risk_pareto

dom_idx = np.random.choice(len(pareto_array), len(pareto_array)//10, 
                           replace = False)
dominated_sample = np.array([pareto_array[i] for i in dom_idx])

# plt.close('all')
fig = plt.figure(figsize=(14, 6))

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 24
subt_font = 18
title_font = 26
import matplotlib as mpl
label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx_pl.min(), xx_pl.max(),
#             yy_pl.min(), yy_pl.max()),
#     aspect="auto",
#     origin="lower",
#     cmap=plt.cm.Blues,
# ) 
ax = fig.add_subplot(1, 2, 1)
ax.scatter(risk_pareto, cost_pareto, marker='s', facecolors='none',
            edgecolors='green', s=20.0, label='Pareto optimal designs')
ax.scatter(risk_pareto, cost_pareto, s=1, color='black')
ax.scatter(dominated_sample[:,1], dominated_sample[:,0], s=1, color='black',
           label='Dominated designs')
ax.set_xlabel('Predicted replacement risk', fontsize=axis_font)
ax.set_ylabel('Construction cost', fontsize=axis_font)
# ax.set_xlim([-0.1, 0.7])
ax.grid(True)
ax.legend(fontsize=label_size)
plt.title('a) Pareto front', fontsize=title_font)
plt.show()


x_var = 'gap_ratio'
xx = X_plot[x_var]
y_var = 'RI'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

cmap = plt.cm.Spectral_r
ax1=fig.add_subplot(1, 2, 2)
lvls = [0.025, 0.05, 0.10, 0.2, 0.3]
sc = ax1.scatter(X_pareto['gap_ratio'], X_pareto['RI'], 
            c=X_pareto['predicted_risk'], s=20.0, cmap=cmap)

# cbar = plt.colorbar(sc, ticks=[0, 0.2, 0.4])
# cs = plt.contour(xx_pl, yy_pl, Z_GRy, linewidths=1.1, cmap='Blues', vmin=-1,
#                   levels=lvls)
# plt.clabel(cs, fontsize=clabel_size)
ax1.set_xlim([0.5, 2.0])
ax1.set_ylim([0.45, 2.3])
ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)
ax1.grid(True)
ax1.set_title(r'b) $T_M / T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=title_font)



from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(ax1,  width="3%", height="20%", loc='lower right',
                    bbox_to_anchor=(-0.05,0.5,1,1), bbox_transform=ax1.transAxes) 
plt.colorbar(sc, cax=cbaxes, orientation='vertical')
cbaxes.set_ylabel('Replacement risk', fontsize=axis_font)
cbaxes.yaxis.set_ticks_position('left')

fig.tight_layout(w_pad=0.0, h_pad=0.0)
plt.show()
'''
X_plot = make_2D_plotting_space(mdl_doe.X, res, x_var='T_ratio', y_var='zeta_e', 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

x_var = 'T_ratio'
xx = X_plot[x_var]
y_var = 'zeta_e'
yy = X_plot[y_var]

x_pl = np.unique(xx)
y_pl = np.unique(yy)

# collapse predictions
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax2=fig.add_subplot(1, 2, 2)
plt.clabel(cs, fontsize=clabel_size)
cs = plt.contour(xx_pl, yy_pl, Z_Tze, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)
plt.clabel(cs, fontsize=clabel_size)
ax2.scatter(X_pareto['T_ratio'], X_pareto['zeta_e'], 
            c=X_pareto['predicted_risk'],
            edgecolors='k', s=20.0, cmap=plt.cm.Spectral_r)
ax2.set_xlabel('T ratio', fontsize=axis_font)
ax2.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
ax2.grid(True)
ax2.set_title(r'b) $GR = 1.0$, $R_y = 2.0$', fontsize=axis_font)


plt.savefig('./figures/pareto.eps')


# 3D

ax=fig.add_subplot(2, 2, 3, projection='3d')

sc = ax.scatter(X_pareto['gap_ratio'], X_pareto['RI'], X_pareto['T_ratio'], 
           c=X_pareto['predicted_risk'], alpha = 1, cmap=plt.cm.Spectral_r)
ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$T_M / T_{fb}$', fontsize=axis_font)
ax.set_title(r'c) $\zeta_M$ not shown', fontsize=title_font)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax=fig.add_subplot(2, 2, 4, projection='3d')

ax.scatter(X_pareto['gap_ratio'], X_pareto['RI'], X_pareto['zeta_e'], 
           c=X_pareto['predicted_risk'], alpha = 1, cmap=plt.cm.Spectral_r)
ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)
# ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$\zeta_M$', fontsize=axis_font)
ax.set_title(r'd) $T_M/T_{fb}$ not shown', fontsize=title_font)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# fig.colorbar(sc, ax=ax)
# plt.savefig('./figures/pareto_full.pdf')

'''
#%% filter design graphic
# TODO: you are here

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

fig = plt.figure(figsize=(16, 5))

#################################
xvar = 'gap_ratio'
yvar = 'RI'

lvls = np.array([0.025, 0.05, 0.1, 0.2])

res = 100
X_plot = make_2D_plotting_space(mdl_impact.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)

xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

ax = fig.add_subplot(1, 4, 1)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

# cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
#                  levels=lvls)

grid_repl = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_repl_hit.gpr,
                       mdl_repl_miss.gpr,
                       outcome=cost_var)

Z = np.array(grid_repl)
Z_cont = Z.reshape(xx_pl.shape)

cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)
# clabels = ax.clabel(cs, fontsize=clabel_size)

nm, lbl = cs.legend_elements()
lbl = ['\% replacement']
plt.legend(nm, lbl, title= '', fontsize= subt_font) 

ax.clabel(cs, fontsize=clabel_size)

ax.grid(visible=True)
ax.set_title('Replacement risk$< 10.0\%$', fontsize=title_font)
# ax.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('red'),
            linestyles=('-'),linewidths=(2,))

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

ax.grid(visible=True)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
ax.set_ylabel(r'$R_y$', fontsize=axis_font)

#### cost now
ax = fig.add_subplot(1, 4, 2)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

lvls=[-0.01, 0.1, 5.0]
cs = ax.contourf(xx_pl, yy_pl, Z_cont, cmap='Greys', levels=lvls)
cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.1], colors=('black'),
            linestyles=('--'),linewidths=(2,))
ax.clabel(cs, fontsize=clabel_size, colors='black')

grid_cost = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_cost_hit.gpr,
                       mdl_cost_miss.gpr,
                       outcome=cost_var)

Z = np.array(grid_cost)
Z_cont = Z.reshape(xx_pl.shape)

lvls = np.array([0.025, 0.05, 0.1, 0.2])
cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)

nm, lbl = cs.legend_elements()
lbl = ['Repair cost ratio']
plt.legend(nm, lbl, title= '', fontsize= subt_font) 
ax.clabel(cs, fontsize=clabel_size)

cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.2], colors=('red'),
            linestyles=('-'),linewidths=(2,))
ax2.clabel(cs, fontsize=clabel_size, colors='red')

ax.grid(visible=True)
ax.set_title('Repair cost $<20\%$ replacement', fontsize=title_font)

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

ax.grid(visible=True)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

#### time now
ax = fig.add_subplot(1, 4, 3)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

lvls=[-0.01, 0.2, 5.0]
cs = ax.contourf(xx_pl, yy_pl, Z_cont, cmap='Greys', levels=lvls)
cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.2], colors=('black'),
            linestyles=('--'),linewidths=(2,))
ax.clabel(cs, fontsize=clabel_size, colors='black')

grid_time = predict_DV(X_plot,
                       mdl_impact.gpc,
                       mdl_time_hit.gpr,
                       mdl_time_miss.gpr,
                       outcome=time_var)

Z = np.array(grid_time)
Z_cont = Z.reshape(xx_pl.shape)

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3])
cs = ax.contour(xx_pl, yy_pl, Z_cont, linewidths=2.0, cmap='Blues', vmin=-1,
                 levels=lvls)

nm, lbl = cs.legend_elements()
lbl = ['Repair time ratio']
plt.legend(nm, lbl, title= '', fontsize= subt_font) 
ax.clabel(cs, fontsize=clabel_size)

cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.203], colors=('red'),
            linestyles=('-'),linewidths=(2,))
ax2.clabel(cs, fontsize=clabel_size, colors='red')

ax.grid(visible=True)
ax.set_title('Repair time $<$ 2 weeks', fontsize=title_font)

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

ax.grid(visible=True)
ax.set_xlabel(r'$GR$', fontsize=axis_font)
# ax.set_ylabel(r'$R_y$', fontsize=axis_font)

#### final space
ax = fig.add_subplot(1, 4, 4)
plt.setp(ax, xticks=np.arange(0.5, 5.0, step=0.5))

lvls=[-0.01, 0.203, 5.0]
cs = ax.contourf(xx_pl, yy_pl, Z_cont, cmap='Greys', levels=lvls)
cs = ax.contour(xx_pl, yy_pl, Z_cont, levels = [0.203], colors=('black'),
            linestyles=('--'),linewidths=(2,))
ax.clabel(cs, fontsize=clabel_size, colors='black')
ax.set_title('Acceptable design space', fontsize=title_font)

ax.set_xlim([0.3, 2.0])
ax.set_ylim([0.5, 2.25])

ax.grid(visible=True)
ax.set_xlabel(r'$GR$', fontsize=axis_font)

fig.tight_layout()
ax.text(1.2, 1.00, 'OK space',
          fontsize=axis_font, color='green')
#%% validation with lognormal fit EDPs
val_dir = '../../data/loss/'

val_inv_file = 'tfp_mf_db_val_inverse_loss.pickle'
baseline_file = 'tfp_mf_db_val_baseline_loss.pickle'

main_obj_val = pd.read_pickle(val_dir+val_inv_file)
val_ln_loss = main_obj_val.loss_data.reset_index(drop=True)
val_ln_run = main_obj_val.ida_results.reset_index(drop=True)

main_obj_val = pd.read_pickle(val_dir+baseline_file)
base_ln_loss = main_obj_val.loss_data.reset_index(drop=True)
base_ln_run = main_obj_val.ida_results.reset_index(drop=True)

#%% full validation (IDA data)

# TODO: validation was ran with changing component generation (distribution)
# this may affect result in the sense that we eventually normalize by an estimated
# total worth of building content (averaged)

val_dir = '../../data/loss/'

val_inv_file = 'tfp_mf_db_valdet_inverse_loss.pickle'
baseline_file = 'tfp_mf_db_valdet_baseline_loss.pickle'

main_obj_val = pd.read_pickle(val_dir+val_inv_file)
val_loss = main_obj_val.loss_data.reset_index(drop=True)
val_run = main_obj_val.ida_results.reset_index(drop=True)

main_obj_val = pd.read_pickle(val_dir+baseline_file)
base_loss = main_obj_val.loss_data.reset_index(drop=True)
base_run = main_obj_val.ida_results.reset_index(drop=True)

# collect means
cost_var_ida = 'cost_50%'
time_var_ida = 'time_l_50%'


df_val = pd.concat([val_run, val_loss], axis=1)
# df_val['max_drift'] = df_val[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
# df_val['collapse_probs'] = ln_dist.cdf(np.array(df_val['max_drift']))
# df_val['repair_time'] = df[time_var]/n_worker_parallel

df_base = pd.concat([base_run, base_loss], axis=1)
# df_base['max_drift'] = df_base[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
# df_base['collapse_probs'] = ln_dist.cdf(np.array(df_base['max_drift']))
# df_base['repair_time'] = df[time_var]/n_worker_parallel

ida_levels = [1.0, 1.5, 2.0]
validation_replacement = np.zeros((3,))
baseline_replacement = np.zeros((3,))
validation_cost  = np.zeros((3,))
baseline_cost = np.zeros((3,))
validation_downtime = np.zeros((3,))
baseline_downtime = np.zeros((3,))

for i, lvl in enumerate(ida_levels):
    val_ida = val_loss[val_loss['ida_level']==lvl]
    base_ida = base_loss[base_loss['ida_level']==lvl]
    
    validation_replacement[i] = val_ida['replacement_freq'].mean()
    validation_downtime[i] = val_ida[time_var_ida].mean()/n_worker_parallel
    validation_cost[i] = val_ida[cost_var_ida].mean()
    
    baseline_downtime[i] = base_ida[time_var_ida].mean()/n_worker_parallel
    baseline_cost[i] = base_ida[cost_var_ida].mean()
    baseline_replacement[i] = base_ida['replacement_freq'].mean()
    

print('==================================')
print('   Validation results  (1.0 MCE)  ')
print('==================================')

inverse_cost = validation_cost[0]
inverse_downtime = validation_downtime[0]
inverse_replacement = validation_replacement[0]
design_tested = df_val[['moat_ampli', 'RI', 'T_ratio' , 'zeta_e']].iloc[0]
design_specifics = df_val[['mu_1', 'mu_2', 'R_1', 'R_2', 'beam', 'column']].iloc[0]

print('====== INVERSE DESIGN ======')
print('Average median repair cost: ',
      f'${inverse_cost:,.2f}')
print('Repair cost ratio: ', 
      f'{inverse_cost/cmp_cost*100:,.3f}%')
print('Average median repair time: ',
      f'{inverse_downtime:,.2f}', 'days')
print('Repair time ratio: ',
      f'{inverse_downtime/cmp_time*100*n_worker_parallel:,.3f}%')
print('Estimated replacement frequency: ',
      f'{inverse_replacement:.2%}')
print(design_tested)
print(design_specifics)

baseline_cost_mce = baseline_cost[0]
baseline_downtime_mce = baseline_downtime[0]
baseline_replacement_mce = baseline_replacement[0]
design_tested = df_base[['moat_ampli', 'RI', 'T_ratio' , 'zeta_e']].iloc[0]
design_specifics = df_base[['mu_1', 'mu_2', 'R_1', 'R_2', 'beam', 'column']].iloc[0]

print('====== BASELINE DESIGN ======')
print('Average median repair cost: ',
      f'${baseline_cost_mce:,.2f}')
print('Repair cost ratio: ', 
      f'{baseline_cost_mce/cmp_cost*100:,.3f}%')
print('Average median repair time: ',
      f'{baseline_downtime_mce:,.2f}', 'days')
print('Repair time ratio: ',
      f'{baseline_downtime_mce/cmp_time*100*n_worker_parallel:,.3f}%')
print('Estimated replacement frequency: ',
      f'{baseline_replacement_mce:.2%}')
print(design_tested)
print(design_specifics)

#%% MLE fragility curves
def neg_log_likelihood_sum(params, im_l, no_a, no_c):
    from scipy import stats
    import numpy as np
    sigma, beta = params
    theoretical_fragility_function = stats.norm(np.log(sigma), beta).cdf(im_l)
    likelihood = stats.binom.pmf(no_c, no_a, theoretical_fragility_function)
    log_likelihood = np.log(likelihood)
    log_likelihood_sum = np.sum(log_likelihood)

    return -log_likelihood_sum

def mle_fit_collapse(ida_levels, pr_collapse):
    from functools import partial
    import numpy as np
    from scipy import optimize
    
    im_log = np.log(ida_levels)
    number_of_analyses = np.array([1000, 1000, 1000 ])
    number_of_collapses = np.round(1000*pr_collapse)
    
    neg_log_likelihood_sum_partial = partial(
        neg_log_likelihood_sum, im_l=im_log, no_a=number_of_analyses, no_c=number_of_collapses)
    
    
    res = optimize.minimize(neg_log_likelihood_sum_partial, (1, 1), method="Nelder-Mead")
    return res.x[0], res.x[1]


from scipy.stats import norm
f = lambda x,theta,beta: norm(np.log(theta), beta).cdf(np.log(x))

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
title_font=20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
# plt.close('all')

fig = plt.figure(figsize=(16, 7))

b_TOT = np.linalg.norm([0.2, 0.2, 0.2, 0.4])

theta_inv, beta_inv = mle_fit_collapse(ida_levels,validation_replacement)

xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_inv, beta_inv)
p2 = f(xx_pr, theta_inv, b_TOT)

MCE_level = float(p[xx_pr==1.0])
MCE_level_unc = float(p2[xx_pr==1.0])
ax1=fig.add_subplot(1, 2, 1)
ax1.plot(xx_pr, p)
# ax1.plot(xx_pr, p2)
ax1.axhline(design_repl_risk, linestyle='--', color='black')
ax1.axvline(1.0, linestyle='--', color='black')
ax1.text(2.2, design_repl_risk+0.02, r'Predicted replacement risk',
          fontsize=subt_font, color='black')
ax1.text(0.6, 0.04, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='steelblue')
# ax1.text(0.2, 0.12, f'{MCE_level_unc:,.4f}',
#           fontsize=subt_font, color='orange')
ax1.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')

ax1.set_ylabel('Replacement probability', fontsize=axis_font)
# ax1.set_xlabel(r'Scale factor', fontsize=axis_font)
ax1.set_title('Inverse design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax1.plot([lvl], [validation_replacement[i]], 
              marker='x', markersize=15, color="red")
ax1.grid()
ax1.set_xlim([0, 4.0])
ax1.set_ylim([0, 1.0])


####
theta_base, beta_base = mle_fit_collapse(ida_levels, baseline_replacement)
xx_pr = np.arange(0.01, 4.0, 0.01)
p = f(xx_pr, theta_base, beta_base)
p2 = f(xx_pr, theta_base, b_TOT)

MCE_level = float(p[xx_pr==1.0])
MCE_level_unc = float(p2[xx_pr==1.0])
ax4=fig.add_subplot(1, 2, 2)
ax4.plot(xx_pr, p, label='Best lognormal fit')
# ax4.plot(xx_pr, p2, label='Adjusted for uncertainty')
ax4.axhline(baseline_repl_risk_pred, linestyle='--', color='black')
ax4.axvline(1.0, linestyle='--', color='black')
ax4.text(0.8, 0.65, r'$MCE_R$ level', rotation=90,
          fontsize=subt_font, color='black')
ax4.text(2.2, baseline_repl_risk_pred+0.02, r'Predicted replacement risk',
          fontsize=subt_font, color='black')
ax4.text(0.6, 0.13, f'{MCE_level:,.4f}',
          fontsize=subt_font, color='steelblue')
# ax4.text(0.2, 0.2, f'{MCE_level_unc:,.4f}',
#           fontsize=subt_font, color='orange')

# ax4.set_ylabel('Collapse probability', fontsize=axis_font)
ax4.set_xlabel(r'Scale factor', fontsize=axis_font)
ax4.set_title('Baseline design', fontsize=title_font)
for i, lvl in enumerate(ida_levels):
    ax4.plot([lvl], [baseline_replacement[i]], 
              marker='x', markersize=15, color="red")
ax4.grid()
ax4.set_xlim([0, 4.0])
ax4.set_ylim([0, 1.0])
# ax4.legend(fontsize=subt_font-2, loc='center right')

fig.tight_layout()
# plt.savefig('./figures/fragility_curve.eps', dpi=1200, format='eps')
plt.show()
#%% cost validation distr

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 18
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

val_ida = val_loss[val_loss['ida_level']==1.0]
base_ida = base_loss[base_loss['ida_level']==1.0]
val_ida['repair_coef'] = val_ida[cost_var_ida]/cmp_cost
base_ida['repair_coef'] = base_ida[cost_var_ida]/cmp_cost

base_repl_cases = base_ida[base_ida[cost_var_ida] == replacement_cost].count()[cost_var_ida]
inv_repl_cases = val_ida[val_ida[cost_var_ida] == replacement_cost].count()[cost_var_ida]
print('Inverse runs requiring replacement:', inv_repl_cases)
print('Baseline runs requiring replacement:', base_repl_cases)

fig, axes = plt.subplots(1, 1, 
                         figsize=(10, 6))
df_dt = pd.DataFrame.from_dict(
    data=dict(Inverse=val_ida['repair_coef'], Baseline=base_ida['repair_coef']),
    orient='index',
).T

ax = sns.stripplot(data=df_dt, orient='h', palette='coolwarm', 
                   edgecolor='black', linewidth=1.0)
ax.set_xlim(0, .75)
meanpointprops = dict(marker='D', markeredgecolor='black', markersize=10,
                      markerfacecolor='navy')
sns.boxplot(data=df_dt, saturation=0.8, ax=ax, orient='h', palette='coolwarm',
            width=0.4, showmeans=True, meanprops=meanpointprops, meanline=False)
# # ax.set_ylabel('Design case', fontsize=axis_font)
ax.set_xlabel(r'$c_r$', fontsize=axis_font)
ax.axvline(design_repair_cost, ymin=0.5, ymax=1, linestyle='--', color='cornflowerblue')
ax.axvline(baseline_repair_cost_pred, ymin=0.0, ymax=0.5, linestyle='--', color='lightsalmon')
ax.grid(visible=True)

custom_lines = [Line2D([-1], [-1], color='white', marker='D', markeredgecolor='black'
                       , markerfacecolor='navy', markersize=10),
                Line2D([-1], [-1], color='cornflowerblue', linestyle='--'),
                Line2D([-1], [-1], color='lightsalmon', linestyle='--'),
                ]

ax.legend(custom_lines, ['Mean', 'Inverse predicted', 'Baseline predicted'], fontsize=subt_font)

ax.text(.30, 0, u'0 replacements', fontsize=axis_font, color='red')
ax.text(.55, 1, u'5 replacements \u2192', fontsize=axis_font, color='red')
# ax.text(14.5, 1.45, r'14 days threshold', fontsize=axis_font, color='black')
plt.show()

#%% time validation distr

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 18
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

val_ida = val_loss[val_loss['ida_level']==1.0]
base_ida = base_loss[base_loss['ida_level']==1.0]
val_ida['repair_coef'] = val_ida[time_var_ida]/cmp_time
base_ida['repair_coef'] = base_ida[time_var_ida]/cmp_time

base_repl_cases = base_ida[base_ida[time_var_ida] == replacement_time].count()[time_var_ida]
inv_repl_cases = val_ida[val_ida[time_var_ida] == replacement_time].count()[time_var_ida]
print('Inverse runs requiring replacement:', inv_repl_cases)
print('Baseline runs requiring replacement:', base_repl_cases)

fig, axes = plt.subplots(1, 1, 
                         figsize=(10, 6))
df_dt = pd.DataFrame.from_dict(
    data=dict(Inverse=val_ida['repair_coef'], Baseline=base_ida['repair_coef']),
    orient='index',
).T

ax = sns.stripplot(data=df_dt, orient='h', palette='coolwarm', 
                   edgecolor='black', linewidth=1.0)
ax.set_xlim(0, 1.50)
meanpointprops = dict(marker='D', markeredgecolor='black', markersize=10,
                      markerfacecolor='navy')
sns.boxplot(data=df_dt, saturation=0.8, ax=ax, orient='h', palette='coolwarm',
            width=0.4, showmeans=True, meanprops=meanpointprops, meanline=False)
# # ax.set_ylabel('Design case', fontsize=axis_font)
ax.set_xlabel(r'$t_r$', fontsize=axis_font)
ax.axvline(design_downtime, ymin=0.5, ymax=1, linestyle='--', color='cornflowerblue')
ax.axvline(baseline_downtime_pred, ymin=0.0, ymax=0.5, linestyle='--', color='lightsalmon')
ax.grid(visible=True)

custom_lines = [Line2D([-1], [-1], color='white', marker='D', markeredgecolor='black'
                       , markerfacecolor='navy', markersize=10),
                Line2D([-1], [-1], color='cornflowerblue', linestyle='--'),
                Line2D([-1], [-1], color='lightsalmon', linestyle='--'),
                ]

ax.legend(custom_lines, ['Mean', 'Inverse predicted', 'Baseline predicted'], fontsize=subt_font)

ax.text(.50, 0, u'0 replacements', fontsize=axis_font, color='red')
ax.text(1.10, 1, u'5 replacements \u2192', fontsize=axis_font, color='red')
# ax.text(14.5, 1.45, r'14 days threshold', fontsize=axis_font, color='black')
plt.show()

#%%

mdl_repl_unconditioned = GP(df)
mdl_repl_unconditioned.set_covariates(covariate_list)
mdl_repl_unconditioned.set_outcome('replacement_freq')
mdl_repl_unconditioned.test_train_split(0.2)
mdl_unconditioned.fit_gpr(kernel_name='rbf_iso')

#%%
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(9, 8))

#################################
xvar = 'gap_ratio'
yvar = 'T_ratio'

res = 75
X_plot = make_2D_plotting_space(mdl_unconditioned.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)


Z, fs1 = mdl_unconditioned.gpr.predict(X_plot, return_std=True)
fs2 = fs1**2
fs2_plot = fs2.reshape(xx_pl.shape)
Z_unconditioned = Z.reshape(xx_pl.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx_pl, yy_pl, fs2_plot, cmap=plt.cm.Spectral_r,
                       linewidth=0, antialiased=False,
                       alpha=0.5)

# ax.scatter(df[xvar], df[yvar], df['replacement_freq'],
#            c=df['replacement_freq'], alpha=0.3,
#            edgecolors='k')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('$GR$', fontsize=axis_font)
ax.set_ylabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_zlabel('Replacement risk variance', fontsize=axis_font)

fig.tight_layout()