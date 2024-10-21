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

main_obj = pd.read_pickle("../../data/loss/structural_db_complete_normloss.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df_raw = main_obj.ops_analysis
df_raw = df_raw.reset_index(drop=True)

# remove the singular outlier point
from scipy import stats
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 5].copy()

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

df_miss = df[df['impacted'] == 0]

# remove the singular outlier point
from scipy import stats
df_no_impact = df_miss[np.abs(stats.zscore(df_miss['cmp_cost_ratio'])) < 5].copy()

df_tfp = df_no_impact[df_no_impact['isolator_system'] == 'TFP']
df_lrb = df_no_impact[df_no_impact['isolator_system'] == 'LRB']
df_cbf = df_no_impact[df_no_impact['superstructure_system'] == 'CBF']
df_mf = df_no_impact[df_no_impact['superstructure_system'] == 'MF']

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
df_no_impact['bin'] = pd.cut(df_no_impact['gap_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 1)
import seaborn as sns
sns.stripplot(data=df_no_impact, x="max_drift", y="bin", orient="h", alpha=0.8, size=5,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df_no_impact,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)

ax.set_ylabel('$GR$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
plt.xlim([0.0, 0.15])

#####
bins = pd.IntervalIndex.from_tuples([(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.25)])
labels=['tiny', 'small', 'okay', 'large']
df_no_impact['bin'] = pd.cut(df_no_impact['RI'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 2)
import seaborn as sns
sns.stripplot(data=df_no_impact, x="max_drift", y="bin", orient="h", size=5, alpha=0.8,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_drift", data=df_no_impact,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$R_y$ range', fontsize=axis_font)
ax.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
# plt.xlim([0.0, 0.15])



#####
bins = pd.IntervalIndex.from_tuples([(1.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 12.0)])
labels=['tiny', 'small', 'okay', 'large']
df_no_impact['bin'] = pd.cut(df_no_impact['T_ratio'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 3)
import seaborn as sns
sns.stripplot(data=df_no_impact, x="max_accel", y="bin", orient="h", size=5, alpha=0.8,
              hue='superstructure_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_accel", data=df_no_impact,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$T_M/T_{fb}$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor acceleration (g)', fontsize=axis_font)
plt.xlim([0.0, 5.0])

#####
bins = pd.IntervalIndex.from_tuples([(0.1, 0.14), (0.14, 0.18), (0.18, 0.22), (0.22, 0.25)])
labels=['tiny', 'small', 'okay', 'large']
df_no_impact['bin'] = pd.cut(df_no_impact['zeta_e'], bins=bins, labels=labels)


ax = fig.add_subplot(2, 2, 4)
import seaborn as sns
sns.stripplot(data=df_no_impact, x="max_velo", y="bin", orient="h", size=5, alpha=0.8,
              hue='isolator_system', ax=ax, legend='brief', palette='seismic')
sns.boxplot(y="bin", x= "max_velo", data=df_no_impact,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax)


ax.set_ylabel('$\zeta_M$ range', fontsize=axis_font)
ax.set_xlabel('Peak floor velocity (in/s)', fontsize=axis_font)
plt.xlim([25, 125.0])
fig.tight_layout(h_pad=2.0)
plt.show()

#%% 3d surf for repair cost - structure
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

color = plt.cm.Set1(np.linspace(0, 1, 10))

ax=fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf[cost_var], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf[cost_var], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

ax.legend(fontsize=label_size)

ax.set_zlim([0.0, 0.1])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost', fontsize=axis_font)

#################################
ax=fig.add_subplot(1, 2, 2, projection='3d')

xvar = 'T_ratio'
yvar = 'zeta_e'

ax.scatter(df_cbf[xvar], df_cbf[yvar], df_cbf[cost_var], color=color[0],
           edgecolors='k', alpha = 0.7, label='CBF')
ax.scatter(df_mf[xvar], df_mf[yvar], df_mf[cost_var], color=color[1],
           edgecolors='k', alpha = 0.7, label='MF')

ax.set_zlim([0.0, 0.1])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost', fontsize=axis_font)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

fig.tight_layout()

#%% ml training

covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_cost_hit = GP(df_hit)
mdl_cost_hit.set_covariates(covariate_list)
mdl_cost_hit.set_outcome(cost_var)
mdl_cost_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_cost_miss = GP(df_no_impact)
mdl_cost_miss.set_covariates(covariate_list)
mdl_cost_miss.set_outcome(cost_var)
mdl_cost_miss.test_train_split(0.2)

mdl_time_hit = GP(df_hit)
mdl_time_hit.set_covariates(covariate_list)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = GP(df_no_impact)
mdl_time_miss.set_covariates(covariate_list)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_unconditioned = GP(df)
mdl_unconditioned.set_covariates(covariate_list)
mdl_unconditioned.set_outcome(cost_var)
mdl_unconditioned.test_train_split(0.2)

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
    
#%% fit regressions for impact / non-impact set

# Fit conditioned DVs using kernel ridge

print('========== Fitting regressions (kernel ridge) ============')

# fit impacted set
mdl_cost_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')

# fit no impact set
mdl_cost_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')

mdl_unconditioned.fit_kernel_ridge(kernel_name='rbf')
print('========== Fitting regressions (GPR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_gpr(kernel_name='rbf_iso')
mdl_time_hit.fit_gpr(kernel_name='rbf_iso')

# fit no impact set
mdl_cost_miss.fit_gpr(kernel_name='rbf_iso')
mdl_cost_miss.fit_gpr_mean_fcn(kernel_name='rbf_iso')
mdl_time_miss.fit_gpr(kernel_name='rbf_iso')


mdl_unconditioned.fit_gpr(kernel_name='rbf_iso')

print('========== Fitting ordinary ridge (OR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_ols_ridge()
mdl_time_hit.fit_ols_ridge()

# fit no impact set
mdl_cost_miss.fit_ols_ridge()
mdl_time_miss.fit_ols_ridge()



#%% 3d surf for cost ratio - non impact set

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
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 4.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_miss.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


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

ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.1])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_miss.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)

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

ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.1])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()


#%% ml training
'''
############
# k ratio
############

covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'k_ratio']

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
df_miss = df[df['impacted'] == 0]

mdl_cost_hit = GP(df_hit)
mdl_cost_hit.set_covariates(covariate_list)
mdl_cost_hit.set_outcome(cost_var)
mdl_cost_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_cost_miss = GP(df_no_impact)
mdl_cost_miss.set_covariates(covariate_list)
mdl_cost_miss.set_outcome(cost_var)
mdl_cost_miss.test_train_split(0.2)

mdl_time_hit = GP(df_hit)
mdl_time_hit.set_covariates(covariate_list)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = GP(df_no_impact)
mdl_time_miss.set_covariates(covariate_list)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_unconditioned = GP(df)
mdl_unconditioned.set_covariates(covariate_list)
mdl_unconditioned.set_outcome(cost_var)
mdl_unconditioned.test_train_split(0.2)

    
#%% fit regressions for impact / non-impact set

# Fit conditioned DVs using kernel ridge

print('========== Fitting regressions (kernel ridge) ============')

# fit impacted set
mdl_cost_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')

# fit no impact set
mdl_cost_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')

mdl_unconditioned.fit_kernel_ridge(kernel_name='rbf')
print('========== Fitting regressions (GPR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_gpr(kernel_name='rbf_iso')
mdl_time_hit.fit_gpr(kernel_name='rbf_iso')

# fit no impact set
mdl_cost_miss.fit_gpr(kernel_name='rbf_iso')
mdl_time_miss.fit_gpr(kernel_name='rbf_iso')


mdl_unconditioned.fit_gpr(kernel_name='rbf_iso')

print('========== Fitting ordinary ridge (OR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_hit.fit_ols_ridge()
mdl_time_hit.fit_ols_ridge()

# fit no impact set
mdl_cost_miss.fit_ols_ridge()
mdl_time_miss.fit_ols_ridge()


#%% 3d surf for cost ratio - non impact set

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
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 5.0, fourth_var_set = 10.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_miss.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)


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

ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.1])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $k_1/k_2 = 10.0$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'k_ratio'

res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_miss.gpr.predict(X_plot)
# Z, stdev = mdl_cost_miss.predict_gpr_mean_fcn(X_plot)

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

ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.1])
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$k_1/k_2$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()
'''

#%% subsets

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']

df_mf_tfp = df_tfp[df_tfp['superstructure_system'] == 'MF']
df_mf_lrb = df_lrb[df_lrb['superstructure_system'] == 'MF']

df_cbf_tfp = df_tfp[df_tfp['superstructure_system'] == 'CBF']
df_cbf_lrb = df_lrb[df_lrb['superstructure_system'] == 'CBF']


df_mf_tfp_i = df_mf_tfp[df_mf_tfp['impacted'] == 1]
df_mf_tfp_o = df_mf_tfp[df_mf_tfp['impacted'] == 0]
df_mf_lrb_i = df_mf_lrb[df_mf_lrb['impacted'] == 1]
df_mf_lrb_o = df_mf_lrb[df_mf_lrb['impacted'] == 0]

df_cbf_tfp_i = df_cbf_tfp[df_cbf_tfp['impacted'] == 1]
df_cbf_tfp_o = df_cbf_tfp[df_cbf_tfp['impacted'] == 0]
df_cbf_lrb_i = df_cbf_lrb[df_cbf_lrb['impacted'] == 1]
df_cbf_lrb_o = df_cbf_lrb[df_cbf_lrb['impacted'] == 0]

#%% regression models: cost
# goal: E[cost|sys=sys, impact=impact]

mdl_cost_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_cost_cbf_lrb_o.set_covariates(covariate_list)
mdl_cost_cbf_lrb_o.set_outcome(cost_var)
mdl_cost_cbf_lrb_o.test_train_split(0.2)

mdl_cost_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_cost_cbf_tfp_o.set_covariates(covariate_list)
mdl_cost_cbf_tfp_o.set_outcome(cost_var)
mdl_cost_cbf_tfp_o.test_train_split(0.2)

mdl_cost_mf_lrb_o = GP(df_mf_lrb_o)
mdl_cost_mf_lrb_o.set_covariates(covariate_list)
mdl_cost_mf_lrb_o.set_outcome(cost_var)
mdl_cost_mf_lrb_o.test_train_split(0.2)

mdl_cost_mf_tfp_o = GP(df_mf_tfp_o)
mdl_cost_mf_tfp_o.set_covariates(covariate_list)
mdl_cost_mf_tfp_o.set_outcome(cost_var)
mdl_cost_mf_tfp_o.test_train_split(0.2)

print('======= outcome regression per system per impact ========')
import time
t0 = time.time()

mdl_cost_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

mdl_cost_cbf_lrb_o.fit_kernel_ridge(kernel_name='rbf')
mdl_cost_cbf_tfp_o.fit_kernel_ridge(kernel_name='rbf')

tp = time.time() - t0

print("GPR training for cost done for 4 models in %.3f s" % tp)

cost_regression_mdls = {'mdl_cost_cbf_lrb_o': mdl_cost_cbf_lrb_o,
                        'mdl_cost_cbf_tfp_o': mdl_cost_cbf_tfp_o,
                        'mdl_cost_mf_lrb_o': mdl_cost_mf_lrb_o,
                        'mdl_cost_mf_tfp_o': mdl_cost_mf_tfp_o}

#%% CBF-TFP and MF-TFP cost

# TODO: here
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
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 4.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_tfp_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_tfp_o.kr.predict(X_plot)

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

ax.scatter(df_cbf_tfp_o[xvar], df_cbf_tfp_o[yvar], df_cbf_tfp_o[cost_var], c=df_cbf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_tfp_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp_o[xvar], df_mf_tfp_o[yvar], df_mf_tfp_o[cost_var], c=df_mf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('TFPs no impact: Blue = CBF, Red = MF', fontsize=subt_font)

#################################


res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 4.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_tfp_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_tfp_o.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.2,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp_o[xvar], df_cbf_tfp_o[yvar], df_cbf_tfp_o[cost_var], c=df_cbf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_tfp_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp_o[xvar], df_mf_tfp_o[yvar], df_mf_tfp_o[cost_var], c=df_mf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.05])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('Same plot zoomed in', fontsize=subt_font)

fig.tight_layout()

#%% CBF-TFP and MF-TFP cost (secondary variables)

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
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_tfp_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_tfp_o.kr.predict(X_plot)

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

ax.scatter(df_cbf_tfp_o[xvar], df_cbf_tfp_o[yvar], df_cbf_tfp_o[cost_var], c=df_cbf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_tfp_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp_o[xvar], df_mf_tfp_o[yvar], df_mf_tfp_o[cost_var], c=df_mf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('TFPs no impact: Blue = CBF, Red = MF', fontsize=subt_font)

#################################


res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_tfp_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_tfp_o.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.2,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_tfp_o[xvar], df_cbf_tfp_o[yvar], df_cbf_tfp_o[cost_var], c=df_cbf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_tfp_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_tfp_o[xvar], df_mf_tfp_o[yvar], df_mf_tfp_o[cost_var], c=df_mf_tfp_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.05])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('Same plot zoomed in', fontsize=subt_font)

fig.tight_layout()

#%% CBF-LRB and MF-LRB cost

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
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 4.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_lrb_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_lrb_o.kr.predict(X_plot)

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

ax.scatter(df_cbf_lrb_o[xvar], df_cbf_lrb_o[yvar], df_cbf_lrb_o[cost_var], c=df_cbf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.1])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('LRBs no impact: Blue = CBF, Red = MF', fontsize=subt_font)

#################################


res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 4.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_lrb_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_lrb_o.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.2,
                       vmin=-0.2)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb_o[xvar], df_cbf_lrb_o[yvar], df_cbf_lrb_o[cost_var], c=df_cbf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.05])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('LRBs no impact: Blue = CBF, Red = MF', fontsize=subt_font)
ax.set_title('Same plot zoomed in', fontsize=subt_font)

fig.tight_layout()

#%% CBF-LRB and MF-LRB cost (secondary variables)

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
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_lrb_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_lrb_o.kr.predict(X_plot)

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

ax.scatter(df_cbf_lrb_o[xvar], df_cbf_lrb_o[yvar], df_cbf_lrb_o[cost_var], c=df_cbf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

ax.set_zlim([0.0, 0.3])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.1])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('LRBs no impact: Blue = CBF, Red = MF', fontsize=subt_font)

#################################


res = 75
X_plot = make_2D_plotting_space(df[covariate_list], res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = mdl_cost_cbf_lrb_o.gpr.predict(X_plot)
# Z = mdl_cost_cbf_lrb_o.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.2,
                       vmin=-0.2)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_cbf_lrb_o[xvar], df_cbf_lrb_o[yvar], df_cbf_lrb_o[cost_var], c=df_cbf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

########
Z = mdl_cost_mf_lrb_o.gpr.predict(X_plot)


x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

Z_surf = np.array(Z).reshape(xx_pl.shape)

surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Reds',
                       linewidth=0, antialiased=False, alpha=0.6,
                       vmin=-0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_mf_lrb_o[xvar], df_mf_lrb_o[yvar], df_mf_lrb_o[cost_var], c=df_mf_lrb_o[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Reds')

ax.set_zlim([0.0, 0.05])

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Reds_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Reds')
######

ax.set_xlabel('$T_M/T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Cost ratio', fontsize=axis_font)
ax.set_title('LRBs no impact: Blue = CBF, Red = MF', fontsize=subt_font)
ax.set_title('Same plot zoomed in', fontsize=subt_font)

fig.tight_layout()