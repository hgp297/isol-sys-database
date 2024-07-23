############################################################################
#               Multiple conditioned data

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

main_obj = pd.read_pickle("../../data/loss/structural_db_parallel_loss.pickle")

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

max_obj = pd.read_pickle("../../data/loss/structural_db_parallel_max_loss.pickle")
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

df['system'] = df['superstructure_system'] +'-' + df['isolator_system']
#%% subsets

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']

df_mf_tfp = df_tfp[df_tfp['superstructure_system'] == 'MF']
df_mf_lrb = df_lrb[df_lrb['superstructure_system'] == 'MF']

df_cbf_tfp = df_tfp[df_tfp['superstructure_system'] == 'CBF']
df_cbf_lrb = df_lrb[df_lrb['superstructure_system'] == 'CBF']

#%%
cmap = plt.cm.tab20

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
fig = plt.figure(figsize=(13, 8))

ax = fig.add_subplot(1, 2, 1)
xvar = 'gap_ratio'
yvar  = 'RI'

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], alpha=0.5, marker='^', color=cmap(0), label='MF-TFP')
ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], alpha=0.5, marker='o', color=cmap(1), label='MF-LRB')

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], alpha=0.5, marker='^', color=cmap(2), label='CBF-TFP')
ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], alpha=0.5, marker='o', color=cmap(3), label='CBF-LRB')

ax.set_xlabel("$GR$", fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.legend()

ax = fig.add_subplot(1, 2, 2)
xvar = 'T_ratio'
yvar  = 'zeta_e'

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], alpha=0.5, marker='^', color=cmap(0), label='MF-TFP')
ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], alpha=0.5, marker='o', color=cmap(1), label='MF-LRB')

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], alpha=0.5, marker='^', color=cmap(2), label='CBF-TFP')
ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], alpha=0.5, marker='o', color=cmap(3), label='CBF-LRB')

ax.set_xlabel("$T_M/T_{fb}$", fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)

#%%
'''
#%% subsets

df_miss = df[df['impacted'] == 0]

# remove the singular outlier point
from scipy import stats
df_no_impact = df_miss[np.abs(stats.zscore(df_miss['cmp_cost_ratio'])) < 5].copy()

df_tfp_miss = df_no_impact[df_no_impact['isolator_system'] == 'TFP']
df_lrb_miss = df_no_impact[df_no_impact['isolator_system'] == 'LRB']
df_cbf_miss = df_no_impact[df_no_impact['superstructure_system'] == 'CBF']
df_mf_miss = df_no_impact[df_no_impact['superstructure_system'] == 'MF']

df_hit = df[df['impacted'] == 1]

df_tfp_hit = df_hit[df_hit['isolator_system'] == 'TFP']
df_lrb_hit = df_hit[df_hit['isolator_system'] == 'LRB']
df_cbf_hit = df_hit[df_hit['superstructure_system'] == 'CBF']
df_mf_hit = df_hit[df_hit['superstructure_system'] == 'MF']

#%% system selector
# consider: replacement freq, num_stories, num_bays, repair cost
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
clf_struct = GP(df)

clf_struct.set_covariates(covariate_list)
clf_struct.set_outcome('superstructure_system', use_ravel=False)
clf_struct.test_train_split(0.2)
clf_struct.fit_gpc(kernel_name='rbf_iso')

#%% fit regressions for impact / non-impact set
mdl_cost_mf = GP(df_mf_hit)
mdl_cost_mf.set_covariates(covariate_list)
mdl_cost_mf.set_outcome(cost_var)
mdl_cost_mf.test_train_split(0.2)

mdl_cost_cbf = GP(df_cbf_hit)
mdl_cost_cbf.set_covariates(covariate_list)
mdl_cost_cbf.set_outcome(cost_var)
mdl_cost_cbf.test_train_split(0.2)

print('========== Fitting regressions (GPR) ============')

# Fit conditioned DVs using GPR

# fit impacted set
mdl_cost_mf.fit_gpr(kernel_name='rbf_iso')

# fit no impact set
mdl_cost_cbf.fit_gpr(kernel_name='rbf_iso')

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
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]


Z = predict_DV(X_plot, clf_struct.gpc, 
               mdl_cost_mf.gpr, mdl_cost_cbf.gpr, outcome='replacement_freq')


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

# ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
#            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.1])

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
X_plot = make_2D_plotting_space(clf_struct.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=covariate_list,
                            third_var_set = 1.0, fourth_var_set = 2.0)
xx = X_plot[xvar]
yy = X_plot[yvar]

Z = predict_DV(X_plot, clf_struct.gpc, 
               mdl_cost_mf.gpr, mdl_cost_cbf.gpr, outcome='replacement_freq')

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

# ax.scatter(df_miss[xvar], df_miss[yvar], df_miss[cost_var], c=df_miss[cost_var],
#            edgecolors='k', alpha = 0.7, cmap='Blues')

# ax.set_zlim([0.0, 0.1])
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
'''