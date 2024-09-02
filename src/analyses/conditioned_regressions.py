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
df['median_cost'] = df_loss['cost_50%']

# but working in parallel (2x faster)
df['replacement_time'] = df['bldg_area']/1000*365
df['total_cmp_time'] = df_loss_max['time_l_50%']
df['cmp_replace_time_ratio'] = df['total_cmp_time']/df['replacement_time']
df['median_time_ratio'] = df_loss['time_l_50%']/df['replacement_time']
df['cmp_time_ratio'] = df_loss['time_l_50%']/df['total_cmp_time']
df['median_time'] = df_loss['time_l_50%']

df['replacement_freq'] = df_loss['replacement_freq']
df['irreparable_freq'] = df['replacement_freq'] - df['collapse_prob']
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

df['replacement_binary'] = round(df['replacement_freq'])
#%% subsets

df_tfp = df[df['isolator_system'] == 'TFP']
df_lrb = df[df['isolator_system'] == 'LRB']

df_cbf = df[df['superstructure_system'] == 'CBF'].reset_index()
df_cbf['dummy_index'] = df_cbf['replacement_freq'] + df_cbf['index']*1e-9
df_mf = df[df['superstructure_system'] == 'MF'].reset_index()
df_mf['dummy_index'] = df_mf['replacement_freq'] + df_mf['index']*1e-9

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

#%% are outcomes different for each system?

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')
import seaborn as sns

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
sns.boxplot(y=cost_var, x= "system", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.6, ax=ax1)
sns.stripplot(x='system', y=cost_var, data=df, ax=ax1, jitter=True,
              alpha=0.3, color='steelblue')
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Repair cost ratio', fontsize=axis_font)
ax1.set_xlabel('System', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y=time_var, x= "system", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.6, ax=ax2)
sns.stripplot(x='system', y=time_var, data=df, ax=ax2, jitter=True,
              alpha=0.3, color='steelblue')
ax2.set_title('Median sequential repair time', fontsize=subt_font)
ax2.set_ylabel('Repair time ratio', fontsize=axis_font)
ax2.set_xlabel('System', fontsize=axis_font)
ax2.set_yscale('log')


# ax3.set_yscale('log')
sns.boxplot(y="replacement_freq", x= "system", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.5, ax=ax3)
sns.stripplot(x='system', y='replacement_freq', data=df, ax=ax3, jitter=True,
              alpha=0.3, color='steelblue')
ax3.set_title('Replacement frequency', fontsize=subt_font)
ax3.set_ylabel('Replacement frequency', fontsize=axis_font)
ax3.set_xlabel('System', fontsize=axis_font)
fig.tight_layout()
plt.show()


#%%

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(16, 7))

# Plot the total crashes
sns.barplot(x="max_drift", y="replacement_freq", data=df_cbf,
            label="PID-related collapse", color="lightsteelblue")

# Plot the crashes where alcohol was involved
sns.barplot(x="max_drift", y="irreparable_freq", data=df_cbf,
            label="RID-related irreparable", color="navy")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([])

ax.legend()
ax.set_title('CBF replacement', fontsize=subt_font)
ax.set_xlabel('Increasing max transient drift \u2192', fontsize=axis_font)
ax.set_ylabel('\% replacement', fontsize=axis_font)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(16, 7))

# Plot the total crashes
sns.barplot(x="max_drift", y="replacement_freq", data=df_mf,
            label="PID-related collapse", color="lightsteelblue")

# Plot the crashes where alcohol was involved
sns.barplot(x="max_drift", y="irreparable_freq", data=df_mf,
            label="RID-related irreparable", color="navy")
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([])

ax.legend()
ax.set_title('MF replacement', fontsize=subt_font)
ax.set_xlabel('Increasing max transient drift \u2192', fontsize=axis_font)
ax.set_ylabel('\% replacement', fontsize=axis_font)

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

# plt.close('all')
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



#%% system classification model 
# goal: Pr[sys|X]
# consider: replacement freq, num_stories, num_bays, repair cost

print('======= system classification ========')
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
clf_sys = GP(df)
clf_sys.set_covariates(covariate_list)
clf_sys.set_outcome('system', use_ravel=False)
clf_sys.test_train_split(0.2)

clf_sys.fit_gpc(kernel_name='rbf_iso')
y_pred = clf_sys.gpc.predict(clf_sys.X_test)
y_prob = clf_sys.gpc.predict_proba(clf_sys.X_test)
y_test = clf_sys.y_test

# output is alphabetical? CBF-LRB, CBF-TFP, MF-LRB, MF-TFP

#%% impact classification model
# for each system, make separate impact classification model
mdl_impact_cbf_lrb = GP(df_cbf_lrb)
mdl_impact_cbf_lrb.set_covariates(covariate_list)
mdl_impact_cbf_lrb.set_outcome('impacted')
mdl_impact_cbf_lrb.test_train_split(0.2)

mdl_impact_cbf_tfp = GP(df_cbf_tfp)
mdl_impact_cbf_tfp.set_covariates(covariate_list)
mdl_impact_cbf_tfp.set_outcome('impacted')
mdl_impact_cbf_tfp.test_train_split(0.2)

mdl_impact_mf_lrb = GP(df_mf_lrb)
mdl_impact_mf_lrb.set_covariates(covariate_list)
mdl_impact_mf_lrb.set_outcome('impacted')
mdl_impact_mf_lrb.test_train_split(0.2)

mdl_impact_mf_tfp = GP(df_mf_tfp)
mdl_impact_mf_tfp.set_covariates(covariate_list)
mdl_impact_mf_tfp.set_outcome('impacted')
mdl_impact_mf_tfp.test_train_split(0.2)

print('======= impact classification per system ========')
import time
t0 = time.time()

mdl_impact_cbf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_impact_cbf_tfp.fit_gpc(kernel_name='rbf_iso')
mdl_impact_mf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_impact_mf_tfp.fit_gpc(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPC training for impact done for 4 models in %.3f s" % tp)

impact_classification_mdls = {'mdl_impact_cbf_lrb': mdl_impact_cbf_lrb,
                        'mdl_impact_cbf_tfp': mdl_impact_cbf_tfp,
                        'mdl_impact_mf_lrb': mdl_impact_mf_lrb,
                        'mdl_impact_mf_tfp': mdl_impact_mf_tfp,}

#%% replacement classification model

mdl_repl_clf_cbf_lrb = GP(df_cbf_lrb)
mdl_repl_clf_cbf_lrb.set_covariates(covariate_list)
mdl_repl_clf_cbf_lrb.set_outcome('replacement_binary')
mdl_repl_clf_cbf_lrb.test_train_split(0.2)

mdl_repl_clf_cbf_tfp = GP(df_cbf_tfp)
mdl_repl_clf_cbf_tfp.set_covariates(covariate_list)
mdl_repl_clf_cbf_tfp.set_outcome('replacement_binary')
mdl_repl_clf_cbf_tfp.test_train_split(0.2)

mdl_repl_clf_mf_lrb = GP(df_mf_lrb)
mdl_repl_clf_mf_lrb.set_covariates(covariate_list)
mdl_repl_clf_mf_lrb.set_outcome('replacement_binary')
mdl_repl_clf_mf_lrb.test_train_split(0.2)

mdl_repl_clf_mf_tfp = GP(df_mf_tfp)
mdl_repl_clf_mf_tfp.set_covariates(covariate_list)
mdl_repl_clf_mf_tfp.set_outcome('replacement_binary')
mdl_repl_clf_mf_tfp.test_train_split(0.2)

print('======= replacement classification per system ========')
import time
t0 = time.time()

mdl_repl_clf_cbf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_repl_clf_cbf_tfp.fit_gpc(kernel_name='rbf_iso')
mdl_repl_clf_mf_lrb.fit_gpc(kernel_name='rbf_iso')
mdl_repl_clf_mf_tfp.fit_gpc(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPC training for impact done for 4 models in %.3f s" % tp)

impact_classification_mdls = {'mdl_repl_clf_cbf_lrb': mdl_repl_clf_cbf_lrb,
                        'mdl_repl_clf_cbf_tfp': mdl_repl_clf_cbf_tfp,
                        'mdl_repl_clf_mf_lrb': mdl_repl_clf_mf_lrb,
                        'mdl_repl_clf_mf_tfp': mdl_repl_clf_mf_tfp}

#%% regression models: cost
# goal: E[cost|sys=sys, impact=impact]

mdl_cost_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_cost_cbf_lrb_i.set_covariates(covariate_list)
mdl_cost_cbf_lrb_i.set_outcome(cost_var)
mdl_cost_cbf_lrb_i.test_train_split(0.2)

mdl_cost_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_cost_cbf_lrb_o.set_covariates(covariate_list)
mdl_cost_cbf_lrb_o.set_outcome(cost_var)
mdl_cost_cbf_lrb_o.test_train_split(0.2)

mdl_cost_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_cost_cbf_tfp_i.set_covariates(covariate_list)
mdl_cost_cbf_tfp_i.set_outcome(cost_var)
mdl_cost_cbf_tfp_i.test_train_split(0.2)

mdl_cost_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_cost_cbf_tfp_o.set_covariates(covariate_list)
mdl_cost_cbf_tfp_o.set_outcome(cost_var)
mdl_cost_cbf_tfp_o.test_train_split(0.2)

mdl_cost_mf_lrb_i = GP(df_mf_lrb_i)
mdl_cost_mf_lrb_i.set_covariates(covariate_list)
mdl_cost_mf_lrb_i.set_outcome(cost_var)
mdl_cost_mf_lrb_i.test_train_split(0.2)

mdl_cost_mf_lrb_o = GP(df_mf_lrb_o)
mdl_cost_mf_lrb_o.set_covariates(covariate_list)
mdl_cost_mf_lrb_o.set_outcome(cost_var)
mdl_cost_mf_lrb_o.test_train_split(0.2)

mdl_cost_mf_tfp_i = GP(df_mf_tfp_i)
mdl_cost_mf_tfp_i.set_covariates(covariate_list)
mdl_cost_mf_tfp_i.set_outcome(cost_var)
mdl_cost_mf_tfp_i.test_train_split(0.2)

mdl_cost_mf_tfp_o = GP(df_mf_tfp_o)
mdl_cost_mf_tfp_o.set_covariates(covariate_list)
mdl_cost_mf_tfp_o.set_outcome(cost_var)
mdl_cost_mf_tfp_o.test_train_split(0.2)

print('======= outcome regression per system per impact ========')
import time
t0 = time.time()

mdl_cost_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_cost_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for cost done for 8 models in %.3f s" % tp)

cost_regression_mdls = {'mdl_cost_cbf_lrb_i': mdl_cost_cbf_lrb_i,
                        'mdl_cost_cbf_lrb_o': mdl_cost_cbf_lrb_o,
                        'mdl_cost_cbf_tfp_i': mdl_cost_cbf_tfp_i,
                        'mdl_cost_cbf_tfp_o': mdl_cost_cbf_tfp_o,
                        'mdl_cost_mf_lrb_i': mdl_cost_mf_lrb_i,
                        'mdl_cost_mf_lrb_o': mdl_cost_mf_lrb_o,
                        'mdl_cost_mf_tfp_i': mdl_cost_mf_tfp_i,
                        'mdl_cost_mf_tfp_o': mdl_cost_mf_tfp_o}

#%% regression models: time
# goal: E[time|sys=sys, impact=impact]

mdl_time_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_time_cbf_lrb_i.set_covariates(covariate_list)
mdl_time_cbf_lrb_i.set_outcome(time_var)
mdl_time_cbf_lrb_i.test_train_split(0.2)

mdl_time_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_time_cbf_lrb_o.set_covariates(covariate_list)
mdl_time_cbf_lrb_o.set_outcome(time_var)
mdl_time_cbf_lrb_o.test_train_split(0.2)

mdl_time_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_time_cbf_tfp_i.set_covariates(covariate_list)
mdl_time_cbf_tfp_i.set_outcome(time_var)
mdl_time_cbf_tfp_i.test_train_split(0.2)

mdl_time_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_time_cbf_tfp_o.set_covariates(covariate_list)
mdl_time_cbf_tfp_o.set_outcome(time_var)
mdl_time_cbf_tfp_o.test_train_split(0.2)

mdl_time_mf_lrb_i = GP(df_mf_lrb_i)
mdl_time_mf_lrb_i.set_covariates(covariate_list)
mdl_time_mf_lrb_i.set_outcome(time_var)
mdl_time_mf_lrb_i.test_train_split(0.2)

mdl_time_mf_lrb_o = GP(df_mf_lrb_o)
mdl_time_mf_lrb_o.set_covariates(covariate_list)
mdl_time_mf_lrb_o.set_outcome(time_var)
mdl_time_mf_lrb_o.test_train_split(0.2)

mdl_time_mf_tfp_i = GP(df_mf_tfp_i)
mdl_time_mf_tfp_i.set_covariates(covariate_list)
mdl_time_mf_tfp_i.set_outcome(time_var)
mdl_time_mf_tfp_i.test_train_split(0.2)

mdl_time_mf_tfp_o = GP(df_mf_tfp_o)
mdl_time_mf_tfp_o.set_covariates(covariate_list)
mdl_time_mf_tfp_o.set_outcome(time_var)
mdl_time_mf_tfp_o.test_train_split(0.2)

print('======= outcome regression per system per impact ========')
import time
t0 = time.time()

mdl_time_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_time_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for time done for 8 models in %.3f s" % tp)

time_regression_mdls = {'mdl_time_cbf_lrb_i': mdl_time_cbf_lrb_i,
                        'mdl_time_cbf_lrb_o': mdl_time_cbf_lrb_o,
                        'mdl_time_cbf_tfp_i': mdl_time_cbf_tfp_i,
                        'mdl_time_cbf_tfp_o': mdl_time_cbf_tfp_o,
                        'mdl_time_mf_lrb_i': mdl_time_mf_lrb_i,
                        'mdl_time_mf_lrb_o': mdl_time_mf_lrb_o,
                        'mdl_time_mf_tfp_i': mdl_time_mf_tfp_i,
                        'mdl_time_mf_tfp_o': mdl_time_mf_tfp_o}

#%% regression models: repl
# goal: E[repl|sys=sys, impact=impact]

mdl_repl_cbf_lrb_i = GP(df_cbf_lrb_i)
mdl_repl_cbf_lrb_i.set_covariates(covariate_list)
mdl_repl_cbf_lrb_i.set_outcome('replacement_freq')
mdl_repl_cbf_lrb_i.test_train_split(0.2)

mdl_repl_cbf_lrb_o = GP(df_cbf_lrb_o)
mdl_repl_cbf_lrb_o.set_covariates(covariate_list)
mdl_repl_cbf_lrb_o.set_outcome('replacement_freq')
mdl_repl_cbf_lrb_o.test_train_split(0.2)

mdl_repl_cbf_tfp_i = GP(df_cbf_tfp_i)
mdl_repl_cbf_tfp_i.set_covariates(covariate_list)
mdl_repl_cbf_tfp_i.set_outcome('replacement_freq')
mdl_repl_cbf_tfp_i.test_train_split(0.2)

mdl_repl_cbf_tfp_o = GP(df_cbf_tfp_o)
mdl_repl_cbf_tfp_o.set_covariates(covariate_list)
mdl_repl_cbf_tfp_o.set_outcome('replacement_freq')
mdl_repl_cbf_tfp_o.test_train_split(0.2)

mdl_repl_mf_lrb_i = GP(df_mf_lrb_i)
mdl_repl_mf_lrb_i.set_covariates(covariate_list)
mdl_repl_mf_lrb_i.set_outcome('replacement_freq')
mdl_repl_mf_lrb_i.test_train_split(0.2)

mdl_repl_mf_lrb_o = GP(df_mf_lrb_o)
mdl_repl_mf_lrb_o.set_covariates(covariate_list)
mdl_repl_mf_lrb_o.set_outcome('replacement_freq')
mdl_repl_mf_lrb_o.test_train_split(0.2)

mdl_repl_mf_tfp_i = GP(df_mf_tfp_i)
mdl_repl_mf_tfp_i.set_covariates(covariate_list)
mdl_repl_mf_tfp_i.set_outcome('replacement_freq')
mdl_repl_mf_tfp_i.test_train_split(0.2)

mdl_repl_mf_tfp_o = GP(df_mf_tfp_o)
mdl_repl_mf_tfp_o.set_covariates(covariate_list)
mdl_repl_mf_tfp_o.set_outcome('replacement_freq')
mdl_repl_mf_tfp_o.test_train_split(0.2)

t0 = time.time()

print('======= outcome regression per system per impact ========')

mdl_repl_cbf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_cbf_tfp_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_lrb_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_lrb_o.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_tfp_i.fit_gpr(kernel_name='rbf_iso')
mdl_repl_mf_tfp_o.fit_gpr(kernel_name='rbf_iso')

tp = time.time() - t0

print("GPR training for replacement done for 8 models in %.3f s" % tp)

repl_regression_mdls = {'mdl_repl_cbf_lrb_i': mdl_repl_cbf_lrb_i,
                        'mdl_repl_cbf_lrb_o': mdl_repl_cbf_lrb_o,
                        'mdl_repl_cbf_tfp_i': mdl_repl_cbf_tfp_i,
                        'mdl_repl_cbf_tfp_o': mdl_repl_cbf_tfp_o,
                        'mdl_repl_mf_lrb_i': mdl_repl_mf_lrb_i,
                        'mdl_repl_mf_lrb_o': mdl_repl_mf_lrb_o,
                        'mdl_repl_mf_tfp_i': mdl_repl_mf_tfp_i,
                        'mdl_repl_mf_tfp_o': mdl_repl_mf_tfp_o}

#%% compare using the two outcomes
print('============= mean squared error of regressions using impact =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - cost
mdl_dummy = GP(df_mf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-TFP, cost:', mse)

# MF LRB - cost
mdl_dummy = GP(df_mf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-LRB, cost', mse)

# CBF TFP - cost
mdl_dummy = GP(df_cbf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-TFP, cost', mse)

# CBF LRB - cost
mdl_dummy = GP(df_cbf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-LRB, cost', mse)

# MF TFP - time
mdl_dummy = GP(df_mf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_mf_tfp.gpc, mdl_time_mf_tfp_i.gpr, mdl_time_mf_tfp_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-TFP, time:', mse)

# MF LRB - time
mdl_dummy = GP(df_mf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_mf_lrb.gpc, mdl_time_mf_lrb_i.gpr, mdl_time_mf_lrb_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-LRB, time', mse)

# CBF TFP - time
mdl_dummy = GP(df_cbf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_cbf_tfp.gpc, mdl_time_cbf_tfp_i.gpr, mdl_time_cbf_tfp_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-TFP, time', mse)

# CBF LRB - time
mdl_dummy = GP(df_cbf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_impact_cbf_lrb.gpc, mdl_time_cbf_lrb_i.gpr, mdl_time_cbf_lrb_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-LRB, time', mse)

print('============= mean squared error of regressions using replacement =======================')
from sklearn.metrics import mean_squared_error
# MF TFP - cost
mdl_dummy = GP(df_mf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-TFP, cost:', mse)

# MF LRB - cost
mdl_dummy = GP(df_mf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-LRB, cost', mse)

# CBF TFP - cost
mdl_dummy = GP(df_cbf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-TFP, cost', mse)

# CBF LRB - cost
mdl_dummy = GP(df_cbf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(cost_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-LRB, cost', mse)

# MF TFP - time
mdl_dummy = GP(df_mf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_mf_tfp.gpc, mdl_time_mf_tfp_i.gpr, mdl_time_mf_tfp_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-TFP, time:', mse)

# MF LRB - time
mdl_dummy = GP(df_mf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_mf_lrb.gpc, mdl_time_mf_lrb_i.gpr, mdl_time_mf_lrb_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('MF-LRB, time', mse)

# CBF TFP - time
mdl_dummy = GP(df_cbf_tfp)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_cbf_tfp.gpc, mdl_time_cbf_tfp_i.gpr, mdl_time_cbf_tfp_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-TFP, time', mse)

# CBF LRB - time
mdl_dummy = GP(df_cbf_lrb)
mdl_dummy.set_covariates(covariate_list)
mdl_dummy.set_outcome(time_var)
mdl_dummy.test_train_split(0.2)
y_pred = predict_DV(mdl_dummy.X_test, mdl_repl_clf_cbf_lrb.gpc, mdl_time_cbf_lrb_i.gpr, mdl_time_cbf_lrb_o.gpr,
                    outcome=time_var, return_var=False)
mse = mean_squared_error(mdl_dummy.y_test, y_pred)
print('CBF-LRB, time', mse)

#%% deprecated model for conditioning on system clf
'''

def predict_outcome(X, system_clf_mdl, impact_clf_mdls, var_regr_mdls,
               outcome='cost_50%', return_var=False):
    """Returns the expected value of the decision variable based on the total
    probability law (law of iterated expectation).
    
    E[cost] = sum_i sum_j E[cost|impact_j] Pr(impact_j) 
    
    Currently, this assumes that the models used are all GPC/GPR.
    
    Parameters
    ----------
    X: pd dataframe of design points
    system_clf_mdl: singular classification model for system selection (probabilistic)
    impact_clf_mdls: one impact classification model per system combination
        name should be 'mdl_impact_'+system_combination
    var_regr_mdls: one regression model per system per impact status
        name should be 'mdl_'+outcome+system_combination+impact (i or o)
    outcome: desired name for outcome variable
    
    Returns
    -------
    expected_DV_df: DataFrame of expected DV with single column name outcome+'_pred'
    """
    
    # predict system selection
    sys_names = ['cbf_lrb', 'cbf_tfp', 'mf_lrb', 'mf_tfp']
    sys_pred = pd.DataFrame(system_clf_mdl.gpc.predict_proba(X), columns=sys_names)
    
    # get name of the regression models, which are the innermost iterated expectaiton
    regr_names = list(var_regr_mdls.keys())
    expected_DV = None
    
    for mdl_name in regr_names:
        
        # E[cost|sys=sys, impact=impact]
        var_pred = var_regr_mdls[mdl_name].gpr.predict(X).ravel()
        
        # get the system name (nested within var_regr_mdls)
        system_var = mdl_name.split('_')[2]+'_'+mdl_name.split('_')[3]
        
        # get the impact status (nested within var_regr_mdls)
        impact_var = mdl_name.split('_')[-1]
        impact_mdl_name = 'mdl_impact_' + system_var
        
        # predict classification either using hit or miss model
        # Pr[impact|sys=sys]
        if impact_var == 'i':
            impact_pred = impact_clf_mdls[impact_mdl_name].gpc.predict_proba(X)[:,1]
        else:
            impact_pred = impact_clf_mdls[impact_mdl_name].gpc.predict_proba(X)[:,0]
            
        # multiply and sum
        if expected_DV is None:
            expected_DV = np.multiply.reduce(
                (var_pred, impact_pred, sys_pred[system_var]))
        else:
            expected_DV += np.multiply.reduce(
                (var_pred, impact_pred, sys_pred[system_var]))
    
    outcome_str = outcome+'_pred'
    expected_DV_df = pd.DataFrame({outcome_str:expected_DV})
    
    if return_var:
        pass
    else:
        return expected_DV_df
'''
  
#%% sample for CBF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_cbf_lrb.gpc, mdl_cost_cbf_lrb_i.gpr, mdl_cost_cbf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_cbf_lrb[xvar], df_cbf_lrb[yvar], df_cbf_lrb[cost_var], c=df_cbf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample for CBF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_cbf_tfp.gpc, mdl_cost_cbf_tfp_i.gpr, mdl_cost_cbf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_cbf_tfp[xvar], df_cbf_tfp[yvar], df_cbf_tfp[cost_var], c=df_cbf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('CBF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample for MF-LRB

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_mf_lrb.gpc, mdl_cost_mf_lrb_i.gpr, mdl_cost_mf_lrb_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_mf_lrb[xvar], df_mf_lrb[yvar], df_mf_lrb[cost_var], c=df_mf_lrb[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-LRB: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% sample for MF-TFP

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_DV(X_plot, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_DV(X_plot, mdl_impact_mf_tfp.gpc, mdl_cost_mf_tfp_i.gpr, mdl_cost_mf_tfp_o.gpr,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
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

ax.scatter(df_mf_tfp[xvar], df_mf_tfp[yvar], df_mf_tfp[cost_var], c=df_mf_tfp[cost_var],
           edgecolors='k', alpha = 0.7, cmap='Blues')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')


ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('MF-TFP: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

#%% 3d surf for cost ratio - mega regression
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(16, 7))

xvar = 'gap_ratio'
yvar = 'RI'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 2.0, fourth_var_set = 0.15)

Z = predict_outcome(X_plot, clf_sys, impact_clf_mdls, cost_regression_mdls,
                    outcome=cost_var, return_var=False)


xx = X_plot[xvar]
yy = X_plot[yvar]
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
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 75
X_plot = make_2D_plotting_space(clf_sys.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 2.0)

Z = predict_outcome(X_plot, clf_sys, impact_clf_mdls, cost_regression_mdls,
                    outcome=cost_var, return_var=False)

xx = X_plot[xvar]
yy = X_plot[yvar]
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
ax.set_zlabel('Repair cost ratio', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

'''