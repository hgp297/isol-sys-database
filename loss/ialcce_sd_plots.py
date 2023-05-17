import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pred import Prediction, predict_DV
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import warnings
warnings.filterwarnings('ignore')

## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

from scipy.stats import lognorm
from math import log, exp
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)


#%%

database_path = './data/tfp_mf/'
database_file = 'run_data.csv'

results_path = './results/tfp_mf/'
results_file = 'loss_estimate_data.csv'

val_dir = './data/tfp_mf_val/'
val_dir_loss = './results/tfp_mf_val/validation/'
val_file = 'addl_TFP_val.csv'

baseline_dir = './data/tfp_mf_val/'
baseline_dir_loss = './results/tfp_mf_val/baseline/'
baseline_file = 'addl_TFP_baseline.csv'

val_loss = pd.read_csv(val_dir_loss+'loss_estimate_data.csv', index_col=None)
base_loss = pd.read_csv(baseline_dir_loss+'loss_estimate_data.csv', index_col=None)

val_run = pd.read_csv(val_dir+val_file, index_col=None)
base_run = pd.read_csv(baseline_dir+baseline_file, index_col=None)

loss_data = pd.read_csv(results_path+results_file, 
                        index_col=None)
full_isolation_data = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df['collapse_probs'] = ln_dist.cdf(np.array(df['max_drift']))

df_val = pd.concat([val_run, val_loss], axis=1)
df_val['max_drift'] = df_val[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_val['collapse_probs'] = ln_dist.cdf(np.array(df_val['max_drift']))

df_base = pd.concat([base_run, base_loss], axis=1)
df_base['max_drift'] = df_base[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_base['collapse_probs'] = ln_dist.cdf(np.array(df_base['max_drift']))

cost_var = 'cost_50%'
time_var = 'time_u_50%'

#%% engineering data
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

bins = pd.IntervalIndex.from_tuples([(0, 0.66), (0.66, 1.0), (1.0, 1.5), (1.5, 4.0)])
labels=['tiny', 'small', 'okay', 'large']
df['gap_bin'] = pd.cut(df['gapRatio'], bins=bins, labels=labels)
df.groupby(['gap_bin']).size()
df_count = df.groupby('gap_bin')['max_drift'].apply(lambda x: (x>=0.10).sum()).reset_index(name='count')

plt.close('all')
fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
import seaborn as sns
sns.stripplot(data=df, x="max_drift", y="gap_bin", orient="h",
            ax=ax1)
sns.boxplot(y="gap_bin", x= "max_drift", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax1)
ax1.set_ylabel('Gap ratio range', fontsize=axis_font)
ax1.set_xlabel('Peak interstory drift', fontsize=axis_font)
plt.xlim([0.0, 0.10])
fig.tight_layout()

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
sns.stripplot(x='impacted', y=cost_var, data=df, ax=ax1, jitter=True)
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Cost [USD]', fontsize=axis_font)
ax1.set_xlabel('Impact', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y=time_var, x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.6, ax=ax2)
sns.stripplot(x='impacted', y=time_var, data=df, ax=ax2, jitter=True)
ax2.set_title('Median sequential repair time', fontsize=subt_font)
ax2.set_ylabel('Time [worker-day]', fontsize=axis_font)
ax2.set_xlabel('Impact', fontsize=axis_font)
ax2.set_yscale('log')

sns.boxplot(y="replacement_freq", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'},
            width=0.5, ax=ax3)
sns.stripplot(x='impacted', y='replacement_freq', data=df, ax=ax3, jitter=True)
ax3.set_title('Replacement frequency', fontsize=subt_font)
ax3.set_ylabel('Replacement frequency', fontsize=axis_font)
ax3.set_xlabel('Impact', fontsize=axis_font)
fig.tight_layout()

#%% ml training

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_hit = Prediction(df_hit)
mdl_hit.set_outcome(cost_var)
mdl_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_miss = Prediction(df_miss)
mdl_miss.set_outcome(cost_var)
mdl_miss.test_train_split(0.2)

hit = Prediction(df_hit)
hit.set_outcome('impacted')
hit.test_train_split(0.2)

miss = Prediction(df_miss)
miss.set_outcome('impacted')
miss.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_miss = Prediction(df_miss)
mdl_miss.set_outcome(cost_var)
mdl_miss.test_train_split(0.2)

mdl_time_hit = Prediction(df_hit)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = Prediction(df_miss)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

mdl_drift_hit = Prediction(df_hit)
mdl_drift_hit.set_outcome('max_drift')
mdl_drift_hit.test_train_split(0.2)

mdl_drift_miss = Prediction(df_miss)
mdl_drift_miss.set_outcome('max_drift')
mdl_drift_miss.test_train_split(0.2)

#%% fit impact (gp classification)

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('impacted')
mdl.test_train_split(0.2)

mdl.fit_gpc(kernel_name='rbf_iso')

# predict the entire dataset
preds_imp = mdl.gpc.predict(mdl.X)
probs_imp = mdl.gpc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

#%% Classification plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
# make grid and plot classification predictions

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
plt.setp((ax1, ax2, ax3), xticks=np.arange(0.5, 4.0, step=0.5))

xvar = 'gapRatio'
yvar = 'RI'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 50
cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax1.clabel(cs, fontsize=label_size)

# sc = ax3.scatter(mdl.X_train[xvar][:plt_density],
#             mdl.X_train[yvar][:plt_density],
#             s=30, c=mdl.y_train[:plt_density],
#             cmap=plt.cm.copper, edgecolors='w')

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax1.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='darkblue', marker='v', edgecolors='k', label='Impacted')

ax1.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='azure', edgecolors='k', label='No impact')

ax1.set_xlim(0.3, 2.5)
ax1.set_title(r'$T_M = 3.25$ s, $\zeta_M = 0.15$', fontsize=subt_font)
ax1.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)

####################################################################
xvar = 'gapRatio'
yvar = 'Tm'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 50
cs = ax2.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax2.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax2.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='darkblue', marker='v', edgecolors='k', label='Impacted')

ax2.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='azure', edgecolors='k', label='No impact')

ax2.set_xlim(0.3, 2.5)
ax2.set_title(r'$R_y= 1.25$ , $\zeta_M = 0.15$', fontsize=subt_font)
ax2.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax2.set_ylabel(r'$T_M$', fontsize=axis_font)

####################################################################
xvar = 'gapRatio'
yvar = 'zetaM'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 50
cs = ax3.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax3.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax3.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='darkblue', marker='v', edgecolors='k', label='Impacted')

ax3.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='azure', edgecolors='k', label='No impact')

# sc = ax3.scatter(mdl.X_train[xvar][:plt_density],
#             mdl.X_train[yvar][:plt_density],
#             s=30, c=mdl.y_train[:plt_density],
#             cmap=plt.cm.copper, edgecolors='w')

ax3.set_xlim(0.3, 2.5)
ax3.set_title(r'$R_y= 1.25$ , $T_M = 3.25$ s', fontsize=subt_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax3.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

ax3.legend(loc="lower right", fontsize=subt_font)

# lg = ax3.legend(*sc.legend_elements(), loc="lower right", title="Impact",
#            fontsize=subt_font)


# lg.get_title().set_fontsize(axis_font) #legend 'Title' fontsize

fig.tight_layout()
plt.show()

#%% regression models

# Fit costs (SVR)

# fit impacted set
mdl_hit.fit_svr()
mdl_hit.fit_kernel_ridge(kernel_name='rbf')

mdl_time_hit.fit_svr()
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')

mdl_drift_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_hit.fit_ols_ridge()

# fit no impact set
mdl_miss.fit_svr()
mdl_miss.fit_kernel_ridge(kernel_name='rbf')

mdl_time_miss.fit_svr()
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')

mdl_drift_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_miss.fit_ols_ridge()

#%% 3d surf
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.close('all')

fig = plt.figure(figsize=(13, 8))

#plt.setp((ax1, ax2), xticks=np.arange(0.5, 4.0, step=0.5),
#        yticks=np.arange(0.5, 2.5, step=0.5))


#################################
xvar = 'gapRatio'
yvar = 'RI'

res = 100
step = 0.01
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

ax1=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(xx, yy, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)

ax1.scatter(df[xvar], df[yvar], df[time_var]/4764.71, color='white',
           edgecolors='k', alpha = 0.7)

xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()
cset = ax1.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax1.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax1.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax1.set_title('a) Downtime: GPC-KR', fontsize=subt_font)

#################################
xvar = 'gapRatio'
yvar = 'RI'

res = 100
step = 0.01
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.kr,
                                     mdl_miss.kr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

ax2=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(xx, yy, Z, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.7,
                       vmin=-0.1)

ax2.scatter(df[xvar], df[yvar], df[cost_var]/8.1e6, color='white',
           edgecolors='k', alpha = 0.7)

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
zlim = ax2.get_zlim()
cset = ax2.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap='Blues')
cset = ax2.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap='Blues')
cset = ax2.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap='Blues')

ax2.set_xlabel('Gap ratio', fontsize=axis_font)
ax2.set_ylabel('$R_y$', fontsize=axis_font)
ax2.set_zlabel('% of replacement cost', fontsize=axis_font)
ax2.set_title('b) Cost: GPC-KR', fontsize=subt_font)

fig.tight_layout()

#%% read out results

design_repair_cost = df_val['cost_mean'].mean()
design_repair_cost_med = df_val['cost_50%'].mean()
design_downtime = df_val['time_u_mean'].mean()
design_downtime_med = df_val['time_u_50%'].mean()
design_collapse_risk = df_val['collapse_freq'].mean()
design_replacement_risk = df_val['replacement_freq'].mean()

print('====== INVERSE DESIGN ======')
print('Estimated mean repair cost: ',
      f'${design_repair_cost:,.2f}')
print('Estimated median repair cost: ',
      f'${design_repair_cost_med:,.2f}')
print('Estimated mean repair time (sequential): ',
      f'{design_downtime:,.2f}', 'worker-days')
print('Estimated median repair time (sequential): ',
      f'{design_downtime_med:,.2f}', 'worker-days')
print('Estimated collapse frequency: ',
      f'{design_collapse_risk:.2%}')
print('Estimated replacement frequency: ',
      f'{design_replacement_risk:.2%}')

baseline_repair_cost = df_base['cost_mean'].mean()
baseline_repair_cost_med = df_base['cost_50%'].mean()
baseline_downtime = df_base['time_u_mean'].mean()
baseline_downtime_med = df_base['time_u_50%'].mean()
baseline_collapse_risk = df_base['collapse_freq'].mean()
baseline_replacement_risk = df_base['replacement_freq'].mean()

print('====== BASELINE DESIGN ======')
print('Estimated mean repair cost: ',
      f'${baseline_repair_cost:,.2f}')
print('Estimated median repair cost: ',
      f'${baseline_repair_cost_med:,.2f}')
print('Estimated mean repair time (sequential): ',
      f'{baseline_downtime:,.2f}', 'worker-days')
print('Estimated median repair time (sequential): ',
      f'{baseline_downtime_med:,.2f}', 'worker-days')
print('Estimated collapse frequency: ',
      f'{baseline_collapse_risk:.2%}')
print('Estimated replacement frequency: ',
      f'{baseline_replacement_risk:.2%}')

#%% fit collapse models (OR)


mdl_drift_hit = Prediction(df_hit)
mdl_drift_hit.set_outcome('max_drift')
mdl_drift_hit.test_train_split(0.2)

mdl_drift_miss = Prediction(df_miss)
mdl_drift_miss.set_outcome('max_drift')
mdl_drift_miss.test_train_split(0.2)

# fit impacted set
mdl_drift_hit.fit_ols_ridge()
        
# fit no impact set
mdl_drift_miss.fit_ols_ridge()

#%% Drift model (GP-OR)
plt.close('all')
X_plot = mdl.make_2D_plotting_space(100)
plt_density = 200
grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.o_ridge,
                        mdl_drift_miss.o_ridge,
                                  outcome='max_drift')

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_drift)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=0, vmax=0.075)

ax.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['max_drift'][:plt_density],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('PID (%)')
ax.set_title('Peak interstory drift prediction (GPC-impact, OR-drift)')
fig.tight_layout()
plt.show()

# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.Blues,
                       linewidth=0, antialiased=False,
                       alpha=0.7, vmin=-0.1, vmax=0.5)

ax.scatter(df['gapRatio'][:plt_density], df['RI'][:plt_density], 
           df['collapse_freq'][:plt_density],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Collapse risk')
ax.set_zlim([0, 1.0])
ax.set_title('Collapse risk prediction, LN transformed from drift (GPC-OR)')
fig.tight_layout()
plt.show()

#%% dirty contours

plt.close('all')
import numpy as np
# x is gap, y is Ry
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.0
y_min = 0.5
y_max = 2.0

lvls = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

res = 100

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

grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.o_ridge,
                        mdl_drift_miss.o_ridge,
                                  outcome='max_drift')


# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
plt.setp((ax1, ax2, ax3), xticks=np.arange(0.5, 4.0, step=0.5))

cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)

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
    probDes = 0.1
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax1.axvline(theGap)
    ax1.text(theGap+0.05, 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90)

df_sc = df[(df['Tm']<=2.65) & (df['zetaM']<=0.17) & (df['zetaM']>=0.13)]

ax1.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_probs'], cmap='Blues',
            s=30, edgecolors='k')

ax1.clabel(cs, fontsize=label_size)
ax1.set_xlim([0.3, 2.0])


ax1.grid(visible=True)
ax1.set_title(r'$T_M = 2.00$ s, $\zeta_M = 0.15$', fontsize=subt_font)
ax1.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)

#####
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.0
y_min = 0.5
y_max = 2.0

res = 100

xx, yy = np.meshgrid(np.linspace(x_min,
                                 x_max,
                                 res),
                     np.linspace(y_min,
                                 y_max,
                                 res))

X_pl = pd.DataFrame({x_var:xx.ravel(),
                     y_var:yy.ravel(),
                     third_var:np.repeat(3.25,
                                         res*res),
                     fourth_var:np.repeat(0.15, 
                                          res*res)})

X_plot = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]

grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.o_ridge,
                        mdl_drift_miss.o_ridge,
                                  outcome='max_drift')


# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

cs = ax2.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)

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
    probDes = 0.1
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax2.axvline(theGap)
    ax2.text(theGap+0.05, 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90)
    
ax2.clabel(cs, fontsize=label_size)
ax2.set_xlim([0.3, 2.0])

df_sc = df[(df['Tm']<=3.4) & (df['Tm']>=3.1) & (df['zetaM']<=0.17) & (df['zetaM']>=0.13)]

ax2.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_probs'], cmap='Blues',
            s=30, edgecolors='k')

ax2.grid(visible=True)
ax2.set_title(r'$T_M = 3.25$ s, $\zeta_M = 0.15$', fontsize=subt_font)
ax2.set_xlabel(r'Gap ratio (GR)', fontsize=axis_font)
ax2.set_ylabel(r'$R_y$', fontsize=axis_font)
#####
x_var = 'gapRatio'
y_var = 'RI'
third_var = 'Tm'
fourth_var = 'zetaM'
x_min = 0.3
x_max = 2.0
y_min = 0.5
y_max = 2.0

res = 100

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

grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.o_ridge,
                        mdl_drift_miss.o_ridge,
                                  outcome='max_drift')


# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84

ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

cs = ax3.contour(xx, yy, Z, linewidths=1.1, cmap='Blues', vmin=-1,
                 levels=lvls)

from scipy.interpolate import RegularGridInterpolator
RyList = [0.5, 1.0, 2.0]
for j in range(len(RyList)):
    RyTest = RyList[j]
    lpBox = Z
    xq = np.linspace(0.3, 1.8, 200)
    
    interp = RegularGridInterpolator((yy[:,0], xx[0,:]), lpBox)
    pts = np.zeros((200,2))
    pts[:,1] = xq
    pts[:,0] = RyTest
    lq = interp(pts)
    probDes = 0.1
    theGapIdx = np.argmin(abs(lq - probDes))
    theGap = xq[theGapIdx]
    ax3.axvline(theGap)
    ax3.text(theGap+0.05, 0.5, r'GR = '+f'{theGap:,.2f}', rotation=90)

ax3.clabel(cs, fontsize=label_size)
ax3.set_xlim([0.3, 2.0])

df_sc = df[(df['Tm']>=3.80) & (df['zetaM']<=0.17) & (df['zetaM']>=0.13)]

sc = ax3.scatter(df_sc[x_var],
            df_sc[y_var],
            c=df_sc['collapse_probs'], cmap='Blues',
            s=30, edgecolors='k')

ax3.grid(visible=True)
ax3.set_title(r'$T_M = 4.00$ s, $\zeta_M = 0.15$', fontsize=subt_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax3.set_ylabel(r'$R_y$', fontsize=axis_font)

handles, labels = sc.legend_elements(prop="colors", alpha=0.6)
legend2 = ax3.legend(handles, labels, loc="lower right", title="% collapse")

fig.tight_layout()

#%%

# TODO: import main run data and sort in order to facilitate upfront costs
