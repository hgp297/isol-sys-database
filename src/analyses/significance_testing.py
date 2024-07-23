############################################################################
#               Statistical testing

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
df = df_raw[np.abs(stats.zscore(df_raw['collapse_prob'])) < 10].copy()

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

# remove outlier point
from scipy import stats
df_no_impact = df_miss[np.abs(stats.zscore(df_miss['cmp_cost_ratio'])) < 5].copy()
df_outlier = df_miss[np.abs(stats.zscore(df_miss['cmp_cost_ratio'])) > 5].copy()

df_tfp = df_no_impact[df_no_impact['isolator_system'] == 'TFP']
df_lrb = df_no_impact[df_no_impact['isolator_system'] == 'LRB']
df_cbf = df_no_impact[df_no_impact['superstructure_system'] == 'CBF']
df_mf = df_no_impact[df_no_impact['superstructure_system'] == 'MF']

#%%  variable testing

print('========= stats for repair cost ==========')
from sklearn import preprocessing

df_test = df_lrb.copy()

X = df_test[['gap_ratio', 'RI', 'T_ratio', 'k_ratio', 'zeta_e' ,'Q']]
y = df_test[cost_var].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.feature_selection import r_regression

r_results = r_regression(X_scaled,y)
print("Pearson's R test: GR, Ry, T_ratio, k_ratio, zeta, Q")
print(["%.4f" % member for member in r_results])

print('========= stats for repair time ==========')
X = df_test[['gap_ratio', 'RI', 'T_ratio', 'k_ratio', 'zeta_e' ,'Q']]
y = df_test[time_var].ravel()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

r_results = r_regression(X_scaled,y)
print("Pearson's R test: GR, Ry, T_ratio, k_ratio, zeta, Q")
print(["%.4f" % member for member in r_results])
#%% Tfbe regression

plt.close('all')

df_cbf_tfbe = df[df['superstructure_system'] == 'CBF']
df_mf_tfbe = df[df['superstructure_system'] == 'MF']

cmap = plt.cm.tab10

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.scatter(df_cbf_tfbe['h_bldg'], df_cbf_tfbe['T_fb'], alpha=0.7, color=cmap(1), label='CBF')
ax.scatter(df_mf_tfbe['h_bldg'], df_mf_tfbe['T_fb'], alpha=0.7, color=cmap(0), label='MF')

hnx = np.linspace(20, 120, 200)
tfbe_mf = 1.4*0.028*(hnx)**(0.8)
tfbe_cbf = 1.4*0.02*(hnx)**(0.75)

ax.plot(hnx, tfbe_mf, color=cmap(0), label=r'$C_u T_a$: MF')
ax.plot(hnx, tfbe_cbf, color=cmap(1), label=r'$C_u T_a$: CBF')

ax.set_ylabel("$T_{fb}$ (s)", fontsize=axis_font)
ax.set_xlabel('$h_n$ (ft)', fontsize=axis_font)

ax.set_xlim([30, 100])
ax.legend()


#%% 1d regression: structure, cost

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig = plt.figure(figsize=(16, 8))

from sklearn import linear_model
regr = linear_model.LinearRegression()

ax = fig.add_subplot(2, 3, 1)
xvar = 'RI'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$R_y$", fontsize=axis_font)
ax.set_ylabel('Cost ratio', fontsize=axis_font)
ax.legend()
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 2)
xvar = 'gap_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$GR$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 3)
xvar = 'T_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$T_M/T_{fb}$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 4)
xvar = 'zeta_e'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$\zeta_e$", fontsize=axis_font)
ax.set_ylabel('Cost ratio', fontsize=axis_font)
ax.legend()
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 5)
xvar = 'k_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$k_1/k_2$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 6)
xvar = 'Q'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_cbf[xvar], df_cbf['cmp_cost_ratio'], alpha=0.7, color=cmap(1), label='CBF')
regr.fit(df_cbf[[xvar]], df_cbf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(1), linewidth=3)

ax.scatter(df_mf[xvar], df_mf['cmp_cost_ratio'], alpha=0.7, color=cmap(0), label='MF')
regr.fit(df_mf[[xvar]], df_mf[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(0), linewidth=3)
ax.set_xlabel("$Q$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

fig.tight_layout()

#%% 1d regression, bearing, cost
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


fig = plt.figure(figsize=(16, 8))

from sklearn import linear_model
regr = linear_model.LinearRegression()

ax = fig.add_subplot(2, 3, 1)
xvar = 'RI'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$R_y$", fontsize=axis_font)
ax.set_ylabel('Cost ratio', fontsize=axis_font)
ax.legend()
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 2)
xvar = 'gap_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$GR$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 3)
xvar = 'T_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$T_M/T_{fb}$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 4)
xvar = 'zeta_e'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$\zeta_e$", fontsize=axis_font)
ax.set_ylabel('Cost ratio', fontsize=axis_font)
ax.legend()
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 5)
xvar = 'k_ratio'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$k_1/k_2$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

ax = fig.add_subplot(2, 3, 6)
xvar = 'Q'
x_1d = np.linspace(df_miss[xvar].min(), df_miss[xvar].max(), 200).reshape(-1, 1)
ax.scatter(df_lrb[xvar], df_lrb['cmp_cost_ratio'], alpha=0.7, color=cmap(2), label='LRB')
regr.fit(df_lrb[[xvar]], df_lrb[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(2), linewidth=3)

ax.scatter(df_tfp[xvar], df_tfp['cmp_cost_ratio'], alpha=0.7, color=cmap(3), label='TFP')
regr.fit(df_tfp[[xvar]], df_tfp[['cmp_cost_ratio']])
y_pred = regr.predict(x_1d)
plt.plot(x_1d, y_pred, color=cmap(3), linewidth=3)
ax.set_xlabel("$Q$", fontsize=axis_font)
ax.set_ylim([0, 0.2])

fig.tight_layout()

