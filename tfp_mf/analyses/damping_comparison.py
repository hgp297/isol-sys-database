############################################################################
#               ML prediction models for log drift

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
g = 386.4
#%% Rayleigh damping data

database_path = './data/'
database_file = 'mik_smrf.csv'

df_rayleigh = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_rayleigh['max_drift'] = df_rayleigh[["driftMax1", 
                                        "driftMax2", 
                                        "driftMax3"]].max(axis=1)
df_rayleigh['max_accel'] = df_rayleigh[["accMax1", 
                                        "accMax2", 
                                        "accMax3"]].max(axis=1)/g
df_rayleigh['log_drift'] = np.log(df_rayleigh['max_drift'])
df_rayleigh['collapse_prob'] = ln_dist.cdf(df_rayleigh['max_drift'])

df_rayleigh = df_rayleigh.tail(200)


#%% stiffness proportional damping data

database_path = './data/'
database_file = 'mik_smrf_SP_damp.csv'

df_stiffness = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_stiffness['max_drift'] = df_stiffness[["driftMax1", 
                                          "driftMax2", 
                                          "driftMax3"]].max(axis=1)
df_stiffness['max_accel'] = df_stiffness[["accMax1", 
                                          "accMax2", 
                                          "accMax3"]].max(axis=1)/g
df_stiffness['log_drift'] = np.log(df_stiffness['max_drift'])
df_stiffness['collapse_prob'] = ln_dist.cdf(df_stiffness['max_drift'])


#%%

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

bins = pd.IntervalIndex.from_tuples([(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.5)])
labels=['tiny', 'small', 'okay', 'large']
df_rayleigh['gap_bin'] = pd.cut(df_rayleigh['gapRatio'], bins=bins, labels=labels)
df_count = df_rayleigh.groupby('gap_bin')['max_drift'].apply(lambda x: (x>=0.10).sum()).reset_index(name='count')
a = df_rayleigh.groupby(['gap_bin']).size()
df_count['percent'] = df_count['count']/a

plt.close('all')
fig, axs = plt.subplots(2, 1, figsize=(12,10))
ax1 = axs[0]
ax2 = axs[1]
import seaborn as sns
sns.stripplot(data=df_rayleigh, x="max_drift", y="gap_bin", orient="h",
              hue='RI', size=10,
              ax=ax1, legend='brief', palette='bone')
sns.boxplot(y="gap_bin", x= "max_drift", data=df_rayleigh,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax1)
ax1.vlines(x=0.078, ymin=-0.5, ymax=3.5, color='black', linestyle=":")
ax1.text(0.079, 3.4, r'50% collapse', fontsize=axis_font, color='black')

plt.setp(ax1.get_legend().get_texts(), fontsize=subt_font) # for legend text
plt.setp(ax1.get_legend().get_title(), fontsize=axis_font)
ax1.get_legend().get_title().set_text(r'$R_y$') # for legend title

ax1.set_ylabel('Gap ratio range', fontsize=axis_font)
ax1.set_xlabel(None)
ax1.set_title(r'Rayleigh damping: 5% and 2% to superstructure modes 1 and 3')
ax1.set_xlim([0.0, 0.2])
ax1.grid()

bins = pd.IntervalIndex.from_tuples([(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.5)])
labels=['tiny', 'small', 'okay', 'large']
df_stiffness['gap_bin'] = pd.cut(df_stiffness['gapRatio'], bins=bins, labels=labels)
df_count = df_stiffness.groupby('gap_bin')['max_drift'].apply(lambda x: (x>=0.10).sum()).reset_index(name='count')
a = df_stiffness.groupby(['gap_bin']).size()
df_count['percent'] = df_count['count']/a

# import seaborn as sns
sns.stripplot(data=df_stiffness, x="max_drift", y="gap_bin", orient="h",
              hue='RI', size=10,
              ax=ax2, legend='brief', palette='bone')
sns.boxplot(y="gap_bin", x= "max_drift", data=df_stiffness,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax2)
ax2.vlines(x=0.078, ymin=-0.5, ymax=3.5, color='black', linestyle=":")
ax2.text(0.079, 3.4, r'50% collapse', fontsize=axis_font, color='black')

plt.setp(ax2.get_legend().get_texts(), fontsize=subt_font) # for legend text
plt.setp(ax2.get_legend().get_title(), fontsize=axis_font)
ax2.get_legend().get_title().set_text(r'$R_y$') # for legend title

ax2.set_xlabel('Peak interstory drift (PID)', fontsize=axis_font)
ax2.set_ylabel('Gap ratio range', fontsize=axis_font)
ax2.set_title(r'Stiffness damping proportional: 5% superstructure mode 1')
ax2.set_xlim([0.0, 0.2])
ax2.grid()
fig.tight_layout()
plt.show()


#%% predict the plotting space

mdl_rl = GP(df_rayleigh)
mdl_rl.set_outcome('collapse_prob')
mdl_rl.fit_gpr(kernel_name='rbf_ard')


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

fmu, fs1 = mdl_rl.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

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
plt.scatter(df_rayleigh['gapRatio'], df_rayleigh['RI'], 
            c=df_rayleigh['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Reds_r)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.grid()
plt.title('Collapse risk, Rayleigh', fontsize=axis_font)

#%% predict the plotting space

mdl_sp = GP(df_stiffness)
mdl_sp.set_outcome('collapse_prob')
mdl_sp.fit_gpr(kernel_name='rbf_ard')


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

fmu, fs1 = mdl_sp.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

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
plt.scatter(df_stiffness['gapRatio'], df_stiffness['RI'], 
            c=df_stiffness['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.contour(xx_pl, yy_pl, Z, levels=[0.025, 0.05, 0.1], cmap=plt.cm.Reds_r)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.grid()
plt.title('Collapse risk, stiffness proportional', fontsize=axis_font)