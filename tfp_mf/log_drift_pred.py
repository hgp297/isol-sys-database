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

#%% pre-doe data

database_path = './data/'
database_file = 'training_set.csv'

df_train = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_train['max_drift'] = df_train[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_train['log_drift'] = np.log(df_train['max_drift'])
df_train['collapse_prob'] = ln_dist.cdf(df_train['max_drift'])

mdl_init = GP(df_train)
mdl_init.set_outcome('log_drift')
mdl_init.fit_gpr(kernel_name='rbf_ard')

#%%


import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
y_hat = mdl_init.gpr.predict(mdl_init.X)
y_true = mdl_init.y

plt.figure()
plt.scatter(y_hat, y_true)

plt.title('Prediction accuracy, pre-DOE')
plt.xlabel('Predicted log drift')
plt.ylabel('True log drift')

# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])
# plt.plot([0, 1.0], [0, 1.0], linestyle='-',color='black')
# plt.plot([0, 1.0], [0, 1.1], linestyle='--',color='black')
# plt.plot([0, 1.0], [0, 0.9], linestyle='--',color='black')

plt.xlim([-6, 0])
plt.ylim([-6, 0])
plt.grid(True)
plt.show()

#%% predict the plotting space
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

fmu, fs1 = mdl_init.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% plots

plt.close('all')
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


Z = np.exp(fmu_subset)
Z = ln_dist.cdf(Z).reshape(xx_pl.shape)

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
plt.scatter(df_train['gapRatio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.contour(xx_pl, yy_pl, Z, levels=[0.1])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.title('Collapse risk', fontsize=axis_font)
plt.show()
#%% post-doe data

# database_path = './data/doe/old/rmse_1_percent/'
database_path = './data/doe/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['log_drift'] = np.log(df_doe['max_drift'])
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('log_drift')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

#%% prediction accuracy doe
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
y_hat = mdl_doe.gpr.predict(mdl_doe.X)
y_true = mdl_doe.y

plt.figure()
plt.scatter(y_hat, y_true)

plt.title('Prediction accuracy, post-DOE')
plt.xlabel('Predicted log drift')
plt.ylabel('True log drift')

# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])
# plt.plot([0, 1.0], [0, 1.0], linestyle='-',color='black')
# plt.plot([0, 1.0], [0, 1.1], linestyle='--',color='black')
# plt.plot([0, 1.0], [0, 0.9], linestyle='--',color='black')

plt.xlim([-6, 0])
plt.ylim([-6, 0])

plt.grid(True)
plt.show()

#%% predict the plotting space

t0 = time.time()

fmu, fs1 = mdl_doe.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% plots

plt.close('all')
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


Z = np.exp(fmu_subset)
Z = ln_dist.cdf(Z).reshape(xx_pl.shape)

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
plt.scatter(df_train['gapRatio'], df_train['RI'], 
            c=df_train['collapse_prob'],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.contour(xx_pl, yy_pl, Z, levels=[0.1])
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.title('Collapse risk', fontsize=axis_font)
