############################################################################
#               ML prediction models for collapse

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) Many models require cross validation of negative weight
#               as well as gamma value for rbf kernels
#               (2) note that KLR works better when there are extremities than
#               SVC in terms of drift-related risks

############################################################################

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

#%% concat with other data
database_path = './data/tfp_mf/'
database_file = 'run_data.csv'

results_path = './results/tfp_mf/'
results_file = 'loss_estimate_data.csv'
loss_data = pd.read_csv(results_path+results_file, 
                        index_col=None)
full_isolation_data = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df.loc[df['collapsed'] == -1, 'collapsed'] = 0
# df['max_accel'] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
# df['max_vel'] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
#%% Prepare data
cost_var = 'cost_50%'
time_var = 'time_u_50%'

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_hit = Prediction(df_hit)
mdl_hit.set_outcome(cost_var)
mdl_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_miss = Prediction(df_miss)
mdl_miss.set_outcome(cost_var)
mdl_miss.test_train_split(0.2)

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('collapsed')
mdl.test_train_split(0.2)
#%% fit collapse (gp classification)

mdl.fit_gpc(kernel_name='rbf_ard', noisy=True)

# predict the entire dataset
preds_col = mdl.gpc.predict(mdl.X)
probs_col = mdl.gpc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_col).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.gpc, contour_pr=0.5)

X_plot = mdl.make_2D_plotting_space(100, y_var='Tm')
mdl.plot_classification(mdl.gpc, yvar='Tm', contour_pr=0.5)

X_plot = mdl.make_2D_plotting_space(100, x_var='gapRatio', y_var='zetaM')
mdl.plot_classification(mdl.gpc, xvar='gapRatio', yvar='zetaM', contour_pr=0.5)

#%% make design space and predict collapse
from pred import get_steel_coefs, calc_upfront_cost

import time

res_des = 30
X_space = mdl.make_design_space(res_des)