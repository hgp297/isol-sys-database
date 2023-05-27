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

# results_path = './results/tfp_mf/'
# results_file = 'loss_estimate_data.csv'
# loss_data = pd.read_csv(results_path+results_file, 
#                         index_col=None)
full_isolation_data = pd.read_csv(database_path+database_file, 
                                  index_col=None)

# df = pd.concat([full_isolation_data, loss_data], axis=1)
df = full_isolation_data
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df.loc[df['collapsed'] == -1, 'collapsed'] = 0
# df['max_accel'] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
# df['max_vel'] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
#%% Prepare data

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('collapsed')
mdl.test_train_split(0.2)

#%% fit collapse (kernel logistic classification)

# # currently only rbf is working
# # TODO: gamma cross validation
# krn = 'rbf'
# gam = None # if None, defaults to 1/n_features = 0.25
# mdl.fit_kernel_logistic(neg_wt=0.75, kernel_name=krn, gamma=gam)

# # predict the entire dataset
# K_data = mdl.get_kernel(mdl.X, kernel_name=krn, gamma=gam)
# preds_imp = mdl.log_reg_kernel.predict(K_data)
# probs_imp = mdl.log_reg_kernel.predict_proba(K_data)

# cmpr = np.array([mdl.y.values.flatten(), preds_imp]).transpose()

# # we've done manual CV to pick the hyperparams that trades some accuracy
# # in order to lower false negatives
# from sklearn.metrics import confusion_matrix

# tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
# print('False negatives: ', fn)
# print('False positives: ', fp)

# # make grid and plot classification predictions
# X_plot = mdl.make_2D_plotting_space(100)
# K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
# mdl.plot_classification(mdl.log_reg_kernel)

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
mdl.plot_classification(mdl.gpc, contour_pr=0.1)

# X_plot = mdl.make_2D_plotting_space(100, y_var='Tm')
# mdl.plot_classification(mdl.gpc, yvar='Tm', contour_pr=0.5)

# X_plot = mdl.make_2D_plotting_space(100, x_var='gapRatio', y_var='zetaM')
# mdl.plot_classification(mdl.gpc, xvar='gapRatio', yvar='zetaM', contour_pr=0.5)


#%% make design space and predict collapse

import time

# res_des = 30
# X_space = mdl.make_design_space(res_des)

res = 75

# xx, yy, uu = np.meshgrid(np.linspace(0.5, 2.0,
#                                           res),
#                               np.linspace(0.1, 0.2,
#                                           res),
#                               np.linspace(2.5, 4.0,
#                                           res))
                             
# X_space = pd.DataFrame({'gapRatio':xx.ravel(),
#                       'RI':np.repeat(2.0,res**3),
#                       'Tm':uu.ravel(),
#                       'zetaM':yy.ravel()})


xx, yy, uu = np.meshgrid(np.linspace(0.5, 2.0,
                                      res),
                          np.linspace(0.5, 2.0,
                                      res),
                          np.linspace(2.5, 4.0,
                                      res))
                             
X_space = pd.DataFrame({'gapRatio':xx.ravel(),
                      'RI':yy.ravel(),
                      'Tm':uu.ravel(),
                      'zetaM':np.repeat(0.2,res**3)})



t0 = time.time()
space_collapse = mdl.gpc.predict_proba(X_space)

tp = time.time() - t0
print("GPC collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))


#%% predictive variance and weighted variance

# latent variance
fmu, fs2 = mdl.predict_gpc_latent(X_space)

#%% plot gpc functions

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

xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

X_subset = X_space[X_space['Tm']==3.25]
fs2_subset = fs2[X_space['Tm']==3.25]
fmu_subset = fmu[X_space['Tm']==3.25]

Z = fs2_subset.reshape(xx_pl.shape)

plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Latent variance', fontsize=axis_font)
plt.colorbar()
plt.show()

Z = fmu_subset.reshape(xx_pl.shape)

plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Latent mean', fontsize=axis_font)
plt.colorbar()
plt.show()

# TODO: transition from latent to predictive mean is in the __gpc code (line 7
# of Algorithm 3.2, GPML) (it's the integral of sigmoid(x)*normpdf(x | fmu, fsigma))

from scipy.stats import logistic

Z = logistic.cdf(fmu_subset.reshape(xx_pl.shape))
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Predictive mean', fontsize=axis_font)
plt.colorbar()
plt.show()


# TODO: reexamine DoE weight
from numpy import exp
T = logistic.ppf(0.1)
pi = 3.14159
Wx = 1/((2*pi*(fs2_subset))**0.5)*exp((-1/2)*((fmu_subset - T)**2/(fs2_subset)))

criterion = np.multiply(Wx, fs2_subset)
idx = np.argmax(criterion)

Z = criterion.reshape(xx_pl.shape)
plt.figure()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx_pl.min(), xx_pl.max(),
            yy_pl.min(), yy_pl.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
) 
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.title('Weighted variance', fontsize=axis_font)
plt.colorbar()
plt.show()

#%% cost efficiency

from pred import get_steel_coefs, calc_upfront_cost
plt.close('all')
steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

risk_thresh = 0.1
space_collapse_pred = pd.DataFrame(space_collapse,
                                   columns=['safe probability', 'collapse probability'])
ok_risk = X_space.loc[space_collapse_pred['collapse probability']<=
                      risk_thresh]

X_design = X_space[X_space.index.isin(ok_risk.index)]
    
# in the filter-design process, only one of cost/dt is likely to control
    
# TODO: more clever selection criteria (not necessarily the cheapest)

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_collapse_risk = space_collapse_pred.iloc[cheapest_design_idx]['collapse probability']

print(best_design)

print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')