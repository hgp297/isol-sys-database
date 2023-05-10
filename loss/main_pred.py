############################################################################
#               ML prediction models for isolator loss data

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
#import collections
#collections.Callable = collections.abc.Callable

#%% concat with other data
database_path = './data/tfp_mf/'
database_file = 'run_data.csv'
loss_data = pd.read_csv('./results/loss_estimate_data.csv', 
                        index_col=None)
full_isolation_data = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
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
mdl.set_outcome('impacted')
mdl.test_train_split(0.2)

#%% fit impact (SVC)
# fit SVM classification for impact
# lower neg_wt = penalize false negatives more

#mdl.fit_svc(neg_wt=1.0, kernel_name='sigmoid')
mdl.fit_svc(neg_wt=1.0, kernel_name='rbf')

# predict the entire dataset
preds_imp = mdl.svc.predict(mdl.X)

# note: SVC probabilities are NOT well calibrated
probs_imp = mdl.svc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.svc)

#%% fit impact (logistic classification)

# fit logistic classification for impact
mdl.fit_log_reg(neg_wt=1.0)

# predict the entire dataset
preds_imp = mdl.log_reg.predict(mdl.X)
probs_imp = mdl.log_reg.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.log_reg)

#%% fit impact (kernel logistic classification)

# currently only rbf is working
# TODO: gamma cross validation
krn = 'rbf'
gam = None # if None, defaults to 1/n_features = 0.25
mdl.fit_kernel_logistic(neg_wt=1.0, kernel_name=krn, gamma=gam)

# predict the entire dataset
K_data = mdl.get_kernel(mdl.X, kernel_name=krn, gamma=gam)
preds_imp = mdl.log_reg_kernel.predict(K_data)
probs_imp = mdl.log_reg_kernel.predict_proba(K_data)

cmpr = np.array([mdl.y.values.flatten(), preds_imp]).transpose()

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
mdl.plot_classification(mdl.log_reg_kernel)

## make grid and plot classification predictions
#X_plot = mdl.make_2D_plotting_space(100, y_var='Tm')
#K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
#mdl.plot_classification(mdl.log_reg_kernel, yvar='Tm')
#%% fit impact (gp classification)

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

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.gpc)

X_plot = mdl.make_2D_plotting_space(100, y_var='Tm')
mdl.plot_classification(mdl.gpc, yvar='Tm')

X_plot = mdl.make_2D_plotting_space(100, x_var='gapRatio', y_var='zetaM')
mdl.plot_classification(mdl.gpc, xvar='gapRatio', yvar='zetaM')
#%% Fit costs (SVR)

# fit impacted set
mdl_hit.fit_svr()
cost_pred_hit = mdl_hit.svr.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_svr()
cost_pred_miss = mdl_miss.svr.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

mdl_miss.make_2D_plotting_space(100)

xx = mdl_miss.xx
yy = mdl_miss.yy
Z = mdl_miss.svr.predict(mdl_miss.X_plot)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss[cost_var],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions given no impact (SVR)')
ax.set_zlim([0, 5e5])
plt.show()

#%% Fit costs (kernel ridge)

kernel_type = 'rbf'

# fit impacted set
mdl_hit.fit_kernel_ridge(kernel_name=kernel_type)
cost_pred_hit = mdl_hit.kr.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_kernel_ridge(kernel_name=kernel_type)
cost_pred_miss = mdl_miss.kr.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

mdl_miss.make_2D_plotting_space(100)

xx = mdl_miss.xx
yy = mdl_miss.yy
Z = mdl_miss.kr.predict(mdl_miss.X_plot)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss[cost_var],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions given no impact (RBF kernel ridge)')
ax.set_zlim([0, 5e5])
plt.show()

#%% Fit costs (GP regression)

kernel_type = 'rq'

# fit impacted set
mdl_hit.fit_gpr(kernel_name=kernel_type)
cost_pred_hit = mdl_hit.gpr.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_gpr(kernel_name=kernel_type)
cost_pred_miss = mdl_miss.gpr.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

mdl_miss.make_2D_plotting_space(100)

xx = mdl_miss.xx
yy = mdl_miss.yy
Z = mdl_miss.gpr.predict(mdl_miss.X_plot)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss[cost_var],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions given no impact (GP regression)')
ax.set_zlim([0, 1e6])
plt.show()

#%% Fit costs (regular ridge)

# sensitive to alpha. keep alpha > 1e-2 since smaller alphas will result in 
# flat line to minimize the outliers' error

# results in a fit similar to kernel ridge

# fit impacted set
mdl_hit.fit_ols_ridge()
cost_pred_hit = mdl_hit.o_ridge.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_ols_ridge()
cost_pred_miss = mdl_miss.o_ridge.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

#%% aggregate the two models
dataset_repair_cost = predict_DV(mdl.X,
                                        mdl.log_reg,
                                        mdl_hit.svr,
                                        mdl_miss.svr)
comparison_cost = np.array([df[cost_var],
                            np.ravel(dataset_repair_cost)]).transpose()

#%% Big cost prediction plot (SVC-SVR)

X_plot = mdl.make_2D_plotting_space(100)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.svc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: SVC-impact, SVR-loss')
plt.show()

#%% Big cost prediction plot (SVC-KR)

X_plot = mdl.make_2D_plotting_space(100)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.svc,
                                     mdl_hit.kr,
                                     mdl_miss.kr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: SVC-impact, KR-loss')
plt.show()

#%% Big cost prediction plot (KLR-SVR)

X_plot = mdl.make_2D_plotting_space(100)
K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
grid_repair_cost = predict_DV(X_plot,
                                     mdl.log_reg_kernel,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: KLR-impact, SVR-loss')
plt.show()

#%% Big cost prediction plot (KLR-KR)

X_plot = mdl.make_2D_plotting_space(100)
K_plot = mdl.get_kernel(X_plot, kernel_name=krn, gamma=gam)
grid_repair_cost = predict_DV(X_plot,
                                     mdl.log_reg_kernel,
                                     mdl_hit.kr,
                                     mdl_miss.kr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: KLR-impact, KR-loss')
plt.show()

#%% Big cost prediction plot (GP-SVR)

X_plot = mdl.make_2D_plotting_space(100)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: GP-impact, SVR-loss')
plt.show()


#%% Big cost prediction plot (GP-KR)

X_plot = mdl.make_2D_plotting_space(100)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.kr,
                                     mdl_miss.kr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: GP-impact, KR-loss')
plt.show()

#%% Big cost prediction plot (GPC-GPR)

X_plot = mdl.make_2D_plotting_space(100)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.gpr,
                                     mdl_miss.gpr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_repair_cost)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean loss ($)')
ax.set_title('Mean cost predictions: GP-impact, GP-loss')
plt.show()

#%% Fit downtime (SVR)
# make prediction objects for impacted and non-impacted datasets
mdl_time_hit = Prediction(df_hit)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = Prediction(df_miss)
mdl_time_miss.set_outcome(time_var)
mdl_time_miss.test_train_split(0.2)

# fit impacted set
mdl_time_hit.fit_svr()
time_pred_hit = mdl_time_hit.svr.predict(mdl_time_hit.X_test)
comparison_time_hit = np.array([mdl_time_hit.y_test, 
                                      time_pred_hit]).transpose()
        
# fit no impact set
mdl_time_miss.fit_svr()
time_pred_miss = mdl_time_miss.svr.predict(mdl_time_miss.X_test)
comparison_time_miss = np.array([mdl_time_miss.y_test, 
                                      time_pred_miss]).transpose()

#%% Fit downtime (KR)

# fit impacted set
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')
time_pred_hit = mdl_time_hit.svr.predict(mdl_time_hit.X_test)
comparison_time_hit = np.array([mdl_time_hit.y_test, 
                                      time_pred_hit]).transpose()
        
# fit no impact set
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')
time_pred_miss = mdl_time_miss.svr.predict(mdl_time_miss.X_test)
comparison_time_miss = np.array([mdl_time_miss.y_test, 
                                      time_pred_miss]).transpose()
#mdl_time_miss.make_2D_plotting_space(100)
#
#xx = mdl_time_miss.xx
#yy = mdl_time_miss.yy
#Z = mdl_time_miss.svr.predict(mdl_time_miss.X_plot)
#Z = Z.reshape(xx.shape)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
## Plot the surface.
#surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss[time_var],
#           edgecolors='k')
#
#ax.set_xlabel('Gap ratio')
#ax.set_ylabel('Ry')
#ax.set_zlabel('Mean downtime (worker-day)')
#ax.set_title('Mean sequential downtime predictions given no impact (SVR)')
#ax.set_zlim([0, 500])
#plt.show()

#%% Big downtime prediction plot (GP-SVR)

X_plot = mdl.make_2D_plotting_space(100)

grid_downtime = predict_DV(X_plot,
                          mdl.gpc,
                          mdl_time_hit.svr,
                          mdl_time_miss.svr,
                          outcome=time_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_downtime)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[time_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean downtime (worker-days)')
ax.set_title('Mean sequential downtime predictions: GP-impact, SVR-time')
plt.show()

#%% Big downtime prediction plot (GP-KR)

X_plot = mdl.make_2D_plotting_space(100)

grid_downtime = predict_DV(X_plot,
                                  mdl.gpc,
                                  mdl_time_hit.kr,
                                  mdl_time_miss.kr,
                                  outcome=time_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_downtime)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[time_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean downtime (worker-days)')
ax.set_title('Mean sequential downtime predictions: GP-impact, KR-time')
plt.show()

#%% Big downtime prediction plot (SVC-KR)

X_plot = mdl.make_2D_plotting_space(100)

grid_downtime = predict_DV(X_plot,
                                  mdl.svc,
                                  mdl_time_hit.kr,
                                  mdl_time_miss.kr,
                                  outcome=time_var)

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_downtime)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df[time_var],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Mean downtime (worker-days)')
ax.set_title('Mean sequential downtime predictions: SVC-impact, KR-time')
plt.show()

#%% fit collapse models (SVR and KR)

# SVR seems to be expensive, may need limiting CV runs

#mdl_clsp = Prediction(df)
#mdl_clsp.set_outcome('max_drift')
#mdl_clsp.test_train_split(0.2)
#mdl_clsp.fit_kernel_ridge()

mdl_drift_hit = Prediction(df_hit)
mdl_drift_hit.set_outcome('max_drift')
mdl_drift_hit.test_train_split(0.2)

mdl_drift_miss = Prediction(df_miss)
mdl_drift_miss.set_outcome('max_drift')
mdl_drift_miss.test_train_split(0.2)

# TODO: fit SVR for drift
# SVR seems to be expensive, may need limiting CV runs
# fit impacted set
mdl_drift_hit.fit_kernel_ridge()
mdl_drift_hit.fit_ols_ridge()
        
# fit no impact set
mdl_drift_miss.fit_kernel_ridge()
mdl_drift_miss.fit_ols_ridge()

#%% Drift model (GP-OR)
X_plot = mdl.make_2D_plotting_space(100)

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
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df['gapRatio'], df['RI'], df['max_drift'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('PID (%)')
ax.set_title('Peak interstory drift prediction (GPC-impact, OR-drift)')
plt.show()

# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*0.9945)
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df['gapRatio'], df['RI'], df['collapse_freq'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Collapse risk')
ax.set_zlim([0, 1.0])
ax.set_title('Collapse risk prediction, LN transformed from drift (GPC-OR)')
plt.show()

#%% drift model (GPC-GPR)
# kernel_type='matern_ard'
kernel_type = 'rq'

# fit impacted set
mdl_drift_hit.fit_gpr(kernel_name=kernel_type)
        
# fit no impact set
mdl_drift_miss.fit_gpr(kernel_name=kernel_type)

X_plot = mdl.make_2D_plotting_space(100)

grid_drift = predict_DV(X_plot,
                        mdl.gpc,
                        mdl_drift_hit.gpr,
                        mdl_drift_miss.gpr,
                                  outcome='max_drift')

xx = mdl.xx
yy = mdl.yy
Z = np.array(grid_drift)
Z = Z.reshape(xx.shape)

plt.close('all')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df['gapRatio'], df['RI'], df['max_drift'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('PID (%)')
ax.set_title('Peak interstory drift prediction (GPC-impact, GPR-drift)')
plt.show()

# drift -> collapse risk
from scipy.stats import lognorm
from math import log, exp
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*0.9945)
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

Z = ln_dist.cdf(np.array(grid_drift))
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df['gapRatio'], df['RI'], df['collapse_freq'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Collapse risk')
ax.set_zlim([0, 1.0])
ax.set_title('Collapse risk prediction, LN transformed from drift (GPC-GPR)')
plt.show()
#%% Testing the design space
import time

res_des = 20
X_space = mdl.make_design_space(res_des)
#K_space = mdl.get_kernel(X_space, kernel_name='rbf', gamma=gam)

# choice SVC for impact bc fast and behavior most closely resembles GPC
# HOWEVER, SVC is poorly calibrated for probablities
# consider using GP if computational resources allow, and GP looks good

# choice SVR bc behavior most closely resembles GPR
# also trend is visible: impact set looks like GPR, nonimpact set favors high R
t0 = time.time()
space_repair_cost = predict_DV(X_space,
                                      mdl.gpc,
                                      mdl_hit.svr,
                                      mdl_miss.svr,
                                      outcome=cost_var)
tp = time.time() - t0
print("GPC-SVR repair cost prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                           tp))

# choice KR bc smoother when predicting downtime
t0 = time.time()
space_downtime = predict_DV(X_space,
                                      mdl.gpc,
                                      mdl_time_hit.kr,
                                      mdl_time_miss.kr,
                                      outcome=time_var)
tp = time.time() - t0
print("GPC-KR downtime prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

# choice O_ridge bc SVR seems to hang, and KR overestimates (may need CV)
t0 = time.time()
space_drift = predict_DV(X_space,
                                      mdl.gpc,
                                      mdl_drift_hit.o_ridge,
                                      mdl_drift_miss.o_ridge,
                                      outcome='max_drift')
tp = time.time() - t0
print("GPC-OR drift prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

#%% Transform predicted drift into probability

from scipy.stats import lognorm
from math import log, exp
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*0.9945)
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

space_collapse_risk = pd.DataFrame(ln_dist.cdf(space_drift),
                                          columns=['collapse_risk_pred'])

#%% Calculate upfront cost of data
# TODO: use PACT to get the replacement cost of these components
# TODO: include the deckings/slabs for more realistic initial costs

def get_steel_coefs(df, steel_per_unit=1.25, W=3037.5, Ws=2227.5):
    col_str = df['col']
    beam_str = df['beam']
    rbeam_str = df['roofBeam']
    
    col_wt = np.array([float(member.split('X',1)[1]) for member in col_str])
    beam_wt = np.array([float(member.split('X',1)[1]) for member in beam_str])
    rbeam_wt = np.array([float(member.split('X',1)[1]) for member in rbeam_str])
    
    # find true steel costs
    n_frames = 4
    n_cols = 12
    L_col = 39.0 #ft
    
    n_beam_per_frame = 6
    L_beam = 30.0 #ft
    
    n_roof_per_frame = 3
    L_roof = 30.0 #ft
    
    bldg_wt = ((L_col * n_cols)*col_wt +
               (L_beam * n_beam_per_frame * n_frames)*beam_wt +
               (L_roof * n_roof_per_frame * n_frames)*rbeam_wt
               )
    
    steel_cost = steel_per_unit*bldg_wt
    
    # find design base shear as a feature
    pi = 3.14159
    g = 386.4
    kM = (1/g)*(2*pi/df['Tm'])**2
    S1 = 1.017
    Dm = g*S1*df['Tm']/(4*pi**2*df['Bm'])
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*df['zetaM'])
    Vs = np.array(Vst/df['RI']).reshape(-1,1)
    
    # linear regress cost as f(base shear)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X=Vs, y=steel_cost)
    return({'coef':reg.coef_, 'intercept':reg.intercept_})
    
# TODO: if full costs are accounted for, then must change land cost to include
# bldg footprint, rather than just moat gap
    
# TODO: add economy of scale for land
    
# TODO: investigate upfront cost's influence by Tm
    
def calc_upfront_cost(X_query, steel_coefs,
                      land_cost_per_sqft=2837/(3.28**2),
                      W=3037.5, Ws=2227.5):
    
    from scipy.interpolate import interp1d
    zeta_ref = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Bm_ref = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    interp_f = interp1d(zeta_ref, Bm_ref)
    Bm = interp_f(X_query['zetaM'])
    
    # calculate moat gap
    pi = 3.14159
    g = 386.4
    S1 = 1.017
    SaTm = S1/X_query['Tm']
    moat_gap = X_query['gapRatio'] * (g*(SaTm/Bm)*X_query['Tm']**2)/(4*pi**2)
    
    # calculate design base shear
    kM = (1/g)*(2*pi/X_query['Tm'])**2
    Dm = g*S1*X_query['Tm']/(4*pi**2*Bm)
    Vb = Dm * kM * Ws / 2
    Vst = Vb*(Ws/W)**(1 - 2.5*X_query['zetaM'])
    Vs = np.array(Vst/X_query['RI']).reshape(-1,1)
    
    steel_cost = np.array(steel_coefs['intercept'] +
                          steel_coefs['coef']*Vs).ravel()
    land_area = 2*(90.0*12.0)*moat_gap - moat_gap**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return(steel_cost + land_cost)

#%% refine space to meet repair cost and downtime requirements
    
steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

percent_of_replacement = 0.2
cost_thresh = percent_of_replacement*8.1e6
ok_cost = X_space.loc[space_repair_cost[cost_var+'_pred']<=cost_thresh]

# <2 weeks for a team of 50
dt_thresh = 50*14
ok_time = X_space.loc[space_downtime[time_var+'_pred']<=dt_thresh]

risk_thresh = 0.025
ok_risk = X_space.loc[space_collapse_risk['collapse_risk_pred']<=
                      risk_thresh]

X_design = X_space[np.logical_and(
        X_space.index.isin(ok_cost.index), 
        X_space.index.isin(ok_time.index),
        X_space.index.isin(ok_risk.index))]
    
# in the filter-design process, only one of cost/dt is likely to control
    
# TODO: more clever selection criteria (not necessarily the cheapest)

# select best viable design
upfront_costs = calc_upfront_cost(X_design, coef_dict)
cheapest_design_idx = upfront_costs.idxmin()
design_upfront_cost = upfront_costs.min()

# least upfront cost of the viable designs
best_design = X_design.loc[cheapest_design_idx]
design_downtime = space_downtime.iloc[cheapest_design_idx].item()
design_repair_cost = space_repair_cost.iloc[cheapest_design_idx].item()
design_collapse_risk = space_collapse_risk.iloc[cheapest_design_idx].item()
design_PID = space_drift.iloc[cheapest_design_idx].item()

print(best_design)

print('Upfront cost of selected design: ',
      f'${design_upfront_cost:,.2f}')
print('Predicted mean repair cost: ',
      f'${design_repair_cost:,.2f}')
print('Predicted repair time (sequential): ',
      f'{design_downtime:,.2f}', 'worker-days')
print('Predicted collapse risk: ',
      f'{design_collapse_risk:.2%}')
print('Predicted peak interstory drift: ',
      f'{design_PID:.2%}')

#%% Fit costs (SVR, across all data)

## fit impacted set
#mdl = Prediction(df)
#mdl.set_outcome(cost_var)
#mdl.test_train_split(0.2)
#mdl.fit_svr()
#
#X_plot = mdl.make_2D_plotting_space(100)
#
#xx = mdl.xx
#yy = mdl.yy
#Z = mdl.svr.predict(mdl.X_plot)
#Z = Z.reshape(xx.shape)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
## Plot the surface.
#surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.scatter(df['gapRatio'], df['RI'], df[cost_var],
#           edgecolors='k')
#
#ax.set_xlabel('Gap ratio')
#ax.set_ylabel('Ry')
#ax.set_zlabel('Mean loss ($)')
#ax.set_title('Mean cost predictions across all data (SVR)')
#plt.show()
        

#%% dirty test prediction plots for cost

##plt.close('all')
#plt.figure()
#plt.scatter(mdl_miss.X_test['RI'], mdl_miss.y_test)
#plt.scatter(mdl_miss.X_test['RI'], cost_pred_miss)
#
#plt.figure()
#plt.scatter(mdl_hit.X_test['RI'], mdl_hit.y_test)
#plt.scatter(mdl_hit.X_test['RI'], cost_pred_hit)

#%% Other plotting

# idea for plots

# stats-style qq error plot
# slice the 3D predictions into contours and label important values
# Tm, zeta plots

# tradeoff curve of upfront-x, repair-y
# for validation, present IDA loss distribution as violin plot
