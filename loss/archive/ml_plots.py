############################################################################
#               ML plotter

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  plots for IALCCE paper

# Open issues:  

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
#    land_area = (90*12 + moat_gap)**2
    land_cost = land_cost_per_sqft/144.0 * land_area
    
    return(steel_cost + land_cost)

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
cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='gist_yarg', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax1.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax1.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='black', marker='v', edgecolors='k', label='Impacted')

ax1.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='lightgray', edgecolors='k', label='No impact')

ax1.set_title(r'$T_M = 3.24$ s, $\zeta_M = 0.155$', fontsize=subt_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
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
cs = ax2.contour(xx, yy, Z, linewidths=1.1, cmap='gist_yarg', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax2.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax2.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='black', marker='v', edgecolors='k', label='Impacted')

ax2.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='lightgray', edgecolors='k', label='No impact')
ax2.set_title(r'$R_y= 1.22$ , $\zeta_M = 0.155$', fontsize=subt_font)
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
cs = ax3.contour(xx, yy, Z, linewidths=1.1, cmap='gist_yarg', vmin=-1,
                 levels=np.linspace(0.1,1.0,num=10))
ax3.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax3.scatter(hit.X_train[xvar][:plt_density],
            hit.X_train[yvar][:plt_density],
            s=30, c='black', marker='v', edgecolors='k', label='Impacted')

ax3.scatter(miss.X_train[xvar][:plt_density],
            miss.X_train[yvar][:plt_density],
            s=30, c='lightgray', edgecolors='k', label='No impact')

# sc = ax3.scatter(mdl.X_train[xvar][:plt_density],
#             mdl.X_train[yvar][:plt_density],
#             s=30, c=mdl.y_train[:plt_density],
#             cmap=plt.cm.copper, edgecolors='w')

ax3.set_title(r'$R_y= 1.22$ , $T_M = 3.24$ s', fontsize=subt_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax3.set_ylabel(r'$\zeta_M$', fontsize=axis_font)

ax3.legend(loc="lower right", fontsize=subt_font)

# lg = ax3.legend(*sc.legend_elements(), loc="lower right", title="Impact",
#            fontsize=subt_font)


# lg.get_title().set_fontsize(axis_font) #legend 'Title' fontsize

fig.tight_layout()
plt.show()

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
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax1)
sns.stripplot(x='impacted', y=cost_var, data=df, ax=ax1,
              color='black', jitter=True)
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Cost [USD]', fontsize=axis_font)
ax1.set_xlabel('Impact', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y=time_var, x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.6, ax=ax2)
sns.stripplot(x='impacted', y=time_var, data=df, ax=ax2,
              color='black', jitter=True)
ax2.set_title('Median sequential repair time', fontsize=subt_font)
ax2.set_ylabel('Time [worker-day]', fontsize=axis_font)
ax2.set_xlabel('Impact', fontsize=axis_font)
ax2.set_yscale('log')

sns.boxplot(y="replacement_freq", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, meanprops={'color': 'black'},
            width=0.5, ax=ax3)
sns.stripplot(x='impacted', y='replacement_freq', data=df, ax=ax3,
              color='black', jitter=True)
ax3.set_title('Replacement frequency', fontsize=subt_font)
ax3.set_ylabel('Replacement frequency', fontsize=axis_font)
ax3.set_xlabel('Impact', fontsize=axis_font)
fig.tight_layout()

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

#plt.close('all')

fig = plt.figure(figsize=(13, 8))

#plt.setp((ax1, ax2), xticks=np.arange(0.5, 4.0, step=0.5),
#        yticks=np.arange(0.5, 2.5, step=0.5))


#################################
xvar = 'gapRatio'
yvar = 'RI'

res = 100
step = 0.01
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

ax1=fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(xx, yy, Z, cmap=plt.cm.gist_gray,
                       linewidth=0, antialiased=False, alpha=0.4)

ax1.scatter(df[xvar], df[yvar], df[cost_var]/8.1e6, color='white',
           edgecolors='k', alpha = 0.5)

xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()
cset = ax1.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.gist_gray)
cset = ax1.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.gist_gray)
cset = ax1.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.gist_gray)

ax1.set_xlabel('Gap ratio', fontsize=axis_font)
ax1.set_ylabel('$R_y$', fontsize=axis_font)
#ax1.set_zlabel('Median loss ($)', fontsize=axis_font)
ax1.set_title('a) Cost: GPC-SVR', fontsize=subt_font)

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

ax2=fig.add_subplot(2, 2, 2, projection='3d')
surf = ax2.plot_surface(xx, yy, Z, cmap=plt.cm.gist_gray,
                       linewidth=0, antialiased=False, alpha=0.4)

ax2.scatter(df[xvar], df[yvar], df[cost_var]/8.1e6, color='white',
           edgecolors='k', alpha = 0.5)

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
zlim = ax2.get_zlim()
cset = ax2.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.gist_gray)
cset = ax2.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.gist_gray)
cset = ax2.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.gist_gray)

ax2.set_xlabel('Gap ratio', fontsize=axis_font)
ax2.set_ylabel('$R_y$', fontsize=axis_font)
ax2.set_zlabel('% of replacement cost', fontsize=axis_font)
ax2.set_title('b) Cost: GPC-KR', fontsize=subt_font)

fig.tight_layout()

#################################
xvar = 'gapRatio'
yvar = 'RI'

X_plot = mdl.make_2D_plotting_space(100)

grid_downtime = predict_DV(X_plot,
                          mdl.gpc,
                          mdl_time_hit.svr,
                          mdl_time_miss.svr,
                          outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_downtime)/4764.71
Z = zz.reshape(xx.shape)

ax3=fig.add_subplot(2, 2, 3, projection='3d')
surf = ax3.plot_surface(xx, yy, Z, cmap=plt.cm.gist_gray,
                       linewidth=0, antialiased=False, alpha=0.4)

ax3.scatter(df[xvar], df[yvar], df[time_var]/4764.71, color='white',
           edgecolors='k', alpha = 0.5)

xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
zlim = ax3.get_zlim()
cset = ax3.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.gist_gray)
cset = ax3.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.gist_gray)
cset = ax3.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.gist_gray)

ax3.set_xlabel('Gap ratio', fontsize=axis_font)
ax3.set_ylabel('$R_y$', fontsize=axis_font)
ax3.set_title('c) Time: GPC-SVR', fontsize=subt_font)

fig.tight_layout()

#################################
xvar = 'gapRatio'
yvar = 'RI'

X_plot = mdl.make_2D_plotting_space(100)

grid_downtime = predict_DV(X_plot,
                          mdl.gpc,
                          mdl_time_hit.kr,
                          mdl_time_miss.kr,
                          outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_downtime)/4764.71
Z = zz.reshape(xx.shape)

ax4=fig.add_subplot(2, 2, 4, projection='3d')
surf = ax4.plot_surface(xx, yy, Z, cmap=plt.cm.gist_gray,
                       linewidth=0, antialiased=False, alpha=0.4)

ax4.scatter(df[xvar], df[yvar], df[time_var]/4764.71, color='white',
           edgecolors='k', alpha = 0.5)

xlim = ax4.get_xlim()
ylim = ax4.get_ylim()
zlim = ax4.get_zlim()
cset = ax4.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.gist_gray)
cset = ax4.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.gist_gray)
cset = ax4.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.gist_gray)

ax4.set_xlabel('Gap ratio', fontsize=axis_font)
ax4.set_ylabel('$R_y$', fontsize=axis_font)
ax4.set_zlabel('% of replacement time', fontsize=axis_font)
ax4.set_title('d) Time: GPC-KR', fontsize=subt_font)

fig.tight_layout()

#%% Big cost prediction plot (GP-SVR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 0.8])

plt.setp((ax1, ax2, ax3),
         yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 0.8])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
#ax1.scatter(df[xvar], df[cost_var]/8.1e6, c=df[yvar],
#           edgecolors='k', cmap='copper')
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('% of replacement cost', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid(visible=True)

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_title('a) Repair cost (GPC-SVR)', fontsize=subt_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid(visible=True)

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid(visible=True)

lines = [ cs.collections[0]]
labels = ['Gap ratios']
ax3.legend(lines, labels, fontsize=label_size)


plt.show()
fig.tight_layout()

#%% Big cost prediction plot (GP-SVR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

# plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
#plt.setp((ax1, ax2, ax3), yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 1.0])

fig, axes = plt.subplots(2, 3, 
     figsize=(13, 9), sharey=True)
ax1 = axes[0][0]
ax2 = axes[0][1]
ax3 = axes[0][2]
ax4 = axes[1][0]
ax5 = axes[1][1]
ax6 = axes[1][2]
plt.setp((ax1, ax2, ax3, ax4, ax5, ax6),
         yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 0.8])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)
#ax1.scatter(df[xvar], df[cost_var]/8.1e6, c=df[yvar],
#           edgecolors='k', cmap='copper')
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('% of replacement cost', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid(visible=True)

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_title('a) Repair cost (GPC-SVR)', fontsize=subt_font)
ax2.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax2.grid(visible=True)

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$R_y$', fontsize=axis_font)
ax3.grid(visible=True)

# lines = [ cs.collections[0]]
# labels = ['Gap ratios']
# ax3.legend(lines, labels, fontsize=label_size)

# dummy legend
ax3.plot(0.65, 0.5, color='gray', label='Gap ratio')
ax3.legend(fontsize=label_size)


#plt.show()
#fig.tight_layout()
##############################

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax4.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)
ax4.clabel(cs, fontsize=label_size)
ax4.set_ylabel('% of replacement time', fontsize=axis_font)
ax4.set_xlabel('$T_M$', fontsize=axis_font)
ax4.grid(visible=True)

xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax5.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)

ax5.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax5.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax5.set_title('b) Downtime (GPC-KR)', fontsize=subt_font)
ax5.grid(visible=True)

xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax6.contour(xx, Z, yy, linewidths=1.1, cmap='gist_yarg',
                 levels=np.arange(0.9, 2.0, step=0.1), vmin=0)
ax6.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax6.set_xlabel('$R_y$', fontsize=axis_font)
ax6.grid(visible=True)

# dummy legend
ax6.plot(0.65, 0.5, color='gray', label='Gap ratio')
ax6.legend(fontsize=label_size)
# ax6.legend(lines, labels, fontsize=label_size)

plt.show()
fig.tight_layout()

#%% Big downtime prediction plot (GP-KR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

#plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 1.0])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('% of replacement time', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid(visible=True)

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid(visible=True)

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid(visible=True)

lines = [ cs.collections[0]]
labels = ['Gap ratios']
ax3.legend(lines, labels, fontsize=label_size)

plt.show()
fig.tight_layout()

#%% Big collapse risk prediction plot (GP-OR)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

#plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

from scipy.stats import lognorm
from math import log, exp

xx = mdl.xx
yy = mdl.yy

beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*0.9945)
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.02, 0.22, step=0.02), ylim=[0.0, 0.2])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('Collapse risk', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid(visible=True)

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

xx = mdl.xx
yy = mdl.yy

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid(visible=True)

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.9, 0.9+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

xx = mdl.xx
yy = mdl.yy

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.9, 2.0, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid(visible=True)

lines = [ cs.collections[0]]
labels = ['Gap ratios']
ax3.legend(lines, labels, fontsize=label_size, loc='upper left')

plt.show()
fig.tight_layout()

#%% read out results
val_dir='./results/tfp_mf_val/validation/'
baseline_dir='./results/tfp_mf_val/baseline/'
df_val = pd.read_csv(val_dir+'loss_estimate_data.csv', index_col=None)
df_base = pd.read_csv(baseline_dir+'loss_estimate_data.csv', index_col=None)

steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)
designs = pd.DataFrame({'gapRatio': [1.383, 1.000],
                           'Tm': [3.93, 3.0],
                           'zetaM': [0.1999, 0.15],
                           'RI': [2.0, 2.0]})
    
upfront_costs = calc_upfront_cost(designs, coef_dict)
upfront_costs = upfront_costs.values
design_repair_cost = df_val['cost_mean'][2]
design_repair_cost_med = df_val['cost_50%'][2]
design_downtime = df_val['time_u_mean'][2]
design_downtime_med = df_val['time_u_50%'][2]
design_collapse_risk = df_val['collapse_freq'][2]
design_replacement_risk = df_val['replacement_freq'][2]
design_upfront_cost = upfront_costs[0]

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
print('Upfront cost: ',
      f'${design_upfront_cost:,.2f}')

baseline_repair_cost = df_base['cost_mean'][2]
baseline_repair_cost_med = df_base['cost_50%'][2]
baseline_downtime = df_base['time_u_mean'][2]
baseline_downtime_med = df_base['time_u_50%'][2]
baseline_collapse_risk = df_base['collapse_freq'][2]
baseline_replacement_risk = df_base['replacement_freq'][2]
baseline_upfront_cost = upfront_costs[1]

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
print('Upfront cost: ',
      f'${baseline_upfront_cost:,.2f}')

# The discrepancy in mean and median cost as compared to the training dataset
# could come from the fact that in validation runs, EDPs are specifically set
# to be generated from the sample in lognormal distribution, whereas they are 
# treated as deterministic samples in the training dataset.
#%% cost tradeoff curve
'''
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 



def make_1d_space(xvar='gapRatio', res=100):
    x_min = min(mdl.X[xvar])
    x_max = max(mdl.X[xvar])
    xx = np.linspace(x_min, x_max, num=res)
    
    if xvar == 'gapRatio':
        yvar = 'RI'
        tvar = 'Tm'
        uvar = 'zetaM'
    elif xvar == 'RI':
        yvar = 'gapRatio'
        tvar = 'Tm'
        uvar = 'zetaM'
    elif xvar == 'Tm':
        yvar = 'gapRatio'
        tvar = 'RI'
        uvar = 'zetaM'
    else:
        yvar = 'gapRatio'
        tvar = 'RI'
        uvar = 'Tm'
    
    X_pl = pd.DataFrame({xvar:xx,
                         yvar:np.repeat(mdl.X[yvar].median(), res), 
                         tvar:np.repeat(mdl.X[tvar].median(), res),
                         uvar:np.repeat(mdl.X[uvar].median(), res)})
    X_pl = X_pl[['gapRatio', 'RI', 'Tm', 'zetaM']]
    return(X_pl)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4),
      sharey=True)

xvar = 'gapRatio'
X_pl = make_1d_space(xvar, res=100)

steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

upfront_costs = calc_upfront_cost(X_pl, coef_dict)
upfront_costs = upfront_costs.values
repair_costs = predict_DV(X_pl, mdl.gpc,
                          mdl_hit.svr,
                          mdl_miss.svr,
                          outcome=cost_var)

ax1.plot(X_pl[xvar], upfront_costs,
         c='black', linestyle='--', label='Upfront cost')
ax1.plot(X_pl[xvar], repair_costs[cost_var+'_pred'],
         c='black', linestyle='-.', label='Repair cost')
ax1.plot(X_pl[xvar], repair_costs[cost_var+'_pred']+upfront_costs,
         c='black', label='Total cost')

ax1.set_ylabel('Cost [USD]', fontsize=axis_font)
ax1.set_xlabel('Gap ratio', fontsize=axis_font)

##########################
xvar='RI'
X_pl = make_1d_space(xvar, res=100)

steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

upfront_costs = calc_upfront_cost(X_pl, coef_dict)
upfront_costs = upfront_costs.values
repair_costs = predict_DV(X_pl, mdl.gpc,
                          mdl_hit.svr,
                          mdl_miss.svr,
                          outcome=cost_var)

ax2.plot(X_pl[xvar], upfront_costs,
         c='black', linestyle='--', label='Upfront cost')
ax2.plot(X_pl[xvar], repair_costs[cost_var+'_pred'],
         c='black', linestyle='-.', label='Repair cost')
ax2.plot(X_pl[xvar], repair_costs[cost_var+'_pred']+upfront_costs,
         c='black', label='Total cost')

ax2.set_xlabel('$R_y$', fontsize=axis_font)

##########################
xvar='Tm'
X_pl = make_1d_space(xvar, res=100)

steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

upfront_costs = calc_upfront_cost(X_pl, coef_dict)
upfront_costs = upfront_costs.values
repair_costs = predict_DV(X_pl, mdl.gpc,
                          mdl_hit.svr,
                          mdl_miss.svr,
                          outcome=cost_var)

ax3.plot(X_pl[xvar], upfront_costs,
         c='black', linestyle='--', label='Upfront cost')
ax3.plot(X_pl[xvar], repair_costs[cost_var+'_pred'],
         c='black', linestyle='-.', label='Repair cost')
ax3.plot(X_pl[xvar], repair_costs[cost_var+'_pred']+upfront_costs,
         c='black', label='Total cost')

ax3.set_xlabel('$T_M$', fontsize=axis_font)

##########################
xvar='zetaM'
X_pl = make_1d_space(xvar, res=100)

steel_price = 2.00
coef_dict = get_steel_coefs(df, steel_per_unit=steel_price)

upfront_costs = calc_upfront_cost(X_pl, coef_dict)
upfront_costs = upfront_costs.values
repair_costs = predict_DV(X_pl, mdl.gpc,
                          mdl_hit.svr,
                          mdl_miss.svr,
                          outcome=cost_var)

ax4.plot(X_pl[xvar], upfront_costs,
         c='black', linestyle='--', label='Upfront cost')
ax4.plot(X_pl[xvar], repair_costs[cost_var+'_pred'],
         c='black', linestyle='-.', label='Repair cost')
ax4.plot(X_pl[xvar], repair_costs[cost_var+'_pred']+upfront_costs,
         c='black', label='Total cost')
ax4.legend(loc='upper right')

ax4.set_xlabel('$\zeta_M$', fontsize=axis_font)
'''
 

#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
import SeabornFig2Grid as sfg

agg = pd.read_csv('./results/val_agg.csv', header=[0,1])
agg.columns = ['cost', 'time_l', 'time_u']
agg_pl = agg.loc[agg['cost'] < 8e6]

print('============ INVERSE DESIGN ============')
agg_test = agg.loc[agg['cost'] < (0.2*8.1e6)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)
agg_test = agg.loc[agg['time_u'] < (700)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)

# plt.close('all')

g1 = sns.JointGrid(data=agg_pl, x='cost', y='time_u')
g1.plot_joint(sns.scatterplot, color='black', alpha = 0.1)

g1.plot_marginals(sns.kdeplot, color='black')

for ax in (g1.ax_joint, g1.ax_marg_x):
    ax.axvline(1.62e6, color='black', ls='--', lw=2)
    
for ax in (g1.ax_joint, g1.ax_marg_y):
    ax.axhline(700, color='black', ls='--', lw=2)
    
g1.ax_joint.annotate('Cost limit', xy=(1.7e6, 2750), fontsize=label_size)
g1.ax_joint.annotate('Downtime limit', xy=(2e6, 750), fontsize=label_size)

g1.ax_joint.grid(visible=True)
g1.ax_joint.set_xlim(-0.25e6, 3.5e6)
g1.ax_joint.set_ylim(-100, 3e3)
g1.set_axis_labels(xlabel='Repair cost [USD]',
                  ylabel='Sequential repair time [worker-days]',
                  fontsize=16)

g1.ax_marg_x.set_axis_off()
g1.ax_marg_y.set_axis_off()
g1.ax_marg_x.set_title('a) Inverse design structure', fontsize=16)
g1.figure.tight_layout()

agg = pd.read_csv('./results/baseline_agg.csv', header=[0,1])
agg.columns = ['cost', 'time_l', 'time_u']
agg_pl = agg.loc[agg['cost'] < 8e6]

print('============ BASELINE DESIGN ============')
agg_test = agg.loc[agg['cost'] < (0.2*8.1e6)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)
agg_test = agg.loc[agg['time_u'] < (700)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)

g2 = sns.JointGrid(data=agg_pl, x='cost', y='time_u')
g2.plot_joint(sns.scatterplot, color='black', alpha = 0.1)
g2.plot_marginals(sns.kdeplot, color='black')

for ax in (g2.ax_joint, g2.ax_marg_x):
    ax.axvline(1.62e6, color='black', ls='--', lw=2)
    
for ax in (g2.ax_joint, g2.ax_marg_y):
    ax.axhline(700, color='black', ls='--', lw=2)
    
g2.ax_joint.annotate('Cost limit', xy=(1.7e6, 2750), fontsize=label_size)
g2.ax_joint.annotate('Downtime limit', xy=(2e6, 750), fontsize=label_size)
    
g2.ax_joint.grid(visible=True)
g2.ax_joint.set_xlim(-0.25e6, 3.5e6)
g2.ax_joint.set_ylim(-100, 3e3)
g2.set_axis_labels(xlabel='Repair cost [USD]',
                  # ylabel='Sequential repair time [worker-days]',
                  fontsize=16)

g2.ax_marg_x.set_axis_off()
g2.ax_marg_y.set_axis_off()
g2.ax_marg_x.set_title('b) Baseline structure', fontsize=16)
g2.figure.tight_layout()

fig = plt.figure(figsize=(13,6))
gs = gridspec.GridSpec(1, 2)

mg0 = sfg.SeabornFig2Grid(g1, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g2, fig, gs[1])

gs.tight_layout(fig)
gs.update(top=0.65, right=0.7)

plt.show()
plt.savefig('./val_joined.eps')

# g1.savefig('g1.png', format='png', dpi=1200)
# plt.close(g1.fig)

# g2.savefig('g2.png', format='png', dpi=1200)
# plt.close(g2.fig)

# ############### 3. CREATE YOUR SUBPLOTS FROM TEMPORAL IMAGES
# import matplotlib.image as mpimg
# f, axarr = plt.subplots(1, 2, figsize=(14, 9))

# axarr[0].imshow(mpimg.imread('g1.png'))
# axarr[1].imshow(mpimg.imread('g2.png'))

# # turn off x and y axis
# [ax.set_axis_off() for ax in axarr.ravel()]

# plt.tight_layout()
# plt.show()

#%%
agg = pd.read_csv('./results/val_agg.csv', header=[0,1])
agg.columns = ['cost', 'time_l', 'time_u']
agg_pl = agg.loc[agg['cost'] < 8e6]

print('============ INVERSE DESIGN ============')
agg_test = agg.loc[agg['cost'] < (0.2*8.1e6)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)
agg_test = agg.loc[agg['time_u'] < (700)]
print('Percent of realizations below cost limit: ', len(agg_test)/10000)

# plt.close('all')
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

sns.jointplot(data=agg_pl, x="cost", y="time_u")