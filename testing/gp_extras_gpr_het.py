import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None  

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

#%%
plt.close('all')

main_obj = pd.read_pickle("../data/tfp_mf_db_doe_prestrat.pickle")

# with open("../../data/tfp_mf_db.pickle", 'rb') as picklefile:
#     main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse()

df = main_obj.doe_analysis
df = df.reset_index(drop=True)


df = df.drop(columns=['index'])
df = df.head(400).copy()

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
    
#%%
stack_obj = pd.read_pickle("../data/tfp_mf_db_stack.pickle")

stack_obj.calculate_collapse()

df_stack = stack_obj.ops_analysis
df_stack = df_stack.reset_index(drop=True)

df_stack = df_stack.drop(columns=['index'])

df_stack['max_drift'] = df_stack.PID.apply(max)
df_stack['log_drift'] = np.log(df_stack['max_drift'])

df_stack['max_velo'] = df_stack.PFV.apply(max)
df_stack['max_accel'] = df_stack.PFA.apply(max)

df_stack['T_ratio'] = df_stack['T_m'] / df_stack['T_fb']
df_stack['T_ratio_e'] = df_stack['T_m'] / df_stack['T_fbe']
pi = 3.14159
g = 386.4

zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df_stack['Bm'] = np.interp(df_stack['zeta_e'], zetaRef, BmRef)

df_stack['gap_ratio'] = (df_stack['constructed_moat']*4*pi**2)/ \
    (g*(df_stack['sa_tm']/df_stack['Bm'])*df_stack['T_m']**2)
    
#%% assign group, calculate log of variance of collapse prob

df_stack['gid'] = (df_stack.groupby(['S_1']).cumcount()==0).astype(int)
df_stack['gid'] = df_stack['gid'].cumsum()

df_stack = df_stack.join(df_stack.groupby(['gid'])['collapse_prob'].var(), on='gid', rsuffix='_r')

df_stack.rename(columns={'collapse_prob_r':'var_collapse_prob'}, inplace=True)
from numpy import log
df_stack['log_var_collapse_prob'] = log(df_stack['var_collapse_prob'])

df_stack['gid'] = (df_stack.groupby(['S_1']).cumcount()==0).astype(int)
df_stack['gid'] = df_stack['gid'].cumsum()
df_stack = df_stack.join(df_stack.groupby(['gid'])['collapse_prob'].mean(), on='gid', rsuffix='_r')
df_stack.rename(columns={'collapse_prob_r':'mean_collapse_prob'}, inplace=True)

df_secondary = df_stack.iloc[::11, :]
df_secondary = df_secondary.drop(columns=['gid'])

df['mean_collapse_prob'] = df['collapse_prob']
df = pd.concat([df, df_stack], axis=0)

#%% gpr extras
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_extras.kernels import HeteroscedasticKernel
from sklearn.gaussian_process import GaussianProcessRegressor

var_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
X_train = df[var_list]
y_train = df['collapse_prob'].ravel()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X = scaler.transform(X_train)

# grab bottom 100 to estimate noise from
X_prototype = scaler.transform(df_secondary[var_list])

#%%
# Gaussian Process with RBF kernel and heteroscedastic noise level
kernel_hetero = C(1.0, (1e-8, 1e8)) * RBF(1, (1e-8, 1e8)) \
    + HeteroscedasticKernel.construct(X_prototype, 1e-3, (1e-10, 1e2),
                                      gamma=5.0, gamma_bounds="fixed")
gp_het = GaussianProcessRegressor(kernel=kernel_hetero, alpha=0)
gp_het.fit(X, y_train)
print("Heteroscedastic kernel: %s" % gp_het.kernel_)
print("Heteroscedastic LML: %.3f" % gp_het.log_marginal_likelihood(gp_het.kernel_.theta))

#%% GP-het plots
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

res = 30
X_plot = make_2D_plotting_space(X_train, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = gp_het.predict(X_plot, return_std=True)
# prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

pred = prob_collapse + prob_std
Z_surf = pred.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.scatter(df[xvar], df[yvar], df['collapse_prob'],
#             edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$\mu(x) + \sigma(x)$', fontsize=axis_font)
# ax.set_zlim([0, 0.3])
ax.set_title('GP-het: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

X_plot = make_2D_plotting_space(X_train, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = gp_het.predict(X_plot, return_std=True)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

pred = prob_collapse + prob_std
Z_surf = pred.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ax.scatter(df[xvar], df[yvar], df['collapse_prob'],
#             edgecolors='k', alpha = 0.7)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$\mu(x) + \sigma(x)$', fontsize=axis_font)
ax.set_title('GP-het: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
# ax.set_zlim([0, 0.3])
fig.tight_layout()
