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
df = df.head(785).copy()

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
df = pd.concat([df, df_secondary], axis=0)

#%% fit secondary GP
var_outcome = 'log_var_collapse_prob'
from doe import GP
mdl_var = GP(df_secondary)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_var.set_covariates(covariate_list)
mdl_var.set_outcome(var_outcome)
mdl_var.fit_gpr(kernel_name='rbf_iso', noise_bound=(1e-5, 1e3))
mdl_var.fit_kernel_ridge(kernel_name='rbf')

# gp_obj = mdl_var.gpr._final_estimator

kr_obj = mdl_var.kr
var_scaler = kr_obj[0]
kr_only = kr_obj[-1:]._final_estimator
dual_coef_var = kr_only.dual_coef_
alpha_kr = kr_only.alpha
gamma_kr = kr_only.gamma

#%% KR models for predicting var

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

res = 15
X_plot = make_2D_plotting_space(mdl_var.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

# var_est = mdl_var.gpr.predict(X_plot, return_std=False)
var_est = mdl_var.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

# Z_surf = np.exp(var_est).reshape(xx_pl.shape)
Z_surf = var_est.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary[var_outcome],
            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$c^2(x)$', fontsize=axis_font)
ax.set_title('$T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 15
X_plot = make_2D_plotting_space(mdl_var.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 0.75)
xx = X_plot[xvar]
yy = X_plot[yvar]


# var_est = mdl_var.gpr.predict(X_plot, return_std=False)
var_est = mdl_var.kr.predict(X_plot)

x_pl = np.unique(xx)
y_pl = np.unique(yy)
xx_pl, yy_pl = np.meshgrid(x_pl, y_pl)

# Z_surf = np.exp(var_est).reshape(xx_pl.shape)
Z_surf = var_est.reshape(xx_pl.shape)

ax=fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(xx_pl, yy_pl, Z_surf, cmap='Blues',
                       linewidth=0, antialiased=False, alpha=0.6)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary[var_outcome],
            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$c^2(x)$', fontsize=axis_font)
ax.set_title('$GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()


#%% homecooked GPR, specific to isotropic RBF

# theta = [log(lengthscale), log(nugget_variance)]
class Homecook_Het_GPR:
    
    def __init__(self, X_train, y_train, mdl_var):
        self.X_train = X_train
        self.y_train = y_train
        self.mdl_var = mdl_var
    
    def lml(self, theta, eval_gradient=False):
        import numpy as np
        
        from scipy.linalg import cho_solve, cholesky
        from sklearn.metrics.pairwise import rbf_kernel
        
        nugget_variance_lumped = np.exp(theta[1])
        lengthscale = np.exp(theta[0])
        
        # K does not contain the leading process variance coef
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled_main = scaler.transform(self.X_train)
        K = rbf_kernel(X_train_scaled_main, gamma=1/(2*lengthscale**2))
        
        if eval_gradient:
            from scipy.spatial.distance import pdist, squareform
            dists = pdist(X_train_scaled_main/lengthscale, metric="sqeuclidean")
            K_gradient = (K * squareform(dists))[:, :, np.newaxis]
        
        n = self.X_train.shape[0]
        
        # pass in secondary model
        kr_obj = self.mdl_var.kr
        var_scaler = kr_obj[0]
        kr_only = kr_obj[-1:]._final_estimator
        dual_coef_var = kr_only.dual_coef_
        
        X_train_scaled_var = var_scaler.transform(self.X_train)
        
        K_var = rbf_kernel(X_train_scaled_var, kr_only.X_fit_, gamma=kr_only.gamma)
        y_pr = K_var @ dual_coef_var
        Cs = np.diag(np.exp(y_pr.ravel()))
        
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        L = cholesky(K + nugget_variance_lumped*np.diag(np.full(n,1/n)) @ Cs, 
                     lower=True, check_finite=False)
        
        M = np.diag(np.full(n,1/n)) @ Cs
        
        if self.y_train.ndim == 1:
            self.y_train = self.y_train[:, np.newaxis]
    
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, True), self.y_train, check_finite=False)
    
        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)
        
        if eval_gradient:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            # compute K^-1 of shape (n_samples, n_samples)
            K_inv = cho_solve(
                (L, True), np.eye(K.shape[0]), check_finite=False
            )
            
            inner_term -= K_inv[..., np.newaxis]
            # Since we are interested about the trace of
            # inner_term @ K_gradient, we don't explicitly compute the
            # matrix-by-matrix operation and instead use an einsum. 
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            # the log likehood gradient is the sum-up across the outputs
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
            
        if eval_gradient:
            breakpoint()
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood
    
    
    
    def constrained_optimization(self, obj_func, initial_theta, bounds):
        import scipy
        opt_res = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
        theta_opt, func_min = opt_res.x, opt_res.fun
        return theta_opt, func_min
    
    def fit_primary(self):
        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml_val, grad = self.lml(
                    theta, eval_gradient=True
                )
                return -lml_val, -grad
            else:
                return -self.lml(theta, eval_gradient=False)
        
        optima = [
            (
                self.constrained_optimization(
                    obj_func, [0.0, 0.0], ((1e-8, 1e5), (1e-8, 1e5))
                )
            )
        ]
        optimized_thetas = optima[0]
        self.theta = optimized_thetas[0]

#%%
# var_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
# X_train = df[var_list]
# y_train = df['collapse_prob'].ravel()
# gph_mdl = Homecook_Het_GPR(X_train, y_train, mdl_var)

# gph_mdl.fit_primary()

#%% fit primary model
from doe import GP
mdl_main = GP(df)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_main.set_covariates(covariate_list)
mdl_main.set_outcome('mean_collapse_prob')

mdl_main.fit_het_gpr(kernel_name='rbf_iso', 
                     mdl_var=mdl_var)

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

res = 15
X_plot = make_2D_plotting_space(mdl_main.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

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
ax.set_zlim([0, 0.3])
ax.set_title('GP-het: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

res = 15
X_plot = make_2D_plotting_space(mdl_main.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 1.0, fourth_var_set = 0.75)
xx = X_plot[xvar]
yy = X_plot[yvar]


prob_collapse, prob_std = mdl_main.gpr_het.predict(X_plot, return_std=True)

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
ax.set_zlim([0, 0.3])
fig.tight_layout()

gp_het = mdl_main.gpr_het[1]

# %% baseline

from doe import GP
mdl_base = GP(df)
covariate_list = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
mdl_base.set_covariates(covariate_list)
mdl_base.set_outcome('collapse_prob')

mdl_base.fit_gpr(kernel_name='rbf_iso')

#%% baseline plots

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

res = 15
X_plot = make_2D_plotting_space(mdl_base.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_base.gpr.predict(X_plot, return_std=True)

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

# ax.scatter(df_secondary[xvar], df_secondary[yvar], df_secondary['log_var_collapse_prob'],
#            edgecolors='k', alpha = 0.7)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('Gap ratio', fontsize=axis_font)
ax.set_ylabel('$R_y$', fontsize=axis_font)
ax.set_zlabel('$\mu(x) + \sigma(x)$', fontsize=axis_font)
ax.set_zlim([0, 0.3])
ax.set_title('GP normal: $T_M/T_{fb} = 3.0$, $\zeta_M = 0.15$', fontsize=subt_font)

#################################
xvar = 'T_ratio'
yvar = 'zeta_e'

X_plot = make_2D_plotting_space(mdl_base.X, res, x_var=xvar, y_var=yvar, 
                            all_vars=['gap_ratio', 'RI', 'T_ratio', 'zeta_e'],
                            third_var_set = 3.0, fourth_var_set = 0.15)
xx = X_plot[xvar]
yy = X_plot[yvar]

prob_collapse, prob_std = mdl_base.gpr.predict(X_plot, return_std=True)

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

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='x', offset=xlim[0], cmap='Blues_r')
cset = ax.contour(xx_pl, yy_pl, Z_surf, zdir='y', offset=ylim[1], cmap='Blues')

ax.set_xlabel('$T_M/ T_{fb}$', fontsize=axis_font)
ax.set_ylabel('$\zeta_M$', fontsize=axis_font)
ax.set_zlabel('$\mu(x) + \sigma(x)$', fontsize=axis_font)
ax.set_zlim([0, 0.3])
ax.set_title('GP normal: $GR = 1.0$, $R_y = 2.0$', fontsize=subt_font)
fig.tight_layout()

gp_hom = mdl_base.gpr[1]