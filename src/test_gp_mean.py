pickle_path = '../data/'

import pandas as pd
import pickle

with open(pickle_path+"tfp_mf_db.pickle", 'rb') as picklefile:
    main_obj = pickle.load(picklefile)
    
main_obj.calculate_collapse(drift_mu_plus_std=0.1)

df = main_obj.ops_analysis

import numpy as np
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
df['Bm'] = np.interp(df['zeta_e'], zetaRef, BmRef)

pi = 3.14159
g = 386.4
df['T_ratio'] = df['T_m'] / df['T_fb']
df['gap_ratio'] = (df['constructed_moat']*4*pi**2)/ \
    (g*(df['sa_tm']/df['Bm'])*df['T_m']**2)

df_train = df.head(df.shape[0]//2)
df_test = df.tail(df.shape[0]//2)

from doe import GP
mdl = GP(df_train)

outcome = 'collapse_prob'
mdl.set_outcome(outcome)

covariate_columns = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']

mdl.set_covariates(covariate_columns)

# https://stackoverflow.com/questions/54951735/how-to-specify-the-prior-for-scikit-learns-gaussian-process-regression
mdl.fit_linear()
# mdl.fit_gpr_mean_fcn(kernel_name='rbf_iso')
mdl.fit_gpr(kernel_name='rbf_iso')

X_test = df_test[covariate_columns]
# y_pred, y_std = mdl.predict_gpr_mean_fcn(X_test)
y_true = df_test[outcome].to_numpy()
y_gpr, y_gpr_std = mdl.gpr.predict(X_test, return_std=True)
compare = pd.DataFrame({'true':y_true, 
                        'predicted_wo_mean': y_gpr})

gp_mdl = mdl.gpr._final_estimator
theta = gp_mdl.kernel_.theta

#%% 
import matplotlib.pyplot as plt
plt.close('all')
X_pl = pd.DataFrame({'gap_ratio':np.linspace(0.5, 3.0, 1000),
                     'RI':np.repeat(2.0, 1000),
                     'T_ratio':np.repeat(3.0, 1000),
                     'zeta_e':np.repeat(0.15, 1000)})

# y_pl_w_mean, y_pl_w_mean_std = mdl.predict_gpr_mean_fcn(X_pl)
y_pl_gpr, y_pl_gpr_std = mdl.gpr.predict(X_pl, return_std=True)

plt.figure()
# plt.plot(X_pl.zeta_e, y_pl_w_mean, 'k', lw=3, zorder=9, label='predicted_w_mean')
plt.plot(X_pl.gap_ratio, y_pl_gpr, 'b', lw=3, zorder=9, label='predicted_wo_mean')
plt.scatter(mdl.X.gap_ratio, mdl.y)
plt.fill_between(X_pl.gap_ratio, y_pl_gpr - y_pl_gpr_std,
                  y_pl_gpr + y_pl_gpr_std,
                  alpha=0.5, color='k', label='+-sigma')
# plt.xlim([0.5, 2.0])
plt.ylim([-0.05, 0.3])

#%%
pickle_path = '../data/'

import pandas as pd
import pickle

with open(pickle_path+"tfp_mf_db_stack.pickle", 'rb') as picklefile:
    main_obj_stack = pickle.load(picklefile)
    
main_obj_stack.calculate_collapse(drift_mu_plus_std=0.1)

#%% secondary GP to predict variance
stack_df = main_obj_stack.ops_analysis

import numpy as np
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
stack_df['Bm'] = np.interp(stack_df['zeta_e'], zetaRef, BmRef)

pi = 3.14159
g = 386.4
stack_df['T_ratio'] = stack_df['T_m'] / stack_df['T_fb']
stack_df['gap_ratio'] = (stack_df['constructed_moat']*4*pi**2)/ \
    (g*(stack_df['sa_tm']/stack_df['Bm'])*stack_df['T_m']**2)
    
stack_df['gap_ratio'] = stack_df['gap_ratio'].apply(pd.to_numeric)
group_var = stack_df.groupby(stack_df['index']).agg(["mean", "median", "min", "var"])
replicates_df = stack_df.groupby('index').first().drop(columns=['gap_ratio'])
replicates_df['gap_ratio'] = group_var[('gap_ratio', 'min')]
replicates_df['var_outcome'] = group_var[(outcome, 'var')]
replicates_df['log_var_outcome'] = np.log(replicates_df['var_outcome'])

plt.figure()
# plt.plot(X_pl.zeta_e, y_pl_w_mean, 'k', lw=3, zorder=9, label='predicted_w_mean')
plt.scatter(replicates_df.gap_ratio, replicates_df.log_var_outcome)

from doe import GP
mdl_var = GP(replicates_df)
mdl_var.set_outcome('log_var_outcome')
mdl_var.set_covariates(covariate_columns)
mdl_var.fit_gpr(kernel_name='rbf_iso')
#%%
import matplotlib.pyplot as plt

y_pl_gpr = mdl.gpr.predict(X_pl, return_std=False)
y_pl_log_var = mdl_var.gpr.predict(X_pl)
y_pl_var = np.exp(y_pl_log_var)
y_pl_std = y_pl_var**0.5

plt.figure()
# plt.plot(X_pl.zeta_e, y_pl_w_mean, 'k', lw=3, zorder=9, label='predicted_w_mean')
plt.plot(X_pl.gap_ratio, y_pl_gpr, 'b', lw=3, zorder=9, label='predicted_wo_mean')
plt.scatter(mdl.X.gap_ratio, mdl.y)
plt.fill_between(X_pl.gap_ratio, y_pl_gpr - y_pl_std,
                  y_pl_gpr + y_pl_std,
                  alpha=0.5, color='k', label='+-sigma')
plt.xlim([0.5, 2.0])
# plt.ylim([-0.05, 0.3])