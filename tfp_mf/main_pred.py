from doe import GP
import pandas as pd

print('======= starting at 200 points =======')
df = pd.read_csv('./data/doe_init.csv')
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df.loc[df['collapsed'] == -1, 'collapsed'] = 0

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
# 0.9945 is inverse normCDF of 0.84
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) 
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
df['collapse_prob'] = ln_dist.cdf(df['max_drift'])

import numpy as np
df['col_thresh'] = 0
df['col_thresh'] = np.where(df["max_drift"] > mean_log_drift, 1,
                                 df["col_thresh"])

mdl = GP(df)
mdl.set_outcome('collapse_prob')
mdl.fit_gpr(kernel_name='rbf_ard')


rsc = mdl.gpr.score(mdl.X, mdl.y)
print('R-squared :', rsc)
y_hat = mdl.gpr.predict(mdl.X)

df['col_thresh_pred'] = 0
df['col_thresh_pred'] = np.where(y_hat > mean_log_drift, 1,
                                   df["col_thresh_pred"])

from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(df['col_thresh'], df['col_thresh_pred'])
bas = balanced_accuracy_score(df['col_thresh'], df['col_thresh_pred'])
print('F1 score: %.3f' % f1)
print('Balanced accuracy score: %.3f' % bas)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
mse = mean_squared_error(mdl.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

y_true = np.array(mdl.y).ravel()
mae = mean_absolute_error(mdl.y, y_hat)
print('Mean absolute error: %.3f' % mae)

# ame is average error in predicting collapse risk (expressed in relative %)
# denominator is 0.001 to prevent extreme outliers
# mape = np.abs(y_true-y_hat)/np.maximum(np.abs(y_true), 1e-3)
# ame = np.average(mape, axis=0)
# print('Average mean error (%%): %.2f' % ame)

print('======= after doe (~630 points) =======')

df_doe = pd.read_csv('./data/doe/mik_smrf_doe.csv')
mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

import numpy as np
df_doe['col_thresh'] = 0
df_doe['col_thresh'] = np.where(df_doe["max_drift"] > mean_log_drift, 1,
                                     df_doe["col_thresh"])

rsc = mdl_doe.gpr.score(mdl_doe.X, mdl_doe.y)
print('R-squared :', rsc)
y_hat = mdl_doe.gpr.predict(mdl_doe.X)


df_doe['col_thresh_pred'] = 0
df_doe['col_thresh_pred'] = np.where(y_hat > mean_log_drift, 1,
                                   df_doe["col_thresh_pred"])

from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(df_doe['col_thresh'], df_doe['col_thresh_pred'])
bas = balanced_accuracy_score(df_doe['col_thresh'], df_doe['col_thresh_pred'])
print('F1 score: %.3f' % f1)
print('Balanced accuracy score: %.3f' % bas)

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(mdl_doe.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

y_true = np.array(mdl_doe.y).ravel()
mae = mean_absolute_error(mdl_doe.y, y_hat)
print('Mean absolute error: %.3f' % mae)

compare = pd.DataFrame([y_hat, y_true]).T
compare.columns = ['predicted %', 'true %']

print('======= testing naive model =======')
y_hat = mdl.gpr.predict(mdl_doe.X)

df_doe['col_thresh_pred'] = 0
col_thresh_naive = np.where(y_hat > mean_log_drift, 1, df_doe["col_thresh_pred"])

from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(df_doe['col_thresh'], col_thresh_naive)
bas = balanced_accuracy_score(df_doe['col_thresh'], col_thresh_naive)
print('F1 score: %.3f' % f1)
print('Balanced accuracy score: %.3f' % bas)

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(mdl_doe.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

y_true = np.array(mdl_doe.y).ravel()
mae = mean_absolute_error(mdl_doe.y, y_hat)
print('Mean absolute error: %.3f' % mae)

compare = pd.DataFrame([y_hat, y_true]).T
compare.columns = ['predicted %', 'true %']

#%%

import matplotlib.pyplot as plt
import pandas as pd

ame_df = pd.read_csv('./data/doe/mean_error.csv', header=None)
ame_df = ame_df.transpose()
ame_df.columns = ['ame']

plt.close('all')

plt.figure()
plt.plot(ame_df.index, ame_df['ame'])
plt.title('Convergence history')
plt.xlabel('Batches')
plt.ylabel('Average mean error (%)')
plt.grid(True)
plt.show()