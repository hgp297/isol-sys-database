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

mdl = GP(df)
mdl.set_outcome('collapse_prob')
mdl.fit_gpr(kernel_name='rbf_ard')

rsc = mdl.gpr.score(mdl.X, mdl.y)
print('R-squared :', rsc)
y_hat = mdl.gpr.predict(mdl.X)

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(mdl.y, y_hat)
print('Mean squared error: %.3f' % mse)

# ame is average error in predicting collapse risk (expressed in relative %)
# denominator is 0.001 to prevent extreme outliers
y_true = np.array(mdl.y).ravel()
mape = np.abs(y_true-y_hat)/np.maximum(np.abs(y_true), 1e-3)
ame = np.average(mape, axis=0)
print('Average mean error (%%): %.2f' % ame)

print('======= after doe (~630 points) =======')

df = pd.read_csv('./data/doe/mik_smrf_doe.csv')
mdl = GP(df)
mdl.set_outcome('collapse_prob')
mdl.fit_gpr(kernel_name='rbf_ard')

rsc = mdl.gpr.score(mdl.X, mdl.y)
print('R-squared :', rsc)
y_hat = mdl.gpr.predict(mdl.X)

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(mdl.y, y_hat)
print('Mean squared error: %.3f' % mse)

# ame is average error in predicting collapse risk (expressed in relative %)
# denominator is 0.001 to prevent extreme outliers
y_true = np.array(mdl.y).ravel()
mape = np.abs(y_true-y_hat)/np.maximum(np.abs(y_true), 1e-3)
ame = np.average(mape, axis=0)
print('Average mean error (%%): %.2f' % ame)