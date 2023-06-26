from doe import GP
import pandas as pd

# doe_path = '../data/doe/old/rmse_1_percent/'
doe_path = '../data/doe/'

print('======= starting at 100 points =======')
df = pd.read_csv('../data/training_set.csv')
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

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
mse = mean_squared_error(mdl.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

y_true = np.array(mdl.y).ravel()
mae = mean_absolute_error(mdl.y, y_hat)
print('Mean absolute error: %.3f' % mae)

print('======= after doe (~210 points) =======')

df_doe = pd.read_csv(doe_path+'rmse_doe_set.csv')

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

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(mdl_doe.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

y_true = np.array(mdl_doe.y).ravel()
mae = mean_absolute_error(mdl_doe.y, y_hat)
print('Mean absolute error: %.3f' % mae)

compare = pd.DataFrame([y_hat, y_true]).T
compare.columns = ['predicted %', 'true %']


#%% plot predictions
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

plt.figure()
plt.scatter(y_hat, y_true)
plt.plot([0, 1.0], [0, 1.0], linestyle='-',color='black')
plt.plot([0, 1.0], [0, 1.1], linestyle='--',color='black')
plt.plot([0, 1.0], [0, 0.9], linestyle='--',color='black')
plt.title('Prediction accuracy')
plt.xlabel('Predicted collapse %')
plt.ylabel('True collapse %')
plt.xlim([0, 0.3])
plt.ylim([0, 0.3])
plt.grid(True)
plt.show()

#%%

import matplotlib.pyplot as plt
import pandas as pd

rmse_df = pd.read_csv(doe_path+'rmse.csv', header=None)
mae_df = pd.read_csv(doe_path+'mae.csv', header=None)
mae_df = mae_df.transpose()
mae_df.columns = ['mae']
rmse_df = rmse_df.transpose()
rmse_df.columns = ['rmse']

plt.close('all')

plt.figure()
plt.plot(rmse_df.index, rmse_df['rmse'])
plt.title('Convergence history')
plt.xlabel('Batches')
plt.ylabel('Root mean squared error (collapse %)')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(rmse_df.index, mae_df['mae'])
plt.title('Convergence history')
plt.xlabel('Batches')
plt.ylabel('Mean absolute error (collapse %)')
plt.grid(True)
plt.show()

#%% fixed test set

print('======= test results trained on 100 points =======')
df_train = pd.read_csv('../data/training_set.csv')
df_train['max_drift'] = df_train[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)

df_test = pd.read_csv('../data/testing_set.csv')
df_test['max_drift'] = df_test[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp
from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
# 0.9945 is inverse normCDF of 0.84
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) 
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

df_train['collapse_prob'] = ln_dist.cdf(df_train['max_drift'])
df_test['collapse_prob'] = ln_dist.cdf(df_test['max_drift'])

# train on the training set
mdl_init = GP(df_train)
mdl_init.set_outcome('collapse_prob')
mdl_init.fit_gpr(kernel_name='rbf_ard')

collapse_threshold = 0.1

test_set = GP(df_test)
test_set.set_outcome('collapse_prob')

# make predictions on test set
y_hat = mdl_init.gpr.predict(test_set.X)



from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(test_set.y, y_hat)
print('Root mean squared error: %.3f' % mse**0.5)

mae = mean_absolute_error(test_set.y, y_hat)
print('Mean absolute error: %.3f' % mae)

df_test['col_thresh'] = 0
df_test['col_thresh'] = np.where(df_test["collapse_prob"] > collapse_threshold, 1,
                                 df_test["col_thresh"])

df_test['col_thresh_pred'] = 0
df_test['col_thresh_pred'] = np.where(y_hat > collapse_threshold, 1,
                                       df_test["col_thresh_pred"])

from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(df_test['col_thresh'], df_test['col_thresh_pred'])
bas = balanced_accuracy_score(df_test['col_thresh'], df_test['col_thresh_pred'])
print('F1 score: %.3f' % f1)
print('Balanced accuracy score: %.3f' % bas)

print('======= test results trained on 210 points =======')
df_doe = pd.read_csv(doe_path+'rmse_doe_set.csv')
df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

# train on the doe set
mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

# make predictions on test set
y_hat_doe = mdl_doe.gpr.predict(test_set.X)
mse = mean_squared_error(test_set.y, y_hat_doe)
print('Root mean squared error: %.3f' % mse**0.5)

mae = mean_absolute_error(test_set.y, y_hat_doe)
print('Mean absolute error: %.3f' % mae)

df_test['col_thresh'] = 0
df_test['col_thresh'] = np.where(df_test["collapse_prob"] > collapse_threshold, 1,
                                  df_test["col_thresh"])

df_test['col_thresh_pred'] = 0
df_test['col_thresh_pred'] = np.where(y_hat_doe > collapse_threshold, 1,
                                       df_test["col_thresh_pred"])

from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(df_test['col_thresh'], df_test['col_thresh_pred'])
bas = balanced_accuracy_score(df_test['col_thresh'], df_test['col_thresh_pred'])
print('F1 score: %.3f' % f1)
print('Balanced accuracy score: %.3f' % bas)

#%% new points

new_pts = df_doe.tail(70)

plt.figure()
plt.scatter(new_pts['gapRatio'], new_pts['RI'], c=new_pts.index)
plt.title('DoE points')
plt.xlabel('Gap ratio')
plt.ylabel('RI')
plt.grid(True)
plt.show()