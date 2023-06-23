#%% possible new pareto-front optimization design


    
#%% main model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from doe import GP
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import warnings
warnings.filterwarnings('ignore')

## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

# collapse as a probability
from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

database_path = './data/doe/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('max_drift')
mdl_doe.fit_gpr(kernel_name='rbf_ard')

#%% pareto front 

# def simple_cull(inputPoints, dominates):
#     paretoPoints = set()
#     candidateRowNr = 0
#     dominatedPoints = set()
#     while True:
#         candidateRow = inputPoints[candidateRowNr]
#         inputPoints.remove(candidateRow)
#         rowNr = 0
#         nonDominated = True
#         while len(inputPoints) != 0 and rowNr < len(inputPoints):
#             row = inputPoints[rowNr]
#             if dominates(candidateRow, row):
#                 # If it is worse on all features remove the row from the array
#                 inputPoints.remove(row)
#                 dominatedPoints.add(tuple(row))
#             elif dominates(row, candidateRow):
#                 nonDominated = False
#                 dominatedPoints.add(tuple(candidateRow))
#                 rowNr += 1
#             else:
#                 rowNr += 1

#         if nonDominated:
#             # add the non-dominated point to the Pareto frontier
#             paretoPoints.add(tuple(candidateRow))

#         if len(inputPoints) == 0:
#             break
#     return paretoPoints, dominatedPoints

# def dominates(row, candidateRow):
#     return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

def simple_cull_df(input_df, dominates, *args):
    pareto_df = pd.DataFrame(columns=input_df.columns)
    candidateRowNr = 0
    dominated_df = pd.DataFrame(columns=input_df.columns)
    while True:
        candidateRow = input_df.iloc[[candidateRowNr]]
        input_df = input_df.drop(index=candidateRow.index)
        rowNr = 0
        nonDominated = True
        while input_df.shape[0] != 0 and rowNr < input_df.shape[0]:
            row = input_df.iloc[[rowNr]]
            if dominates(candidateRow, row, *args):
                # If it is worse on all features remove the row from the array
                input_df = input_df.drop(index=row.index)
                dominated_df = dominated_df.append(row)
            elif dominates(row, candidateRow, *args):
                nonDominated = False
                dominated_df = dominated_df.append(candidateRow)
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            pareto_df = pareto_df.append(candidateRow)

        if input_df.shape[0] == 0:
            break
    return pareto_df, dominated_df

def dominates_pd(row, candidate_row, mdl, cost, coefs):
    row_pr = mdl.predict(row).item()
    cand_pr = mdl.predict(candidate_row).item()
    
    row_cost = cost(row, coefs).item()
    cand_cost = cost(candidate_row, coefs).item()
    
    return ((row_pr < cand_pr) and (row_cost < cand_cost))

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

#%% try for design
res = 30
xx, yy, uu = np.meshgrid(np.linspace(0.3, 2.0,
                                      res),
                              np.linspace(0.5, 2.0,
                                          res),
                              np.linspace(2.5, 4.0,
                                          res))
                             
X_space = pd.DataFrame({'gapRatio':xx.ravel(),
                      'RI':yy.ravel(),
                      'Tm':uu.ravel(),
                      'zetaM':np.repeat(0.2, res**3)})

# xx, yy, uu, vv = np.meshgrid(np.linspace(0.3, 2.0,
#                                       res),
#                               np.linspace(0.5, 2.0,
#                                           res),
#                               np.linspace(2.5, 4.0,
#                                           res),
#                               np.linspace(0.1, 0.2,
#                                           res))
                             
# X_space = pd.DataFrame({'gapRatio':xx.ravel(),
#                       'RI':yy.ravel(),
#                       'Tm':uu.ravel(),
#                       'zetaM':vv.ravel()})

from pred import get_steel_coefs, calc_upfront_cost
steel_price = 2.00
coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)

import time
t0 = time.time()

# pareto, domed = simple_cull_df(X_space, dominates_pd, 
#                                mdl_doe.gpr, calc_upfront_cost, coef_dict)

fmu, fs1 = mdl_doe.gpr.predict(X_space, return_std=True)
fs2 = fs1**2

tp = time.time() - t0

# print("Culled %d points in %.3f s" % (X_space.shape[0], tp))
print("GPR collapse prediction for %d inputs in %.3f s" % (X_space.shape[0],
                                                               tp))

constr_costs = calc_upfront_cost(X_space, coef_dict)

#%% culling

cost_pareto = np.array([constr_costs, fmu]).transpose()

t0 = time.time()
pareto_mask = is_pareto_efficient(cost_pareto)
tp = time.time() - t0

print("Culled %d points in %.3f s" % (X_space.shape[0], tp))

X_pareto = X_space.iloc[pareto_mask]

#%%

prob_target = 0.1
drifts = mdl_doe.gpr.predict(X_pareto, return_std=False)
probs = ln_dist.cdf(drifts)
X_feasible = X_pareto[probs < prob_target]

#%% plot

plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 20
subt_font = 18
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


plt.scatter(X_feasible['gapRatio'], X_feasible['RI'], 
            c=drifts[probs < prob_target],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$R_y$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.title('Pareto front of designs', fontsize=axis_font)
plt.show()



plt.figure()
plt.scatter(X_feasible['gapRatio'], X_feasible['Tm'], 
            c=drifts[probs < prob_target],
            edgecolors='k', s=20.0, cmap=plt.cm.Reds_r)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$T_M$', fontsize=axis_font)
plt.xlim([0.3, 2.0])
plt.title('Pareto front of designs', fontsize=axis_font)
plt.show()

#%% 3d plot

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X_feasible['gapRatio'], X_feasible['Tm'], X_feasible['RI'], 
           c=drifts[probs < prob_target],
           edgecolors='k', alpha = 0.7)
plt.xlabel('Gap ratio', fontsize=axis_font)
plt.ylabel(r'$T_M$', fontsize=axis_font)
ax.set_xlim([0.3, 2.0])
ax.set_zlabel(r'$R_y$', fontsize=axis_font)
#%% example
# inputPoints = [[1,1,1], [1,2,3], [3,2,1], [4,1,1]]
# paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

# print("*"*8 + " non-dominated answers " + ("*"*8))
# for p in paretoPoints:
#     print(p)
# print("*"*8 + " dominated answers " + ("*"*8))
# for p in dominatedPoints:
#     print(p)