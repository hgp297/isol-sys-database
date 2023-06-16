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

database_path = './data/doe/old/rmse_1_percent/'
database_file = 'rmse_doe_set.csv'

df_doe = pd.read_csv(database_path+database_file, 
                                  index_col=None)

df_doe['max_drift'] = df_doe[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df_doe['collapse_prob'] = ln_dist.cdf(df_doe['max_drift'])

mdl_doe = GP(df_doe)
mdl_doe.set_outcome('collapse_prob')
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

#%% try for design
res = 3
xx, yy, uu = np.meshgrid(np.linspace(0.3, 2.0,
                                      res),
                          np.linspace(0.5, 2.0,
                                      res),
                          np.linspace(2.5, 4.0,
                                      res))
                             
X_space = pd.DataFrame({'gapRatio':xx.ravel(),
                      'RI':yy.ravel(),
                      'Tm':uu.ravel(),
                      'zetaM':np.repeat(0.15,res**3)})

from pred import get_steel_coefs, calc_upfront_cost
steel_price = 2.00
coef_dict = get_steel_coefs(df_doe, steel_per_unit=steel_price)

import time
t0 = time.time()
pareto, domed = simple_cull_df(X_space, dominates_pd, 
                               mdl_doe.gpr, calc_upfront_cost, coef_dict)
tp = time.time() - t0

print("Culled %d points in %.3f s" % (X_space.shape[0], tp))
#%% example
# inputPoints = [[1,1,1], [1,2,3], [3,2,1], [4,1,1]]
# paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

# print("*"*8 + " non-dominated answers " + ("*"*8))
# for p in paretoPoints:
#     print(p)
# print("*"*8 + " dominated answers " + ("*"*8))
# for p in dominatedPoints:
#     print(p)