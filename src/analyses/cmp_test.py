############################################################################
#               Components testing file

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2023

# Description:  Working file to find component contents of structure

# Open issues:  (1) 

############################################################################
import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30

# and import pelicun classes and methods
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment

import warnings
warnings.filterwarnings('ignore')

#%% run info
data = pd.read_csv('../../data/structural_db_conv.csv')
cbf_run = data.iloc[0]
mf_run = data.iloc[-1]

#%% get database
# initialize, no printing outputs, offset fixed with current components
PAL = Assessment({
    "PrintLog": False, 
    "Seed": 985,
    "Verbose": False,
    "DemandOffset": {"PFA": 0, "PFV": 0}
})

# generate structural components and join with NSCs
P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

#%% nqe main data
nqe_data = pd.read_csv('../../resource/loss/fema_nqe_cmp.csv')
nqe_data = nqe_data.replace({'All Zero': 0}, regex=True)
nqe_data = nqe_data.replace({'2 Points = 0': 0}, regex=True)
nqe_data = nqe_data.replace({np.nan: 0})
nqe_data['directional'] = nqe_data['directional'].replace(
    {'YES': True, 'NO': False})

ta = nqe_data[['lab_std', 'health_std', 'edu_std', 'res_std',
          'office_std', 'retail_std', 'warehouse_std', 'hotel_std']].apply(
              pd.to_numeric, errors='coerce')
nqe_data[['lab_std', 'health_std', 'edu_std', 'res_std',
          'office_std', 'retail_std', 'warehouse_std', 'hotel_std']] = ta

#%%
# p90 low situations - calculator (values replaced in sheet)
'''
from scipy.stats import lognorm, norm
from scipy.optimize import curve_fit
f = lambda x,mu,sigma: lognorm(sigma,mu).cdf(x)
fn = lambda x, mu, sigma: norm(mu, sigma).cdf(x)

quantile_data = [6, 16.2, 27]
theta_n, beta_n = curve_fit(fn, np.log(quantile_data), [0.1, 0.5, 0.9])[0]
print(theta_n)
print(beta_n)
xx_pr = np.arange(0.001, 1.5, 0.001)
xx_pr_log = np.log(xx_pr)
p = fn(xx_pr_log, theta_n, beta_n)

import matplotlib.pyplot as plt
plt.close('all')

fig = plt.figure(figsize=(13, 10))
plt.plot(xx_pr, p)
plt.grid(True)
'''

# modular office needs definition


# rounding


#%% nqe function

cbf_floors = cbf_run.num_stories
# commercial, ed, health, hospitality, res, research, retail, warehouse
fl_usage = [0.8, 0, 0, 0, 0, 0, 0.2, 0]
bldg_usage = [fl_usage]*cbf_floors

def normative_quantity_estimation(run_info, usage, database):
    superstructure_system = run_info.superstructure_system
    
    
def commercial(run_info, scale):
    cmp = pd.DataFrame()
# inputs
# floors, floor usage, system

# need
# cmp, units, location, direction, mean, stdev, distr, blocks, description