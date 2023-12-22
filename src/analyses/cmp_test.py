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
nqe_data.set_index('cmp', inplace=True)
nqe_data = nqe_data.replace({'All Zero': 0}, regex=True)
nqe_data = nqe_data.replace({'2 Points = 0': 0}, regex=True)
nqe_data = nqe_data.replace({np.nan: 0})
nqe_data['directional'] = nqe_data['directional'].replace(
    {'YES': True, 'NO': False})

nqe_meta = nqe_data[[c for c in nqe_data if not (
    c.endswith('mean') or c.endswith('std'))]]
nqe_mean = nqe_data[[c for c in nqe_data if c.endswith('mean')]]
nqe_std = nqe_data[[c for c in nqe_data if c.endswith('std')]].apply(
    pd.to_numeric, errors='coerce')

# divide WT row by 1000 to convert to kVA
nqe_mean[nqe_meta['unit'] == 'WT'] = nqe_mean[nqe_meta['unit'] == 'WT']/1000
nqe_meta = nqe_meta.replace({'WT': 'KV'})

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
cbf_area = cbf_run.L_bldg**2 # sq ft
# lab, health, ed, res, office, retail, warehouse, hotel
fl_usage = [0., 0., 0., 0., 0.7, 0.3, 0., 0.]
bldg_usage = [fl_usage]*cbf_floors

area_usage = np.array(fl_usage)*cbf_area

# estimate mean qty for floor area
# has not adjusted for 'quantity' column
def floor_mean(area_usage, mean_data):
    return mean_data * area_usage

# get std for floor (sum of lognormals)
def floor_std(area_usage, std_data, floor_mean):
    # get boolean if things are present, then multiply with stdev
    import numpy as np
    has_stuff = floor_mean.copy()
    has_stuff[has_stuff != 0] = 1
    
    # variance per floor
    var_present = np.square(std_data.values * has_stuff.values)
    
    # var_xy = var_x + var_y; std = sqrt(var)
    std_cmp = np.sqrt(np.sum(var_present, axis=1))
    
    return pd.Series(std_cmp, index=std_data.index)
    
def floor_qty_estimate(area_usage, mean_data, std_data, meta_data):
    fl_cmp_by_usage = floor_mean(area_usage, mean_data)
    fl_std = floor_std(area_usage, std_data, fl_cmp_by_usage)
    
    # sum across all usage and adjust for base quantity
    fl_cmp_qty = fl_cmp_by_usage.sum(axis=1) * meta_data['quantity']
    return(fl_cmp_by_usage, fl_cmp_qty, fl_std)
    
fl_mean = floor_mean(area_usage, nqe_mean)
fl_std = floor_std(area_usage, nqe_std, fl_mean)
fl_cmp_qty = fl_mean.sum(axis=1)*nqe_meta['quantity']
cmp_df = pd.concat([fl_cmp_qty, fl_std], axis=1)
def normative_quantity_estimation(run_info, usage, nqe_mean, nqe_std, nqe_meta):
    floor_area = run_info.L_bldg**2 # sq ft

    
    
    cmp_marginal = pd.DataFrame(columns=['Units', 'Location',
                                         'Direction','Theta_0', 'Theta_1',
                                         'Family', 'Blocks', 'Comment'])
    fema_units = nqe_meta['unit']
    nqe_meta[['pact_unit', 'pact_block_qty']] = nqe_meta['PACT_block'].str.split(
        ' ', n=1, expand=True)
    nqe_meta['pact_block_qty'] = pd.to_numeric(nqe_meta['pact_block_qty'])
    pact_units = fema_units.replace({'SF': 'ft2',
                                     'LF': 'ft'})
    # perform floor estimation
    for fl, fl_usage in enumerate(bldg_usage):
        area_usage = np.array(fl_usage)*floor_area
        
        fl_cmp_by_cat, fl_cmp_total, fl_cmp_std = floor_qty_estimate(
            area_usage, nqe_mean, nqe_std)
        
        loc_series = np.repeat(fl+1, len(fl_cmp_total))
        
        dir_map = {True:'1,2', False:'0'}
        dir_series = nqe_meta.directional.map(dir_map)
        
        has_stdev = fl_cmp_std != 0
        family_map = {True:'lognormal', False:''}
        family_series = has_stdev.map(family_map)
        
        # TODO: convert from FEMA units to PACT units to PBEE units
        # TODO: handle blocks
        fl_cmp_df = pd.concat([pact_units, loc_series, 
                               dir_series, fl_cmp_total, fl_cmp_std,
                               family_series, nqe_meta.PACT_name], axis=1)
        
        # TODO: concatenate
    
    # TODO: rounding
    # TODO: special components (bldg-wide and roof-only)
    

# inputs
# floors, floor usage, system

# need
# cmp, units, location, direction, mean, stdev, distr, blocks, description