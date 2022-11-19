############################################################################
#               Pelicun (adapted for old project)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: November 2022

# Description:  Trial run of pelicun applied to old CE227 project

# Open issues:  (1) 

############################################################################

# import helpful packages for numerical analysis
import sys

import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import pprint

# and for plotting
from plotly import graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# and import pelicun classes and methods
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment

#%%

# initialize a pelicun Assessment
PAL = Assessment({
    "PrintLog": True, 
    "Seed": 985,
    "Verbose": False,
})

#%%

# Demands

raw_demands = pd.read_csv('MCEr_EDP.csv', index_col=0)
for EDP in raw_demands:
    if 'PFA' in EDP:
        raw_demands[EDP] = raw_demands[EDP]/386.4
  
raw_demands.loc['Units'] = ['g','g','g','g','rad','rad','rad','g','g','g','g','rad','rad','rad']

raw_demands["new"] = range(1,len(raw_demands)+1)
raw_demands.loc[raw_demands.index=='Units', 'new'] = 0
raw_demands = raw_demands.sort_values("new").drop('new', axis=1)



#%%

PAL.demand.load_sample(raw_demands)
PAL.demand.calibrate_model(
    {
        "ALL": {
            "DistributionFamily": "lognormal"
        },
        "PID": {
            "DistributionFamily": "lognormal",
            "TruncateLower": "",
            "TruncateUpper": "0.06"
        }
    }
)

#%%

# choose a sample size for the analysis
sample_size = 10000

# generate demand sample
PAL.demand.generate_sample({"SampleSize": sample_size})

# extract the generated sample
# Note that calling the save_sample() method is better than directly pulling the 
# sample attribute from the demand object because the save_sample method converts
# demand units back to the ones you specified when loading in the demands.
demand_sample = PAL.demand.save_sample()

demand_sample.head()

#%%

# get residual drift estimates 
delta_y = 0.0075
PID = demand_sample['PID']

RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y})

# and join them with the demand_sample
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

demand_sample_ext.describe().T

#%%

# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA']] = 'g'
demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'


PAL.demand.load_sample(demand_sample_ext)

#%%

# load the component configuration
cmp_marginals = pd.read_csv('CMP_marginals.csv', index_col=0)

cmp_marginals.tail(10)

# to make the convenience keywords work in the model, 
# we need to specify the number of stories
PAL.stories = 3

# now load the model into Pelicun
PAL.asset.load_cmp_model({'marginals': cmp_marginals})

# run into error if theta is outputted as a series instead of a float64