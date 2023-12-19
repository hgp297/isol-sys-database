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