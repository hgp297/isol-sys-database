############################################################################
#               Damage estimation tool

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: November 2022

# Description:  Main pelicun working script

# Open issues:  (1) 

############################################################################

# import helpful packages for numerical analysis
import numpy as np
import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 100

# and for plotting

# and import pelicun classes and methods
from pelicun.assessment import Assessment


# ### Load demand distribution data


all_demands = pd.read_csv('engineering_demands_TFP.csv')


# Pelicun uses SimCenter's naming convention for demands:
# - The first number represents the event_ID. This can be used to differentiate between multiple stripes of an analysis, or multiple consecutive events in a main-shock - aftershock sequence, for example. Pelicun does not use the first number internally.
# - The type of the demand identifies the EDP or IM. The following options are available:
#     * 'Story Drift Ratio' :             'PID',
#     * 'Peak Interstory Drift Ratio':    'PID',
#     * 'Roof Drift Ratio' :              'PRD',
#     * 'Peak Roof Drift Ratio' :         'PRD',
#     * 'Damageable Wall Drift' :         'DWD',
#     * 'Racking Drift Ratio' :           'RDR',
#     * 'Peak Floor Acceleration' :       'PFA',
#     * 'Peak Floor Velocity' :           'PFV',
#     * 'Peak Gust Wind Speed' :          'PWS',
#     * 'Peak Inundation Height' :        'PIH',
#     * 'Peak Ground Acceleration' :      'PGA',
#     * 'Peak Ground Velocity' :          'PGV',
#     * 'Spectral Acceleration' :         'SA',
#     * 'Spectral Velocity' :             'SV',
#     * 'Spectral Displacement' :         'SD',
#     * 'Peak Spectral Acceleration' :    'SA',
#     * 'Peak Spectral Velocity' :        'SV',
#     * 'Peak Spectral Displacement' :    'SD',
#     * 'Permanent Ground Deformation' :  'PGD',
#     * 'Mega Drift Ratio' :              'PMD',
#     * 'Residual Drift Ratio' :          'RID',
#     * 'Residual Interstory Drift Ratio':'RID'
# - The third part is an integer the defines the location where the demand was recorded. In buildings, locations are typically floors, but in other assets, locations could reference any other part of the structure.
# - The last part is an integer the defines the direction of the demand. Typically 1 stands for horizontal X and 2 for horizontal Y, but any other numbering convention can be used.
# 
# The location and direction numbers need to be in line with the component definitions presented later.

# initialize a pelicun Assessment
PAL = Assessment({"PrintLog": True, "Seed": 985,})