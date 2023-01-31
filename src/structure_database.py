############################################################################
#               Database object for structural systems

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Object acts as central aggregate for designs, datasets

# Open issues:  (1) 

############################################################################
import pandas as pd

class Database:
    
    # sets up the problem by generating building specifications
    
    def __init__(self, n_points=400, seed=985):
        
        from scipy.stats import qmc
        import numpy as np
        
        ######################################################################
        # generalized design parameters
        ######################################################################
        
        # S_1: site design spectral acc
        # T_m: effective bearing period
        # zeta: effective bearing damping
        # k_ratio: post yielding bearing stiffness ratio
        # T_d: post yielding bearing period
        # Q: normalized back strength
        # moat_ampli: moat gap
        # RI: building strength
        
        self.variable_ranges   = {
            'S_1' : [0.8, 1.3],
            'T_m' : [2.5, 4.0],
            'zeta_M' : [0.10, 0.20],
            'k_ratio' :[2, 20],
            'T_d' : [2.5, 6.0],
            'Q': [0.02, 0.2],
            'moat_ampli' : [0.3, 1.8],
            'RI' : [0.5, 2.0]
        }

        # create array of limits, then run LHS
        var_names      = list(self.variable_ranges.keys())
        var_bounds     = np.asarray(list(self.variable_ranges.values()),
                                    dtype=np.float64).T
        
        l_bounds = var_bounds[0,]
        u_bounds = var_bounds[1,]
        
        dim_vars = len(self.variable_ranges)
        sampler = qmc.LatinHypercube(d=dim_vars, seed=seed)
        sample = sampler.random(n=n_points)
        
        self.input_list = var_names
        self.input_values = qmc.scale(sample, l_bounds, u_bounds)
        
        ######################################################################
        # system selection variables
        ######################################################################
        # structural systems: 1 - MF, 2 - BF, 3 - BRB, 4 - SW
        # isolator systems: 1 - TFP, 2 - LRB
        
        system_dict   = {
            'structural_system' : [1, 4],
            'isolator_system' : [1, 2],
            'num_bays' : [3, 8],
            'num_stories' : [3, 5]
        }

        # generate random integers within the bounds and place into array
        self.system_names = list(system_dict.keys())       
        num_categories = len(system_dict)
        self.system_selection = np.empty([n_points, num_categories])
        
        # set seed
        np.random.seed(seed)
        
        for index, (key, bounds) in enumerate(system_dict.items()):
            self.system_selection[:,index] = np.random.randint(bounds[0], 
                                                               high=bounds[1]+1, 
                                                               size=n_points)
        
        