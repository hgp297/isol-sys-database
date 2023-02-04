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
        
        self.param_ranges   = {
            'S_1' : [0.8, 1.3],
            'T_m' : [3.0, 5.0],
            'zeta_M' : [0.10, 0.20],
            'k_ratio' :[5, 100],
            'T_d' : [2.0, 6.0],
            'Q': [0.02, 0.16],
            'moat_ampli' : [0.3, 1.8],
            'RI' : [0.5, 2.0]
        }

        # create array of limits, then run LHS
        self.param_names      = list(self.param_ranges.keys())
        param_bounds     = np.asarray(list(self.param_ranges.values()),
                                    dtype=np.float64).T
        
        l_bounds = param_bounds[0,]
        u_bounds = param_bounds[1,]
        
        dim_params = len(self.param_ranges)
        sampler = qmc.LatinHypercube(d=dim_params, seed=seed)
        sample = sampler.random(n=n_points)
        
        self.param_values = qmc.scale(sample, l_bounds, u_bounds)
        
        ######################################################################
        # system selection params
        ######################################################################
        
        config_dict   = {'num_bays' : [3, 8],
            'num_stories' : [3, 5]
        }

        # generate random integers within the bounds and place into array
        self.config_names = list(config_dict.keys())       
        num_categories = len(config_dict)
        self.config_selection = np.empty([n_points, num_categories])
        
        # set seed
        np.random.seed(seed)
        
        for index, (key, bounds) in enumerate(config_dict.items()):
            self.config_selection[:,index] = np.random.randint(bounds[0], 
                                                               high=bounds[1]+1, 
                                                               size=n_points)
            
        import random
        random.seed(seed)
        struct_sys_list = ['MF', 'CBF']
        isol_sys_list = ['TFP', 'LRB']
        
        structs = random.choices(struct_sys_list, k=n_points)
        isols = random.choices(isol_sys_list, k=n_points)
        self.system_selection = np.array([structs, isols]).T
        self.system_names = ['superstructure_system', 'isolator_system']
        
        all_inputs = np.concatenate((self.system_selection,
                                     self.config_selection,
                                     self.param_values), axis=1)
        self.input_df = pd.DataFrame(all_inputs)
        self.input_df.columns = self.system_names + self.config_names + self.param_names
        
###############################################################################
# Designing isolation systems
###############################################################################

        
    def design_bearings(self):
        df_in = self.input_df
        
        # separate df into isolator systems
        df_tfp = df_in[df_in['isolator_system'] == 'TFP']
        df_lrb = df_in[df_in['isolator_system'] == 'LRB']
        
    
    # units are kips, ft
    def define_gravity_loads(self, D_load=None, L_load=None,
                             S_s=2.282, S_1 = 1.017,
                             n_floors=3, n_bays=3, L_bay=30.0, h_story=13.0,
                             n_frames=2):
        
        import numpy as np
        
        # assuming 100 psf D and 50 psf L for floors 
        # assume that D already includes structural members
        if D_load is None:
            D_load = np.repeat(100.0/1000, n_floors+1)
        if L_load is None:
            L_load = np.repeat(50.0/1000, n_floors+1)
            
        # roof loading is lighter
        D_load[-1] = 75.0/1000
        L_load[-1] = 20.0/1000
        
        # assuming square building
        A_bldg = (L_bay*n_bays)**2
        
        # seismic weight: ASCE 7-22, Ch. 12.7.2
        W = np.sum(D_load*A_bldg)
        
        # assume lateral frames are placed on the edge
        trib_width_lat = L_bay/2
        
        # line loads for lateral frame
        w_D = D_load*trib_width_lat
        w_L = L_load*trib_width_lat
        w_Ev = 0.2*S_s*w_D
        
        
        w_case_1 = 1.4*w_D
        w_case_2 = 1.2*w_D + 1.6*w_L # includes both case 2 and 3
        # case 4 and 5 do not control (wind)
        w_case_6 = 1.2*w_D + w_Ev + 0.5*w_L
        w_case_7 = 0.9*w_D - w_Ev
        
        w_on_frame = np.maximum.reduce([w_case_1,
                                        w_case_2,
                                        w_case_6,
                                        w_case_7])
        
        # leaning columns
        L_bldg = n_bays*L_bay
        
        # area assigned to lateral frame minus area already modeled by line loads
        trib_width_LC = (L_bldg/n_frames) - trib_width_lat 
        trib_area_LC = trib_width_LC * L_bldg
        
        # point loads for leaning column
        P_D = D_load*trib_area_LC
        P_L = L_load*trib_area_LC
        P_Ev = 0.2*S_s*P_D
        
        
        P_case_1 = 1.4*P_D
        P_case_2 = 1.2*P_D + 1.6*P_L # includes both case 2 and 3
        # case 4 and 5 do not control (wind)
        P_case_6 = 1.2*P_D + P_Ev + 0.5*P_L
        P_case_7 = 0.9*P_D - P_Ev
        
        P_on_leaning_column = np.maximum.reduce([P_case_1,
                                        P_case_2,
                                        P_case_6,
                                        P_case_7])