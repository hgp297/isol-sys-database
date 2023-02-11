############################################################################
#               Database object for structural systems

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Object acts as central aggregate for designs, datasets

# Open issues:  (1) ranges of params dependent on system

############################################################################

class Database:
    
    # sets up the problem by generating building specifications
    
    def __init__(self, n_points=5000, seed=985):
        
        from scipy.stats import qmc
        import numpy as np
        import pandas as pd
        
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
            'T_m' : [2.5, 5.0],
            'k_ratio' :[3.0, 30.0],
            'Q': [0.05, 0.12],
            'moat_ampli' : [0.8, 1.8],
            'RI' : [0.5, 2.0]
        }

        # create array of limits, then run LHS
        param_names      = list(self.param_ranges.keys())
        param_bounds     = np.asarray(list(self.param_ranges.values()),
                                    dtype=np.float64).T
        
        l_bounds = param_bounds[0,]
        u_bounds = param_bounds[1,]
        
        dim_params = len(self.param_ranges)
        sampler = qmc.LatinHypercube(d=dim_params, seed=seed)
        sample = sampler.random(n=n_points)
        
        params = qmc.scale(sample, l_bounds, u_bounds)
        param_selection = pd.DataFrame(params)
        
        
        ######################################################################
        # system selection params
        ######################################################################
        
        config_dict   = {'num_bays' : [3, 6],
            'num_stories' : [3, 6]
        }

        # generate random integers within the bounds and place into array
        config_names = list(config_dict.keys())       
        num_categories = len(config_dict)
        config_selection = np.empty([n_points, num_categories])
        
        # set seed
        np.random.seed(seed)
        
        for index, (key, bounds) in enumerate(config_dict.items()):
            config_selection[:,index] = np.random.randint(bounds[0], 
                                                               high=bounds[1]+1, 
                                                               size=n_points)
        config_selection = pd.DataFrame(config_selection)
        
        import random
        random.seed(seed)
        struct_sys_list = ['MF', 'CBF']
        isol_sys_list = ['TFP', 'LRB']
        
        structs = random.choices(struct_sys_list, k=n_points)
        isols = random.choices(isol_sys_list, k=n_points)
        system_selection = pd.DataFrame(np.array([structs, isols]).T)
        system_names = ['superstructure_system', 'isolator_system']
        
        self.raw_input = pd.concat([system_selection,
                                   config_selection,
                                   param_selection], axis=1)
        self.raw_input.columns = system_names + config_names + param_names
        
###############################################################################
# Designing isolation systems
###############################################################################

    # use filter_designs=True if only realistic/physically sensible designs are
    # retained. This may result in the LHS distribution being uneven.
        
    def design_bearings(self, filter_designs=True):
        import time
        import pandas as pd
        
        df_in = self.raw_input
        
        # get loading conditions
        from load_calc import define_gravity_loads
        loading_df = df_in.apply(lambda row: define_gravity_loads(S_1=row['S_1'],
                                                      n_floors=row['num_stories'],
                                                      n_bays=row['num_bays']),
                                 axis='columns', result_type='expand')
        
        loading_df.columns = ['seismic_weight', 'w_floor', 'P_leaning_col']
        
        df_in = pd.concat([df_in, loading_df['seismic_weight']], axis=1)
        
        # separate df into isolator systems
        import design as ds
        df_tfp = df_in[df_in['isolator_system'] == 'TFP']
        
        
        # attempt to design all TFPs
        t0 = time.time()
        all_tfp_designs = df_tfp.apply(lambda row: ds.design_TFP(row['T_m'],
                                                             row['S_1'],
                                                             row['Q'],
                                                             row['k_ratio']),
                                       axis='columns', result_type='expand')
        
        all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                   'T_e', 'k_e', 'zeta_e']
        
        if filter_designs == False:
            tfp_designs = all_tfp_designs
        else:
            # keep the designs that look sensible
            tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                              (all_tfp_designs['R_1'] <= 50.0) &
                                              (all_tfp_designs['R_2'] <= 200.0) &
                                              (all_tfp_designs['zeta_e'] <= 0.30)]
        
        tp = time.time() - t0
        
        print("Designs completed for %d TFPs in %.2f s" %
              (tfp_designs.shape[0], tp))
        
        # get the design params of those bearings
        a = df_tfp[df_tfp.index.isin(tfp_designs.index)]
        
        self.tfp_designs = pd.concat([a, tfp_designs], axis=1)
        
        df_lrb = df_in[df_in['isolator_system'] == 'LRB']
        
        # attempt to design all LRBs
        t_rb = 10.0
        t0 = time.time()
        all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row['T_m'],
                                                                 row['S_1'],
                                                                 row['Q'],
                                                                 row['k_ratio'],
                                                                 row['num_bays'],
                                                                 row['seismic_weight'],
                                                                 t_r=t_rb),
                                       axis='columns', result_type='expand')
        
        
        all_lrb_designs.columns = ['d_bearing', 'd_lead', 
                                   'T_e', 'k_e', 'zeta_e', 'buckling_fail']
        
        if filter_designs == False:
            lrb_designs = all_lrb_designs
        else:
            # keep the designs that look sensible
            lrb_designs = all_lrb_designs.loc[(all_lrb_designs['d_bearing'] >=
                                               3*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_bearing'] <=
                                               6*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_lead'] <= t_rb) &
                                              (all_lrb_designs['buckling_fail'] == 0)]
            
            lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
            
        tp = time.time() - t0
        
        print("Designs completed for %d LRBs in %.2f s" %
              (lrb_designs.shape[0], tp))
        
        b = df_lrb[df_lrb.index.isin(lrb_designs.index)]
        
        self.lrb_designs = pd.concat([b, lrb_designs], axis=1)