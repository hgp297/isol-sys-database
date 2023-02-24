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
        
        config_dict   = {
            'num_bays' : [3, 6],
            'num_stories' : [3, 6],
            'num_frames' : [2, 2]
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
        
        # temp add in for constants
        
        self.raw_input['L_bay'] = 30.0
        self.raw_input['h_story'] = 13.0
        self.raw_input['S_s'] = 2.2815
        
###############################################################################
# Designing isolation systems
###############################################################################

    # use filter_designs=True if only realistic/physically sensible designs are
    # retained. This may result in the LHS distribution being uneven.
    
    # loads are also defined here
    # TODO: assertions
    
    def design_bearings(self, filter_designs=True):
        import time
        import pandas as pd
        
        df_raw = self.raw_input
        
        # get loading conditions
        from loads import define_gravity_loads
        df_raw[['W', 
               'W_s', 
               'w_fl', 
               'P_lc']] = df_raw.apply(lambda row: define_gravity_loads(row),
                                       axis='columns', result_type='expand')
        
        # separate df into isolator systems
        import design as ds
        df_tfp = df_raw[df_raw['isolator_system'] == 'TFP']
        
        
        # attempt to design all TFPs
        t0 = time.time()
        all_tfp_designs = df_tfp.apply(lambda row: ds.design_TFP(row),
                                       axis='columns', result_type='expand')
        
        all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                   'T_e', 'k_e', 'zeta_e', 'D_m']
        
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
        
        df_lrb = df_raw[df_raw['isolator_system'] == 'LRB']
        
        # attempt to design all LRBs
        t_rb = 10.0
        t0 = time.time()
        all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row,
                                                                 t_r=t_rb),
                                       axis='columns', result_type='expand')
        
        
        all_lrb_designs.columns = ['d_bearing', 'd_lead', 
                                   'T_e', 'k_e', 'zeta_e', 'D_m', 'buckling_fail']
        
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
        
    def design_structure(self, filter_designs=True):
        import pandas as pd
        import time
        
        # combine both set of isolator designs
        df_in = pd.concat([self.tfp_designs, self.lrb_designs], axis=0)
        
        from loads import define_lateral_forces
        
        # TODO: check n_frames, current defaults: n_frames = 2, 30ft bay, 13 ft stories
        df_in[['wx', 
               'hx', 
               'h_col', 
               'hsx', 
               'Fx', 
               'Vs']] = df_in.apply(lambda row: define_lateral_forces(row),
                                    axis='columns', result_type='expand')
        
        # separate by superstructure systems
        smrf_df = df_in[df_in['superstructure_system'] == 'MF']
        cbf_df = df_in[df_in['superstructure_system'] == 'CBF']
        
        # attempt to design all moment frames
        t0 = time.time()
        import design as ds
        
        smrf_df[['beam',
                 'roof',
                 'column',
                 'flag']] = smrf_df.apply(lambda row: ds.design_MF(row),
                                               axis='columns', 
                                               result_type='expand')
        tp = time.time() - t0
      
        self.mf_designs = smrf_df
        
        # TODO: filter out bad designs (scwb)
        
        print("Designs completed for %d moment frames in %.2f s" %
              (smrf_df.shape[0], tp))