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
    # n_points: points to generate
    # n_buffer: generate n_buffer*n_points designs to account for discards
    # struct_sys_list: systems list
    # isol_wts: currently hardcoded for ['TFP', 'LRB'], sample 1:3 TFP:LRB ratio
    # to account for harder LRB designs being discarded
    
    def __init__(self, n_points=400, seed=985, n_buffer=15,
                 struct_sys_list=['MF', 'CBF'], isol_wts=[1,3]):
        
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
        
        # LRB
            # T_m = [ < 4.0]
            # k_ratio = [ < 10.0]
            
        # TFP (find documentation in books)
            # T_m = 
            # k_ratio = [ < 100.0]
            
        self.param_ranges   = {
            'S_1' : [0.8, 1.3],
            'T_m' : [2.0, 5.0],
            'k_ratio' :[5.0, 18.0],
            'Q': [0.05, 0.12],
            'moat_ampli' : [0.5, 1.2],
            'RI' : [0.5, 2.25],
            'L_bldg': [75.0, 250.0],
            'h_bldg': [30.0, 100.0]
        }

        # create array of limits, then run LHS
        param_names      = list(self.param_ranges.keys())
        param_bounds     = np.asarray(list(self.param_ranges.values()),
                                    dtype=np.float64).T
        
        l_bounds = param_bounds[0,]
        u_bounds = param_bounds[1,]
        
        self.n_points = n_points
        
        # roughly need 7x points to fill desired 
        self.n_generated = n_points*n_buffer
        
        dim_params = len(self.param_ranges)
        sampler = qmc.LatinHypercube(d=dim_params, seed=seed)
        sample = sampler.random(n=self.n_generated)
        
        params = qmc.scale(sample, l_bounds, u_bounds)
        param_selection = pd.DataFrame(params)
        param_selection.columns = param_names
        
        ######################################################################
        # system selection params
        ######################################################################
        
        # FEMA P-695 studies for bay length selection
        config_dict   = {
            'num_frames' : [2, 2]
        }

        # generate random integers within the bounds and place into array
        config_names = list(config_dict.keys())       
        num_categories = len(config_dict)
        config_selection = np.empty([self.n_generated, num_categories])
        
        # set seed
        np.random.seed(seed)
        
        for index, (key, bounds) in enumerate(config_dict.items()):
            config_selection[:,index] = np.random.randint(bounds[0], 
                                                               high=bounds[1]+1, 
                                                               size=self.n_generated)
        config_selection = pd.DataFrame(config_selection)
        
        
        
        import random
        random.seed(seed)
        
        # upweigh LRBs to ensure fair split
        isol_sys_list = ['TFP', 'LRB']
        # isol_wts = [1, 3]
        
        structs = random.choices(struct_sys_list, k=self.n_generated)
        isols = random.choices(isol_sys_list, k=self.n_generated, weights=isol_wts)
        system_selection = pd.DataFrame(np.array([structs, isols]).T)
        system_names = ['superstructure_system', 'isolator_system']
        
        self.raw_input = pd.concat([system_selection,
                                   config_selection,
                                   param_selection], axis=1)
        self.raw_input.columns = system_names + config_names + param_names
        
        # temp add in for constants
        # from numpy import ceil, floor
        
        # find the number of bay (try to keep around 3 to 8)
        target_Lbay = 30.0
        target_hstory = 14.0
        self.raw_input['num_bays'] = self.raw_input.apply(
            lambda row: round(row['L_bldg']/target_Lbay), axis=1)
        self.raw_input['num_stories'] = self.raw_input.apply(
            lambda row: round(row['h_bldg']/target_hstory), axis=1)
        self.raw_input['L_bay'] = (self.raw_input['L_bldg'] / 
                                   self.raw_input['num_bays'])
        self.raw_input['h_story'] = (self.raw_input['h_bldg'] / 
                                     self.raw_input['num_stories'])
        self.raw_input['S_s'] = 2.2815
        
###############################################################################
# Designing isolation systems
###############################################################################

    # use filter_designs=True if only realistic/physically sensible designs are
    # retained. This may result in the LHS distribution being uneven.
    
    # loads are also defined here
    
    def design_bearings(self, filter_designs=True):
        import time
        import pandas as pd
        
        df_raw = self.raw_input
        
        # get loading conditions
        from loads import define_gravity_loads
        df_raw[['W', 
               'W_s', 
               'w_fl', 
               'P_lc',
               'all_w_cases',
               'all_Plc_cases']] = df_raw.apply(lambda row: define_gravity_loads(row),
                                                axis='columns', result_type='expand')
        
        # separate df into isolator systems
        import design as ds
        df_tfp = df_raw[df_raw['isolator_system'] == 'TFP']
        
        
        # attempt to design all TFPs
        if df_tfp.shape[0] > 0:
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
                                                  (all_tfp_designs['R_2'] <= 180.0) &
                                                  (all_tfp_designs['zeta_e'] <= 0.25)]
            
            tp = time.time() - t0
            
            print("Designs completed for %d TFPs in %.2f s" %
                  (tfp_designs.shape[0], tp))
            
            # get the design params of those bearings
            a = df_tfp[df_tfp.index.isin(tfp_designs.index)]
        
            self.tfp_designs = pd.concat([a, tfp_designs], axis=1)
        else:
            self.tfp_designs = None
        
        df_lrb = df_raw[df_raw['isolator_system'] == 'LRB']
            
        
        # attempt to design all LRBs
        if df_lrb.shape[0] > 0:
            t0 = time.time()
            all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row),
                                           axis='columns', result_type='expand')
            
            
            all_lrb_designs.columns = ['d_bearing', 'd_lead', 't_r', 't', 'n_layers',
                                       'N_lb', 'S_pad', 'S_2',
                                       'T_e', 'k_e', 'zeta_e', 'D_m', 'buckling_fail']
            
            if filter_designs == False:
                lrb_designs = all_lrb_designs
            else:
                # keep the designs that look sensible
                # limits from design example CE 223
                lrb_designs = all_lrb_designs.loc[(all_lrb_designs['d_bearing'] >=
                                                   3*all_lrb_designs['d_lead']) &
                                                  (all_lrb_designs['d_bearing'] <=
                                                   6*all_lrb_designs['d_lead']) &
                                                  (all_lrb_designs['d_lead'] <= 
                                                    all_lrb_designs['t_r']) &
                                                  (all_lrb_designs['t_r'] > 4.0) &
                                                  (all_lrb_designs['t_r'] < 35.0) &
                                                  (all_lrb_designs['buckling_fail'] == 0) &
                                                  (all_lrb_designs['zeta_e'] <= 0.25)]
                
                lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
                
            tp = time.time() - t0
            
            print("Designs completed for %d LRBs in %.2f s" %
                  (lrb_designs.shape[0], tp))
            
            b = df_lrb[df_lrb.index.isin(lrb_designs.index)]
            
            self.lrb_designs = pd.concat([b, lrb_designs], axis=1)
        else:
            self.lrb_designs = None
            
    def design_structure(self, filter_designs=True):
        import pandas as pd
        import time
        
        # combine both set of isolator designs
        df_in = pd.concat([self.tfp_designs, self.lrb_designs], axis=0)
        
        from loads import define_lateral_forces
        
        # assumes that there is at least one design
        df_in[['wx', 
               'hx', 
               'h_col', 
               'hsx', 
               'Fx', 
               'Vs',
               'T_fbe']] = df_in.apply(lambda row: define_lateral_forces(row),
                                    axis='columns', result_type='expand')
        
        # separate by superstructure systems
        smrf_df = df_in[df_in['superstructure_system'] == 'MF']
        cbf_df = df_in[df_in['superstructure_system'] == 'CBF']
        
        # attempt to design all moment frames
        if smrf_df.shape[0] > 0:
            t0 = time.time()
            import design as ds
            
            all_mf_designs = smrf_df.apply(lambda row: ds.design_MF(row),
                                           axis='columns', 
                                           result_type='expand')
            
            all_mf_designs.columns = ['beam', 'column', 'flag']
            
            if filter_designs == False:
                mf_designs = all_mf_designs
            else:
                # keep the designs that look sensible
                mf_designs = all_mf_designs.loc[all_mf_designs['flag'] == False]
                mf_designs = mf_designs.dropna(subset=['beam','column'])
             
            mf_designs = mf_designs.drop(['flag'], axis=1)
            tp = time.time() - t0
          
            # get the design params of those bearings
            a = smrf_df[smrf_df.index.isin(mf_designs.index)]
            
            self.mf_designs = pd.concat([a, mf_designs], axis=1)
            
            print("Designs completed for %d moment frames in %.2f s" %
                  (smrf_df.shape[0], tp))
        else:
            self.mf_designs = None
            
        # attempt to design all CBFs
        if cbf_df.shape[0] > 0:
            t0 = time.time()
            all_cbf_designs = cbf_df.apply(lambda row: ds.design_CBF(row),
                                            axis='columns', 
                                            result_type='expand')
            all_cbf_designs.columns = ['brace', 'beam', 'column']
            if filter_designs == False:
                cbf_designs = all_cbf_designs
            else:
                # keep the designs that look sensible
                cbf_designs = all_cbf_designs.dropna(subset=['beam','column','brace'])
                
            
            tp = time.time() - t0
            
            # get the design params of those bearings
            a = cbf_df[cbf_df.index.isin(cbf_designs.index)]
            self.cbf_designs = pd.concat([a, cbf_designs], axis=1)
            
            print("Designs completed for %d braced frames in %.2f s" %
                  (cbf_df.shape[0], tp))
        else:
            self.cbf_designs = None
        
        # join both systems (assumes that there's at least one design)
        all_des = pd.concat([self.mf_designs, self.cbf_designs], 
                                     axis=0)
        # retained designs
        # ensure even distribution between number of systems
        n_systems = len(pd.unique(df_in['superstructure_system']))
        self.retained_designs = all_des.groupby(
            'superstructure_system', group_keys=False).apply(
            lambda x: x.sample(n=int(self.n_points/n_systems)))
        self.generated_designs = all_des
        
        print('======================================')
        print('Final database: %d structures.' % len(self.retained_designs))
        print('%d moment frames | %d braced frames' % 
              (len(self.retained_designs[
                  self.retained_designs['superstructure_system'] == 'MF']),
               len(self.retained_designs[
                   self.retained_designs['superstructure_system'] == 'CBF'])))
        print('%d LRBs | %d TFPs' % 
              (len(self.retained_designs[
                  self.retained_designs['isolator_system'] == 'LRB']),
               len(self.retained_designs[
                   self.retained_designs['isolator_system'] == 'TFP'])))
        print('======================================')
        
    def scale_gms(self):
        
        
        # only scale motions that will be retained
        all_des = self.retained_designs.copy()
        
        # set seed to ensure same GMs are selected
        from random import seed
        seed(985)
        
        # scale and select ground motion
        # TODO: this section is inefficient
        from gms import scale_ground_motion
        import time
        t0 = time.time()
        all_des[['gm_selected',
                 'scale_factor',
                 'sa_avg']] = all_des.apply(lambda row: scale_ground_motion(row),
                                            axis='columns', result_type='expand')
        tp = time.time() - t0
        print("Scaled ground motions for %d structures in %.2f s" %
              (all_des.shape[0], tp))
        
        self.retained_designs = all_des
        
    def analyze_db(self, output_str, save_interval=10,
                   data_path='../data/',
                   gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/'):
        
        from experiment import run_nlth
        import pandas as pd
        
        all_designs = self.retained_designs
        
        db_results = None
        
        for index, design in all_designs.iterrows():
            i_run = all_designs.index.get_loc(index)
            print('========= Run %d of %d ==========' % 
                  (i_run+1, len(all_designs)))
            bldg_result = run_nlth(design, gm_path)
            
            # TODO: find a way to keep indices
            # if initial run, start the dataframe with headers
            if db_results is None:
                db_results = pd.DataFrame(bldg_result).T
            else:
                db_results = pd.concat([db_results,bldg_result.to_frame().T], 
                                       sort=False)
                
            if (len(db_results)%save_interval == 0):
                db_results.to_csv(data_path+'temp_save.csv', index=False)
        
        db_results.to_csv(data_path+output_str, index=False)
        self.ops_analysis = db_results
        
        
    def calculate_collapse(self):
        df = self.ops_analysis
        
        from experiment import collapse_fragility
        df[['max_drift',
           'collapse_prob']] = df.apply(lambda row: collapse_fragility(row),
                                                axis='columns', result_type='expand')
                               
        from numpy import log
        df['log_collapse_prob'] = log(df['collapse_prob'])
        
        self.ops_analysis = df
                
    def perform_doe(self, target_prob=0.5, n_set=200, batch_size=10):
        
        try:
            whole_set = self.ops_analysis
        except:
            print('Cannot perform DoE without analysis results')
            
        # TODO: temporary assign variable here, also need to calculate gap ratio
        whole_set['T_ratio'] = whole_set['T_m'] / whole_set['T_fb']
        
        # n_set is both test_train split
        ml_set = whole_set.sample(n=n_set, replace=False, random_state=985)
        
        # split 50/50 for 
        df_train = ml_set.head(int(n_set/2))
        df_test = ml_set.tail(int(n_set/2))
        
        from experiment import run_doe
        
        df_doe, rmse_hist, mae_hist = run_doe(target_prob, df_train, df_test, 
                                             batch_size=batch_size, error_tol=1e-2, 
                                             maxIter=600, conv_tol=1e-5)
        
        self.doe_analysis = df_doe
        self.rmse_hist = rmse_hist
        self.mae_hist = mae_hist
        pass