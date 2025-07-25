############################################################################
#               Database object for structural systems

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Object acts as central aggregate for designs, datasets

# Open issues:  (1) ranges of params dependent on system

############################################################################

import numpy as np
import pandas as pd

# TODO: package inverse design/ML models into this
class Database:
    
    # sets up the problem by generating building specifications
    # n_points: points to generate
    # n_buffer: generate n_buffer*n_points designs to account for discards
    # struct_sys_list: systems list
    # isol_wts: currently hardcoded for ['TFP', 'LRB'], sample 1:3 TFP:LRB ratio
    # to account for harder LRB designs being discarded
    
    def __init__(self, n_points=400, seed=985, n_buffer=15,
                 struct_sys_list=['MF', 'CBF'], isol_sys_list=['TFP','LRB'],
                 isol_wts=[1.0,2.5]):
        
        from scipy.stats import qmc
        
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
            
        # self.param_ranges   = {
        #     'S_1' : [0.8, 1.3],
        #     'T_m' : [2.5, 5.0],
        #     'k_ratio' :[5.0, 18.0],
        #     'moat_ampli' : [0.5, 1.2],
        #     'RI' : [0.5, 2.25],
        #     'L_bldg': [75.0, 250.0],
        #     'h_bldg': [30.0, 100.0],
        #     'zeta_e': [0.1, 0.25]
        # }
        
        # # expanded bounds, with allowable drift
        # self.param_ranges   = {
        #     'S_1' : [0.5, 1.3],
        #     'T_m' : [2.5, 5.0],
        #     'k_ratio' :[5.0, 18.0],
        #     'moat_ampli' : [0.5, 1.5],
        #     'RI' : [0.3, 2.5],
        #     'L_bldg': [75.0, 250.0],
        #     'h_bldg': [30.0, 100.0],
        #     'zeta_e': [0.1, 0.25],
        #     'delta_a': [0.005, 0.02]
        # }
        
        # expanded bounds using S1_ampli
        self.param_ranges   = {
            'S_1' : [0.8, 1.3],
            'S_1_ampli': [0.25, 2.0],
            'T_m' : [2.5, 5.0],
            'k_ratio' :[5.0, 18.0],
            'moat_ampli' : [0.5, 1.5],
            'RI' : [0.3, 2.5],
            'L_bldg': [75.0, 250.0],
            'h_bldg': [30.0, 100.0],
            'zeta_e': [0.1, 0.25]
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
        import random
        random.seed(seed)
        
        for index, (key, bounds) in enumerate(config_dict.items()):
            config_selection[:,index] = np.random.randint(bounds[0], 
                                                               high=bounds[1]+1, 
                                                               size=self.n_generated)
        config_selection = pd.DataFrame(config_selection)
        
        # upweigh LRBs to ensure fair split
        # isol_sys_list = ['TFP', 'LRB']
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
        
        df_raw = self.raw_input
        self.tfp_designs, self.lrb_designs = design_bearing_util(
            df_raw, filter_designs=filter_designs)
        
            
    def design_structure(self, filter_designs=True):
        import pandas as pd
        
        # combine both set of isolator designs
        df_in = pd.concat([self.tfp_designs, self.lrb_designs], axis=0)
        
        self.mf_designs, self.cbf_designs = design_structure_util(
            df_in, filter_designs=filter_designs)
        
        # join both systems (assumes that there's at least one design)
        all_des = pd.concat([self.mf_designs, self.cbf_designs], 
                                     axis=0)
        # retained designs
        # ensure even distribution between number of systems
        all_des['combined_system'] = all_des['superstructure_system'] + '-' + all_des['isolator_system']
        n_systems = len(pd.unique(all_des['combined_system']))
                
        # add a duplicate struct_system column to facilitate groupby dropping (whY???)
        all_des['system_drop'] = all_des['combined_system'].copy()
        # all_des['supersystem_drop'] = all_des['superstructure_system'].copy()
        
        self.retained_designs = all_des.groupby(
            'system_drop', group_keys=False).apply(
            lambda x: x.sample(n=int(self.n_points/n_systems), random_state=985), include_groups=False)
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
        
    def scale_gms(self, repeat=False, seed=985):
        
        
        # only scale motions that will be retained
        all_des = self.retained_designs.copy()
        
        # put many GMs on same design if testing record-to-record variance
        if repeat != False:
            all_des = all_des.loc[all_des.index.repeat(repeat)]
            
        # set seed to ensure same GMs are selected
        import random
        random.seed(seed)
        
        # scale and select ground motion
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
                   gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/',
                   output_path='./outputs/'):
        
        from experiment import run_nlth
        import pandas as pd
        
        all_designs = self.retained_designs
        all_designs = all_designs.reset_index()
        db_results = None
        
        import os
        import shutil
        
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        
        for index, design in all_designs.iterrows():
            i_run = all_designs.index.get_loc(index)
            print('========= Run %d of %d ==========' % 
                  (i_run+1, len(all_designs)))
            bldg_result = run_nlth(design=design, gm_path=gm_path, output_path=output_path)
            
            # if initial run, start the dataframe with headers
            if db_results is None:
                db_results = pd.DataFrame(bldg_result).T
            else:
                db_results = pd.concat([db_results,bldg_result.to_frame().T], 
                                       sort=False)
                
            if (len(db_results)%save_interval == 0):
                db_results.to_csv(output_path+'temp_save.csv', index=False)
        
        db_results.to_csv(data_path+output_str, index=False)
        self.ops_analysis = db_results
        
        
    def calculate_collapse(self, mf_reference_drift=0.1, cbf_reference_drift=0.05):
        df = self.ops_analysis
        
        from experiment import collapse_fragility
        df[['max_drift',
           'collapse_prob']] = df.apply(
               lambda row: collapse_fragility(
                   row, mf_drift_mu_plus_std=mf_reference_drift,
                   cbf_drift_90=cbf_reference_drift),
               axis='columns', result_type='expand')
                               
        from numpy import log
        df['log_collapse_prob'] = log(df['collapse_prob'])
        
        self.ops_analysis = df
                
    def perform_doe(self, target_prob=0.5, n_set=200, max_iters=1000,
                    batch_size=10, kernel='rbf_iso', strategy='balanced'):
        
        try:
            whole_set = self.ops_analysis
        except:
            print('Cannot perform DoE without analysis results')
            
        # n_set is both test_train split
        ml_set = whole_set.sample(n=n_set, replace=False, random_state=985)
        
        # split 50/50 for holdout set
        df_train = ml_set.head(int(n_set/2))
        df_test = ml_set.tail(int(n_set/2))
        
        self.training_set = df_train
        self.testing_set = df_test
        
        from experiment import run_doe
        
        df_doe, rmse_hist, mae_hist, nrmse_hist, hyperparam_list = run_doe(
            target_prob, df_train, df_test, batch_size=batch_size, error_tol=1e-2, 
            maxIter=max_iters, conv_tol=1e-4, kernel=kernel, doe_strat=strategy)
        
        self.doe_analysis = df_doe
        self.rmse_hist = rmse_hist
        self.mae_hist = mae_hist
        self.nrmse_hist = nrmse_hist
        self.hyperparam_list = hyperparam_list
        
    # TODO: unify design procedure for main, DoE, IDAs, validation
    def set_design(self, design, title):
        pass
    
    def prepare_pushover(self, design_df):
        import pandas as pd
        
        config_dict = {
            'S_1' : 1.017,
            'k_ratio' : 10,
            'Q': 0.06,
            'num_frames' : 2,
            'num_bays' : 4,
            'num_stories' : 4,
            'L_bay': 30.0,
            'h_story': 13.0,
            'isolator_system' : 'TFP',
            'superstructure_system' : 'MF',
            'S_s' : 2.2815
        }
        
        config_dict['L_bldg'] = config_dict['num_bays'] * config_dict['L_bay']
        config_dict['h_bldg'] = config_dict['num_stories'] * config_dict['h_story']
        
        work_df = pd.DataFrame(config_dict, index=[0])
        from loads import estimate_period
        work_df['T_fbe'] = estimate_period(work_df.iloc[0])
        
        work_df = pd.concat([work_df, design_df.set_index(work_df.index)], 
                            axis=1)
        
        import design as ds
        from loads import define_lateral_forces, define_gravity_loads
        from gms import scale_ground_motion
        
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
        work_df['moat_ampli'] = work_df['gap_ratio']
        
        # design
        work_df[['W', 
               'W_s', 
               'w_fl', 
               'P_lc',
               'all_w_cases',
               'all_Plc_cases']] = work_df.apply(lambda row: define_gravity_loads(row),
                                                axis='columns', result_type='expand')
                     
        try:
            all_tfp_designs = work_df.apply(lambda row: ds.design_TFP(row),
                                           axis='columns', result_type='expand')
        except:
            print('Bearing design failed.')
            return
        
        all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                   'T_e', 'k_e', 'Q', 'zeta_e', 'D_m']
        
        tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                          (all_tfp_designs['R_1'] <= 50.0) &
                                          (all_tfp_designs['R_2'] <= 190.0) &
                                          (all_tfp_designs['zeta_e'] <= 0.27)]
        
        # retry if design didn't work
        if tfp_designs.shape[0] == 0:
            print('Bearing design failed.')
            return
        
        tfp_designs = tfp_designs.drop(columns=['zeta_e'])
        work_df = pd.concat([work_df, tfp_designs.set_index(work_df.index)], 
                            axis=1)
        
        # get lateral force and design structures
        work_df[['wx', 
               'hx', 
               'h_col', 
               'hsx', 
               'Fx', 
               'Vs',
               'T_fbe']] = work_df.apply(lambda row: define_lateral_forces(row),
                                    axis='columns', result_type='expand')
        work_df['Vb'] = work_df['k_e'] * work_df['W'] * work_df['D_m'] / work_df['num_frames']
        all_mf_designs = work_df.apply(lambda row: ds.design_MF(row),
                                         axis='columns', 
                                         result_type='expand')
          
        all_mf_designs.columns = ['beam', 'column', 'flag']
        
        # keep the designs that look sensible
        mf_designs = all_mf_designs.loc[all_mf_designs['flag'] == False]
        mf_designs = mf_designs.dropna(subset=['beam','column'])
         
        mf_designs = mf_designs.drop(['flag'], axis=1)
        
        if mf_designs.shape[0] == 0:
            print('Bearing design failed.')
            return
      
        # get the design params of those bearings
        work_df = pd.concat([work_df, mf_designs.set_index(work_df.index)], 
                            axis=1)
        
        self.pushover_design = work_df
        
    def prepare_idas(self, design_dict, levels=[1.0, 1.5, 2.0]):
        
        config_dict = {
            'S_1' : 1.017,
            'L_bldg': 120.0,
            'h_bldg': 52.0,
            'num_frames' : 2,
            'num_bays' : 4,
            'num_stories' : 4,
            'L_bay': 30.0,
            'h_story': 13.0,
            'S_s' : 2.2815
        }
        
        work_df = pd.DataFrame(config_dict, index=[0])
        work_df = pd.concat([work_df, design_dict.set_index(work_df.index)], 
                            axis=1)
        
        from loads import estimate_period
        work_df['T_fbe'] = estimate_period(work_df.iloc[0])
        
        
        import design as ds
        from loads import define_lateral_forces, define_gravity_loads
        from gms import scale_ground_motion
        
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
        work_df['moat_ampli'] = work_df['gap_ratio']
        
        all_tfps, all_lrbs = design_bearing_util(work_df, filter_designs=False)
        
        if work_df['isolator_system'].item() == 'TFP':
            # keep the designs that look sensible
            tfp_designs = all_tfps.loc[(all_tfps['R_1'] >= 10.0) &
                                       (all_tfps['R_1'] <= 50.0) &
                                       (all_tfps['R_2'] <= 190.0) &
                                       (all_tfps['zeta_loop'] <= 0.27)]
            
            
            if tfp_designs.shape[0] == 0:
                all_tfps, lrb_designs = design_bearing_util(
                    work_df, filter_designs=False, mu_1_force=0.06)
                
                # keep the designs that look sensible
                tfp_designs = all_tfps.loc[(all_tfps['R_1'] >= 10.0) &
                                           (all_tfps['R_1'] <= 50.0) &
                                           (all_tfps['R_2'] <= 190.0) &
                                           (all_tfps['zeta_loop'] <= 0.27)]
            
            # retry if design didn't work
            if tfp_designs.shape[0] == 0:
                print('Bearing design failed')
                return
            
            work_df = tfp_designs.copy()
            
        else:
            lrb_designs = all_lrbs.loc[(all_lrbs['d_bearing'] >=
                                               3*all_lrbs['d_lead']) &
                                              (all_lrbs['d_bearing'] <=
                                               6*all_lrbs['d_lead']) &
                                              (all_lrbs['d_lead'] <= 
                                                all_lrbs['t_r']) &
                                              (all_lrbs['t_r'] > 4.0) &
                                              (all_lrbs['t_r'] < 35.0) &
                                              (all_lrbs['buckling_fail'] == 0) &
                                              (all_lrbs['zeta_loop'] <= 0.27)]
            
            lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
            
            # retry if design didn't work
            if lrb_designs.shape[0] == 0:
                print('Bearing design failed')
                return
            
            work_df = lrb_designs.copy()
        
        mf_designs, cbf_designs = design_structure_util(
            work_df, filter_designs=True)
        
        if work_df['superstructure_system'].item() == 'MF':
            work_df = mf_designs.copy()
        else:
            work_df = cbf_designs.copy()
        
        
        gm_series, sf_series, sa_avg = scale_ground_motion(work_df.iloc[0], return_list=True)
        ida_base = pd.concat([gm_series, sf_series], axis=1)
        ida_base['sa_avg'] = sa_avg
        ida_base.columns = ['gm_selected', 'scale_factor', 'sa_avg']
        ida_base = ida_base.reset_index(drop=True)
        
        ida_gms = None
        
        # prepare the sets of ida levels
        for lvl in levels:
            ida_level = ida_base[['scale_factor', 'sa_avg']].copy()
            ida_level = ida_level*lvl
            ida_level['ida_level'] = lvl
            ida_level['gm_selected'] = ida_base['gm_selected']
            if ida_gms is None:
                ida_gms = ida_level.copy()
            else:
                ida_gms = pd.concat([ida_gms, ida_level], axis=0)
            
        ida_gms = ida_gms.reset_index(drop=True)
        
        ida_df = pd.DataFrame(np.repeat(work_df.values, ida_gms.shape[0], axis=0))
        ida_df.columns = work_df.columns
        
        self.ida_df = pd.concat([ida_df, ida_gms], axis=1)
      
    def prepare_ida_legacy(self, design_df, levels=[1.0, 1.5, 2.0]):
        
        config_dict = {
            'S_1' : 1.017,
            'k_ratio' : 10,
            'Q': 0.06,
            'L_bldg': 120.0,
            'h_bldg': 52.0,
            'num_frames' : 2,
            'num_bays' : 4,
            'num_stories' : 4,
            'L_bay': 30.0,
            'h_story': 13.0,
            'isolator_system' : 'TFP',
            'superstructure_system' : 'MF',
            'S_s' : 2.2815
        }
        
        work_df = pd.DataFrame(config_dict, index=[0])
        from loads import estimate_period
        work_df['T_fbe'] = estimate_period(work_df.iloc[0])
        
        work_df = pd.concat([work_df, design_df.set_index(work_df.index)], 
                            axis=1)
        
        import design as ds
        from loads import define_lateral_forces, define_gravity_loads
        from gms import scale_ground_motion
        
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
        work_df['moat_ampli'] = work_df['gap_ratio']
        
        # design
        work_df[['W', 
               'W_s', 
               'w_fl', 
               'P_lc',
               'all_w_cases',
               'all_Plc_cases']] = work_df.apply(lambda row: define_gravity_loads(row),
                                                axis='columns', result_type='expand')
                     
        try:
            all_tfp_designs = work_df.apply(lambda row: ds.design_TFP_legacy(row),
                                           axis='columns', result_type='expand')
        except:
            print('Bearing design failed.')
            return
        
        all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                   'T_e', 'k_e', 'zeta_e', 'D_m']
        
        tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                          (all_tfp_designs['R_1'] <= 50.0) &
                                          (all_tfp_designs['R_2'] <= 190.0) &
                                          (all_tfp_designs['zeta_e'] <= 0.27)]
        
        # retry if design didn't work
        if tfp_designs.shape[0] == 0:
            print('Bearing design failed.')
            return
        
        tfp_designs = tfp_designs.drop(columns=['zeta_e'])
        work_df = pd.concat([work_df, tfp_designs.set_index(work_df.index)], 
                            axis=1)
        
        # get lateral force and design structures
        work_df[['wx', 
               'hx', 
               'h_col', 
               'hsx', 
               'Fx', 
               'Vs',
               'T_fbe']] = work_df.apply(lambda row: define_lateral_forces(row),
                                    axis='columns', result_type='expand')
                                          
        all_mf_designs = work_df.apply(lambda row: ds.design_MF(row),
                                         axis='columns', 
                                         result_type='expand')
          
        all_mf_designs.columns = ['beam', 'column', 'flag']
        
        # keep the designs that look sensible
        mf_designs = all_mf_designs.loc[all_mf_designs['flag'] == False]
        mf_designs = mf_designs.dropna(subset=['beam','column'])
         
        mf_designs = mf_designs.drop(['flag'], axis=1)
        
        if mf_designs.shape[0] == 0:
            print('Bearing design failed.')
            return
      
        # get the design params of those bearings
        work_df = pd.concat([work_df, mf_designs.set_index(work_df.index)], 
                            axis=1)
        
        gm_series, sf_series, sa_avg = scale_ground_motion(work_df.iloc[0], return_list=True)
        ida_base = pd.concat([gm_series, sf_series], axis=1)
        ida_base['sa_avg'] = sa_avg
        ida_base.columns = ['gm_selected', 'scale_factor', 'sa_avg']
        ida_base = ida_base.reset_index(drop=True)
        
        ida_gms = None
        # prepare the sets of ida levels
        for lvl in levels:
            ida_level = ida_base[['scale_factor', 'sa_avg']].copy()
            ida_level = ida_level*lvl
            ida_level['ida_level'] = lvl
            ida_level['gm_selected'] = ida_base['gm_selected']
            if ida_gms is None:
                ida_gms = ida_level.copy()
            else:
                ida_gms = pd.concat([ida_gms, ida_level], axis=0)
            
        ida_gms = ida_gms.reset_index(drop=True)
        
        ida_df = pd.DataFrame(np.repeat(work_df.values, ida_gms.shape[0], axis=0))
        ida_df.columns = work_df.columns
        
        self.ida_df = pd.concat([ida_df, ida_gms], axis=1)
    
    def analyze_ida(self, output_str, save_interval=10,
                   data_path='../data/validation/',
                   gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/'):
        
        from experiment import run_nlth
        import pandas as pd
        
        all_designs = self.ida_df
        all_designs = all_designs.reset_index()
        
        db_results = None
        print('========= Validation IDAs ==========')
        
        for index, design in all_designs.iterrows():
            i_run = all_designs.index.get_loc(index)
            print('========= Run %d of %d ==========' % 
                  (i_run+1, len(all_designs)))
            
            print('IDA level: %.1f' % design.ida_level)
            bldg_result = run_nlth(design, gm_path)
            
            # if initial run, start the dataframe with headers
            if db_results is None:
                db_results = pd.DataFrame(bldg_result).T
            else:
                db_results = pd.concat([db_results,bldg_result.to_frame().T], 
                                       sort=False)
                
            if (len(db_results)%save_interval == 0):
                db_results.to_csv(data_path+'ida_temp_save.csv', index=False)
        
        db_results.to_csv(data_path+output_str, index=False)
        
        self.ida_results = db_results
        
    # this runs Pelicun in the deterministic style. collect_IDA flag used if running
    # validation IDAs
    def run_pelicun(self, df, collect_IDA=False,
                    cmp_dir='../resource/loss/', max_loss_df=None):
        # run info

        # and import pelicun classes and methods
        from pelicun.assessment import Assessment
        from loss import Loss_Analysis
        
        from scipy.stats import ecdf, norm
        
        # make lambda function for generic lognormal distribution
        lognorm_f = lambda x,theta,beta: norm(np.log(theta), beta**0.5).cdf(np.log(x))
        
        # make lambda function for generic weibull distribution
        from scipy.stats import weibull_min
        weibull_f = lambda x,k,lam,loc: weibull_min(k, loc=loc, scale=lam).cdf(x)
        
        from scipy.stats import kstest

        # get database
        # initialize, no printing outputs, offset fixed with current components
        PAL = Assessment({
            "PrintLog": False, 
            "Seed": 985,
            "Verbose": False,
            "DemandOffset": {"PFA": 0, "PFV": 0}
        })

        # generate structural components and join with NSCs
        P58_metadata = PAL.get_default_metadata('loss_repair_DB_FEMA_P58_2nd')
        
        additional_frag_db = pd.read_csv(cmp_dir+'custom_component_fragilities.csv',
                                          header=[0,1], index_col=0)
        
        # lab, health, ed, res, office, retail, warehouse, hotel
        fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
        
        df = df.reset_index(drop=True)
        
        # estimate loss for set
        all_losses = []
        loss_cmp_group = []
        col_list = []
        irr_list = []
        
        # lognormal parameters
        theta_cost_list = []
        beta_cost_list = []
        theta_time_list = []
        beta_time_list = []
        
        # quantiles
        cost_quantile_list = []
        time_quantile_list = []    
        
        # weibull parameters
        k_cost_list = []
        k_time_list = []
        lam_cost_list = []
        lam_time_list = []
        
        # weibull truncated parameters
        k_trunc_cost_list = []
        k_trunc_time_list = []
        lam_trunc_cost_list = []
        lam_trunc_time_list = []
        
        # ks stats
        weibull_cost_ks_pvalue_list = []
        weibull_trunc_cost_ks_pvalue_list = []
        lognormal_cost_ks_pvalue_list = []
        
        weibull_time_ks_pvalue_list = []
        weibull_trunc_time_ks_pvalue_list = []
        lognormal_time_ks_pvalue_list = []
        
        # AIC 
        weibull_cost_aic_list = []
        weibull_trunc_cost_aic_list = []
        lognormal_cost_aic_list = []
        
        weibull_time_aic_list = []
        weibull_trunc_time_aic_list = []
        lognormal_time_aic_list = []
        
        if collect_IDA:
            IDA_list = []
            
        # if max loss df is provided, we now use the median loss of the total 
        # damage scenario as the replacement consequences
        if max_loss_df is not None:
            
            '''
            # if the max_loss_df is only one row, replicate it (for validation run)
            if max_loss_df.shape[0] < 2:
                len_df = df.shape[0]
                col_names = max_loss_df.columns
                max_loss_df = pd.DataFrame(np.repeat(max_loss_df.values, len_df, axis=0))
                max_loss_df.columns = col_names
            '''
                
            max_costs = max_loss_df['cost_50%']
            max_time = max_loss_df['time_l_50%']
        else:
            max_costs = None*df.shape[0]
            max_time = None*df.shape[0]
            
            
        for run_idx in df.index:
            print('========================================')
            print('Estimating loss for run index', run_idx+1)
            
            run_data = df.loc[run_idx]
            
            floors = run_data.num_stories
            
            bldg_usage = [fl_usage]*floors

            loss = Loss_Analysis(run_data)
            loss.nqe_sheets()
            loss.normative_quantity_estimation(bldg_usage, P58_metadata)

            loss.process_EDP()
            
            run_max_cost = max_costs.loc[run_idx]
            run_max_time = max_time.loc[run_idx]
            
            # if collect_IDA, a validation run is indicated, treat EDP as deterministic
            if collect_IDA:
                [cmp, dmg, loss, loss_cmp, agg, 
                  collapse_rate, irr_rate] = loss.estimate_damage(
                      custom_fragility_db=additional_frag_db, mode='generate',
                      cmp_replacement_cost=run_max_cost, cmp_replacement_time=run_max_time)
            # else, try to fit lognormal around edp
            else:
                [cmp, dmg, loss, loss_cmp, agg, 
                 collapse_rate, irr_rate] = loss.estimate_damage(
                     custom_fragility_db=additional_frag_db, mode='sample_lognormal',
                     cmp_replacement_cost=run_max_cost, cmp_replacement_time=run_max_time)
                 
            # Collect quantiles
            loss_summary = agg.describe([0.1, 0.5, 0.9])
            cost = loss_summary['repair_cost']['50%']
            time_l = loss_summary[('repair_time', 'parallel')]['50%']
            time_u = loss_summary[('repair_time', 'sequential')]['50%']
            
            q_array = np.arange(0.05, 1.0, 0.05)
            loss_quantiles  = agg.quantile(q_array)
            
            cost_quantile_list.append(loss_quantiles['repair_cost'])
            time_quantile_list.append(loss_quantiles['repair_time']['parallel'])
            
            print('Median repair cost: ', 
                  f'${cost:,.2f}')
            print('Median lower bound repair time: ', 
                  f'{time_l:,.2f}', 'worker-days')
            print('Median upper bound repair time: ', 
                  f'{time_u:,.2f}', 'worker-days')
            print('Collapse frequency: ', 
                  f'{collapse_rate:.2%}')
            print('Irreparable RID frequency: ', 
                  f'{irr_rate:.2%}')
            print('Replacement frequency: ', 
                  f'{collapse_rate+irr_rate:.2%}')
            
            all_losses.append(loss_summary)
            loss_cmp_group.append(loss_cmp)
            col_list.append(collapse_rate)
            irr_list.append(irr_rate)
            
            ######## cost distro stats ##################
            # collect cost distribution stats for time analysis (weibull)
            # adding a dollar to make sure no divide by zero
            my_y_var = agg['repair_cost']+1
            k_cost, lam_cost = mle_weibull(my_y_var)
            loc_trunc_cost = my_y_var.min()
            k_trunc_cost, lam_trunc_cost = mle_weibull(my_y_var - loc_trunc_cost)
            
            log_lik_weibull_cost = -nlls_weibull((k_cost, lam_cost), my_y_var)
            log_lik_weibull_trunc_cost = -nlls_weibull(
                (k_trunc_cost, lam_trunc_cost), my_y_var-loc_trunc_cost)
            
            # closed-form MLE calculation of lognormal parameters
            # collect cost distribution stats for time analysis (lognormal)
            theta_cost = np.exp(np.log(my_y_var).mean())
            beta_cost = np.log(my_y_var).var()
            log_lik_ln_cost = -nlls_normal(
                (np.log(theta_cost), beta_cost), np.log(my_y_var))
            
            # AIC, with weibull estimating 2 parameters
            k = 2
            AIC_weibull_cost = 2*k - 2*log_lik_weibull_cost
            AIC_weibull_trunc_cost = 2*k - 2*log_lik_weibull_trunc_cost
            
            # AIC, with lognormal estimating 2 parameters
            # "On the likelihood of a time series model" (Akaike, 1978, pg. 224) 
            # describes adjusting the AIC for a transformed variable
            k = 2
            log_lik_ln_cost_adj = np.sum(2*np.log(my_y_var + 1))
            AIC_ln_cost = 2*k - 2*log_lik_ln_cost + log_lik_ln_cost_adj
            
            # Set up Kolmogorov-Smirnov test
            
            # null hypothesis: cost is distributed weibull
            res = ecdf(my_y_var)
            ecdf_values = res.cdf.quantiles
            # if p < 0.05, the alternative is true (cost is not weibull)
            ks_results_weibull_cost = kstest(
                ecdf_values, 
                weibull_min(k_cost, loc=0, scale=lam_cost).cdf)
            
            # if p < 0.05, the alternative is true (cost is not truncated weibull)
            ks_results_weibull_trunc_cost = kstest(
                ecdf_values, 
                weibull_min(k_trunc_cost, loc=loc_trunc_cost, scale=lam_trunc_cost).cdf)
            
            # null hypothesis: cost is distributed lognormal
            ln_f = norm(np.log(theta_cost), beta_cost**0.5).cdf
            # if p < 0.05, the alternative is true (cost is not lognormal)
            ks_results_lognormal_cost = kstest(np.log(ecdf_values), ln_f)
            
            ######## downtime distro stats ##################
            # collect downtime distribution stats for time analysis (weibull)
            # making sure one worker stays for 30 mins to ensure no divide by zero
            my_y_var = agg[('repair_time', 'parallel')]+0.5
            k_time, lam_time = mle_weibull(my_y_var)
            loc_trunc_time = my_y_var.min()
            k_trunc_time, lam_trunc_time = mle_weibull(my_y_var - loc_trunc_time)
            
            log_lik_weibull_time = -nlls_weibull((k_time, lam_time), my_y_var)
            log_lik_weibull_trunc_time = -nlls_weibull(
                (k_time, lam_time), my_y_var-loc_trunc_time)
            
            # closed-form MLE calculation of lognormal parameters
            # collect downtime distribution stats for time analysis (lognormal)
            theta_time = np.exp(np.log(my_y_var).mean())
            beta_time = np.log(my_y_var).var()
            log_lik_ln_time = -nlls_normal((np.log(theta_time), beta_time), np.log(my_y_var))
            
            # AIC, with weibull estimating 2 parameters
            k = 2
            AIC_weibull_time = 2*k - 2*log_lik_weibull_time
            AIC_weibull_trunc_time = 2*k - 2*log_lik_weibull_trunc_time
            
            # AIC, with lognormal estimating 2 parameters
            k = 2
            log_lik_ln_time_adj = np.sum(2*np.log(my_y_var + 1))
            AIC_ln_time = 2*k - 2*log_lik_ln_time + log_lik_ln_time_adj
            
            # null hypothesis: time is distributed weibull
            res = ecdf(my_y_var)
            ecdf_values = res.cdf.quantiles
            # if p < 0.05, the alternative is true (time is not weibull)
            ks_results_weibull_time = kstest(
                ecdf_values, 
                weibull_min(k_time, loc=0, scale=lam_time).cdf)
            
            # if p < 0.05, the alternative is true (time is not truncated weibull)
            ks_results_weibull_trunc_time = kstest(
                ecdf_values, 
                weibull_min(k_trunc_time, loc=loc_trunc_time, scale=lam_trunc_time).cdf)
            
            # null hypothesis: time is distributed lognormal
            ln_f = norm(np.log(theta_time), beta_time**0.5).cdf
            # if p < 0.05, the alternative is true (time is not lognormal)
            ks_results_lognormal_time = kstest(np.log(ecdf_values), ln_f)
            
            # # old: relied on cdf fitting rather than true MLE
            # res = ecdf(my_y_var)
            # ecdf_prob = res.cdf.probabilities
            # ecdf_values = res.cdf.quantiles
            # theta_init = ecdf_values.mean()
            
            # try:
            #     theta_cost, beta_cost = mle_fit_general(ecdf_values,ecdf_prob, 
            #                                             x_init=(theta_init, 1.0))
            # except:
            #      theta_cost = theta_init
            #      beta_cost = 0.1
            
            
            # aggregate findings to list
            theta_cost_list.append(theta_cost)
            beta_cost_list.append(beta_cost)
            theta_time_list.append(theta_time)
            beta_time_list.append(beta_time)
            
            k_cost_list.append(k_cost)
            lam_cost_list.append(lam_cost)
            k_time_list.append(k_time)
            lam_time_list.append(lam_time)
            
            # weibull truncated parameters
            k_trunc_cost_list.append(k_trunc_cost)
            k_trunc_time_list.append(k_trunc_time)
            lam_trunc_cost_list.append(lam_trunc_cost)
            lam_trunc_time_list.append(lam_trunc_time)
            
            # ks stats
            weibull_cost_ks_pvalue_list.append(ks_results_weibull_cost.pvalue)
            weibull_trunc_cost_ks_pvalue_list.append(ks_results_weibull_trunc_cost.pvalue)
            lognormal_cost_ks_pvalue_list.append(ks_results_lognormal_cost.pvalue)
            
            weibull_time_ks_pvalue_list.append(ks_results_weibull_time.pvalue)
            weibull_trunc_time_ks_pvalue_list.append(ks_results_weibull_trunc_time.pvalue)
            lognormal_time_ks_pvalue_list.append(ks_results_lognormal_time.pvalue)
            
            # AIC 
            weibull_cost_aic_list.append(AIC_weibull_cost)
            weibull_trunc_cost_aic_list.append(AIC_weibull_trunc_cost)
            lognormal_cost_aic_list.append(AIC_ln_cost)
            
            weibull_time_aic_list.append(AIC_weibull_time)
            weibull_trunc_time_aic_list.append(AIC_weibull_trunc_time)
            lognormal_time_aic_list.append(AIC_ln_time)
            
            if collect_IDA:
                IDA_list.append(run_data['ida_level'])
                
                 
        
        # concat list of df into one df
        loss_df = pd.concat(all_losses)
        group_df = pd.concat(loss_cmp_group)
        
        loss_header = ['cost_mean', 'cost_std', 'cost_min',
                       'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
                       'time_l_mean', 'time_l_std', 'time_l_min',
                       'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
                       'time_u_mean', 'time_u_std', 'time_u_min',
                       'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']
        
        
        
        all_rows = []
        
        for row_idx in range(len(loss_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = loss_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], 
                                     run_df.iloc[2]])
                
                all_rows.append(new_row)
                
        loss_df_data = pd.concat(all_rows, axis=1).T
        loss_df_data.columns = loss_header
        
        loss_df_data['collapse_freq'] = col_list
        loss_df_data['irreparable_freq'] = irr_list
        loss_df_data['replacement_freq'] = [x + y for x, y
                                            in zip(col_list, irr_list)]
        
        # lognormal fit
        loss_df_data['cost_theta'] = theta_cost_list
        loss_df_data['cost_beta'] = beta_cost_list
        loss_df_data['time_l_theta'] = theta_time_list
        loss_df_data['time_l_beta'] = beta_time_list
        
        # weibull fit
        loss_df_data['cost_k'] = k_cost_list
        loss_df_data['cost_lam'] = lam_cost_list
        loss_df_data['time_l_k'] = k_time_list
        loss_df_data['time_l_lam'] = lam_time_list
        
        # weibull truncated fit
        # remember that loc_trunc is the y_var.min()
        loss_df_data['cost_k_trunc'] = k_trunc_cost_list
        loss_df_data['cost_lam_trunc'] = lam_trunc_cost_list
        loss_df_data['time_l_k_trunc'] = k_trunc_time_list
        loss_df_data['time_l_lam_trunc'] = lam_trunc_time_list
        
        # ks pvalues
        loss_df_data['cost_weibull_ks_pvalue'] = weibull_cost_ks_pvalue_list
        loss_df_data['cost_weibull_trunc_ks_pvalue'] = weibull_trunc_cost_ks_pvalue_list
        loss_df_data['cost_lognormal_ks_pvalue'] = lognormal_cost_ks_pvalue_list
        
        loss_df_data['time_l_weibull_ks_pvalue'] = weibull_time_ks_pvalue_list
        loss_df_data['time_l_weibull_trunc_ks_pvalue'] = weibull_trunc_time_ks_pvalue_list
        loss_df_data['time_l_lognormal_ks_pvalue'] = lognormal_time_ks_pvalue_list
        
        # AIC
        loss_df_data['cost_weibull_aic'] = weibull_cost_aic_list
        loss_df_data['cost_weibull_trunc_aic'] = weibull_trunc_cost_aic_list
        loss_df_data['cost_lognormal_aic'] = lognormal_cost_aic_list
        
        loss_df_data['time_l_weibull_aic'] = weibull_time_aic_list
        loss_df_data['time_l_weibull_trunc_aic'] = weibull_trunc_time_aic_list
        loss_df_data['time_l_lognormal_aic'] = lognormal_time_aic_list
        
        # quantiles
        loss_df_data['cost_quantiles'] = cost_quantile_list
        loss_df_data['time_l_quantiles'] = time_quantile_list
        
        if collect_IDA:
            loss_df_data['ida_level'] = IDA_list
            
        # clean loss_by_group results=
        group_header = ['B_mean', 'B_std', 'B_min',
                       'B_25%', 'B_50%', 'B_75%', 'B_max',
                       'C_mean', 'C_std', 'C_min',
                       'C_25%', 'C_50%', 'C_75%', 'C_max',
                       'D_mean', 'D_std', 'D_min',
                       'D_25%', 'D_50%', 'D_75%', 'D_max',
                       'E_mean', 'E_std', 'E_min',
                       'E_25%', 'E_50%', 'E_75%', 'E_max']
        
        all_rows = []
        
        for row_idx in range(len(group_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = group_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], run_df.iloc[2], 
                                     run_df.iloc[3]])
                
                all_rows.append(new_row)
                
        group_df_data = pd.concat(all_rows, axis=1).T
        group_df_data.columns = group_header
        
        self.loss_data = pd.concat([loss_df_data, group_df_data], axis=1)
        
    # This runs Pelicun by fitting a lognormal distribution through the MCE EDPs
    # TODO: store distribution statistics
    def validate_pelicun(self, df, cmp_dir='../resource/loss/'):
        # run info
        import pandas as pd

        # and import pelicun classes and methods
        from pelicun.assessment import Assessment
        from loss import Loss_Analysis

        # get database
        # initialize, no printing outputs, offset fixed with current components
        PAL = Assessment({
            "PrintLog": False, 
            "Seed": 985,
            "Verbose": False,
            "DemandOffset": {"PFA": 0, "PFV": 0}
        })

        # generate structural components and join with NSCs
        P58_metadata = PAL.get_default_metadata('loss_repair_DB_FEMA_P58_2nd')
        
        additional_frag_db = pd.read_csv(cmp_dir+'custom_component_fragilities.csv',
                                          header=[0,1], index_col=0)
        
        # lab, health, ed, res, office, retail, warehouse, hotel
        fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
        
        df = df.reset_index(drop=True)
        
        # estimate loss for set
        all_losses = []
        loss_cmp_group = []
        col_list = []
        irr_list = []
        
        ida_levels = df['ida_level'].unique().tolist()
        
        for lvl_i, lvl in enumerate(ida_levels):
            df_lvl = df[df['ida_level'] == lvl]
            
            print('========================================')
            print('Estimating loss for IDA level', lvl)
            
            # grab a representative row to calculate non EDP things
            run_data = df_lvl.iloc[0]
            
            floors = run_data.num_stories
            
            bldg_usage = [fl_usage]*floors

            loss = Loss_Analysis(run_data)
            loss.nqe_sheets()
            loss.normative_quantity_estimation(bldg_usage, P58_metadata)

            # if validation, pass in the entire df
            loss.process_EDP(df_edp=df_lvl)
            
            [cmp, dmg, loss, loss_cmp, agg, 
             collapse_rate, irr_rate] = loss.estimate_damage(
                 custom_fragility_db=additional_frag_db, mode='validation')
                 
            loss_summary = agg.describe([0.1, 0.5, 0.9])
            cost = loss_summary['repair_cost']['50%']
            time_l = loss_summary[('repair_time', 'parallel')]['50%']
            time_u = loss_summary[('repair_time', 'sequential')]['50%']
            
            print('Median repair cost: ', 
                  f'${cost:,.2f}')
            print('Median lower bound repair time: ', 
                  f'{time_l:,.2f}', 'worker-days')
            print('Median upper bound repair time: ', 
                  f'{time_u:,.2f}', 'worker-days')
            print('Collapse frequency: ', 
                  f'{collapse_rate:.2%}')
            print('Irreparable RID frequency: ', 
                  f'{irr_rate:.2%}')
            print('Replacement frequency: ', 
                  f'{collapse_rate+irr_rate:.2%}')
            
            all_losses.append(loss_summary)
            loss_cmp_group.append(loss_cmp)
            col_list.append(collapse_rate)
            irr_list.append(irr_rate)
                 
        
        # concat list of df into one df
        loss_df = pd.concat(all_losses)
        group_df = pd.concat(loss_cmp_group)
        
        loss_header = ['cost_mean', 'cost_std', 'cost_min',
                       'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
                       'time_l_mean', 'time_l_std', 'time_l_min',
                       'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
                       'time_u_mean', 'time_u_std', 'time_u_min',
                       'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']
        
        
        
        all_rows = []
        
        for row_idx in range(len(loss_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = loss_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], 
                                     run_df.iloc[2]])
                
                all_rows.append(new_row)
                
        loss_df_data = pd.concat(all_rows, axis=1).T
        loss_df_data.columns = loss_header
        
        loss_df_data['collapse_freq'] = col_list
        loss_df_data['irreparable_freq'] = irr_list
        loss_df_data['replacement_freq'] = [x + y for x, y
                                            in zip(col_list, irr_list)]
        
        loss_df_data['ida_level'] = ida_levels
        
        
        # clean loss_by_group results=
        group_header = ['B_mean', 'B_std', 'B_min',
                       'B_25%', 'B_50%', 'B_75%', 'B_max',
                       'C_mean', 'C_std', 'C_min',
                       'C_25%', 'C_50%', 'C_75%', 'C_max',
                       'D_mean', 'D_std', 'D_min',
                       'D_25%', 'D_50%', 'D_75%', 'D_max',
                       'E_mean', 'E_std', 'E_min',
                       'E_25%', 'E_50%', 'E_75%', 'E_max']
        
        all_rows = []
        
        for row_idx in range(len(group_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = group_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], run_df.iloc[2], 
                                     run_df.iloc[3]])
                
                all_rows.append(new_row)
                
        group_df_data = pd.concat(all_rows, axis=1).T
        group_df_data.columns = group_header
            
        self.loss_data = pd.concat([loss_df_data, group_df_data], axis=1)
        
    def calc_cmp_max(self, df,
                    cmp_dir='../resource/loss/',
                    validation_run=False):
        # run info
        import pandas as pd

        # and import pelicun classes and methods
        from pelicun.assessment import Assessment
        from loss import Loss_Analysis

        # get database
        # initialize, no printing outputs, offset fixed with current components
        PAL = Assessment({
            "PrintLog": False, 
            "Seed": 985,
            "Verbose": False,
            "DemandOffset": {"PFA": 0, "PFV": 0}
        })

        # generate structural components and join with NSCs
        P58_metadata = PAL.get_default_metadata('loss_repair_DB_FEMA_P58_2nd')
        
        additional_frag_db = pd.read_csv(cmp_dir+'custom_component_fragilities.csv',
                                          header=[0,1], index_col=0)
        
        # lab, health, ed, res, office, retail, warehouse, hotel
        fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
        
        df = df.reset_index(drop=True)
        
        # estimate loss for set
        all_losses = []
        loss_cmp_group = []
        col_list = []
        irr_list = []
        
        # if validation mode, only need to run one building
        if validation_run:
            df = df.head(1)
        
        for run_idx in df.index:
            print('========================================')
            print('Estimating maximum loss for run index', run_idx+1)
            
            run_data = df.loc[run_idx]
            
            floors = run_data.num_stories
            
            bldg_usage = [fl_usage]*floors

            loss = Loss_Analysis(run_data)
            loss.nqe_sheets()
            loss.normative_quantity_estimation(bldg_usage, P58_metadata)

            loss.process_EDP()
            
            [cmp, dmg, loss, loss_cmp, agg, 
             collapse_rate, irr_rate] = loss.estimate_damage(
                 custom_fragility_db=additional_frag_db, mode='maximize')
            
            loss_summary = agg.describe([0.1, 0.5, 0.9])
            cost = loss_summary['repair_cost']['50%']
            time_l = loss_summary[('repair_time', 'parallel')]['50%']
            time_u = loss_summary[('repair_time', 'sequential')]['50%']
            
            print('Median repair cost: ', 
                  f'${cost:,.2f}')
            print('Median lower bound repair time: ', 
                  f'{time_l:,.2f}', 'worker-days')
            print('Median upper bound repair time: ', 
                  f'{time_u:,.2f}', 'worker-days')
            print('Collapse frequency: ', 
                  f'{collapse_rate:.2%}')
            print('Irreparable RID frequency: ', 
                  f'{irr_rate:.2%}')
            print('Replacement frequency: ', 
                  f'{collapse_rate+irr_rate:.2%}')
            
            all_losses.append(loss_summary)
            loss_cmp_group.append(loss_cmp)
            col_list.append(collapse_rate)
            irr_list.append(irr_rate)
                 
        
        # concat list of df into one df
        loss_df = pd.concat(all_losses)
        group_df = pd.concat(loss_cmp_group)
        
        loss_header = ['cost_mean', 'cost_std', 'cost_min',
                       'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
                       'time_l_mean', 'time_l_std', 'time_l_min',
                       'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
                       'time_u_mean', 'time_u_std', 'time_u_min',
                       'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']
        
        
        
        all_rows = []
        
        for row_idx in range(len(loss_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = loss_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], 
                                     run_df.iloc[2]])
                
                all_rows.append(new_row)
                
        loss_df_data = pd.concat(all_rows, axis=1).T
        loss_df_data.columns = loss_header
        
        loss_df_data['collapse_freq'] = col_list
        loss_df_data['irreparable_freq'] = irr_list
        loss_df_data['replacement_freq'] = [x + y for x, y
                                            in zip(col_list, irr_list)]
        
        
        # clean loss_by_group results=
        group_header = ['B_mean', 'B_std', 'B_min',
                       'B_25%', 'B_50%', 'B_75%', 'B_max',
                       'C_mean', 'C_std', 'C_min',
                       'C_25%', 'C_50%', 'C_75%', 'C_max',
                       'D_mean', 'D_std', 'D_min',
                       'D_25%', 'D_50%', 'D_75%', 'D_max',
                       'E_mean', 'E_std', 'E_min',
                       'E_25%', 'E_50%', 'E_75%', 'E_max']
        
        all_rows = []
        
        for row_idx in range(len(group_df)):
            if row_idx % 8 == 0:
                # get the block with current run, drop the 'Count'
                run_df = group_df[row_idx:row_idx+8]
                run_df = run_df.transpose()
                run_df = run_df.drop(columns=['count'])
                new_row = pd.concat([run_df.iloc[0], 
                                     run_df.iloc[1], run_df.iloc[2], 
                                     run_df.iloc[3]])
                
                all_rows.append(new_row)
                
        group_df_data = pd.concat(all_rows, axis=1).T
        group_df_data.columns = group_header
        if validation_run:
            df_temp = pd.concat([loss_df_data, group_df_data], axis=1)
            len_to_rep = self.ida_results.shape[0]
            col_names = df_temp.columns
            max_loss_df = pd.DataFrame(np.repeat(df_temp.values, len_to_rep, axis=0))
            max_loss_df.columns = col_names
            self.max_loss = max_loss_df
        else:
            self.max_loss = pd.concat([loss_df_data, group_df_data], axis=1)
        
    
#%% design tools

def design_bearing_util(raw_input, filter_designs=True, mu_1_force=None):
    import time
    import pandas as pd
    
    df_raw = raw_input
    
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
        all_tfp_designs = df_tfp.apply(lambda row: ds.design_TFP(row, mu_1=mu_1_force),
                                        axis='columns', result_type='expand')
        
        
        # all_tfp_designs = df_tfp.apply(lambda row: ds.design_TFP_legacy(row),
        #                                 axis='columns', result_type='expand')
        
        all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                    'T_e', 'k_e', 'Q', 'zeta_loop', 'D_m']
        
        if filter_designs == False:
            tfp_designs = all_tfp_designs
        else:
            # keep the designs that look sensible
            tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                              (all_tfp_designs['R_1'] <= 50.0) &
                                              (all_tfp_designs['R_2'] <= 180.0) &
                                              (all_tfp_designs['zeta_loop'] <= 0.25)]
        
        tp = time.time() - t0
        
        print("Designs completed for %d TFPs in %.2f s" %
              (tfp_designs.shape[0], tp))
        
        # get the design params of those bearings
        a = df_tfp[df_tfp.index.isin(tfp_designs.index)]
    
        tfp_designs = pd.concat([a, tfp_designs], axis=1)
    else:
        tfp_designs = None
    
    df_lrb = df_raw[df_raw['isolator_system'] == 'LRB']
        
    # attempt to design all LRBs
    if df_lrb.shape[0] > 0:
        t0 = time.time()
        all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row),
                                       axis='columns', result_type='expand')
        
        
        all_lrb_designs.columns = ['d_bearing', 'd_lead', 't_r', 't', 'n_layers',
                                   'N_lb', 'S_pad', 'S_2',
                                   'T_e', 'k_e', 'Q', 'zeta_loop', 'D_m', 'buckling_fail']
        
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
                                              (all_lrb_designs['zeta_loop'] <= 0.25)]
            
            lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
            
        # if failed (particularly for inverse design), reduce bearings
        if lrb_designs.shape[0] < 1:
            all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row, reduce_bearings=True),
                                            axis='columns', result_type='expand')
            
            
            all_lrb_designs.columns = ['d_bearing', 'd_lead', 't_r', 't', 'n_layers',
                                        'N_lb', 'S_pad', 'S_2',
                                        'T_e', 'k_e', 'Q', 'zeta_loop', 'D_m', 'buckling_fail']
            
            # keep the designs that look sensible
            lrb_designs = all_lrb_designs.loc[(all_lrb_designs['d_bearing'] >=
                                                3*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_bearing'] <=
                                                6*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_lead'] <= 
                                                all_lrb_designs['t_r']) &
                                              (all_lrb_designs['t_r'] > 4.0) &
                                              (all_lrb_designs['t_r'] < 35.0) &
                                              (all_lrb_designs['buckling_fail'] == 0) &
                                              (all_lrb_designs['zeta_loop'] <= 0.25)]
            
            lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
            
        # check to see if bearing design succeeded
        # hacky solution: just skip the displacement check
        if lrb_designs.shape[0] < 1:
            all_lrb_designs = df_lrb.apply(lambda row: ds.design_LRB(row, bypass_disp_check=True),
                                            axis='columns', result_type='expand')
            
            
            all_lrb_designs.columns = ['d_bearing', 'd_lead', 't_r', 't', 'n_layers',
                                        'N_lb', 'S_pad', 'S_2',
                                        'T_e', 'k_e', 'Q', 'zeta_loop', 'D_m', 'buckling_fail']
            
            # keep the designs that look sensible
            lrb_designs = all_lrb_designs.loc[(all_lrb_designs['d_bearing'] >=
                                                3*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_bearing'] <=
                                                6*all_lrb_designs['d_lead']) &
                                              (all_lrb_designs['d_lead'] <= 
                                                all_lrb_designs['t_r']) &
                                              (all_lrb_designs['t_r'] > 4.0) &
                                              (all_lrb_designs['t_r'] < 35.0) &
                                              (all_lrb_designs['buckling_fail'] == 0) &
                                              (all_lrb_designs['zeta_loop'] <= 0.25)]
            
            lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
            
        
        
            
        tp = time.time() - t0
        
        print("Designs completed for %d LRBs in %.2f s" %
              (lrb_designs.shape[0], tp))
        
        b = df_lrb[df_lrb.index.isin(lrb_designs.index)]
        
        lrb_designs = pd.concat([b, lrb_designs], axis=1)
    else:
        lrb_designs = None
        
    return(tfp_designs, lrb_designs)

def design_structure_util(df_in, filter_designs=True, db_string='../resource/'):
    import pandas as pd
    import time
    
    from loads import define_lateral_forces
    import design as ds
    
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
        
        all_mf_designs = smrf_df.apply(lambda row: ds.design_MF(row, db_string=db_string),
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
        
        mf_designs = pd.concat([a, mf_designs], axis=1)
        
        print("Designs completed for %d moment frames in %.2f s" %
              (smrf_df.shape[0], tp))
    else:
        mf_designs = None
        
    # attempt to design all CBFs
    if cbf_df.shape[0] > 0:
        t0 = time.time()
        all_cbf_designs = cbf_df.apply(lambda row: ds.design_CBF(row, db_string=db_string),
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
        cbf_designs = pd.concat([a, cbf_designs], axis=1)
        
        print("Designs completed for %d braced frames in %.2f s" %
              (cbf_df.shape[0], tp))
    else:
        cbf_designs = None
        
    return mf_designs, cbf_designs
    
def prepare_ida_util(design_dict, levels=[1.0, 1.5, 2.0],
                     config_dict={'S_1' : 1.017,
                                 'L_bldg': 120.0,
                                 'h_bldg': 52.0,
                                 'num_frames' : 2,
                                 'num_bays' : 4,
                                 'num_stories' : 4,
                                 'L_bay': 30.0,
                                 'h_story': 13.0,
                                 'S_s' : 2.2815},
                     db_string='../resource/'):
    
    # this is here because val_ida_thread needs to see ida_levels
    # ida levels should already have been stored in `levels`
    if 'ida_levels' in config_dict:
        # remove ida levels from config dict because indexed df can't handle lists
        config_dict.pop('ida_levels')
    
        
    work_df = pd.DataFrame(config_dict, index=[0])
    design_df = pd.DataFrame(design_dict, index=[0])
    work_df = pd.concat([work_df, design_df.set_index(work_df.index)], 
                        axis=1)

    
    # for TFPs, T_m is the "unadjusted" design period for effect of vertical live load
    # if pre-calculated Tfbe passed, use it (from regression)
    try:
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
     # if no pre-calculated Tfbe is seen, use the Cu Ta method
    except:
        from loads import estimate_period
        # ad-hoc adjust Tfbe here
        if work_df['superstructure_system'].item() == 'MF':
            work_df['T_fbe'] = estimate_period(work_df.iloc[0]) / 1.4*1.8
        else:
            work_df['T_fbe'] = estimate_period(work_df.iloc[0])
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
        
    
    work_df['moat_ampli'] = work_df['gap_ratio']
    
    # initially, do not filter designs to allow for bearing reduction
    # this is only used for IDA inverse design, where bearing design is finicky
    all_tfps, all_lrbs = design_bearing_util(work_df, filter_designs=False)
    
    # manually perform bearing design filters
    if work_df['isolator_system'].item() == 'TFP':
        # keep the designs that look sensible
        tfp_designs = all_tfps.loc[(all_tfps['R_1'] >= 10.0) &
                                   (all_tfps['R_1'] <= 50.0) &
                                   (all_tfps['R_2'] <= 190.0) &
                                   (all_tfps['zeta_loop'] <= 0.27)]
        
        # retry if design didn't work
        if tfp_designs.shape[0] == 0:
            all_tfps, lrb_designs = design_bearing_util(
                work_df, filter_designs=False, mu_1_force=0.06)
            
            # keep the designs that look sensible
            tfp_designs = all_tfps.loc[(all_tfps['R_1'] >= 10.0) &
                                       (all_tfps['R_1'] <= 50.0) &
                                       (all_tfps['R_2'] <= 190.0) &
                                       (all_tfps['zeta_loop'] <= 0.27)]
            
        if tfp_designs.shape[0] == 0:
            all_tfps, lrb_designs = design_bearing_util(
                work_df, filter_designs=False, mu_1_force=0.03)
            
            # keep the designs that look sensible
            tfp_designs = all_tfps.loc[(all_tfps['R_1'] >= 10.0) &
                                       (all_tfps['R_1'] <= 50.0) &
                                       (all_tfps['R_2'] <= 190.0) &
                                       (all_tfps['zeta_loop'] <= 0.27)]
        
        # retry if design didn't work
        if tfp_designs.shape[0] == 0:
            print('Bearing design failed')
            return
        
        work_df = tfp_designs.copy()
    
    # manually perform bearing design filters    
    else:
        lrb_designs = all_lrbs.loc[(all_lrbs['d_bearing'] >=
                                           3*all_lrbs['d_lead']) &
                                          (all_lrbs['d_bearing'] <=
                                           6*all_lrbs['d_lead']) &
                                          (all_lrbs['d_lead'] <= 
                                            all_lrbs['t_r']) &
                                          (all_lrbs['t_r'] > 4.0) &
                                          (all_lrbs['t_r'] < 35.0) &
                                          (all_lrbs['buckling_fail'] == 0) &
                                          (all_lrbs['zeta_loop'] <= 0.27)]
        
        lrb_designs = lrb_designs.drop(columns=['buckling_fail'])
        
        # filter_designs flag allows for design util to enter reduction loop
        # retry if design didn't work, reducing bearings
        if lrb_designs.shape[0] == 0:
            all_tfps, all_lrbs = design_bearing_util(work_df, filter_designs=True)
            lrb_designs = all_lrbs.copy()
        
        # if lrb_designs.shape[0] == 0:
        #     all_tfps, all_lrbs = design_bearing_util(work_df, filter_designs=True)
            
        #     lrb_designs = all_lrbs.loc[(all_lrbs['d_bearing'] >=
        #                                        3*all_lrbs['d_lead']) &
        #                                       (all_lrbs['d_bearing'] <=
        #                                        6*all_lrbs['d_lead']) &
        #                                       (all_lrbs['d_lead'] <= 
        #                                         all_lrbs['t_r']) &
        #                                       (all_lrbs['t_r'] > 4.0) &
        #                                       (all_lrbs['t_r'] < 35.0) &
        #                                       (all_lrbs['zeta_loop'] <= 0.27)]
            
        if lrb_designs.shape[0] == 0:
            print('Bearing design failed')
            return
        
        work_df = lrb_designs.copy()
    
    mf_designs, cbf_designs = design_structure_util(
        work_df, filter_designs=True, db_string=db_string)
    
    if work_df['superstructure_system'].item() == 'MF':
        work_df = mf_designs.copy()
    else:
        work_df = cbf_designs.copy()
        
    if work_df.shape[0] == 0:
        print('Structure design failed.')
        return
    
    
    if 'T_fbe' in config_dict:
        work_df['T_fbe'] = config_dict['T_fbe']
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
    else:
        from loads import estimate_period
        # ad-hoc adjust Tfbe here
        if work_df['superstructure_system'].item() == 'MF':
            work_df['T_fbe'] = estimate_period(work_df.iloc[0]) / 1.4*1.8
        else:
            work_df['T_fbe'] = estimate_period(work_df.iloc[0])
        work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
    
    
    
    from gms import scale_ground_motion
    gm_dir = db_string+'/ground_motions/gm_db.csv'
    spec_dir = db_string+'/ground_motions/gm_spectra.csv'
    gm_series, sf_series, sa_avg = scale_ground_motion(work_df.iloc[0], return_list=True,
                                                       db_dir=gm_dir, spec_dir=spec_dir)
    ida_base = pd.concat([gm_series, sf_series], axis=1)
    ida_base['sa_avg'] = sa_avg
    ida_base.columns = ['gm_selected', 'scale_factor', 'sa_avg']
    ida_base = ida_base.reset_index(drop=True)
    
    ida_gms = None
    
    # prepare the sets of ida levels
    for lvl in levels:
        ida_level = ida_base[['scale_factor', 'sa_avg']].copy()
        ida_level = ida_level*lvl
        ida_level['ida_level'] = lvl
        ida_level['gm_selected'] = ida_base['gm_selected']
        if ida_gms is None:
            ida_gms = ida_level.copy()
        else:
            ida_gms = pd.concat([ida_gms, ida_level], axis=0)
        
    ida_gms = ida_gms.reset_index(drop=True)
    
    ida_df = pd.DataFrame(np.repeat(work_df.values, ida_gms.shape[0], axis=0))
    ida_df.columns = work_df.columns
    
    ida_df = pd.concat([ida_df, ida_gms], axis=1)
    return(ida_df)


def prepare_ida_legacy_util(design_dict, levels=[1.0, 1.5, 2.0], 
                            config_dict=None, db_string='../resource/'):
    
    if config_dict is None:    
        config_dict = {
            'S_1' : 1.017,
            'k_ratio' : 10,
            'Q': 0.06,
            'L_bldg': 120.0,
            'h_bldg': 52.0,
            'num_frames' : 2,
            'num_bays' : 4,
            'num_stories' : 4,
            'L_bay': 30.0,
            'h_story': 13.0,
            'isolator_system' : 'TFP',
            'superstructure_system' : 'MF',
            'S_s' : 2.2815
        }
    
    work_df = pd.DataFrame(config_dict, index=[0])
    design_df = pd.DataFrame(design_dict, index=[0])
    work_df = pd.concat([work_df, design_df.set_index(work_df.index)], 
                        axis=1)
    
    
    from loads import estimate_period
    work_df['T_fbe'] = estimate_period(work_df.iloc[0])
    
    # work_df = pd.concat([work_df, design_df.set_index(work_df.index)], 
    #                     axis=1)
    
    import design as ds
    from loads import define_lateral_forces, define_gravity_loads
    from gms import scale_ground_motion
    
    work_df['T_m'] = work_df['T_fbe']*work_df['T_ratio']
    work_df['moat_ampli'] = work_df['gap_ratio']
    
    # design
    work_df[['W', 
            'W_s', 
            'w_fl', 
            'P_lc',
            'all_w_cases',
            'all_Plc_cases']] = work_df.apply(lambda row: define_gravity_loads(row),
                                            axis='columns', result_type='expand')
                    
    try:
        all_tfp_designs = work_df.apply(lambda row: ds.design_TFP_legacy(row),
                                        axis='columns', result_type='expand')
    except:
        print('Bearing design failed.')
        return
    
    all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                'T_e', 'k_e', 'Q', 'zeta_e', 'D_m']
    
    tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                        (all_tfp_designs['R_1'] <= 50.0) &
                                        (all_tfp_designs['R_2'] <= 190.0) &
                                        (all_tfp_designs['zeta_e'] <= 0.27)]
    
    # retry if design didn't work
    if tfp_designs.shape[0] == 0:
        print('Bearing design failed.')
        return
    
    tfp_designs = tfp_designs.drop(columns=['zeta_e'])
    work_df = pd.concat([work_df, tfp_designs.set_index(work_df.index)], 
                        axis=1)
    
    # get lateral force and design structures
    work_df[['wx', 
            'hx', 
            'h_col', 
            'hsx', 
            'Fx', 
            'Vs',
            'T_fbe']] = work_df.apply(lambda row: define_lateral_forces(row),
                                axis='columns', result_type='expand')
                                        
    all_mf_designs = work_df.apply(lambda row: ds.design_MF(row, db_string=db_string),
                                        axis='columns', 
                                        result_type='expand')
        
    all_mf_designs.columns = ['beam', 'column', 'flag']
    
    # keep the designs that look sensible
    mf_designs = all_mf_designs.loc[all_mf_designs['flag'] == False]
    mf_designs = mf_designs.dropna(subset=['beam','column'])
        
    mf_designs = mf_designs.drop(['flag'], axis=1)
    
    if mf_designs.shape[0] == 0:
        print('Bearing design failed.')
        return
    
    # get the design params of those bearings
    work_df = pd.concat([work_df, mf_designs.set_index(work_df.index)], 
                        axis=1)
    
    gm_dir = db_string+'/ground_motions/gm_db.csv'
    spec_dir = db_string+'/ground_motions/gm_spectra.csv'
    gm_series, sf_series, sa_avg = scale_ground_motion(work_df.iloc[0], return_list=True,
                                                       db_dir=gm_dir, spec_dir=spec_dir)
    ida_base = pd.concat([gm_series, sf_series], axis=1)
    ida_base['sa_avg'] = sa_avg
    ida_base.columns = ['gm_selected', 'scale_factor', 'sa_avg']
    ida_base = ida_base.reset_index(drop=True)
    
    ida_gms = None
    # prepare the sets of ida levels
    for lvl in levels:
        ida_level = ida_base[['scale_factor', 'sa_avg']].copy()
        ida_level = ida_level*lvl
        ida_level['ida_level'] = lvl
        ida_level['gm_selected'] = ida_base['gm_selected']
        if ida_gms is None:
            ida_gms = ida_level.copy()
        else:
            ida_gms = pd.concat([ida_gms, ida_level], axis=0)
        
    ida_gms = ida_gms.reset_index(drop=True)
    
    ida_df = pd.DataFrame(np.repeat(work_df.values, ida_gms.shape[0], axis=0))
    ida_df.columns = work_df.columns
    
    ida_df = pd.concat([ida_df, ida_gms], axis=1)
    
    return(ida_df)
        
        
#%% PBE fitting tools

# function to optimize weibull k
# if log(0) causes underflow, just increase findings by a dollar
def weibull_k_fcn(k, x):
    if np.isinf(np.sum(x**k * np.log(x))/np.sum(x**k) - 1/k - np.log(x).mean()):
        return np.sum(x**k * np.log(x+1))/np.sum(x**k) - 1/k - np.log(x+1).mean()
    elif np.isnan(np.sum(x**k * np.log(x))/np.sum(x**k) - 1/k - np.log(x).mean()):
        return np.sum(x**k * np.log(x+1))/np.sum(x**k) - 1/k - np.log(x+1).mean()
    else:
        return np.sum(x**k * np.log(x))/np.sum(x**k) - 1/k - np.log(x).mean()

# negative log likelihood: lognormal
def nlls_normal(params, x):
    theta, sigma_sq = params
    n = len(x)
    pi = 3.14159
    log_likelihood_sum = -n/2*np.log(2*pi) - n/2*np.log(sigma_sq) - 1/(2*sigma_sq)*np.sum((x - theta)**2)
    return -log_likelihood_sum

# function to find MLE weibull params
def mle_weibull(x):
    from scipy.optimize import fsolve
    root = fsolve(weibull_k_fcn, x0=1.0, args=(x))
    k_mle = float(root)
    n = len(x)
    lam_mle = (1/n*np.sum(x**k_mle))**(1/k_mle)
    return k_mle, lam_mle
    

# negative log likelihood: weibull without shifting
# if shifting, fit with loc=min(x)
# (shift implies that P(X < x) = 0 if x is less than loc)
def nlls_weibull(params, x):

    k, lam = params
    n = len(x)
    log_likelihood_sum = (
        n*(np.log(k) - k*np.log(lam)) + (k - 1)*np.sum(np.log(x)) 
        -np.sum((x / lam)**k ) 
        )
    
    # if log(0) causes underflow, just increase findings by a dollar
    if np.isinf(log_likelihood_sum):
        log_likelihood_sum = (
            n*(np.log(k) - k*np.log(lam)) + (k - 1)*np.sum(np.log(x+1)) 
            -np.sum((x / lam)**k ) 
            )
        
    if np.isnan(log_likelihood_sum):
        log_likelihood_sum = (
            n*(np.log(k) - k*np.log(lam)) + (k - 1)*np.sum(np.log(x+1)) 
            -np.sum((x / lam)**k ) 
            )
        
    return -log_likelihood_sum

# old fit for weibull (not analytical solution)
def mle_fit_weibull(x_values, x_init=None):
    from functools import partial
    from scipy.optimize import basinhopping
    
    neg_log_likelihood_sum_partial = partial(
        nlls_weibull, x=x_values)
    
    # k (shape), lam (scale), loc (shift)
    if x_init is None:
        x0 = (1., 1.)
    else:
        x0 = x_init
        
    bnds = ((0.01, 5.0), (1.0, 1e8))
    # bnds = None
    
    # use basin hopping to avoid local minima
    minimizer_kwargs={'bounds':bnds}
    res = basinhopping(neg_log_likelihood_sum_partial, x0, minimizer_kwargs=minimizer_kwargs,
                        niter=100, seed=985)
    
    return res.x[0], res.x[1]