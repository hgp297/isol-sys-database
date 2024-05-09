############################################################################
#               Experiments

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: August 2023

# Description:  Functions as control for Opensees experiments
############################################################################

# prepare the pandas output of the run
def prepare_results(output_path, design, T_1, Tfb, run_status):
    
    import pandas as pd
    import numpy as np
    from gms import get_gm_ST, get_ST
    
    num_stories = design['num_stories']
    
    # gather EDPs from opensees output
    # also collecting story 0, which is isol layer
    story_names = ['story_'+str(story)
                   for story in range(0,num_stories+1)]
    story_names.insert(0, 'time')
    
    isol_dof_names = ['time', 'horizontal', 'vertical', 'rotation']
    # forceColumns = ['time', 'iAxial', 'iShearX', 'iShearY',
    #                 'iMomentX','iMomentY', 'iMomentZ',
    #                 'jAxial','jShearX', 'jShearY',
    #                 'jMomentX', 'jMomentY', 'jMomentZ']
    
    # displacements
    inner_col_disp = pd.read_csv(output_path+'inner_col_disp.csv', sep=' ',
                                 header=None, names=story_names)
    outer_col_disp = pd.read_csv(output_path+'outer_col_disp.csv', sep=' ',
                                 header=None, names=story_names)
    
    # velocities (relative)
    inner_col_vel = pd.read_csv(output_path+'inner_col_vel.csv', sep=' ',
                                 header=None, names=story_names)
    outer_col_vel = pd.read_csv(output_path+'outer_col_vel.csv', sep=' ',
                                 header=None, names=story_names)
    
    # accelerations (absolute)
    inner_col_acc = pd.read_csv(output_path+'inner_col_acc.csv', sep=' ',
                                 header=None, names=story_names)
    outer_col_acc = pd.read_csv(output_path+'outer_col_acc.csv', sep=' ',
                                 header=None, names=story_names)
    
    # isolator layer displacement
    isol_disp = pd.read_csv(output_path+'isolator_displacement.csv', sep=' ',
                            header=None, names=isol_dof_names)
    
    # maximum displacement in isol layer
    isol_max_horiz_disp = isol_disp['horizontal'].abs().max()
    
    # drift ratios recorded. diff takes difference with adjacent column
    ft = 12
    h_story = design['h_story']
    inner_col_drift = inner_col_disp.diff(axis=1).drop(columns=['time', 'story_0'])/(h_story*ft)
    outer_col_drift = outer_col_disp.diff(axis=1).drop(columns=['time', 'story_0'])/(h_story*ft)
    
    g = 386.4
    inner_col_acc = inner_col_acc.drop(columns=['time'])/g
    outer_col_acc = outer_col_acc.drop(columns=['time'])/g
    
    inner_col_vel = inner_col_vel.drop(columns=['time'])
    outer_col_vel = outer_col_vel.drop(columns=['time'])
    
    ss_type = design['superstructure_system']
    if ss_type == 'MF':
        ok_thresh = 0.20
    else:
        ok_thresh = 0.075
    # if run was OK, we collect true max values
    if run_status == 0:
        PID = np.maximum(inner_col_drift.abs().max(), 
                         outer_col_drift.abs().max()).tolist()
        PFV = np.maximum(inner_col_vel.abs().max(), 
                         outer_col_vel.abs().max()).tolist()
        PFA = np.maximum(inner_col_acc.abs().max(), 
                         outer_col_acc.abs().max()).tolist()
        RID = np.maximum(inner_col_drift.iloc[-1].abs(), 
                         outer_col_drift.iloc[-1].abs()).tolist()
        
    # if run failed, we find the state corresponding to 0.20 drift across all
    # assumes that once drift crosses 0.20, it only increases (no other floor
    # will exceed 0.20 AND be the highest)
    else:
        drift_df = pd.concat([inner_col_drift, outer_col_drift], axis=1)
        worst_drift = drift_df.abs().max(axis=1)
        drift_sort = worst_drift.iloc[(worst_drift-ok_thresh).abs().argsort()[:1]]
        ok_state = drift_sort.index.values
        
        PID = np.maximum(inner_col_drift.iloc[ok_state.item()].abs(), 
                         outer_col_drift.iloc[ok_state.item()].abs()).tolist()
        
        PFV = np.maximum(inner_col_vel.iloc[ok_state.item()].abs(), 
                         outer_col_vel.iloc[ok_state.item()].abs()).tolist()
        
        PFA = np.maximum(inner_col_acc.iloc[ok_state.item()].abs(), 
                         outer_col_acc.iloc[ok_state.item()].abs()).tolist()
        
        # if collapse, just collect PID as residual
        RID = PID
    
    impact_cols = ['time', 'dirX_left', 'dirX_right']
    impact_force = pd.read_csv(output_path+'impact_forces.csv',
                               sep = ' ', header=None, names=impact_cols)
    impact_thresh = 100   # kips
    if(any(abs(impact_force['dirX_left']) > impact_thresh) or
       any(abs(impact_force['dirX_right']) > impact_thresh)):
        impact_bool = 1
    else:
        impact_bool = 0
        
    Tms_interest = np.array([design['T_m'], 1.0, Tfb])
    
    # be careful not to double calculate damping effect
    # Sa_gm = get_gm_ST(design, Tms_interest)
    Sa_gm = get_ST(design, Tms_interest)
    
    Sa_Tm = Sa_gm[0]
    Sa_1 = Sa_gm[1]
    Sa_Tfb = Sa_gm[2]
    
    # Sa_Tm = get_ST(design, design['T_m'])
    # Sa_1 = get_ST(design, 1.0)
    # Sa_Tfb = get_ST(design, Tfb)
        
    import numpy as np
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    Bm = np.interp(design['zeta_e'], zetaRef, BmRef)
    
    pi = 3.14159
    g = 386.4
    gap_ratio = (design['moat_ampli']*design['D_m']*4*pi**2)/ \
        (g*(Sa_Tm/Bm)*design['T_m']**2)
    
    result_dict = {'sa_tm': Sa_Tm,
                   'sa_1': Sa_1,
                   'sa_tfb': Sa_Tfb,
                   'constructed_moat': design['moat_ampli']*design['D_m'],
                   'T_1': T_1,
                   'T_fb': Tfb,
                   'T_ratio' : design['T_m']/Tfb,
                   'gap_ratio' : gap_ratio,
                   'max_isol_disp': isol_max_horiz_disp,
                   'PID': PID,
                   'PFV': PFV,
                   'PFA': PFA,
                   'RID': RID,
                   'impacted': impact_bool,
                   'run_status': run_status
        }
    result_series = pd.Series(result_dict)
    final_series = pd.concat([design, result_series])
    return(final_series)
    
def collapse_fragility(run, drift_at_mu_plus_std=0.1):
    system = run.superstructure_system
    peak_drift = max(run.PID)
    n_stories = run.num_stories
    
    # collapse as a probability
    from math import log, exp
    from scipy.stats import lognorm
    from scipy.stats import norm
    
    # TODO: change this for taller buildings
    # MF: set 84% collapse at 0.10 drift, 0.25 beta
    if system == 'MF':
        inv_norm = norm.ppf(0.84)
        if n_stories < 4:
            beta_drift = 0.25
        else:
            beta_drift = 0.35
        mean_log_drift = exp(log(drift_at_mu_plus_std) - beta_drift*inv_norm) 
        
    # CBF: set 90% collapse at 0.05 drift, 0.55 beta
    else:
        inv_norm = norm.ppf(0.90)
        beta_drift = 0.55
        mean_log_drift = exp(log(0.05) - beta_drift*inv_norm) 
        
    ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
    collapse_prob = ln_dist.cdf(peak_drift)
    
    return(peak_drift, collapse_prob)
    
# TODO: run pushover

# run the experiment, GM name and scale factor must be baked into design

def run_nlth(design, 
             gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/',
             output_path='./outputs/'):
    
    from building import Building
    
    # generate the building, construct model
    bldg = Building(design)
    bldg.model_frame()
    
    # apply gravity loads, perform eigenvalue analysis, add damping
    bldg.apply_grav_load()
    T_1 = bldg.run_eigen()
    Tfb = bldg.provide_damping(80, method='SP',
                               zeta=[0.05], modes=[1])
    
    # run ground motion
    if bldg.superstructure_system == 'MF':
        dt_default = 0.005
    else:
        dt_default = 0.005
    run_status = bldg.run_ground_motion(design['gm_selected'], 
                                   design['scale_factor'], 
                                   dt_default,
                                   gm_dir=gm_path,
                                   data_dir=output_path)
    
    # lower dt if convergence issues
    if run_status != 0:
        if bldg.superstructure_system == 'MF':
            print('Lowering time step...')
            
            bldg = Building(design)
            bldg.model_frame()
            
            # apply gravity loads, perform eigenvalue analysis, add damping
            bldg.apply_grav_load()
            T_1 = bldg.run_eigen()
            Tfb = bldg.provide_damping(80, method='SP',
                                       zeta=[0.05], modes=[1])
            
            run_status = bldg.run_ground_motion(design['gm_selected'], 
                                                design['scale_factor'], 
                                                0.001,
                                                gm_dir=gm_path,
                                                data_dir=output_path)
        else:
            # print('Cutting time did not work.')
            print('Lowering time step and convergence mode CBF...')
            
            bldg = Building(design)
            bldg.model_frame(convergence_mode=True)
            
            # apply gravity loads, perform eigenvalue analysis, add damping
            bldg.apply_grav_load()
            T_1 = bldg.run_eigen()
            Tfb = bldg.provide_damping(80, method='SP',
                                        zeta=[0.05], modes=[1])
            
            run_status = bldg.run_ground_motion(design['gm_selected'], 
                                                design['scale_factor'], 
                                                0.001,
                                                gm_dir=gm_path,
                                                data_dir=output_path)
        
    # CBF if still no converge, give up
    if run_status != 0:
        if bldg.superstructure_system == 'MF':
            print('Lowering time step one last time...')
            
            bldg = Building(design)
            bldg.model_frame()
            
            # apply gravity loads, perform eigenvalue analysis, add damping
            bldg.apply_grav_load()
            T_1 = bldg.run_eigen()
            Tfb = bldg.provide_damping(80, method='SP',
                                       zeta=[0.05], modes=[1])
            
            run_status = bldg.run_ground_motion(design['gm_selected'], 
                                                design['scale_factor'], 
                                                0.0005,
                                                gm_dir=gm_path,
                                                data_dir=output_path)
        else:
            print('CBF did not converge ...')
            
            # bldg = Building(design)
            # bldg.model_frame(convergence_mode=True)
            
            # # apply gravity loads, perform eigenvalue analysis, add damping
            # bldg.apply_grav_load()
            # T_1 = bldg.run_eigen()
            # Tfb = bldg.provide_damping(80, method='SP',
            #                             zeta=[0.05], modes=[1])
            
            # run_status = bldg.run_ground_motion(design['gm_selected'], 
            #                                     design['scale_factor'], 
            #                                     0.0005,
            #                                     gm_dir=gm_path,
            #                                     data_dir=output_path)
    if run_status != 0:
        print('Recording run and moving on.')
       
    # add a little delay to prevent weird overwriting
    import time
    time.sleep(3)
    
    results_series = prepare_results(output_path, design, T_1, Tfb, run_status)
    return(results_series)
    

def run_doe(prob_target, df_train, df_test, sample_bounds=None,
            batch_size=10, error_tol=0.15, maxIter=1000, conv_tol=1e-2,
            kernel='rbf_ard'):
    
    import random
    import numpy as np
    import pandas as pd
    
    gm_path='../resource/ground_motions/PEERNGARecords_Unscaled/'
    
    np.random.seed(986)
    random.seed(986)
    from doe import GP
    from db import Database
    
    # sample_bounds = test_set.X.agg(['min', 'max'])
    
    # set bounds for DoE
    if sample_bounds is None:
        sample_bounds = pd.DataFrame({'gap_ratio': [0.5, 2.0],
                                      'RI': [0.5, 2.25],
                                      'T_ratio': [2.0, 5.0],
                                      'zeta_e': [0.10, 0.25]}, index=['min', 'max'])
        covariate_columns = ['gap_ratio', 'RI', 'T_ratio', 'zeta_e']
    
    else:
        covariate_columns = sample_bounds.columns
    
    test_set = GP(df_test)
    test_set.set_covariates(covariate_columns)
    
    outcome = 'collapse_prob'
    
    test_set.set_outcome(outcome)
    
    buffer = 4
    doe_reserve_db = Database(maxIter, n_buffer=buffer, seed=131, 
                        struct_sys_list=['MF'], isol_wts=[1, 0])
    
    # drop covariates 
    reserve_df = doe_reserve_db.raw_input
    reserve_df = reserve_df.drop(columns=['T_m', 'moat_ampli'])
    pregen_designs = reserve_df.drop(columns=[col for col in reserve_df 
                                              if col in covariate_columns])
    
    from loads import estimate_period
    pregen_designs['T_fbe'] = pregen_designs.apply(lambda row: estimate_period(row),
                                                     axis='columns', result_type='expand')
    
    rmse = 1.0
    batch_idx = 0
    batch_no = 0
    
    rmse_list = []
    mae_list = []
    nrmse_list = []
    
    if kernel == 'rbf_iso':
        hyperparam_list = np.empty((0,3), float)
    else:
        hyperparam_list = np.empty((0,6), float)
    
    doe_idx = 0
    
    
    import design as ds
    from loads import define_lateral_forces, define_gravity_loads
    from gms import scale_ground_motion
    
    while doe_idx < maxIter:
        
        print('========= DoE batch %d ==========' % 
              (batch_no+1))
        
        if (batch_idx % (batch_size) == 0):
            
            mdl = GP(df_train)
            
            mdl.set_outcome(outcome)
            
            mdl.set_covariates(covariate_columns)
            mdl.fit_gpr(kernel_name=kernel)
            
            y_hat = mdl.gpr.predict(test_set.X)
            
            print('===== Training model size:', mdl.X.shape[0], '=====')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np
            compare = pd.DataFrame(test_set.y).copy()
            compare['predicted'] = y_hat
            mse = mean_squared_error(test_set.y, y_hat)
            
            # SOURCE: Yi & Taflanidis (2023)
            gp_obj = mdl.gpr._final_estimator
            L = gp_obj.L_
            K_mat = L @ L.T
            alpha_ = gp_obj.alpha_.flatten()
            K_inv_diag = np.linalg.inv(K_mat).diagonal()
            theta = gp_obj.kernel_.theta
            
            hyperparam_list = np.append(hyperparam_list, [theta], 
                                        axis=0)
            
            NRMSE_cv = ((np.sum(np.divide(alpha_, K_inv_diag)**2)/len(alpha_))**0.5/
                        (max(mdl.y[outcome]) - min(mdl.y[outcome])))
            
            # nrmse = rmse/(max(mdl.y[outcome]) - min(mdl.y[outcome]))
            print('Test set NRMSE_cv: %.3f' % NRMSE_cv)
            
            rmse = mse**0.5
            print('Test set RMSE: %.3f' % rmse)

            mae = mean_absolute_error(test_set.y, y_hat)
            print('Test set MAE: %.3f' % mae)
            
            # if len(rmse_list) == 0:
            #     conv = rmse
            # else:
            #     conv = abs(rmse - rmse_list[-1])/rmse_list[-1]
            
            # TODO: more intelligent convergence criteria
            # if rmse < error_tol:
            if len(nrmse_list) == 0:
                conv = NRMSE_cv
            else:
                conv = abs(NRMSE_cv - nrmse_list[-1])/nrmse_list[-1]
            
            if NRMSE_cv < error_tol:
                print('Stopping criterion reached. Ending DoE...')
                print('Number of added points: ' + str((batch_idx)*(batch_no)))
                
                rmse_list.append(rmse)
                nrmse_list.append(NRMSE_cv)
                mae_list.append(mae)
                
                return (df_train, rmse_list, mae_list, nrmse_list, hyperparam_list)
            elif conv < conv_tol:
                print('NRMSE_cv did not improve beyond convergence tolerance. Ending DoE...')
                print('Number of added points: ' + str((batch_idx)*(batch_no)))
                
                rmse_list.append(rmse)
                nrmse_list.append(NRMSE_cv)
                mae_list.append(mae)
                
                return (df_train, rmse_list, mae_list, nrmse_list, hyperparam_list)
            else:
                pass
            batch_idx = 0
            df_train.to_csv('../data/doe/temp_save.csv', index=False)
            
            x_next = mdl.doe_rejection_sampler(batch_size, prob_target, 
                                                sample_bounds, design_filter=True)
            
            # x_next = mdl.doe_mse_loocv(sample_bounds, design_filter=True)
            
            next_df = pd.DataFrame(x_next, columns=covariate_columns)
            print('Convergence not reached yet. Resetting batch index to 0...')
    
        ######################## DESIGN FOR DOE SET ###########################
        #
        # Currently somewhat hardcoded for TFP-MF
    
        # get first set of randomly generated params and merge with a buffer
        # amount of DoE found points (to account for failed designs)
        
        for idx, next_row in next_df.iterrows():
            
            print('========= Run %d of batch %d ==========' % 
                  (batch_idx+1, batch_no+1))
            
            while pregen_designs.shape[0] > 0:
                
                # pop off a pregen design and try to design with it
                work_df = pregen_designs.head(1)
                pregen_designs.drop(pregen_designs.head(1).index, inplace=True)
                
                # next_row = next_df.iloc[[batch_idx]].set_index(work_df.index)
                row_df = pd.DataFrame(next_row).T
                work_df = pd.concat([work_df, row_df.set_index(work_df.index)], 
                                    axis=1)
                
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
                    continue
                
                all_tfp_designs.columns = ['mu_1', 'mu_2', 'R_1', 'R_2', 
                                           'T_e', 'k_e', 'zeta_e', 'D_m']
                
                tfp_designs = all_tfp_designs.loc[(all_tfp_designs['R_1'] >= 10.0) &
                                                  (all_tfp_designs['R_1'] <= 50.0) &
                                                  (all_tfp_designs['R_2'] <= 190.0) &
                                                  (all_tfp_designs['zeta_e'] <= 0.27)]
                
                # retry if design didn't work
                if tfp_designs.shape[0] == 0:
                    continue
                
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
                    continue
              
                # get the design params of those bearings
                work_df = pd.concat([work_df, mf_designs.set_index(work_df.index)], 
                                    axis=1)
                
                
                work_df[['gm_selected',
                         'scale_factor',
                         'sa_avg']] = work_df.apply(lambda row: scale_ground_motion(row),
                                                    axis='columns', result_type='expand')
                   
                break
            
            # TODO: we cannot have the exact T_ratio and gap_ratio as DOE called for
            # gap ratio is affected by a stochastic gm_sa_tm
            # T_ratio is affected by the fact that the true Tfb is not the estimated Tfb
            
            # drop the "called-for" values and record the "as constructed" values
            
            work_df = work_df.drop(columns=['gap_ratio', 'T_ratio'])
            bldg_result = run_nlth(work_df.iloc[0], gm_path)
            result_df = pd.DataFrame(bldg_result).T
            
            
            result_df[['max_drift',
               'collapse_prob']] = result_df.apply(lambda row: collapse_fragility(row),
                                                    axis='columns', result_type='expand')
                                   
            from numpy import log
            result_df['log_collapse_prob'] = log(result_df['collapse_prob'])
            
            # if run is successful and is batch marker, record error metric
            if (batch_idx % (batch_size) == 0):
                rmse_list.append(rmse)
                mae_list.append(mae)
                nrmse_list.append(NRMSE_cv)
            
            batch_idx += 1
            doe_idx += 1
    
            # attach to existing data
            df_train = pd.concat([df_train, result_df], axis=0)
        
        batch_no += 1
    print('DoE did not converge within maximum iteration specified.')
    return df_train, rmse_list, mae_list, nrmse_list, hyperparam_list
