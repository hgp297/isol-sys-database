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
    from gms import get_gm_ST
    
    # TODO: collect Sa values, collect validation indicator (IDA level)
    num_stories = design['num_stories']
    
    # gather EDPs from opensees output
    # also collecting story 0, which is isol layer
    story_names = ['story_'+str(story)
                   for story in range(0,num_stories+1)]
    story_names.insert(0, 'time')
    
    isol_dof_names = ['time', 'horizontal', 'vertical', 'rotation']
    forceColumns = ['time', 'iAxial', 'iShearX', 'iShearY',
                    'iMomentX','iMomentY', 'iMomentZ',
                    'jAxial','jShearX', 'jShearY',
                    'jMomentX', 'jMomentY', 'jMomentZ']
    
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
        ok_thresh = 0.1
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
        
    # TODO: CBFs may not be stable enough to record high drift even if failure
    # Find way to record it
    
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
    Sa_gm = get_gm_ST(design, Tms_interest)
    
    Sa_Tm = Sa_gm[0]
    Sa_1 = Sa_gm[1]
    Sa_Tfb = Sa_gm[2]
        
    # Sa_Tm = get_ST(design, design['T_m'])
    # Sa_1 = get_ST(design, 1.0)
    
    result_dict = {'sa_tm': Sa_Tm,
                   'sa_1': Sa_1,
                   'sa_tfb': Sa_Tfb,
                   'constructed_moat': design['moat_ampli']*design['D_m'],
                   'T_1': T_1,
                   'T_fb': Tfb,
                   'max_isol_disp': isol_max_horiz_disp,
                   'PID': PID,
                   'PFV': PFV,
                   'PFA': PFA,
                   'RID': RID,
                   'impacted': impact_bool,
                   'run_status': run_status
        }
    result_series = pd.Series(result_dict)
    
    final_series = design.append(result_series)
    return(final_series)
    
    
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
    
    # TODO: if validating hard run, start at .0005 dt (for CBF, e.g.)
    # run ground motion
    run_status = bldg.run_ground_motion(design['gm_selected'], 
                                   design['scale_factor'], 
                                   0.005,
                                   gm_dir=gm_path,
                                   data_dir=output_path)
    
    # lower dt if convergence issues
    if run_status != 0:
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
        
    # # TODO: integrate this VariableTransient
    # if run_status != 0:
    #     print('Lowering time step last time...')
        
    #     bldg = Building(design)
    #     bldg.model_frame()
        
    #     # apply gravity loads, perform eigenvalue analysis, add damping
    #     bldg.apply_grav_load()
    #     T_1 = bldg.run_eigen()
    #     Tfb = bldg.provide_damping(80, method='SP',
    #                                zeta=[0.05], modes=[1])
        
    #     run_status = bldg.run_ground_motion(design['gm_selected'], 
    #                                         design['scale_factor'], 
    #                                         0.0005,
    #                                         gm_dir=gm_path,
    #                                         data_dir=output_path)
    if run_status != 0:
        print('Recording run and moving on.')
       
    # add a little delay to prevent weird overwriting
    import time
    time.sleep(3)
    
    results_series = prepare_results(output_path, design, T_1, Tfb, run_status)
    return(results_series)
    
