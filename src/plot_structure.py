############################################################################
#               Troubleshooting plots

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: Aug 2023

# Description:  Various plots for structure

# Open issues:  

############################################################################

import pandas as pd
import matplotlib.pyplot as plt

def plots(run):
    plt.close('all')
    data_dir = './outputs/'
    structure_type = run.superstructure_system
    
    L_bay = run.L_bay
    h_story = run.h_story
    
    if structure_type == 'CBF':
        res_columns = ['stress1', 'strain1', 'stress2', 'strain2', 
                       'stress3', 'strain3', 'stress4', 'strain4']
        left_brace_res = pd.read_csv(data_dir+'brace_left.csv', sep=' ', 
                                     header=None, names=res_columns)
        right_brace_res = pd.read_csv(data_dir+'brace_right.csv', sep=' ', 
                                     header=None, names=res_columns)
        
        ghost_columns = ['time', 'axial_strain']
        left_brace_def = pd.read_csv(data_dir+'left_ghost_deformation.csv', sep=' ', 
                                     header=None, names=ghost_columns)
        right_brace_def = pd.read_csv(data_dir+'right_ghost_deformation.csv', sep=' ', 
                                     header=None, names=ghost_columns)
        
        from building import get_shape 
        selected_brace = get_shape(run.brace[0],'brace')
        d_brace = selected_brace.iloc[0]['b']
        t_brace = selected_brace.iloc[0]['tdes']
        
        A_flange = d_brace*t_brace
        A_web = (d_brace-2*t_brace)*t_brace
        
        total_axial_force = (left_brace_res['stress1']*A_flange +
                             left_brace_res['stress2']*A_web +
                             left_brace_res['stress3']*A_web +
                             left_brace_res['stress4']*A_flange)
        
        # stress strain
        fig = plt.figure()
        plt.plot(-left_brace_res['strain1'], -left_brace_res['stress1'])
        plt.title('Axial stress-strain brace (midpoint, top fiber)')
        plt.ylabel('Stress (ksi)')
        plt.xlabel('Strain (in/in)')
        plt.grid(True)
    
    isol_columns = ['time', 'x', 'z', 'rot']
    isol_disp = pd.read_csv(data_dir+'isolator_displacement.csv', sep=' ', 
                                 header=None, names=isol_columns)
    
    force_columns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 
                    'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']
    isol_force = pd.read_csv(data_dir+'isolator_forces.csv', sep=' ', 
                                 header=None, names=force_columns)
    # All hystereses
    isol_type = run.isolator_system
    if isol_type == 'LRB':
        fig = plt.figure()
        plt.plot(isol_disp['x'], isol_force['jFy'])
        plt.title('Isolator hystereses (LRB)')
        plt.xlabel('Displ (in)')
        plt.ylabel('V/N')
        plt.grid(True)
    else:
        fig = plt.figure()
        plt.plot(isol_disp['x'], isol_force['jFy']/isol_force['iFx'])
        plt.title('Isolator hystereses (TFP)')
        plt.xlabel('Displ (in)')
        plt.ylabel('V/N')
        plt.grid(True)
        
    wall_columns = ['time', 'left_x', 'left_z', 'right_x', 'right_z']
    impact_forces = pd.read_csv(data_dir+'impact_forces.csv', sep=' ', 
                                 header=None, names=wall_columns)
    
    # wall
    wall_columns = ['time', 'left_x', 'right_x']
    impact_forces = pd.read_csv(data_dir+'impact_forces.csv', sep=' ', 
                                 header=None, names=wall_columns)
    impact_disp = pd.read_csv(data_dir+'impact_disp.csv', sep=' ', 
                                 header=None, names=wall_columns)

    fig = plt.figure()
    plt.plot(impact_forces['time'], impact_forces['left_x'])
    plt.title('Left wall impact')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (kip)')
    plt.grid(True)

    fig = plt.figure()
    plt.plot(impact_disp['left_x'], impact_forces['left_x'])
    plt.plot(-impact_disp['right_x'], impact_forces['right_x'])
    plt.title('Impact hysteresis')
    plt.xlabel('Displ (in)')
    plt.ylabel('Force (kip)')
    plt.grid(True)
    
    diaph_columns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz']
    diaph_forces = pd.read_csv(data_dir+'diaphragm_forces.csv', sep=' ', 
                                 header=None, names=diaph_columns)
    
    fig = plt.figure()
    plt.plot(diaph_forces['time'], diaph_forces['iFx'])
    plt.title('Diaphragm forces')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (kip)')
    plt.grid(True)