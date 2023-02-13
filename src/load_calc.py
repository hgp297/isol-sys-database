############################################################################
#               Loading calculations

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2023

# Description:  Gravity and lateral load calculators

# Open issues:  (1) 

############################################################################

# length units are in ft
# load units are in kips

def define_gravity_loads(D_load=None, L_load=None,
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
    W_seis = np.sum(D_load*A_bldg)
    
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
    
    return(W_seis, w_on_frame, P_on_leaning_column)