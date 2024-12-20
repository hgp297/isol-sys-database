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

def define_gravity_loads(config_df, D_load=None, L_load=None):
    
    n_floors = config_df['num_stories']
    L_bay = config_df['L_bay']
    n_bays = config_df['num_bays']
    S_s = config_df['S_s']
    n_frames = config_df['num_frames']
    
    import numpy as np
    
    # assuming 100 psf D and 50 psf L for floors 
    # assume that D already includes structural members
    # D_loads in kip/ft^2
    if D_load is None:
        D_load = np.repeat(100.0/1000, n_floors+1)
    if L_load is None:
        L_load = np.repeat(50.0/1000, n_floors+1)
        
    # roof loading is lighter
    D_load[-1] = 75.0/1000
    L_load[-1] = 20.0/1000
    
    # assuming square building
    A_bldg = (L_bay*n_bays)**2 # ft^2
    
    # seismic weight: ASCE 7-22, Ch. 12.7.2 (kips)
    W_seis = np.sum(D_load*A_bldg)
    W_super = np.sum(D_load[1:]*A_bldg)
    
    # assume lateral frames are placed on the edge
    trib_width_lat = L_bay/2
    
    # line loads for lateral frame (kip/ft)
    w_D = D_load*trib_width_lat
    w_L = L_load*trib_width_lat
    w_Ev = 0.2*S_s*w_D
    
    
    w_case_1 = 1.4*w_D
    w_case_2 = 1.2*w_D + 1.6*w_L # includes both case 2 and 3
    # case 4 and 5 do not control (wind)
    w_case_6 = 1.2*w_D + w_Ev + 0.5*w_L
    w_case_7 = 0.9*w_D - w_Ev
    w_case_NLTH = 1.0*w_D + 0.5*w_L
    
    w_on_frame = np.maximum.reduce([w_case_1,
                                    w_case_2,
                                    w_case_6,
                                    w_case_7])
    
    # leaning columns
    L_bldg = n_bays*L_bay
    
    # area assigned to lateral frame minus area already modeled by line loads
    trib_width_LC = (L_bldg/n_frames) - trib_width_lat 
    trib_area_LC = trib_width_LC * L_bldg
    
    # point loads for leaning column (kips)
    P_D = D_load*trib_area_LC
    P_L = L_load*trib_area_LC
    P_Ev = 0.2*S_s*P_D
    
    # TODO: here, vertical load might be changed for isolation (Section 17.2.7)
    # particularly E_v and the NLTH case
    
    # this reduction takes E_v from 0.2 S_ds = 0.456 to 0.12 S_ms = 0.410
    
    P_case_1 = 1.4*P_D
    P_case_2 = 1.2*P_D + 1.6*P_L # includes both case 2 and 3
    # case 4 and 5 do not control (wind)
    P_case_6 = 1.2*P_D + P_Ev + 0.5*P_L
    P_case_7 = 0.9*P_D - P_Ev
    P_case_NLTH = 1.0*P_D + 0.5*P_L
    
    P_on_leaning_column = np.maximum.reduce([P_case_1,
                                    P_case_2,
                                    P_case_6,
                                    P_case_7])
    
    all_w_cases = {
        '1.4D': w_case_1,
        '1.2D+1.6L': w_case_2,
        '1.2D+0.5L+1.0E': w_case_6,
        '0.9D-1.0E': w_case_7,
        '1.0D+0.5L': w_case_NLTH}
    
    all_plc_cases = {
        '1.4D': P_case_1,
        '1.2D+1.6L': P_case_2,
        '1.2D+0.5L+1.0E': P_case_6,
        '0.9D-1.0E': P_case_7,
        '1.0D+0.5L': P_case_NLTH}
    
    return(W_seis, W_super, w_on_frame, P_on_leaning_column,
           all_w_cases, all_plc_cases)

# from Table 12.8-2
def get_Ct(frame_type):
    return {
        'MF': 0.028,
        'CBF' : 0.02,
        'BRB' : 0.03, 
        'SW' : 0.02
    }.get(frame_type, 0.02)

def get_x_Tfb(frame_type):
    return {
        'MF': 0.8,
        'CBF' : 0.75,
        'BRB' : 0.75, 
        'SW' : 0.75
    }.get(frame_type, 0.75)

def estimate_period(input_df, use_Cu=True, unit_in_ft=True):
    struct_type = input_df['superstructure_system']
    if unit_in_ft:
        h_n = input_df['h_bldg']
    else:
        h_n = input_df['h_bldg']/12
    # approximate fixed based fundamental period
    Ct = get_Ct(struct_type)
    x_Tfb = get_x_Tfb(struct_type)
    T_a = Ct*(h_n**x_Tfb)
    
    if use_Cu:
        if struct_type == 'CBF':
            C_u = 1.4
        # this is a departure from code value from my own set of data
        elif struct_type == 'MF':
            C_u = 1.4
    else:
        C_u = 1.0
        
    return(C_u*T_a)

# returns the required story forces per frame per floor
# units are kips and inches
def define_lateral_forces(input_df, D_load=None, L_load=None):
    
    D_m = input_df['D_m']
    K_e = input_df['k_e']
    zeta_e = input_df['zeta_e']
    R_y = input_df['RI']
    struct_type = input_df['superstructure_system']
    n_floors = input_df['num_stories']
    n_bays = input_df['num_bays']
    n_frames = input_df['num_frames']
    L_bay = input_df['L_bay']
    h_story = input_df['h_story']
    
    import numpy as np
    
    # assuming 100 psf D and 50 psf L for floors 
    # assume that D already includes structural members
    # here we do not tack on diaphragm level because these are story lateral forces
    if D_load is None:
        D_load = np.repeat(100.0/1000, n_floors)
        # roof loading is lighter
        D_load[-1] = 75.0/1000
    if L_load is None:
        L_load = np.repeat(50.0/1000, n_floors)
        L_load[-1] = 20.0/1000
    
    # assuming square building
    A_bldg = (L_bay*n_bays)**2
    
    # seismic weight: ASCE 7-22, Ch. 12.7.2
    W_tot = input_df['W']
    W_s = input_df['W_s']
    
    # assuming square building
    A_bldg = (L_bay*n_bays)**2
    
    ft = 12.0
    
    wx = D_load*A_bldg                          # Floor seismic weights
    hsx = np.repeat(h_story*ft, n_floors)     # Column heights
    hx = np.arange(1, n_floors+1) * hsx         # Floor elevations
    h_col = np.repeat(h_story*ft, n_floors)   # Column moment arm heights
    h_col[-1] = h_story/2*ft

    # unnormalize stiffness
    K = K_e * W_tot
    Vb = (D_m * K)/n_frames
    Vst = (Vb*(W_s/W_tot)**(1 - 2.5*zeta_e))
    Vs = (Vst/R_y)
    F1 = (Vb - Vst)/R_y

    # approximate fixed based fundamental period
    Ct = get_Ct(struct_type)
    x_Tfb = get_x_Tfb(struct_type)
    h_n = np.sum(hsx)/12.0
    if struct_type == 'CBF':
        C_u = 1.4
    # this is a departure from code value from my own set of data
    elif struct_type == 'MF':
        C_u = 1.4
    T_a = Ct*(h_n**x_Tfb)
    T_fb = C_u*T_a

    k       = 14*zeta_e*T_fb

    hxk     = hx**k

    CvNum   = wx*hxk
    CvDen   = np.sum(CvNum)

    Cvx     = CvNum/CvDen

    Fx      = Cvx*Vs
    
    return(wx, hx, h_col, hsx, Fx, Vs, T_fb)