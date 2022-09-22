############################################################################
#               Superstructure design algorithm

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) forces need to adjust Ry for structural system

############################################################################




def get_properties(shape):
    if (len(shape) == 0):
        raise IndexError('No shape fits the requirements.')
    Zx      = float(shape.iloc[0]['Zx'])
    Ag      = float(shape.iloc[0]['A'])
    Ix      = float(shape.iloc[0]['Ix'])
    bf      = float(shape.iloc[0]['bf'])
    tf      = float(shape.iloc[0]['tf'])
    return(Ag, bf, tf, Ix, Zx)

def calculate_strength(shape, L_bay):
    # returns critical moments and shears given shape
    Zx      = float(shape.iloc[0]['Zx'])

    ksi = 1.0
    Fy = 50.0*ksi
    Fu = 65.0*ksi

    Ry_ph = 1.1
    Cpr = (Fy + Fu)/(2*Fy)

    Mn          = Zx*Fy
    Mpr         = Mn*Ry_ph*Cpr
    Vpr         = 2*Mpr/L_bay
    return(Mn, Mpr, Vpr)

# from Table 12.8-2
def get_Ct(frame_type):
    return {
        'SMRF': 0.028,
        'CBF' : 0.03,
        'BRB' : 0.03, 
        'SW' : 0.02
    }.get(frame_type, 0.02)

def get_x_Tfb(frame_type):
    return {
        'SMRF': 0.8,
        'CBF' : 0.75,
        'BRB' : 0.75, 
        'SW' : 0.75
    }.get(frame_type, 0.75)

# returns the required story forces per frame per floor
def get_story_forces(D_m, K_e, W_tot, W_s, n_frames, zeta_e, R_y, struct_type):
    import numpy as np
    
    kip = 1.0
    ft = 12.0
    
    wx      = np.array([810.0*kip, 810.0*kip, 607.5*kip])           # Floor seismic weights
    hx      = np.array([13.0*ft, 26.0*ft, 39.0*ft])                 # Floor elevations
    hCol    = np.array([13.0*ft, 13.0*ft, 7.5*ft])                  # Column moment arm heights
    hsx     = np.array([13.0*ft, 13.0*ft, 13.0*ft])                 # Column heights
    wLoad   = np.array([2.72*kip/ft, 2.72*kip/ft, 1.94*kip/ft])     # Floor line loads

    Vb      = (D_m * K_e)/n_frames
    Vst     = (Vb*(W_s/W_tot)**(1 - 2.5*zeta_e))
    Vs      = (Vst/R_y)
    F1      = (Vb - Vst)/R_y

    # approximate fixed based fundamental period
    Ct = get_Ct(struct_type)
    x_Tfb = get_x_Tfb(struct_type)
    h_n = np.sum(hsx)/12.0
    T_fb = Ct*(h_n**x_Tfb)

    k       = 14*zeta_e*T_fb

    hxk     = hx**k

    CvNum   = wx*hxk
    CvDen   = np.sum(CvNum)

    Cvx     = CvNum/CvDen

    Fx      = Cvx*Vs
    
    return(wx, hx, hCol, hsx, wLoad, Fx, Vs)

def get_MRF_element_forces(hsx, Fx, R_y, n_bays):
    import numpy as np
    
    nFloor      = len(hsx)
    Cd          = R_y

    thetaMax    = 0.015         # ASCE Table 12.12-1 drift limits
    delx        = thetaMax*hsx
    delxe       = delx*(1/Cd)   # assumes Ie = 1.0

    # element lateral force
    Q           = np.empty(nFloor)
    Q[-1]       = Fx[-1]

    for i in range(nFloor-2, -1, -1):
        Q[i] = Fx[i] + Q[i+1]

    q           = Q/n_bays

    return(delxe, q)

def get_required_modulus(q, hCol, hsx, delxe, L_bay, wLoad):
    # beam-column relationships
    alpha       = 0.8           # strain ratio
    dcdb        = 0.5           # depth ratio
    beta        = dcdb/alpha

    # required I
    ksi = 1.0
    E           = 29000*ksi
    # story beams
    Ib          = q*hCol**2/(12*delxe*E)*(hCol/beta + L_bay)                             
    Ib[-1]      = q[-1]*hsx[-1]**2/(12*delxe[-1]*E)*(hsx[-1]/(2*beta) + L_bay)
    # roof beams, using hsx since flexibility method assumes h = full column
    Ic          = Ib*beta

    # required Z
    Fy              = 50*ksi
    MGrav           = wLoad*L_bay**2/12
    MEq             = q*hCol/2
    Mu              = MEq + MGrav
    Zb              = Mu/(0.9*Fy)

    return(Ib, Ic, Zb)

def select_member(member_list, req_var, req_val):
    # req_var is string 'Ix' or 'Zx'
    # req_val is value
    qualified_list = member_list[member_list[req_var] > req_val]
    sorted_weight = qualified_list.sort_values(by=['W'])
    selected_member = sorted_weight.iloc[:1]
    return(selected_member, qualified_list)

# Zx check


def zx_check(current_member, member_list, Z_beam_req):
    (beam_Ag, beam_bf, beam_tf, 
        beam_Ix, beam_Zx) = get_properties(current_member)

    if(beam_Zx < Z_beam_req):
        selected_member, qualified_list = select_member(member_list, 
            'Zx', Z_beam_req)
    else:
        selected_member = current_member
        qualified_list = member_list

    return(selected_member, qualified_list)

# PH location check

def ph_shear_check(current_member, member_list, line_load, L_bay):
    
    import numpy as np
    
    ksi = 1.0
    Fy = 50.0*ksi
    Fu = 65.0*ksi

    # PH location check

    Ry_ph = 1.1
    Cpr = (Fy + Fu)/(2*Fy)

    (M_n, M_pr, V_pr) = calculate_strength(current_member, L_bay)

    ph_VGrav = line_load*L_bay/2
    ph_VBeam = 2*M_pr/(0.9*L_bay)  # 0.9L_bay for plastic hinge length
    ph_location = ph_VBeam > ph_VGrav

    if not np.all(ph_location):
        # print('Detected plastic hinge away from ends. Reselecting...')
        Z_beam_ph_req = max(line_load*L_bay**2/(4*Fy*Ry_ph*Cpr))
        selected_member, ph_list = select_member(member_list, 
            'Zx', Z_beam_ph_req)

    else:
        selected_member = current_member
        ph_list = member_list

    # beam shear
    (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_member)

    (M_n, M_pr, V_pr) = calculate_strength(selected_member, L_bay)

    A_web        = A_g - 2*(t_f*b_f)
    V_n          = 0.9*A_web*0.6*Fy

    beam_shear_fail   = V_n < V_pr

    if beam_shear_fail:
        # print('Beam shear check failed. Reselecting...')
        Ag_req   = 2*V_pr/(0.9*0.6*Fy)    # Assume web is half of gross area

        selected_member, shear_list = select_member(ph_list, 
            'A', Ag_req)

    else:
        shear_list = ph_list

    return(selected_member, shear_list)

# SCWB design
def select_column(wLoad, L_bay, h_col, current_beam, current_roof, col_list, 
    I_col_req, I_beam_req):

    import numpy as np
    
    ksi = 1.0
    Fy = 50.0*ksi
    # V_grav = 2 * wx * L / 2 (shear induced by both beams on column)
    V_grav      = wLoad*L_bay

    nFloor = len(wLoad)

    (M_n_beam, M_pr_beam, V_pr_beam) = calculate_strength(current_beam, L_bay)
    (M_n_roof, M_pr_roof, V_pr_roof) = calculate_strength(current_roof, L_bay)

    # find axial demands
    Pr              = np.empty(nFloor)
    Pr[-1]          = V_pr_beam + V_grav[-1]
    for i in range(nFloor-2, -1, -1):
        Pr[i] = V_grav[i] + V_pr_beam + Pr[i + 1]

    # initial guess: use columns that has similar Ix to beam
    qualified_Ix = col_list[col_list['Ix'] > I_beam_req]
    selected_col = qualified_Ix.iloc[(qualified_Ix['Ix'] - I_beam_req).abs().argsort()[:1]]  # select the first few that qualifies Ix

    (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_col)
    M_pr = Z_x*(Fy - Pr/A_g)

    # find required Zx for SCWB to be true
    scwb_Z_req        = np.max(M_pr_beam/(Fy - Pr[:-1]/A_g))

    # select column based on SCWB
    selected_col, passed_Ix_cols = select_member(col_list, 
        'Ix', I_col_req)
    selected_col, passed_Zx_cols = select_member(passed_Ix_cols, 
        'Zx', scwb_Z_req)

    (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_col)
    M_pr = Z_x*(Fy - Pr/A_g)
    

    # check final SCWB
    ratio           = np.empty(nFloor-1)
    
    for i in range(nFloor-2, -1, -1):
        ratio[i] = (M_pr[i+1] + M_pr[i])/(2*M_pr_beam)
        if (ratio[i] < 1.0):
            print('SCWB check failed at floor ' + str(nFloor+1) + ".")

    A_web = A_g - 2*(t_f*b_f)
    V_n  = 0.9*A_web*0.6*Fy
    V_pr = max(M_pr/h_col)
    col_shear_fail   = V_n < V_pr

    if col_shear_fail:
        # print('Column shear check failed. Reselecting...')
        Ag_req   = 2*V_pr/(0.9*0.6*Fy)    # Assume web is half of gross area

        selected_col, shear_list = select_member(passed_Zx_cols, 
            'A', Ag_req)

    else:
        shear_list = passed_Zx_cols

    return(selected_col, shear_list)

############################################################################
#              ASCE 7-16: Capacity design
############################################################################
    
def design_MF(D_m, K_e, W_tot, W_s, n_frames, zeta_e, R_y,
              n_bays, L_bay, struct_type):
    
    import pandas as pd

    # ASCE 7-16: Story forces
    wx, hx, hCol, hsx, wLoad, Fx, Vs = get_story_forces(D_m, K_e, W_tot, W_s, 
                                                    n_frames, zeta_e, 
                                                    R_y, struct_type)

    delxe, q = get_MRF_element_forces(hsx, Fx, R_y, n_bays)
    
    # get required section specs
    Ib, Ic, Zb = get_required_modulus(q, hCol, hsx, delxe, L_bay, wLoad)

    I_beam_req        = Ib.max()
    I_col_req         = Ic.max()
    Z_beam_req        = Zb.max()
    
    I_roof_beam_req    = Ib[-1]
    Z_roof_beam_req    = Zb[-1]
    
    # import shapes 
    
    beam_shapes      = pd.read_csv('../inputs/beamShapes.csv',
        index_col=None, header=0)
    sorted_beams     = beam_shapes.sort_values(by=['Ix'])
    
    col_shapes       = pd.read_csv('../inputs/colShapes.csv',
        index_col=None, header=0)
    sorted_cols      = col_shapes.sort_values(by=['Ix'])



    # Floor beams

    selected_beam, passed_Ix_beams = select_member(sorted_beams, 
                                                   'Ix', I_beam_req)

    selected_beam, passed_Zx_beams = zx_check(selected_beam, 
                                              passed_Ix_beams, Z_beam_req)

    story_loads = wLoad[:-1]
    selected_beam, passed_checks_beams = ph_shear_check(selected_beam, 
                                                        passed_Zx_beams, 
                                                        story_loads, L_bay)


    # Roof beams

    selected_roof_beam, passed_Ix_roof_beams = select_member(sorted_beams, 
        'Ix', I_roof_beam_req)
    
    selected_roof_beam, passed_Zx_roof_beams = zx_check(selected_roof_beam, 
        passed_Ix_roof_beams, Z_roof_beam_req)
    
    roof_load = wLoad[-1]
    
    selected_roof_beam, passed_checks_roof_beams = ph_shear_check(selected_roof_beam, 
        passed_Zx_roof_beams, roof_load, L_bay)


    
    # columns
    selected_column, passed_check_cols = select_column(wLoad, L_bay, 
        hCol, selected_beam, selected_roof_beam, sorted_cols, 
        I_col_req, I_beam_req)
    
    return(selected_beam, selected_roof_beam, selected_column)

def main():
    ft = 12.0

    D_m = 22.12
    K_e = 40.073
    zeta_e = 0.15
    W_tot = 3530
    n_frames = 2
    W_s = 2227.5
    R_y = 1.0


    n_bays = 3
    L_bay = 30*ft
    
    frame_type = 'SMRF'
    
    beam, roof, col = design_MF(D_m, K_e, W_tot, W_s, n_frames, zeta_e, R_y,
                  n_bays, L_bay, frame_type)
    
    return(beam, roof, col)
    
# beam, roof, col = main()
# if __name__ == '__main__':
#     design_LRB()