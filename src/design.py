############################################################################
#               Design algorithms

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  Main design algorithm for bearings and superstructure frames

# Open issues:  (1) checks not done yet
#               (2) for LRB case, check the load division/bearing layout effects

############################################################################

def get_layout(n_bays):
    # place bearings on edge
    num_bearings = (n_bays+1)**2
    # from numpy import ceil
    num_lb = 4*n_bays
    # num_lb = ceil(0.4*num_bearings / 2.)*2
    num_rb = num_bearings - num_lb
    return(int(num_lb), int(num_rb))

# perform one iteration of LRB design to return a damping coefficient
def iterate_LRB(zeta_guess, S_1, T_m, Q_L, rho_k, W_tot):
    
    from numpy import interp
    g  = 386.4
    pi = 3.14159
    
    # from ASCE Ch. 17, get damping multiplier
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
    B_m      = interp(zeta_guess, zetaRef, BmRef)
    
    # design displacement
    D_m = g*S_1*T_m/(4*pi**2*B_m)
    k_M = (2*pi/T_m)**2 * (W_tot/g)
    
    # from Q, zeta, and T_m
    k_2 = (k_M*D_m - Q_L)/D_m
    
    # yielding force
    k_1 = rho_k * k_2
    D_y = Q_L/(k_1 - k_2)
    
    zeta_E = (4*Q_L*(D_m - D_y)) / (2*pi*k_M*D_m**2)
    
    err = (zeta_E - zeta_guess)**2
    
    return(err)

# from specified parameters, find the height that converges on lead height
# (and rubber height) that achieves the k_1 specified
def iterate_bearing_height(tr_guess, D_m, k_M, Q_L, rho_k, N_lb, S_des=15.0):
    k_2 = (k_M*D_m - Q_L)/D_m
    
    # required area of lead per bearing
    f_y_Pb = 1.5 # ksi, shear yield strength
    A_Pb = (Q_L/f_y_Pb) / N_lb # in^2
    pi = 3.14159
    d_Pb = (4*A_Pb/pi)**(0.5)
    a = d_Pb/2
    
    # 60 psi rubber
    # select thickness
    
    G_r = 0.060 # ksi, shear modulus 
    A_r = k_2 * tr_guess / (G_r * N_lb)
    d_r = (4*(A_r + A_Pb)/pi)**(0.5)
    b_s = (d_r - 0.5)/2
    
    # yielding force
    k_1 = rho_k * k_2
    # assume lead has shear modulus of 150 MPa ~ 21 ksi
    G_Pb = 21.0 # ksi
    h_Pb = (G_Pb * A_Pb + A_r * G_r)*N_lb/k_1
    
    # try for shape factor of 15
    t_pad_req = b_s / (2*S_des)
    # t_pad_req = (b_s - a)/(2*S_pad_trial)
    
    from math import floor
    n_layers = floor(tr_guess/t_pad_req)
    n_shims = n_layers - 1
    
    # assume 12 layers (11 shims at ~3.5mm thickness)
    # assume lead core goes 0.5 inch into each endplate
    tr_req = h_Pb - (n_shims*0.13) - 1.0
    
    err = (tr_req - tr_guess)**2
    
    return(err)

def design_LRB(param_df):
    
    # read in parameters
    T_m = param_df['T_m']
    S_1 = param_df['S_1']
    Q = param_df['Q']
    rho_k = param_df['k_ratio']
    n_bays = param_df['num_bays']
    W_tot = param_df ['W']
    
    # number of LRBs vs non LRBs
    N_lb, N_sl = get_layout(n_bays)
    
    Q_L = Q * W_tot
    
    # converge design on damping
    # design will achieve T_m, Q, rho_k as specified
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(iterate_LRB, args=(S_1, T_m, Q_L, rho_k, W_tot),
                          bounds=(0.01, 0.35), method='bounded')

    zeta_m = res.x
    
    from numpy import interp
    g  = 386.4
    pi = 3.14159
    
    # from ASCE Ch. 17, get damping multiplier
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
    B_m      = interp(zeta_m, zetaRef, BmRef)
    
    # design displacement
    D_m = g*S_1*T_m/(4*pi**2*B_m)
    k_M = (2*pi/T_m)**2 * (W_tot/g)

    k_2 = (k_M*D_m - Q_L)/D_m
    
    # required area of lead per bearing
    f_y_Pb = 1.5 # ksi, shear yield strength
    A_Pb = (Q_L/f_y_Pb) / N_lb # in^2
    d_Pb = (4*A_Pb/pi)**(0.5)
    
    # converge on t_r necessary to achieve rho_k
    S_pad_trial = 20.0
    res = minimize_scalar(iterate_bearing_height,
                          args=(D_m, k_M, Q_L, rho_k, N_lb, S_pad_trial),
                          bounds=(0.01, 1e3), method='bounded')
    t_r = res.x
    
    # 60 psi rubber
    # select thickness
    
    G_r = 0.060 # ksi, shear modulus
    A_r = k_2 * t_r / (G_r * N_lb)
    d_r = (4*(A_r + A_Pb)/pi)**(0.5)
    
    # yielding force
    k_1 = rho_k * k_2
    D_y = Q_L/(k_1 - k_2)
    
    # final values
    k_e = (Q_L + k_2*D_m)/D_m
    T_e = 2*pi*(W_tot/(g*k_e))**0.5
    W_e = 4*Q_L*(D_m - D_y)
    zeta_E = W_e/(2*pi*k_e*D_m**2)
    
    #################################################
    # buckling checks
    #################################################
    
    flag = 0
    # assume small strain G is 75% larger
    G_ss = 1.75*G_r
    # incompressibility
    K_inc = 290 # ksi
    
    
    # shape factor (circular)
    # assumes 12 layers
    a = d_Pb/2
    b = d_r/2
    b_s = (d_r - 0.5)/2
    
    # try for shape factor of 15
    # S_pad_trial = 30.0
    # t_pad_req = (b_s - a)/(2*S_pad_trial)
    t_pad_req = b_s/(2*S_pad_trial)
    
    from math import floor
    n_layers = floor(t_r/t_pad_req)
    
    # if nonsense n_layers reach, stop calculations now (to be discarded)
    if n_layers < 1:
        return(1.0, 1.0, 1.0, 1.0, 1, T_e, k_e, zeta_E, D_m, 1)
    
    n_shims = n_layers - 1
    t = t_r/n_layers
    
    # assume shim is half inch less than rubber diameter
    # buckling values are calculated for rubber area overlapping with shims
    # the following values are annular
    
    I = pi/4 * (b_s**4 - a**4)
    A = pi*(b_s**2 - a**2)
    h = t_r + n_shims*0.13 # 11x0.1 in thick shim
    # S_pad = (b_s - a)/(2*t)
    S_pad = b_s/(2*t)
    eta = a/b_s
    th = (48*G_ss/K_inc)**(0.5)*S_pad/(1 - eta)
    
    # modified Bessel functions
    from scipy.special import kv, i1, iv, i0
    
    #################################################
    # compressive behavior
    #################################################
    
    # shape factor adjusts for annular shape
    from math import log
    lam = (b**2 + a**2 - ((b**2 - a**2)/(log(b/a))))/((b - a)**2)
    E_pc = 6*lam*G_ss*S_pad**2
    
    # this seems to adjusts for incompressibility but is ad hoc
    # E_c = (E_pc*K_inc)/(E_pc + K_inc) 
    
    # full solution from Kelly & Konstantinidis
    C1p = ((1/((12*G_ss/K_inc)**0.5*(1 + eta)*S_pad)) * 
           (kv(0, th) - kv(0, eta*th)) / 
           (i0(th)*kv(0, eta*th) - i0(eta*th)*kv(0, th)))
    
    C2p = ((1/((12*G_ss/K_inc)**0.5*(1 + eta)*S_pad)) * 
           (i0(th) - i0(eta*th)) / 
           (i0(th)*kv(0, eta*th) - i0(eta*th)*kv(0, th)))
    
    E_c = (K_inc*(1 + C1p*(iv(1, th) - eta*iv(1,eta*th)) +
                  C2p*(kv(1, th) - eta*kv(1, eta*th))))
    
    #################################################
    # bending behavior
    #################################################
    
    # from Kelly & Konstantinidis
    # first calculate the incompressible case
    
    # for an annular pad
    # this is equivalent to pi*G/8 *(b**2 - a**2)**3/t**2
    EI_eff_inc = 2*G_ss*S_pad**2*I*(1 + eta)**2/(1 + eta**2)
    
    
    
    B1p = (4/(th*(1 - eta**4)) * 
            (-kv(1, eta*th) + eta*kv(1,th)) / 
            (i1(eta*th)*kv(1, th) - i1(th)*kv(1,eta*th)))
    
    B2p = (4/(th*(1 - eta**4)) * 
            (i1(eta*th) - eta*i1(th)) / 
            (i1(eta*th)*kv(1, th) - i1(th)*kv(1,eta*th)))
    
    EI_comp_ratio = (K_inc/(2*G_ss*S_pad**2) * 
                      (1 + eta**2)/((1 + eta)**2) * 
                      (1 - B1p*(iv(2, th) - eta**2*iv(2, eta*th)) +
                      B2p*(kv(2, th) - eta**2*kv(2, eta*th))))
    
    EI_eff_comp = EI_eff_inc * EI_comp_ratio
    
    # assume 1.0 inch cover plates
    # height of lead core
    # h_Pb = t_r + 11*0.1 + 1.0
    
    # global buckling check, uses EI_s = 1/3 E_c I h/tr
    # full unsimplified equation
    P_S = G_ss*A*h/t_r
    
    # # is this specific to circular only?
    # P_E = pi**2/(h**2)/3*E_c*I*h/t_r
    
    # annular
    P_E = pi**2*EI_eff_comp*h/t_r/(h**2)
    
    # full solution critical load
    P_crit = (-P_S + (P_S**2 + 4*P_S*P_E)**0.5)/2
    
    # # truncated solution for sqrt(P_S P_E) assuming that S > 5
    # P_crit = pi**2*G_ss*(b_s**2 - a**2)**2/(2*(2**0.5)*t_r*t)
    
    # # already accounts for A_s effective shear area
    # P_crit = pi/t_r * ((E_c * I/3)*G_ss*A)**(0.5)
    
    w_floor = param_df['w_fl'] # k/ft
    L_bay = param_df['L_bay'] # ft
    P_estimate = sum(w_floor)*L_bay
    
    # P_estimate = W_tot/(N_lb + N_sl)
    
    if P_estimate/P_crit > 1.0:
        flag = 1
    
    # normalize stiffness by weight
    k_e_norm = k_e/W_tot
    
    # shortcut for circular bearing 
    # compare the strength with an equivalent circular bearing
    p_crit_compare = P_crit/(pi*b_s**2)
    S_2 = b_s/t_r
    p_crit_circ = G_ss*pi*S_pad*S_2/(2*2**0.5)
    return(d_r, d_Pb, t_r, t, n_layers, T_e, k_e_norm, zeta_E, D_m, flag)
    



# perform one iteration of TFP design to return a damping coefficient
def iterate_TFP(zeta_guess, mu_1, S_1, T_m, Q, rho_k):
    from numpy import interp
    g  = 386.4
    pi = 3.14159
    
    # from ASCE Ch. 17, get damping multiplier
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
    # from T_m, zeta_M, S_1
    B_m = interp(zeta_guess, zetaRef, BmRef)
    D_m = g*S_1*T_m/(4*pi**2*B_m)
    
    # specify sliders
    h_1 = 1.0
    h_2 = 4.0
    
    k_M = (2*pi/T_m)**2 * (1/g)
    
    u_y = 0.01
    
    k_0 = mu_1/u_y
    
    # from Q and D_m
    k_2 = (k_M*D_m - Q)/D_m
    R_2 = 1/(2*k_2) + h_2
    
    # from rho_k
    u_a = Q/(k_2*(rho_k-1))
    k_a = rho_k*k_2
    mu_2 = u_a*k_a
    R_1 = u_a/(2*(mu_2-mu_1)) + h_1
    
    # effective design values
    a = 1/(2*R_1)
    b = 1/(2*R_2)
    k_e = (mu_2 + b*(D_m - u_a))/D_m
    W_e = 4*(mu_2 - b*u_a)*D_m - 4*(a-b)*u_a**2 - 4*(k_0 -a)*u_y**2
    zeta_E   = W_e/(2*pi*k_e*D_m**2)
    
    err = (zeta_E - zeta_guess)**2
    
    return(err)
    
def design_TFP(param_df):
    
    # read in parameters
    T_m = param_df['T_m']
    S_1 = param_df['S_1']
    Q = param_df['Q']
    rho_k = param_df['k_ratio']
    
    # guess
    import random
    # random.seed(985)
    mu_Q_coef = random.uniform(0.3, 0.6)
    mu_1 = mu_Q_coef*Q
    
    # converge design on damping
    # design will achieve T_m, Q, rho_k as specified
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(iterate_TFP, args=(mu_1, S_1, T_m, Q, rho_k),
                             bounds=(0.01, 0.35), method='bounded')

    zeta_m = res.x
    
    # finish design on converged damping
    from numpy import interp
    g  = 386.4
    pi = 3.14159
    
    # from ASCE Ch. 17, get damping multiplier
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
    # from T_m, zeta_M, S_1
    B_m = interp(zeta_m, zetaRef, BmRef)
    D_m = g*S_1*T_m/(4*pi**2*B_m)
    
    k_M = (2*pi/T_m)**2 * (1/g)
    
    # specify sliders
    h_1 = 1.0
    h_2 = 4.0
    
    # W_m = zeta_M*(2*pi*k_M*D_m**2)
    
    u_y = 0.01
    
    k_0 = mu_1/u_y
    
    # from Q and D_m
    k_2 = (k_M*D_m - Q)/D_m
    R_2 = 1/(2*k_2) + h_2
    
    # from rho_k
    u_a = Q/(k_2*(rho_k-1))
    k_a = rho_k*k_2
    mu_2 = u_a*k_a
    R_1 = u_a/(2*(mu_2-mu_1)) + h_1
    
    # effective design values
    a = 1/(2*R_1)
    b = 1/(2*R_2)
    k_e = (mu_2 + b*(D_m - u_a))/D_m
    W_e = 4*(mu_2 - b*u_a)*D_m - 4*(a-b)*u_a**2 - 4*(k_0 -a)*u_y**2
    zeta_E   = W_e/(2*pi*k_e*D_m**2)
    T_e = 2*pi*(1/(g*k_e))**0.5
    
    return(mu_1, mu_2, R_1, R_2, T_e, k_e, zeta_E, D_m)

def get_properties(shape):
    # if (len(shape) == 0):
    #     raise IndexError('No shape fits the requirements.')
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

def get_MRF_element_forces(hsx, Fx, R_y, n_bays):
    import numpy as np
    
    nFloor      = len(hsx)
    # take ratio of Cd -> Ry from Table 12.2-1
    Cd_code = 5.5
    Ry_code = 8.0
    Cd          = (Cd_code/Ry_code)*R_y
    
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
    # ensure units are in inches
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
    
    from numpy import nan
    if len(qualified_list) < 1:
        return(nan, nan)
    
    sorted_weight = qualified_list.sort_values(by=['W'])
    selected_member = sorted_weight.iloc[:1]
    return(selected_member, qualified_list)

# Zx check


def zx_check(current_member, member_list, Z_beam_req):
    
    from numpy import nan
    if len(current_member) < 1:
        return(nan, nan)
    
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
    
    from numpy import nan
    if len(current_member) < 1:
        return(nan, nan)
    
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
        Z_beam_ph_req = np.max(line_load*L_bay**2/(4*Fy*Ry_ph*Cpr))
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

    beam_shear_fail   = V_n < (V_pr + np.max(ph_VGrav))

    if beam_shear_fail:
        # print('Beam shear check failed. Reselecting...')
        Ag_req   = 2*(V_pr + np.max(ph_VGrav))/(0.9*0.6*Fy)    # Assume web is half of gross area

        selected_member, shear_list = select_member(ph_list, 
            'A', Ag_req)

    else:
        shear_list = ph_list

    return(selected_member, shear_list)

# SCWB design
def select_column(wLoad, L_bay, h_col, current_beam, current_roof, col_list, 
    I_col_req, I_beam_req):

    import numpy as np
    
    if (current_beam is np.nan) or (current_roof is np.nan):
        return(np.nan, np.nan, False)
    
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
        
    # TODO: check SMRF column axial capacity

    # initial guess: use columns that has similar Ix to beam
    qualified_Ix = col_list[col_list['Ix'] > I_beam_req]
    from numpy import nan
    if len(qualified_Ix) < 1:
        return (nan, nan, False)
    selected_col = qualified_Ix.loc[[(qualified_Ix.Ix - 
                                       I_beam_req).abs().idxmin()]] # select the first few that qualifies Ix
    
    (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_col)
    M_pr = Z_x*(Fy - Pr/A_g)

    # find required Zx for SCWB to be true
    scwb_Z_req        = np.max(M_pr_beam/(Fy - Pr[:-1]/A_g))

    # select column based on SCWB
    selected_col, passed_Ix_cols = select_member(col_list, 
        'Ix', I_col_req)
    selected_col, passed_Zx_cols = select_member(passed_Ix_cols, 
        'Zx', scwb_Z_req)
    
    import pandas as pd
    if isinstance(selected_col, pd.DataFrame):
        (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_col)
    M_pr = Z_x*(Fy - Pr/A_g)
    

    # check final SCWB
    ratio           = np.empty(nFloor-1)
    
    scwb_flag = False
    for i in range(nFloor-2, -1, -1):
        ratio[i] = (M_pr[i+1] + M_pr[i])/(2*M_pr_beam)
        if (ratio[i] < 1.0):
            # print('SCWB check failed at floor ' + str(nFloor+1) + ".")
            scwb_flag = True

    A_web = A_g - 2*(t_f*b_f)
    V_n  = 0.9*A_web*0.6*Fy
    V_pr = np.max(M_pr/h_col)
    col_shear_fail   = V_n < V_pr

    if col_shear_fail:
        # print('Column shear check failed. Reselecting...')
        Ag_req   = 2*V_pr/(0.9*0.6*Fy)    # Assume web is half of gross area

        selected_col, shear_list = select_member(passed_Zx_cols, 
            'A', Ag_req)

    else:
        shear_list = passed_Zx_cols

    return(selected_col, shear_list, scwb_flag)

############################################################################
#              ASCE 7-22: Capacity design for moment frame
############################################################################
    
def design_MF(input_df, db_string='../resource/'):
    
    # ensure everything is in inches, kip/in
    ft = 12.0
    R_y = input_df['RI']
    n_bays = input_df['num_bays']
    L_bay = input_df['L_bay']*ft 
    hsx = input_df['hsx']
    Fx = input_df['Fx']
    h_col = input_df['h_col']
    
    load_cases = input_df['all_w_cases']
    case_1 = load_cases['1.2D+0.5L+1.0E']/12
    case_2 = load_cases['0.9D-1.0E']/12
    
    import numpy as np
    w_load = np.maximum(case_1, case_2)
    # w_load = input_df['w_fl']/ft
    
    import pandas as pd

    # ASCE 7-22: Story forces

    delxe, q = get_MRF_element_forces(hsx, Fx, R_y, n_bays)
    
    # get required section specs
    Ib, Ic, Zb = get_required_modulus(q, h_col, hsx, delxe, L_bay, w_load)

    I_beam_req        = Ib.max()
    I_col_req         = Ic.max()
    Z_beam_req        = Zb.max()
    
    I_roof_beam_req    = Ib[-1]
    Z_roof_beam_req    = Zb[-1]
    
    # import shapes 
    
    beam_shapes      = pd.read_csv(db_string+'beamShapes.csv',
        index_col=None, header=0)
    sorted_beams     = beam_shapes.sort_values(by=['Ix'])
    
    col_shapes       = pd.read_csv(db_string+'colShapes.csv',
        index_col=None, header=0)
    sorted_cols      = col_shapes.sort_values(by=['Ix'])


    # Floor beams
    selected_beam, passed_Ix_beams = select_member(sorted_beams, 
                                                   'Ix', I_beam_req)

    selected_beam, passed_Zx_beams = zx_check(selected_beam, 
                                              passed_Ix_beams, Z_beam_req)

    story_loads = w_load[:-1]
    selected_beam, passed_checks_beams = ph_shear_check(selected_beam, 
                                                        passed_Zx_beams, 
                                                        story_loads, L_bay)

    # Roof beams

    selected_roof_beam, passed_Ix_roof_beams = select_member(sorted_beams, 
        'Ix', I_roof_beam_req)
    
    selected_roof_beam, passed_Zx_roof_beams = zx_check(selected_roof_beam, 
        passed_Ix_roof_beams, Z_roof_beam_req)
    
    roof_load = w_load[-1]
    
    selected_roof_beam, passed_checks_roof_beams = ph_shear_check(selected_roof_beam, 
        passed_Zx_roof_beams, roof_load, L_bay)
    
    # columns
    selected_column, passed_check_cols, scwb_flag = select_column(w_load, L_bay, 
        h_col, selected_beam, selected_roof_beam, sorted_cols, 
        I_col_req, I_beam_req)
    
    # return only string to keep data management clean
    if isinstance(selected_beam, pd.DataFrame):
        selected_beam = selected_beam.iloc[0]['AISC_Manual_Label']
    if isinstance(selected_roof_beam, pd.DataFrame):
        selected_roof_beam = selected_roof_beam.iloc[0]['AISC_Manual_Label']
    if isinstance(selected_column, pd.DataFrame):
        selected_column = selected_column.iloc[0]['AISC_Manual_Label']
    
    return(selected_beam, selected_roof_beam, selected_column, scwb_flag)

############################################################################
#              ASCE 7-22: Capacity design for braced frame
############################################################################

def get_CBF_element_forces(hsx, Fx, R_y, n_bay_braced=2):
    import numpy as np
    
    nFloor      = len(hsx)
    # take ratio of Cd -> Ry from Table 12.2-1
    Cd_code = 5.0
    Ry_code = 6.0
    Cd          = (Cd_code/Ry_code)*R_y

    thetaMax    = 0.015         # ASCE Table 12.12-1 drift limits (risk cat III)
    delx        = thetaMax*hsx
    delxe       = delx*(1/Cd)   # assumes Ie = 1.0
    
    # element lateral force
    Q           = np.empty(nFloor)
    Q[-1]       = Fx[-1]

    for i in range(nFloor-2, -1, -1):
        Q[i] = Fx[i] + Q[i+1]

    q           = Q/n_bay_braced # stacked horizontal force per bay

    return(delxe, q)

def get_brace_demands(Fx, del_xe, q, h_story, L_bay, w_1, w_2):
    
    # angle
    from math import atan, sin, cos
    theta = atan(h_story/(L_bay/2))
    
    # required axial stiffness of braces for each bay at each level
    E = 29000.0 # ksi
    A_brace_req = q/(2*cos(theta)**2) * (L_bay)/(del_xe * E)
    # required stress capacity for buckling
    F_cr = q / A_brace_req # ksi
    del_buckling = F_cr*L_bay / (E*cos(theta))
    
    # w1 is 1.2D+0.5L, w2 is 0.9D
    # w is already for edge frame (kip/ft)
    
    # assuming frame is inner bay of edge frame
    # assuming col-brace-col is simply supported beam
    C_1 = w_1*(L_bay/2)/sin(theta)
    C_2 = w_2*(L_bay/2)/sin(theta)
    C_E = q/cos(theta)
    
    C_max = C_1 + C_E
    T_max = C_2 - C_E
    
    return(A_brace_req, C_max, T_max, del_buckling)

# def compressive_brace_loss(x, C, h, L):
#     phi_Pn = compressive_strength(x[0], x[1], h, L)
    
#     # we minimize the distance to true desired strength
#     # we also seek to minimize Ag and ry to keep the shape size as small as possible
#     loss = (phi_Pn - C)**2 + (x[0]**2 + x[1]**2)**(0.5)
#     return(loss)

# TODO: adjust this so that Ry_hss is not used for all compression design
def compressive_strength(Ag, ry, Lc_r, Ry=1.0):
    
    Ry_hss = Ry
    
    E = 29000.0 # ksi
    pi = 3.14159
    Fy = 50.0 # ksi for ASTM A500 brace
    Fy_pr = Fy*Ry_hss
    
    Fe = pi**2*E/(Lc_r**2)
    
    if (Lc_r <= 4.71*(E/Fy)**0.5):
        F_cr = 0.658**(Fy_pr/Fe)*Fy_pr
    else:
        F_cr = 0.877*Fe
        
    phi = 0.9
    phi_Pn = phi * Ag * F_cr
    return(phi_Pn)

def select_compression_member(mem_list, Lc, C_design):
    
    import numpy as np
    if mem_list is np.nan:
        return np.nan, np.nan
    
    mem_list['Lc_r'] = Lc/mem_list['ry']
    
    mem_list['phi_Pn'] = mem_list.apply(lambda row: 
                                            compressive_strength(
                                                row['A'],
                                                row['ry'],
                                                row['Lc_r']),
                                            axis='columns')
    
    # choose compact designs that can withstand C_max
    # compact requirement from AISC 341-16, Sec F2.5
    qualified_list = mem_list[(mem_list['Lc_r'] < 200.0) & 
                              (mem_list['phi_Pn'] >= C_design)]
    
    if len(qualified_list) < 1:
        return(np.nan, np.nan)
    else:
        selected_mem = qualified_list.iloc[:1]
        return(selected_mem, qualified_list)
 
# design both columns and beams

def capacity_CBF_design(selected_brace, Q_per_bay, w_cases, 
                        h_story, L_bay, n_bays,
                        beam_list, col_list):
    
    # angle
    from math import atan, sin, cos
    theta = atan(h_story/(L_bay/2))
    
    import numpy as np
    if selected_brace is np.nan:
        return np.nan, np.nan, np.nan, np.nan
    
    k_brace = 1.0 # for pinned-pinned connection, steel manual Table C-A-7.1
    Lc = (h_story**2 + (L_bay/2)**2)**(0.5)*k_brace
    rad_gy = selected_brace['ry'].iloc[0]
    Ag = selected_brace['A'].iloc[0]
    Lc_r = Lc / rad_gy

    Ry_hss = 1.4
    Fy = 50.0 #ksi
    
    # probable capacities for compr, tension, buckled compr 
    Cpr = compressive_strength(Ag, rad_gy, Lc_r, Ry=Ry_hss) / 0.9 / 0.877
    Tpr = Ag * Fy * Ry_hss
    Cpr_pr = Cpr * 0.3
    
    # axial demand on beam
    # Q_per_bay is the stacked force per bay
    L_bldg = n_bays * L_bay
    q_distr = Q_per_bay/L_bldg # distributed axial force per bay
    Q_beam = q_distr * L_bay # axial force felt in each beam
    Fh_braces = (Tpr + Cpr) * cos(theta) # horizontal force from brace action
    
    Pu_beam = Fh_braces - Q_beam
    
    # shear/moment demand on beam
    w1 = w_cases['1.2D+0.5L+1.0E']/12 # raw is kip/ft, convert to kip/in
    w2 = w_cases['0.9D-1.0E']/12
    w3 = w_cases['1.2D+1.6L']/12
    
    M_grav1 = w1*L_bay**2/8 # kip-in
    M_grav2 = w2*L_bay**2/8 # kip-in
    M_grav3 = w3*L_bay**2/8 # kip-in
    
    Fv_braces = (Tpr - Cpr_pr)*sin(theta)
    M_seis = Fv_braces * L_bay / 4 # kip-in 
    
    # apply load combos
    M_max = np.maximum.reduce([M_grav1+M_seis,
                               abs(M_grav2-M_seis),
                               M_grav3])
    
    # find required Zx that satisfies M_max
    Z_req = np.max(M_max/Fy)

    # select beam based on required moment capacity
    selected_beam, passed_Zx_beams = select_member(beam_list, 
        'Zx', Z_req)
    
    
    
    # axial check
    rad_gy_beam = selected_beam['ry'].iloc[0]
    Ag_beam = selected_beam['A'].iloc[0]
    Lc_r_beam = L_bay / rad_gy_beam
    Pn_beam = compressive_strength(Ag_beam, rad_gy_beam, Lc_r_beam)
    
    P_beam_des = np.max(Pu_beam)
    
    if P_beam_des > Pn_beam:
        selected_beam, passed_axial_beams = select_compression_member(passed_Zx_beams, 
                                                                      L_bay, 
                                                                      P_beam_des)
    else:
        passed_axial_beams = passed_Zx_beams
    
    # shear check
    # beam shear
    Mn_beam, Mpr_beam, Vpr_beam = calculate_strength(selected_beam, L_bay)
    V_grav = np.max(w1*L_bay/2)
    
    (A_g, b_f, t_f, I_x, Z_x) = get_properties(selected_beam)

    A_web        = A_g - 2*(t_f*b_f)
    Fy = 50.0
    V_n          = 0.9*A_web*0.6*Fy

    beam_shear_fail   = V_n < (Vpr_beam + V_grav)
    if beam_shear_fail:
        # print('Beam shear check failed. Reselecting...')
        Ag_req   = 2*Vpr_beam/(0.9*0.6*Fy)    # Assume web is half of gross area

        selected_beam, beam_shear_list = select_member(passed_axial_beams, 
                                                       'A', Ag_req)
    else:
        beam_shear_list = passed_axial_beams
    
    # using the 1.2D+0.5L case
    Fv_buck = (Tpr - Cpr_pr)*sin(theta)
    Fv_TC = (Tpr - Cpr)*sin(theta)
    
    P_col_grav = w1 * L_bay
    P_case_T = P_col_grav + Fv_buck/2
    P_case_C = P_col_grav - Fv_TC/2
    P_case_buck = P_col_grav - Fv_buck/2
    P_case_TC = P_col_grav + Fv_buck/2 - Fv_TC/2
    P_case_Tbuck = P_col_grav + Fv_buck/2 - Fv_buck/2
    
    # top floor has no direct brace action (just transferred from beam)
    P_case_T[:-1] = P_case_T[:-1] - Tpr*sin(theta)
    P_case_C[:-1] = P_case_C[:-1] + Cpr*sin(theta)
    P_case_buck[:-1] = P_case_buck[:-1] + Cpr_pr*sin(theta)
    P_case_TC[:-1] = P_case_TC[:-1] - Tpr*sin(theta) + Cpr*sin(theta)
    P_case_Tbuck[:-1] = P_case_Tbuck[:-1] - Tpr*sin(theta) + Cpr_pr*sin(theta)
    
    import numpy as np
    Pu_col = np.maximum.reduce([P_case_T,
                                P_case_C,
                                P_case_buck,
                                P_case_Tbuck,
                                P_case_TC])
    
    Tu_col = np.minimum.reduce([P_case_T,
                                P_case_C,
                                P_case_buck,
                                P_case_Tbuck,
                                P_case_TC])
    
    # T_des_col = np.sum(Tu_col)
    C_des_col = np.sum(Pu_col)
    k_col = 1.0 
    Lc_col = h_story*k_col
    selected_col, col_compr_list = select_compression_member(col_list, Lc_col, C_des_col)
    
    return(selected_beam, beam_shear_list, selected_col, col_compr_list)

    # if max(C_max) > phi_Pn:
    #     import numpy as np
    #     from scipy.optimize import minimize
    #     x0 = np.array([Ag, rad_gy])
    #     res = minimize(compressive_brace_loss, x0, method='nelder-mead',
    #                    args=(max(C_max), h_story, L_bay))
    #     ans = res.x
        
    #     Ag_req_est = ans[0]
    #     ry_req_est = ans[1]
    #     selected_brace, passed_A_braces = select_member(member_list, 
    #                                                     'A', Ag_req_est)
    #     Ag = selected_brace['A'].iloc[0]
    #     ry = selected_brace['ry'].iloc[0]
    #     test_strength = compressive_strength(Ag, ry, h_story, L_bay)
    #     selected_brace, passed_ry_braces = select_member(passed_A_braces, 
    #                                                      'ry', ry_req_est)
        
    #     Ag = selected_brace['A'].iloc[0]
    #     ry = selected_brace['ry'].iloc[0]
    #     phi_Pn = compressive_strength(Ag, ry, h_story, L_bay)
        

def design_CBF(input_df, db_string='../resource/'):
    
    # ensure everything is in inches, kip/in
    ft = 12.0
    R_y = input_df['RI']
    n_bays = input_df['num_bays']
    L_bay = input_df['L_bay']*ft 
    hsx = input_df['hsx']
    Fx = input_df['Fx']
    # h_col = input_df['h_col']
    h_story = input_df['h_story']*ft
    
    # cases specific to earthquake design
    load_cases = input_df['all_w_cases']
    case_1 = load_cases['1.2D+0.5L+1.0E']/12
    case_2 = load_cases['0.9D-1.0E']/12
    
    # # gravity case for all uses
    # # import numpy as np
    # w_grav = input_df['w_fl']/12
    
    import pandas as pd

    # ASCE 7-22: Story forces
    # currently, design only two bays of braces
    n_braced = round(n_bays/2.25)
    # n_braced = 2
    delxe, Q_per_bay = get_CBF_element_forces(hsx, Fx, R_y, n_braced)
    
    
    
    A_brace, C_max, T_max, del_b = get_brace_demands(Fx, delxe, Q_per_bay, 
                                                     h_story, L_bay,
                                                     case_1, case_2)
    
    # import shapes 
    
    brace_shapes      = pd.read_csv(db_string+'braceShapes.csv',
        index_col=None, header=0)
    sorted_braces     = brace_shapes.sort_values(by=['A'])

    beam_shapes      = pd.read_csv(db_string+'beamShapes.csv',
        index_col=None, header=0)
    sorted_beams     = beam_shapes.sort_values(by=['A'])
    
    col_shapes       = pd.read_csv(db_string+'colShapes.csv',
        index_col=None, header=0)
    sorted_cols      = col_shapes.sort_values(by=['A'])

    # Braces
    # select compact braces that has required compression capacity
    k_brace = 1.0 
    Lc_brace = (h_story**2 + (L_bay/2)**2)**(0.5)*k_brace
    C_brace = max(C_max)
    selected_brace, qualified_braces = select_compression_member(sorted_braces, 
                                                                 Lc_brace, 
                                                                 C_brace)
        
    import numpy as np
    A_min = np.max(A_brace)
    if selected_brace['A'].iloc[0] < A_min:
        selected_brace = select_member(qualified_braces, 'A', A_min)
        
    # beam and column capacity design
    selected_beam, qualified_beams, selected_col, qualified_cols = capacity_CBF_design(
        selected_brace, Q_per_bay, load_cases, 
        h_story, L_bay, n_bays, sorted_beams, sorted_cols)
    
    
    # return only string to keep data management clean
    if isinstance(selected_beam, pd.DataFrame):
        selected_beam = selected_beam.iloc[0]['AISC_Manual_Label']
    if isinstance(selected_col, pd.DataFrame):
        selected_col = selected_col.iloc[0]['AISC_Manual_Label']
    if isinstance(selected_brace, pd.DataFrame):
        selected_brace = selected_brace.iloc[0]['AISC_Manual_Label']
    
    return(selected_brace, selected_beam, selected_col)
    