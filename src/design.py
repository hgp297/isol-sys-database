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
    num_bearings = (n_bays+1)**2
    from numpy import ceil
    num_lb = ceil(0.4*num_bearings / 2.)*2
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

# rho_k is not important in rubber bearings
# how to divide force among bearings?
def design_LRB(param_df, t_r=10.0):
    
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
    f_y_Pb = 1.5 # ksi
    A_Pb = (Q_L/f_y_Pb) / N_lb # in^2
    d_Pb = (4*A_Pb/pi)**(0.5)
    
    # 60 psi rubber
    # select thickness
    
    G_r = 0.060 # ksi
    A_r = k_2 * t_r / (G_r * N_lb)
    d_r = (4*A_r/pi)**(0.5)
    
    # yielding force
    k_1 = rho_k * k_2
    D_y = Q_L/(k_1 - k_2)
    
    # final values
    k_e = (Q_L + k_2*D_m)/D_m
    T_e = 2*pi*(W_tot/(g*k_e))**0.5
    W_e = 4*Q_L*(D_m - D_y)
    zeta_E = W_e/(2*pi*k_e*D_m**2)
    
    flag = 0
    
    # shape factor
    t = t_r/12
    S = (d_r/2)/(2*t)
    
    # assume small strain G is 75% larger
    G_ss = 1.75*G_r
    # incompressibility
    K_inc = 290 # ksi
    E_c = (6*G_ss*S**2*K_inc)/(6*G_ss*S**2 + K_inc)
    
    # assume shim is half inch less than rubber diameter
    I = pi/4 *((d_r - 0.5)/2)**4
    A_s = pi/4 * (d_r - 0.5)**2
    
    # buckling check
    P_crit = pi/t_r * ((E_c * I/3)*G_r*A_s)**(0.5)
    P_estimate = W_tot/(N_lb + N_sl)
    
    if P_estimate/P_crit > 1.0:
        flag = 1
    
    # # shear check
    # gamma_c = P_crit / (G_r * A_r * S)
    # limit_aashto = 0.5*7
    # gamma_s_limit = limit_aashto - gamma_c
    
    # print('Effective period:', T_e)
    # print('Effective damping:', zeta_E)
    
    # normalize stiffness by weight
    k_e_norm = k_e/W_tot
    
    return(d_r, d_Pb, T_e, k_e_norm, zeta_E, D_m, flag)
    

# def design_LR(T_m, zeta_m, W_tot, r_init, S_1, t_r, N_rb, N_Pb):

#     from numpy import interp
#     inch = 1.0
#     kip = 1.0
#     ft = 12.0*inch
#     g  = 386.4
#     pi = 3.14159
    
#     # from ASCE Ch. 17, get damping multiplier
#     zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
#     BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    
#     B_m      = interp(zeta_m, zetaRef, BmRef)
    
#     # design displacement
#     D_m = g*S_1*T_m/(4*pi**2*B_m)
    
#     # stiffness
#     K_eff = (2*pi/T_m)**2 * W_tot/g # k/in
    
#     # EDC
#     W_D = 2*pi*K_eff*D_m**2*zeta_m
    
#     # first guess
#     Q_d = W_D/(4*D_m) # kip
    
#     err = 1.0
#     tol = 0.001
    
#     # converge Q_d, K_2, D_y, r_init = K1/K2
#     while err > tol:
#         K_2 = K_eff - Q_d/D_m
#         D_y = Q_d/((r_init-1)*K_2)
#         Q_d_new = pi*K_eff*D_m**2*zeta_m/(2*(D_m-D_y))
#         #Q_d_new = W_D/(4*D_m)
    
#         err = abs(Q_d_new - Q_d)/Q_d
    
#         Q_d = Q_d_new
    
#     # required area of lead per bearing
#     f_y_Pb = 1.5 # ksi
#     A_Pb = (Q_d/f_y_Pb) / N_Pb # in^2
#     d_Pb = (4*A_Pb/pi)**(0.5)
    
#     # yielding force
#     K_1 = r_init * K_2
#     F_y = K_1*D_y
    
#     # rubber stiffness per bearing
#     K_r = (K_eff - Q_d / D_m)/ N_rb
    
#     # 60 psi rubber
#     # select thickness
    
#     G_r = 0.060 * kip # ksi
#     A_r = K_r * t_r / G_r
#     d_r = (4*A_r/pi)**(0.5)
    
#     # final values
#     K_e = N_rb * K_r + Q_d/D_m
#     W_e = 4*Q_d*(D_m - D_y)
#     zeta_e = W_e/(2*pi*K_e*D_m**2)
    
#     # check slenderness
#     # check lead vs main bearing ratio
    
#     # buckling check
    
#     # shape factor
#     t = t_r/12
#     S = (d_r/2)/(2*t)
    
#     # assume small strain G is 75% larger
#     G_ss = 1.75*G_r
#     # incompressibility
#     K_inc = 290 # ksi
#     E_c = (6*G_ss*S**2*K_inc)/(6*G_ss*S**2 + K_inc)
    
#     # assume shim is half inch less than rubber diameter
#     I = pi/4 *((d_r - 0.5)/2)**4
#     A_s = pi/4 * (d_r - 0.5)**2
    
#     P_crit = pi/t_r * ((E_c * I/3)*G_r*A_s)**(0.5)
    
#     # shear check
#     gamma_c = P_crit / (G_r * A_r * S)
#     limit_aashto = 0.5*7
#     gamma_s_limit = limit_aashto - gamma_c
    
#     # slenderness check
#     slen_ratio = d_r / d_Pb
    
#     return(d_Pb, d_r)

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
    
    k_M = (2*pi/T_m)**2 * (1/g)
    
    u_y = 0.01
    
    k_0 = mu_1/u_y
    
    # from Q and D_m
    k_2 = (k_M*D_m - Q)/D_m
    R_2 = 1/(2*k_2)
    
    # from rho_k
    u_a = Q/(k_2*(rho_k-1))
    k_a = rho_k*k_2
    mu_2 = u_a*k_a
    R_1 = u_a/(2*(mu_2-mu_1))
    
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
    
    # W_m = zeta_M*(2*pi*k_M*D_m**2)
    
    u_y = 0.01
    
    k_0 = mu_1/u_y
    
    # from Q and D_m
    k_2 = (k_M*D_m - Q)/D_m
    R_2 = 1/(2*k_2)
    
    # from rho_k
    u_a = Q/(k_2*(rho_k-1))
    k_a = rho_k*k_2
    mu_2 = u_a*k_a
    R_1 = u_a/(2*(mu_2-mu_1))
    
    # effective design values
    a = 1/(2*R_1)
    b = 1/(2*R_2)
    k_e = (mu_2 + b*(D_m - u_a))/D_m
    W_e = 4*(mu_2 - b*u_a)*D_m - 4*(a-b)*u_a**2 - 4*(k_0 -a)*u_y**2
    zeta_E   = W_e/(2*pi*k_e*D_m**2)
    T_e = 2*pi*(1/(g*k_e))**0.5
    
    # u_a = (4*Q*D_m - W_m)/(4*(Q - mu_1))
    # mu_2 = Q + u_a/(2*R_2)
    # R_1 = u_a/(2*(mu_2 - mu_1))
    
    # u_a = D_m - W_m/(4*Q)
    # mu_2 = Q + u_a/(2*R_2)
    
    # aa = (-2/R_1)
    # bb = 4*(1/(2*R_1) - k_2)*D_m
    # cc = 4*mu_1 - W_m
    
    # up = (-bb + (bb**2 - 4*aa*cc) / (2*aa))
    # un = (-bb - (bb**2 - 4*aa*cc) / (2*aa))
    
    # u_a = max(up, un)
    
    # mu_2 = mu_1 + 1/(2*R_1)*u_a
    
    # # from rho_k
    # k_a = rho_k * k_2
    # u_a = 2*mu_1*R_1/(2*k_a*R_1 - 1)
    # mu_2 = u_a * k_a
    
    # print('Effective period:', T_e)
    # print('Effective damping:', zeta_E)
    
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
#              ASCE 7-16: Capacity design
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
    w_load = input_df['w_fl']/ft
    
    import pandas as pd

    # ASCE 7-16: Story forces

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
    
if __name__ == '__main__':
    print('====== sample TFP design ======')
    T_m = 4.5
    S_1 = 1.017
    Q = 0.06
    rho_k = 7.0
    mu_1, mu_2, R_1, R_2, T_e, k_e, zeta_E, D_m = design_TFP(T_m, S_1, Q, rho_k)
    print('Friction coefficients:', mu_1, mu_2)
    print('Radii of curvature:', R_1, R_2)
    
    print('====== sample LRB design ======')
    T_m = 2.5
    S_1 = 1.017
    Q = 0.06
    rho_k = 21.0
    n_bay = 3
    W_sample = 3000.0
    d_r, d_Pb, T_e, k_e, zeta_E, D_m, checks = design_LRB(T_m, S_1, Q, rho_k,
                                                     n_bay, W_sample,t_r=10.0)
    print('Bearing diameter:', d_r)
    print('Lead core diameter:', d_Pb)
    
    # design_LRB()