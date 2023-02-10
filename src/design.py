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
def design_LRB(T_m, S_1, Q, rho_k, n_bays, W_tot, t_r=10.0):
    
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
    
    return(d_r, d_Pb, T_e, k_e, zeta_E, flag)
    

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
    
def design_TFP(T_m, S_1, Q, rho_k):
    
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
    
    return(mu_1, mu_2, R_1, R_2, T_e, k_e, zeta_E)
    
if __name__ == '__main__':
    print('====== sample TFP design ======')
    T_m = 4.5
    S_1 = 1.017
    Q = 0.06
    rho_k = 7.0
    mu_1, mu_2, R_1, R_2, T_e, k_e, zeta_E = design_TFP(T_m, S_1, Q, rho_k)
    print('Friction coefficients:', mu_1, mu_2)
    print('Radii of curvature:', R_1, R_2)
    
    print('====== sample LRB design ======')
    T_m = 2.5
    S_1 = 1.017
    Q = 0.06
    rho_k = 21.0
    n_bay = 3
    W_sample = 3000.0
    d_r, d_Pb, T_e, k_e, zeta_E, checks = design_LRB(T_m, S_1, Q, rho_k,
                                                     n_bay, W_sample,t_r=10.0)
    print('Bearing diameter:', d_r)
    print('Lead core diameter:', d_Pb)
    
    # design_LRB()