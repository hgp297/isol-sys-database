############################################################################
#               Temporary RC shear wall design script

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) forces need to adjust Ry for structural system

############################################################################

from design_superstructure import get_story_forces
import numpy as np

def get_bar_area(bar_num):
    return {
        3 : 0.11,
        4 : 0.2,
        5 : 0.31, 
        6 : 0.44,
        7 : 0.6,
        8 : 0.79,
        9 : 1.0,
        10 : 1.27
    }.get(bar_num)

def get_section_forces(c, d, f_y, As, f_prime_c, alpha, beta_1, b):
    eps_s = np.zeros(3) # Similar to A_s and d
    f_s = np.zeros(3)
    S_s = np.zeros(3)

    # Initialize strains
    eps_cu = -0.003
    E_s = 29000000.0
    eps_y = f_y/E_s

    for i in range(len(d)):
            
            eps_s[i] = eps_cu*(c - d[i])/c # strains
            
            if eps_s[i] > eps_y:             # check for yielding
                f_s[i] = f_y
            elif eps_s[i] < -eps_y:
                f_s[i] = -f_y
            else:
                f_s[i] = eps_s[i] * E_s    # stresses
            
            if f_s[i] > 0:
                S_s[i] = As[i]*f_s[i]      # forces
            else:
                S_s[i] = As[i]*(-f_s[i] - alpha*f_prime_c)
                S_s[i] = -S_s[i]
    
    # concrete compression
    Cc = alpha*f_prime_c*beta_1*c*b
    
    return(S_s, eps_s, Cc)
    
# Function that outputs sum of axial force with N.A as input
def check_axial_neutral_axis(c, P, d, f_y, As, f_prime_c, alpha, beta_1, b):
    
    # get steel forces
    S_s, eps_s, Cc = get_section_forces(c, d, f_y, As, f_prime_c, alpha, beta_1, b)

    # assumes P is loaded at the center of the section
    # Here, P is added because a negative compressive P is subtracted 
    # (P is positive)
    err = sum(S_s) - Cc + P    
                                
    # returning err^2 to make function convex and minimize at 0
    return(err**2)

# Function takes a P_n and finds corresponding c and M_n, hardcoded for 3 regions
def PM_section_analyze(P, d, f_y, As, f_prime_c, alpha, beta_1, b, h):

    # Initialize strains
    E_s = 29000000.0
    eps_y = f_y/E_s

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(check_axial_neutral_axis, args=(P, d, f_y, As,
        f_prime_c, alpha, beta_1, b), bounds=(0.01, h), method='bounded')
    
    c = res.x
    
    S_s, eps_s, Cc = get_section_forces(c, d, f_y, As, f_prime_c, alpha, beta_1, b)

    phi = 0.65 + (eps_s[-1] - eps_y)/(0.005 - eps_y)*0.250

    phi = max(phi, 0.65)
    phi = min(phi, 0.9)

    M = 0.0

    for j in range(len(d)):
        M = M + abs(S_s[j])*abs(d[j] - h/2)

    M = M + abs(Cc)*(h/2 - (beta_1*c)/2)

    M_r = M*phi

    return(M_r, c)


ft = 12.0
ksi = 1000.0
psi = 1.0
kip = 1000.0

D_m = 22.12
K_e = 40.073
zeta_e = 0.15
W_tot = 3530
n_frames = 2
W_s = 2227.5
R_y = 2.0

wx, hx, hCol, hsx, wLoad, Fx, V_s = get_story_forces(D_m, K_e, W_tot, W_s, 
                                                    n_frames, zeta_e, 
                                                    R_y, 'SW')
# unit conversion to pounds/in
wx = wx*kip
wLoad = wLoad*kip/ft
#wLoad = np.array([0.93, 0.93, 0.71])*kip/ft
print(Fx)
Fx = Fx*kip
V_u = V_s*kip

L_bay = 22.5*ft
l_w = L_bay


from math import ceil

h_w = np.sum(hsx)

is_squat = (h_w <= 2*l_w)

# squat wall design
l_be = 0.2*l_w

f_c = 4000.0*psi
f_yc = 60000.0*psi
f_ys = 60000.0*psi

# minimum reinforcement?
# bar spacing
# design spec: try changing s_t as a function of b_w or l_w
# spacings and ratios: ACI 318-19 11.6.2, 11.7.3.1, 2.2

# transverse
bar_t = 4
b_w = 10.0
s_t = 16.0
rho_t = max((2*get_bar_area(bar_t))/(s_t*b_w), 0.0025)
print('Transv reinf:', rho_t)

# longitudinal
bar_l = 5
s_l = 18.0
rho_l = max(2*get_bar_area(bar_l)/(s_l*b_w), 0.0025)
rho_l_min = max(0.0025+0.5*(2.5-b_w/l_w)*(rho_t - 0.0025), 0.0025)

rho_l = max(rho_l, rho_l_min)
print('Longi reinf:', rho_l)


# required area
# all assuming normalweight concrete
# A_cv = V_u / (4*(f_c**(0.5)))
# b_w = ceil(A_cv / l_w)



# nominal shear strength and transverse steel
# 
if (h_w/l_w <= 1.5):
    alpha_c = 3.0 * psi
elif (h_w/l_w >= 2.0):
    alpha_c = 2.0 * psi
else:
    alpha_c = np.interp(h_w/l_w, [1.5, 2.5], [3.0, 2.0])

phi = 0.6 # specific to squat wall shear
rho_t_v = 1/f_ys * (V_u/(phi*l_w*b_w) - alpha_c * (f_c**(0.5)))
rho_t = max(rho_t_v, 0.0025)

bar_t = 4
s_t = min(2*get_bar_area(bar_t)/(rho_t*b_w), 18.0)

V_n = (alpha_c * (f_c**(0.5)) + rho_t*f_ys)*l_w*b_w

# max shear strength check
shear_upper_combined = 8*b_w*l_w*(f_c**(0.5)) < n_frames*V_n
shear_upper_indiv = 10*b_w*l_w*(f_c**(0.5)) < V_n
if shear_upper_combined:
    print('Upper limit shear strength exceeded by combined walls')
elif shear_upper_indiv:
    print('Upper limit shear strength exceeded by single wall')

two_curtains = V_u > 2*b_w*l_w*(f_c**(0.5))

# P-M design, prelim proportioning
# assuming rho_l minimum (fixing T_s1, then solving for T_s2 to achieve M_n)
# T_s1 is main section, T_s2 is boundary element
# loads already account combos and LL reduction

M_u = np.sum(Fx*hx)
M_n = M_u / 0.90 # phi for tension controlled
P_u = np.sum(wLoad)*(90*ft)
beta_1 = 0.85 # ACI 318-19, T 22.2.2.4.3

rho_l = max(0.0025, rho_t)

A_s1 = (0.6*l_w)*b_w*rho_l
T_s1 = A_s1*f_ys
T_s2_req = 1/(0.8*l_w)*(M_n - P_u*(0.4*l_w) - T_s1*(0.4*l_w))
A_s2_req = T_s2_req/f_ys

bar_be = 5
num_be = ceil(A_s2_req/get_bar_area(bar_be))
A_s2 = num_be*get_bar_area(bar_be)
s_be_temp = (0.2*l_w)/(num_be/2)
s_be = min(s_be_temp, 18.0)

# now use section analysis to check P-M strength. Compare against M_u, P_u
# use matlab code as template
d = np.array([0.1*l_w, 0.5*l_w, 0.9*l_w])
A_s = np.array([A_s2, A_s1, A_s2])
beta_1 = 0.85 - (0.05)*(f_c - 4000.0)/1000.0
M_r, c = PM_section_analyze(P_u, d, f_ys, A_s, f_yc, 0.85, beta_1, b_w, l_w)


# 