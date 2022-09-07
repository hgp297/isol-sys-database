############################################################################
#               Bearing design algorithm

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) 

############################################################################

T_m = 3.0
zeta_m = 0.15
W_tot = 3530
r_init = 10
S_1 = 1.017
N_rb = 12
N_Pb = 4
t_r = 10.0

# def design_LRB(T_m, zeta_m, W_tot, r_init, S_1, t_r, N_rb, N_Pb):

from numpy import interp
inch = 1.0
kip = 1.0
ft = 12.0*inch
g  = 386.4
pi = 3.14159

# from ASCE Ch. 17, get damping multiplier
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

B_m      = interp(zeta_m, zetaRef, BmRef)

# design displacement
D_m = g*S_1*T_m/(4*pi**2*B_m)

# stiffness
K_eff = (2*pi/T_m)**2 * W_tot/g # k/in

# EDC
W_D = 2*pi*K_eff*D_m**2*zeta_m

# first guess
Q_d = W_D/(4*D_m) # kip

err = 1.0
tol = 0.001

# converge Q_d, K_2, D_y
while err > tol:
    K_2 = K_eff - Q_d/D_m
    D_y = Q_d/((r_init-1)*K_2)
    Q_d_new = W_D/(4*D_m)

    err = abs(Q_d_new - Q_d)/Q_d

    Q_d = Q_d_new

# required area of lead per bearing
f_y_Pb = 1.5 # ksi
A_Pb = (Q_d/f_y_Pb) / N_Pb # in^2
d_Pb = (4*A_Pb/pi)**(0.5)

# rubber stiffness per bearing
K_r = (K_eff - Q_d / D_m)/ N_rb

# 60 psi rubber
# select thickness

G_r = 0.060 * kip # ksi
A_r = K_r * t_r / G_r
d_r = (4*A_r/pi)**(0.5)

# final values
K_e = N_rb * K_r + Q_d/D_m
W_e = 4*Q_d*(D_m - D_y)
zeta_e = W_e/(2*pi*K_e*D_m**2)

# check slenderness
# check lead vs main bearing ratio

# buckling check

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

P_crit = pi/t_r * ((E_c * I/3)*G_r*A_s)**(0.5)

# shear check
gamma_c = P_crit / (G_r * A_r * S)
limit_aashto = 0.5*7
gamma_s_limit = limit_aashto - gamma_c

# slenderness check
slen_ratio = d_r / d_Pb



# if __name__ == '__main__':
#     design_LRB()