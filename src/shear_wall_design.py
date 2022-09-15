############################################################################
#               Temporary RC shear wall design script

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) forces need to adjust Ry for structural system

############################################################################

from design_superstructure import get_story_forces

ft = 12.0
ksi = 1.0

D_m = 22.12
K_e = 40.073
zeta_e = 0.15
W_tot = 3530
n_frames = 2
W_s = 2227.5
R_y = 1.0

wx, hx, hCol, hsx, wLoad, Fx = get_story_forces(D_m, K_e, W_tot, W_s, 
                                                    n_frames, zeta_e, 
                                                    R_y, 'SW')

L_bay = 22.5*ft
l_w = L_bay

import numpy as np

h_w = np.sum(hsx)

is_squat = (h_w <= 2*l_w)

# squat wall design
l_be = 0.2*l_w

b_w = 10.0

f_c = 4.0*ksi
f_y = 60.0*ksi
