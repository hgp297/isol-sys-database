############################################################################
#               Bearing design algorithm

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) 

############################################################################

def design_LRB(Tm, zeta, W_tot):

    inch = 1.0
    ft = 12.0*inch
    g  = 386.4
    pi = 3.14159

    # minimum required lead core diameter
    n_lead = 10
    psi_lead = 1.0

    # assumes 1.3 ksi shear yield stress of lead

    dL_min = (4*n_lead*psi_lead*Qd)/(pi*(n_lead - 1)*1.3)



