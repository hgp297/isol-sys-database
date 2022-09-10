############################################################################
#               Superstructure design algorithm

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2022

# Description:  

# Open issues:  (1) checks not done yet
#               (2) design shims

############################################################################

kip = 1.0
ft = 12.0
ksi = 1.0

D_m = 22.12
K_e = 40.073
zeta_e = 0.15
W_tot = 3530
n_frames = 2
W_s = 2227.5
R_y = 1.0

T_fb = 1.0

n_bays = 3
L_bay = 30*ft


def get_properties(shape):
    if (len(shape) == 0):
        raise IndexError('No shape fits the requirements.')
    Zx      = float(shape.iloc[0]['Zx'])
    Ag      = float(shape.iloc[0]['A'])
    Ix      = float(shape.iloc[0]['Ix'])
    bf      = float(shape.iloc[0]['bf'])
    tf      = float(shape.iloc[0]['tf'])
    return(Ag, bf, tf, Ix, Zx)

def calculate_strength(shape, VGrav):
    Zx      = float(shape.iloc[0]['Zx'])

    Mn          = Zx*Fy
    Mpr         = Mn*Ry*Cpr
    Vpr         = 2*Mpr/L_bay
    beamVGrav   = 2*VGrav
    return(Mn, Mpr, Vpr, beamVGrav)

def get_story_forces(D_m, K_e, W_tot, W, n_frames, zeta_e, R_y, T_fb):
    wx      = np.array([810.0*kip, 810.0*kip, 607.5*kip])           # Floor seismic weights
    hx      = np.array([13.0*ft, 26.0*ft, 39.0*ft])                 # Floor elevations
    hCol    = np.array([13.0*ft, 13.0*ft, 7.5*ft])                  # Column moment arm heights
    hsx     = np.array([13.0*ft, 13.0*ft, 13.0*ft])                 # Column heights
    wLoad   = np.array([2.72*kip/ft, 2.72*kip/ft, 1.94*kip/ft])     # Floor line loads

    Vb      = (D_m * K_e)/n_frames
    Vst     = (Vb*(W_s/W_tot)**(1 - 2.5*zeta_e))
    Vs      = (Vst/R_y)
    F1      = (Vb - Vst)/R_y

    k       = 14*zeta_e*T_fb

    hxk     = hx**k

    CvNum   = wx*hxk
    CvDen   = np.sum(CvNum)

    Cvx     = CvNum/CvDen

    Fx      = Cvx*Vs
    
    return(wx, hx, hCol, hsx, wLoad, Fx)

def get_MRF_element_forces(wx, hsx, Fx, R_y, n_bays):
    nFloor      = len(wx)
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

def get_required_modulus(q, hCol, hsx, delxe, L_bay, wLoad, E, Fy, Fu):
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
    Fu              = 65*ksi
    MGrav           = wLoad*L_bay**2/12
    VGravStory      = max(wLoad*L_bay/2)
    VGravRoof       = wLoad[-1]*L_bay/2
    MEq             = q*hCol/2
    Mu              = MEq + MGrav
    Zb              = Mu/(0.9*Fy)

    return(Ib, Ic, Iz)

# def design_MF(D_m, K_e, W_tot, W_s, n_frames, zeta_e, R_y, T_fb,
#               n_bays, L_bay):
    

import numpy as np
import pandas as pd

############################################################################
#              ASCE 7-16: Story forces
############################################################################

wx, hx, hCol, hsx, wLoad, Fx = get_story_forces(D_m, K_e, W_tot, W_s, n_frames, 
                                                zeta_e, R_y, T_fb)

############################################################################
#              ASCE 7-16: Steel moment frame design
############################################################################

delxe, q = get_MRF_element_forces(wx, hsx, Fx, R_y, n_bays)

Fy = 50.0
Fu = 65.0

Ib, Ic, Iz = get_required_modulus(q, hCol, hsx, delxe, L_bay, wLoad, E, Fy, Fu)

############################################################################
#              ASCE 7-16: Import shapes
############################################################################

IBeamReq        = Ib.max()
IColReq         = Ic.max()
ZBeamReq        = Zb.max()

IBeamRoofReq    = Ib[-1]
ZBeamRoofReq    = Zb[-1]

beamShapes      = pd.read_csv('../inputs/beamShapes.csv',
    index_col=None, header=0)
sortedBeams     = beamShapes.sort_values(by=['Ix'])

colShapes       = pd.read_csv('../inputs/colShapes.csv',
    index_col=None, header=0)
sortedCols      = colShapes.sort_values(by=['Ix'])

############################################################################
#              ASCE 7-16: Capacity design
############################################################################

############################################################################
#              Floor beams

# select beams that qualify Ix requirement
qualifiedIx     = sortedBeams[sortedBeams['Ix'] > IBeamReq] # eliminate all shapes with insufficient Ix
sortedWeight    = qualifiedIx.sort_values(by=['W'])         # select lightest from list     
selectedBeam    = sortedWeight.iloc[:1]

############################################################################
#              Beam checks

# Zx check
(beamAg, beambf, beamtf, beamIx, beamZx) = get_properties(selectedBeam)

if(beamZx < ZBeamReq):
    # print("The beam does not meet Zx requirement. Reselecting...")
    qualifiedZx         = qualifiedIx[qualifiedIx['Zx'] > ZBeamReq]  # narrow list further down to only sufficient Zx
    sortedWeight        = qualifiedZx.sort_values(by=['W'])          # select lightest from list
    selectedBeam        = sortedWeight.iloc[:1]
    (beamAg, beambf, beamtf, beamIx, beamZx) = get_properties(selectedBeam)

Ry              = 1.1
Cpr             = (Fy + Fu)/(2*Fy)

(beamMn, beamMpr, beamVpr, beamVGrav)   = calculate_strength(selectedBeam,
    VGravStory)

# PH location check
phVGrav         = wLoad[:-1]*L_bay/2
phVBeam         = 2*beamMpr/(0.9*L_bay)  # 0.9L_bay for plastic hinge length
phLocation      = phVBeam > phVGrav

if False in phLocation:
    # print('Detected plastic hinge away from ends. Reselecting...')
    ZBeamPHReq          = max(wLoad*L_bay**2/(4*Fy*Ry*Cpr))
    qualifiedZx         = qualifiedIx[qualifiedIx['Zx'] > ZBeamPHReq] # narrow list further down to only sufficient Zx
    sortedWeight        = qualifiedZx.sort_values(by=['W'])           # select lightest from list
    selectedBeam        = sortedWeight.iloc[:1]
    (beamAg, beambf, beamtf, beamIx, beamZx)    = get_properties(selectedBeam)
    (beamMn, beamMpr, beamVpr, beamVGrav) = calculate_strength(selectedBeam,
        VGravStory)

# beam shear
beamAweb        = beamAg - 2*(beamtf*beambf)
beamVn          = 0.9*beamAweb*0.6*Fy

beamShearFail   = beamVn < beamVpr

while beamShearFail:
    # print('Beam shear check failed. Reselecting...')
    AgReq               = 2*beamVpr/(0.9*0.6*Fy)                                # Assume web is half of gross area

    if 'qualifiedZx' in dir():
        qualifiedAg         = qualifiedZx[qualifiedZx['A'] > AgReq]             # narrow Zx list down to sufficient Ag
    else:
        qualifiedAg         = qualifiedIx[qualifiedIx['A'] > AgReq]             # if Zx check wasn't done previously, use Ix list

    sortedWeight        = qualifiedAg.sort_values(by=['W'])                     # select lightest from list
    selectedBeam        = sortedWeight.iloc[:1]

    # recheck beam shear
    (beamAg, beambf, beamtf, beamIx, beamZx) = get_properties(selectedBeam)
    
    beamAweb        = beamAg - 2*(beamtf*beambf)
    beamVn          = 0.9*beamAweb*0.6*Fy

    (beamMn, beamMpr, beamVpr, beamVGrav) = calculate_strength(selectedBeam,
        VGravStory)
    
    beamShearFail   = beamVn < beamVpr

############################################################################
#              Roof beams

# select beams that qualify Ix requirement
qualifiedIx         = sortedBeams[sortedBeams['Ix'] > IBeamRoofReq]         # eliminate all shapes with insufficient Ix
sortedWeight        = qualifiedIx.sort_values(by=['W'])                     # select lightest from list     
selectedRoofBeam    = sortedWeight.iloc[:1]

############################################################################
#              Roof beam checks

# Zx check
(roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) = get_properties(selectedRoofBeam)

if(roofBeamZx < ZBeamRoofReq):
    # print("The beam does not meet Zx requirement. Reselecting...")
    qualifiedZx         = qualifiedIx[qualifiedIx['Zx'] > ZBeamRoofReq]         # narrow list further down to only sufficient Zx
    sortedWeight        = qualifiedZx.sort_values(by=['W'])                     # select lightest from list
    selectedRoofBeam        = sortedWeight.iloc[:1]
    (roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) = get_properties(selectedRoofBeam)

(roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav)   = calculate_strength(selectedRoofBeam,
    VGravRoof)

# PH location check
phVGravRoof         = wLoad[-1]*L_bay/2
phVBeamRoof         = 2*roofBeamMpr/(0.9*L_bay)                                  # 0.9L_bay for plastic hinge length
phLocationRoof      = phVBeamRoof > phVGravRoof

if not phLocationRoof:
    # print('Detected plastic hinge away from ends. Reselecting...')
    ZBeamPHReq          = wLoad[-1]*L_bay**2/(4*Fy*Ry*Cpr)
    qualifiedZx         = qualifiedIx[qualifiedIx['Zx'] > ZBeamPHReq]               # narrow list further down to only sufficient Zx
    sortedWeight        = qualifiedZx.sort_values(by=['W'])                         # select lightest from list
    selectedRoofBeam        = sortedWeight.iloc[:1]
    (roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx)    = get_properties(selectedRoofBeam)
    (roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav)           = calculate_strength(selectedRoofBeam, VGravRoof)

# roof beam shear check
roofBeamAweb        = roofBeamAg - 2*(roofBeamtf*roofBeambf)
roofBeamVn          = 0.9*roofBeamAweb*0.6*Fy

roofBeamShearFail   = roofBeamVn < roofBeamVpr

while roofBeamShearFail:
    # print('Beam shear check failed. Reselecting...')
    roofAgReq               = 2*roofBeamVpr/(0.9*0.6*Fy)                        # Assume web is half of gross area

    if 'qualifiedZx' in dir():
        qualifiedAg         = qualifiedZx[qualifiedZx['A'] > roofAgReq]         # narrow Zx list down to sufficient Ag
    else:
        qualifiedAg         = qualifiedIx[qualifiedIx['A'] > roofAgReq]         # if Zx check wasn't done previously, use Ix list

    sortedWeight        = qualifiedAg.sort_values(by=['W'])                     # select lightest from list
    selectedRoofBeam        = sortedWeight.iloc[:1]

    # recheck beam shear
    (roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx)    = get_properties(selectedRoofBeam)
    (roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav) = calculate_strength(selectedRoofBeam,
        VGravRoof)
    
    roofBeamAweb        = roofBeamAg - 2*(roofBeamtf*roofBeambf)
    roofBeamVn          = 0.9*roofBeamAweb*0.6*Fy
    
    roofBeamShearFail   = roofBeamVn < roofBeamVpr

############################################################################
#              Columns

# SCWB design

Pr              = np.empty(nFloor)
Pr[-1]          = beamVpr + roofBeamVGrav

for i in range(nFloor-2, -1, -1):
    Pr[i] = beamVGrav + beamVpr + Pr[i + 1]

# guess: use columns that has similar Ix to beam
qualifiedIx     = sortedCols[sortedCols['Ix'] > IBeamReq]                               # eliminate all shapes with insufficient Ix
selectedCol     = qualifiedIx.iloc[(qualifiedIx['Ix'] - IBeamReq).abs().argsort()[:1]]  # select the first few that qualifies Ix

(colAg, colbf, coltf, colIx, colZx) = get_properties(selectedCol)

colMpr          = colZx*(Fy - Pr/colAg)

# find required Zx for SCWB to be true
scwbZReq        = np.max(beamMpr/(Fy - Pr[:-1]/colAg))

# select column based on SCWB
qualifiedIx     = sortedCols[sortedCols['Ix'] > IColReq]            # eliminate all shapes with insufficient Ix
qualifiedZx     = qualifiedIx[qualifiedIx['Zx'] > scwbZReq]         # eliminate all shapes with insufficient Z for SCWB
sortedWeight    = qualifiedZx.sort_values(by=['W'])                 # select lightest from list
selectedCol     = sortedWeight.iloc[:1]

(colAg, colbf, coltf, colIx, colZx) = get_properties(selectedCol)

colMpr          = colZx*(Fy - Pr/colAg)

# check final SCWB
ratio           = np.empty(nFloor-1)
for i in range(nFloor-2, -1, -1):
    ratio[i] = (colMpr[i+1] + colMpr[i])/(2*beamMpr)
    if (ratio[i] < 1.0):
        print('SCWB check failed at floor ' + str(nFloor+1) + ".")

# column shear
colAweb         = colAg - 2*(coltf*colbf)
colVn           = 0.9*colAweb*0.6*Fy
colVpr          = max(colMpr/hCol)

colShearFail    = colVn < colVpr

while colShearFail:
    # print('Column shear check failed. Reselecting...')
    AgReq       = 2*colVpr/(0.9*0.6*Fy)                                     # Assume web is half of gross area

    qualifiedAg         = qualifiedZx[qualifiedZx['A'] > AgReq]             # narrow Zx list down to sufficient Ag

    sortedWeight        = qualifiedAg.sort_values(by=['W'])                 # select lightest from list
    selectedCol         = sortedWeight.iloc[:1]

    # recheck column shear
    (colAg, colbf, coltf, colIx, colZx) = get_properties(selectedCol)
    
    colAweb         = colAg - 2*(coltf*colbf)
    colVn           = 0.9*colAweb*0.6*Fy

    colMpr          = colZx*(Fy - Pr/colAg)
    colVpr          = max(colMpr/hCol)
    
    colShearFail    = colVn < colVpr

# if __name__ == '__main__':
#     design_LRB()