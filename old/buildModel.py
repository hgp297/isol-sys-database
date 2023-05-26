############################################################################
#               Build model (revamped with modified IK)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: April 2022

# Description:  Script models designed structure (superStructDesign.py) in OpenSeesPy.
#               Returns loading if function called
#               Adapted for DoE

# Open issues:  (1) 

# following example on http://opensees.berkeley.edu/wiki/index.php/Elastic_Frame_Example
# nonlinear model example from https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_5._2D_Frame,_3-story_3-bay,_Reinforced-Concrete_Section_%26_Steel_W-Section#Elastic_Element
# using Ex5.Frame2D.build.InelasticSection.tcl
############################################################################


# import OpenSees and libraries
import openseespy.opensees as ops

############################################################################
#              Utilities
############################################################################

# get shape properties
def getProperties(shape):
    Ag      = float(shape.iloc[0]['A'])
    Ix      = float(shape.iloc[0]['Ix'])
    Iy      = float(shape.iloc[0]['Iy'])
    Zx      = float(shape.iloc[0]['Zx'])
    Sx      = float(shape.iloc[0]['Sx'])
    d       = float(shape.iloc[0]['d'])
    bf      = float(shape.iloc[0]['bf'])
    tf      = float(shape.iloc[0]['tf'])
    tw      = float(shape.iloc[0]['tw'])
    return(Ag, Ix, Iy, Zx, Sx, d, bf, tf, tw)

# create or destroy fixed base, for eigenvalue analysis
def refix(nodeTag, action):
    for j in range(1,7):
        ops.remove('sp', nodeTag, j)
    if(action == "fix"):
        ops.fix(nodeTag,  1, 1, 1, 1, 1, 1)
    if(action == "unfix"):
        ops.fix(nodeTag,  0, 1, 0, 1, 0, 1)

# returns load to analysis script
def giveLoads():
    return(w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3)

# add superstructure damping (dependent on eigenvalue anly)
def provideSuperDamping(regTag, w2, zetai=0.05, zetaj=0.05, modes=[1,3]):
    # Pick your modes and damping ratios
    wi = w2[modes[0]-1]**0.5; zetai = 0.05 # 5% in mode 1
    wj = w2[modes[1]-1]**0.5; zetaj = 0.02 # 2% in mode 3
    
    import numpy as np
    
    A = np.array([[1/wi, wi],[1/wj, wj]])
    b = np.array([zetai,zetaj])
    
    x = np.linalg.solve(A,2*b)
    
    # alphaM      = 0.0
    betaK       = 0.0
    betaKInit   = 0.0
    # a1          = 2*zetaTarget/omega1
    ops.region(regTag, '-ele',
        111, 112, 113, 114,
        121, 122, 123, 124,
        131, 132, 133, 134,
        221, 222, 223,
        231, 232, 233,
        241, 242, 243,
        5118, 5128, 5138, 5148,
        5216, 5226, 5236, 5246,
        5218, 5228, 5238, 5248,
        5316, 5326, 5336, 5346,
        5318, 5328, 5338, 5348,
        5416, 5426, 5436, 5446,
        5219, 5229, 5239,
        5227, 5237, 5247,
        5319, 5329, 5339,
        5327, 5337, 5347,
        5419, 5429, 5439,
        5427, 5437, 5447,
        '-rayleigh', x[0], betaK, betaKInit, x[1])

# TODO: RBS?
# Current values: Evaluation of seismic collapse performance of steel special moment resisting
# frames using FEMA P695 (ATC-63) methodology (Zareian et al., 2010)

def getModifiedIK(shape, L):
    # reference Lignos & Krawinkler (2011)
    Fy = 50 # ksi
    Es = 29000 # ksi

    Sx = float(shape['Sx'])
    Iz = float(shape['Ix'])
    d = float(shape['d'])
    htw = float(shape['h/tw'])
    bftf = float(shape['bf/2tf'])
    ry = float(shape['ry'])
    c1 = 25.4
    c2 = 6.895

    My = Fy * Sx  *1.1
    thy = My/(6*Es*Iz/L)
    Ke = My/thy
    # consider using Lb = 0 for beams bc of slab?
    Lb = L
    kappa = 0.4

    if d > 21.0:
        Lam = (536*(htw)**(-1.26)*(bftf)**(-0.525)
            *(Lb/ry)**(-0.130)*(c2*Fy/355)**(-0.291))
        thp = (0.318*(htw)**(-0.550)*(bftf)**(-0.345)
            *(Lb/ry)**(-0.0230)*(L/d)**(0.090)*(c1*d/533)**(-0.330)*
            (c2*Fy/355)**(-0.130))
        thpc = (7.50*(htw)**(-0.610)*(bftf)**(-0.710)
            *(Lb/ry)**(-0.110)*(c1*d/533)**(-0.161)*
            (c2*Fy/355)**(-0.320))
    else:
        Lam = 495*(htw)**(-1.34)*(bftf)**(-0.595)*(c2*Fy/355)**(-0.360)
        thp = (0.0865*(htw)**(-0.365)*(bftf)**(-0.140)
            *(L/d)**(0.340)*(c1*d/533)**(-0.721)*
            (c2*Fy/355)**(-0.230))
        thpc = (5.63*(htw)**(-0.565)*(bftf)**(-0.800)
            *(c1*d/533)**(-0.280)*(c2*Fy/355)**(-0.430))

    # Lam = 1000
    # thp = 0.025
    # thpc = 0.3
    thu = 0.2

    return(Ke, My, Lam, thp, thpc, kappa, thu)

############################################################################
#              Start model
############################################################################

def build():

    # remove existing model
    ops.wipe()
    ops.wipeAnalysis()

    ############################################################################
    #              Definitions
    ############################################################################

    # units: in, kip, s
    # dimensions
    inch    = 1.0
    in4     = inch*inch*inch*inch
    ft      = 12.0*inch
    sec     = 1.0
    g       = 386.4*inch/(sec**2)
    kip     = 1.0
    ksi     = kip/(inch**2)

    # set modelbuilder
    # command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    import superStructDesign as sd
    (mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol) = sd.design()

    (AgCol, IzCol, IyCol, ZxCol, SxCol, dCol, bfCol, tfCol, twCol) = getProperties(selectedCol)
    (AgBeam, IzBeam, IyBeam, ZxBeam, SxBeam, dBeam, bfBeam, tfBeam, twBeam) = getProperties(selectedBeam)
    (AgRoofBeam, IzRoofBeam, IyRoofBeam, ZxRoofBeam, SxRoofBeam, dRoofBeam, bfRoofBeam, tfRoofBeam, twRoofBeam) = getProperties(selectedRoofBeam)

    ############################################################################
    #              Model construction
    ############################################################################

    # assuming mass only includes a) dead load, no weight, no live load, unfactored, or b) dead load + live load, factored
    # masses represent half the building's mass

    global w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3

    m0Inner = 81.7*kip/g
    m1Inner = 81.7*kip/g
    m2Inner = 81.7*kip/g
    m3Inner = 58.1*kip/g

    m0Outer = 40.9*kip/g
    m1Outer = 40.9*kip/g
    m2Outer = 40.9*kip/g
    m3Outer = 29.0*kip/g

    w0 = 2.72*kip/(1*ft)
    w1 = 2.72*kip/(1*ft)
    w2 = 2.72*kip/(1*ft)
    w3 = 1.94*kip/(1*ft)

    # multiply to account for half-building tfp loading
    pOuter = (w0 + w1 + w2 + w3)*15*ft*3
    pInner = (w0 + w1 + w2 + w3)*30*ft*3

    # Leaning column loads

    pLc0 = 482.0*kip
    pLc1 = 482.0*kip
    pLc2 = 482.0*kip
    pLc3 = 340.0*kip

    pLc  = pLc0 + pLc1 + pLc2 + pLc3

    mLc0 = 490.4*kip/g
    mLc1 = 490.4*kip/g
    mLc2 = 490.4*kip/g
    mLc3 = 348.4*kip/g

    LBeam = 30.0*ft
    LCol = 13.0*ft

################################################################################
# define nodes
################################################################################

    # create nodes
    # node number is XY, (X-1)th floor, Yth column
    # support nodes have the coordinates XYA
    # A is 6,7,8,9 for S,W,N,E respectively

    # command: node(nodeID, x-coord, y-coord, z-coord)
    ops.node(11,     0*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(12,     1*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(13,     2*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(14,     3*LBeam,    0.0*ft,     0*LCol+1.0*ft)

    ops.node(21,     0*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(22,     1*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(23,     2*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(24,     3*LBeam,    0.0*ft,     1*LCol+1.0*ft)

    ops.node(31,    0*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(32,    1*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(33,    2*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(34,    3*LBeam,    0.0*ft,     2*LCol+1.0*ft)

    ops.node(41,    0*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(42,    1*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(43,    2*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(44,    3*LBeam,    0.0*ft,     3*LCol+1.0*ft)

    # additional nodes to facilitate springs
    # diaphragm level
    ops.node(118,     0*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(128,     1*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(138,     2*LBeam,    0.0*ft,     0*LCol+1.0*ft)
    ops.node(148,     3*LBeam,    0.0*ft,     0*LCol+1.0*ft)

    # first floor
    ops.node(216,     0*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(226,     1*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(236,     2*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(246,     3*LBeam,    0.0*ft,     1*LCol+1.0*ft)

    ops.node(227,     1*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(237,     2*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(247,     3*LBeam,    0.0*ft,     1*LCol+1.0*ft)

    ops.node(218,     0*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(228,     1*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(238,     2*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(248,     3*LBeam,    0.0*ft,     1*LCol+1.0*ft)

    ops.node(219,     0*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(229,     1*LBeam,    0.0*ft,     1*LCol+1.0*ft)
    ops.node(239,     2*LBeam,    0.0*ft,     1*LCol+1.0*ft)

    # second floor
    ops.node(316,     0*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(326,     1*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(336,     2*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(346,     3*LBeam,    0.0*ft,     2*LCol+1.0*ft)

    ops.node(327,     1*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(337,     2*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(347,     3*LBeam,    0.0*ft,     2*LCol+1.0*ft)

    ops.node(318,     0*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(328,     1*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(338,     2*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(348,     3*LBeam,    0.0*ft,     2*LCol+1.0*ft)

    ops.node(319,     0*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(329,     1*LBeam,    0.0*ft,     2*LCol+1.0*ft)
    ops.node(339,     2*LBeam,    0.0*ft,     2*LCol+1.0*ft)

    # third (roof) floor
    ops.node(416,     0*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(426,     1*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(436,     2*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(446,     3*LBeam,    0.0*ft,     3*LCol+1.0*ft)

    ops.node(427,     1*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(437,     2*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(447,     3*LBeam,    0.0*ft,     3*LCol+1.0*ft)

    ops.node(419,     0*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(429,     1*LBeam,    0.0*ft,     3*LCol+1.0*ft)
    ops.node(439,     2*LBeam,    0.0*ft,     3*LCol+1.0*ft)

    # Base isolation layer node i
    ops.node(1,    0*LBeam,    0.0*ft,     0*LCol)
    ops.node(2,    1*LBeam,    0.0*ft,     0*LCol)
    ops.node(3,    2*LBeam,    0.0*ft,     0*LCol)
    ops.node(4,    3*LBeam,    0.0*ft,     0*LCol)

    # Leaning columns
    # ops.node(5, 4*LBeam,   0.0*ft, 0.0*ft)

    ops.node(15, 4*LBeam,   0.0*ft, 0*LCol+1.0*ft)
    # ops.node(158, 4*LBeam,   0.0*ft, 0*LCol+1.0*ft)
    
    ops.node(256, 4*LBeam,   0.0*ft, 1*LCol+1.0*ft)
    ops.node(25, 4*LBeam,   0.0*ft, 1*LCol+1.0*ft)
    ops.node(258, 4*LBeam,   0.0*ft, 1*LCol+1.0*ft)

    ops.node(356, 4*LBeam,   0.0*ft, 2*LCol+1.0*ft)
    ops.node(35, 4*LBeam,   0.0*ft, 2*LCol+1.0*ft)
    ops.node(358, 4*LBeam,   0.0*ft, 2*LCol+1.0*ft)

    ops.node(456, 4*LBeam,   0.0*ft, 3*LCol+1.0*ft)
    ops.node(45, 4*LBeam,   0.0*ft, 3*LCol+1.0*ft)

    # Moat wall
    ops.node(81, 0*LBeam,   0.0*ft,     0*LCol+1.0*ft)
    ops.node(84, 3*LBeam,   0.0*ft,     0*LCol+1.0*ft)


    # assign masses, in direction of motion and stiffness
    # DOF list: X, Y, Z, rotX, rotY, rotZ
    negligible = 1e-15

    ops.mass(11, m0Outer, m0Outer, m0Outer, negligible, negligible, negligible)
    ops.mass(12, m0Inner, m0Inner, m0Inner, negligible, negligible, negligible)
    ops.mass(13, m0Inner, m0Inner, m0Inner, negligible, negligible, negligible)
    ops.mass(14, m0Outer, m0Outer, m0Outer, negligible, negligible, negligible)

    ops.mass(21, m1Outer, m1Outer, m1Outer, negligible, negligible, negligible)
    ops.mass(22, m1Inner, m1Inner, m1Inner, negligible, negligible, negligible)
    ops.mass(23, m1Inner, m1Inner, m1Inner, negligible, negligible, negligible)
    ops.mass(24, m1Outer, m1Outer, m1Outer, negligible, negligible, negligible)

    ops.mass(31, m2Outer, m2Outer, m2Outer, negligible, negligible, negligible)
    ops.mass(32, m2Inner, m2Inner, m2Inner, negligible, negligible, negligible)
    ops.mass(33, m2Inner, m2Inner, m2Inner, negligible, negligible, negligible)
    ops.mass(34, m2Outer, m2Outer, m2Outer, negligible, negligible, negligible)

    ops.mass(41, m3Outer, m3Outer, m3Outer, negligible, negligible, negligible)
    ops.mass(42, m3Inner, m3Inner, m3Inner, negligible, negligible, negligible)
    ops.mass(43, m3Inner, m3Inner, m3Inner, negligible, negligible, negligible)
    ops.mass(44, m3Outer, m3Outer, m3Outer, negligible, negligible, negligible)

    ops.mass(15, mLc0, mLc0, mLc0, negligible, negligible, negligible)
    ops.mass(25, mLc1, mLc1, mLc1, negligible, negligible, negligible)
    ops.mass(35, mLc2, mLc2, mLc2, negligible, negligible, negligible)
    ops.mass(45, mLc3, mLc3, mLc3, negligible, negligible, negligible)

################################################################################
# constraints
################################################################################

    # constraints
    # command: fix(nodeID, DOF1, DOF2, DOF3) 0 = free, 1 = fixed
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(2, 1, 1, 1, 1, 1, 1)
    ops.fix(3, 1, 1, 1, 1, 1, 1)
    ops.fix(4, 1, 1, 1, 1, 1, 1)
    # ops.fix(5, 1, 1, 1, 1, 1, 1)

    # stopgap solution: restrain all nodes from moving in y-plane and rotating about X & Z axes

    ops.fix(11, 0, 1, 0, 1, 0, 1)
    ops.fix(12, 0, 1, 0, 1, 0, 1)
    ops.fix(13, 0, 1, 0, 1, 0, 1)
    ops.fix(14, 0, 1, 0, 1, 0, 1)
    
    ops.fix(15, 0, 1, 1, 1, 0, 1)

    ops.fix(21, 0, 1, 0, 1, 0, 1)
    ops.fix(22, 0, 1, 0, 1, 0, 1)
    ops.fix(23, 0, 1, 0, 1, 0, 1)
    ops.fix(24, 0, 1, 0, 1, 0, 1)
    ops.fix(25, 0, 1, 0, 1, 0, 1)

    ops.fix(31, 0, 1, 0, 1, 0, 1)
    ops.fix(32, 0, 1, 0, 1, 0, 1)
    ops.fix(33, 0, 1, 0, 1, 0, 1)
    ops.fix(34, 0, 1, 0, 1, 0, 1)
    ops.fix(35, 0, 1, 0, 1, 0, 1)

    ops.fix(41, 0, 1, 0, 1, 0, 1)
    ops.fix(42, 0, 1, 0, 1, 0, 1)
    ops.fix(43, 0, 1, 0, 1, 0, 1)
    ops.fix(44, 0, 1, 0, 1, 0, 1)
    ops.fix(45, 0, 1, 0, 1, 0, 1)

    # moat wall restraints
    ops.fix(81, 1, 1, 1, 1, 1, 1)
    ops.fix(84, 1, 1, 1, 1, 1, 1)

################################################################################
# Tags
################################################################################

    # General elastic section (non-plastic beam columns, leaning columns)
    LCSpringMatTag          = 51
    elasticMatTag            = 52

    # Steel material tag
    steelColTag = 31
    steelBeamTag = 32
    steelRoofBeamTag = 33

    # Isolation layer tags
    frn1ModelTag    = 41
    frn2ModelTag    = 42
    frn3ModelTag    = 43
    fpsMatPTag      = 44
    fpsMatMzTag     = 45
    
    # Impact material tags
    impactMatTag = 91


################################################################################
# define materials
################################################################################

    # define material: steel
    Es  = 29000*ksi     # initial elastic tangent
    nu  = 0.2          # Poisson's ratio
    Gs  = Es/(1 + nu) # Torsional stiffness modulus
    J   = 1e10          # Set large torsional stiffness

    # Frame link
    ARigid = 1000.0         # define area of truss section (make much larger than A of frame elements)
    IRigid = 1e6*in4        # moment of inertia for p-delta columns  (make much larger than I of frame elements)
    ops.uniaxialMaterial('Elastic', elasticMatTag, Es)

################################################################################
# define beams and columns
################################################################################
    

    # Modified IK steel
    cIK = 1.0
    DIK = 1.0
    (KeCol, MyCol, LamCol, 
     thpCol, thpcCol, kappaCol, thuCol) = getModifiedIK(selectedCol, LCol)
    (KeBeam, MyBeam, LamBeam, 
     thpBeam, thpcBeam, kappaBeam, thuBeam) = getModifiedIK(selectedBeam, LBeam)
    (KeRoofBeam, MyRoofBeam, LamRoofBeam, 
     thpRoofBeam, thpcRoofBeam, kappaRoofBeam, thuRoofBeam) = getModifiedIK(selectedRoofBeam, LBeam)

    # calculate modified section properties to account for spring stiffness being in series with the elastic element stiffness
    # Ibarra, L. F., and Krawinkler, H. (2005). "Global collapse of frame structures under seismic excitations,"
    n = 10 # stiffness multiplier for rotational spring

    IzCol_mod = IzCol*(n+1)/n
    IzBeam_mod = IzBeam*(n+1)/n
    IzRoofBeam_mod = IzRoofBeam*(n+1)/n

    IyCol_mod = IyCol*(n+1)/n
    IyBeam_mod = IyBeam*(n+1)/n
    IyRoofBeam_mod = IyRoofBeam*(n+1)/n

    KeCol = n*6.0*Es*IzCol/(0.8*LCol)
    KeBeam = n*6.0*Es*IzBeam/(0.8*LBeam)
    KeRoofBeam = n*6.0*Es*IzRoofBeam/(0.8*LBeam)

    McMy = 1.1 # ratio of capping moment to yield moment, Mc / My
    a_mem_col = (n+1.0)*(MyCol*(McMy-1.0))/(KeCol*thpCol)
    b_col = a_mem_col/(1.0+n*(1.0-a_mem_col))

    a_mem_beam = (n+1.0)*(MyCol*(McMy-1.0))/(KeBeam*thpBeam)
    b_beam = a_mem_beam/(1.0+n*(1.0-a_mem_beam))

    ops.uniaxialMaterial('Bilin', steelColTag, KeCol, b_col, b_col, 
                         MyCol, -MyCol, LamCol, 
                         0, 0, 0, 
                         cIK, cIK, cIK, cIK, 
                         thpCol, thpCol, thpcCol, thpcCol, 
                         kappaCol, kappaCol, thuCol, thuCol, DIK, DIK)

    ops.uniaxialMaterial('Bilin', steelBeamTag, KeBeam, b_beam, b_beam, 
                         MyBeam, -MyBeam, LamBeam, 
                         0, 0, 0, 
                         cIK, cIK, cIK, cIK, 
                         thpBeam, thpBeam, thpcBeam, thpcBeam, 
                         kappaBeam, kappaBeam, thuBeam, thuBeam, DIK, DIK)

    ops.uniaxialMaterial('Bilin', steelRoofBeamTag, KeRoofBeam, b_beam, b_beam, 
                         MyRoofBeam, -MyRoofBeam, LamRoofBeam, 
                         0, 0, 0, 
                         cIK, cIK, cIK, cIK, 
                         thpRoofBeam, thpRoofBeam, thpcRoofBeam, thpcRoofBeam, 
                         kappaRoofBeam, kappaRoofBeam, thuRoofBeam, thuRoofBeam, DIK, DIK)

    # Create springs at column and beam ends
    # Springs follow Modified Ibarra Krawinkler model
    def rotSpringModIK(eleID, matID, nodeI, nodeJ, memTag):
        # columns
        if memTag == 1:
            ops.element('zeroLength', eleID, nodeI, nodeJ,
                '-mat', elasticMatTag, elasticMatTag, elasticMatTag, 
                elasticMatTag, elasticMatTag, matID, 
                '-dir', 1, 2, 3, 4, 5, 6,
                '-orient', 0, 0, 1, 1, 0, 0,
                '-doRayleigh', 1)           # Create zero length element (spring), rotations allowed about local z axis
        # beams
        if memTag == 2:
            ops.element('zeroLength', eleID, nodeI, nodeJ,
                '-mat', elasticMatTag, elasticMatTag, elasticMatTag, 
                elasticMatTag, elasticMatTag, matID, 
                '-dir', 1, 2, 3, 4, 5, 6, 
                '-orient', 1, 0, 0, 0, 0, 1,
                '-doRayleigh', 1)           # Create zero length element (spring), rotations allowed about local z axis
        # ops.equalDOF(nodeI, nodeJ, 1, 2, 3, 4, 6)

    # TODO: should we physically move these hinges?
    
    # column springs
    # diaphragm level top
    colMem = 1
    beamMem = 2
    rotSpringModIK(5118, steelColTag, 11, 118, colMem)
    rotSpringModIK(5128, steelColTag, 12, 128, colMem)
    rotSpringModIK(5138, steelColTag, 13, 138, colMem)
    rotSpringModIK(5148, steelColTag, 14, 148, colMem)

    # floor 1 bottom
    rotSpringModIK(5216, steelColTag, 21, 216, colMem)
    rotSpringModIK(5226, steelColTag, 22, 226, colMem)
    rotSpringModIK(5236, steelColTag, 23, 236, colMem)
    rotSpringModIK(5246, steelColTag, 24, 246, colMem)

    # floor 1 top
    rotSpringModIK(5218, steelColTag, 21, 218, colMem)
    rotSpringModIK(5228, steelColTag, 22, 228, colMem)
    rotSpringModIK(5238, steelColTag, 23, 238, colMem)
    rotSpringModIK(5248, steelColTag, 24, 248, colMem)

    # floor 2 bottom
    rotSpringModIK(5316, steelColTag, 31, 316, colMem)
    rotSpringModIK(5326, steelColTag, 32, 326, colMem)
    rotSpringModIK(5336, steelColTag, 33, 336, colMem)
    rotSpringModIK(5346, steelColTag, 34, 346, colMem)

    # floor 2 top
    rotSpringModIK(5318, steelColTag, 31, 318, colMem)
    rotSpringModIK(5328, steelColTag, 32, 328, colMem)
    rotSpringModIK(5338, steelColTag, 33, 338, colMem)
    rotSpringModIK(5348, steelColTag, 34, 348, colMem)

    # floor 3 bottom
    rotSpringModIK(5416, steelColTag, 41, 416, colMem)
    rotSpringModIK(5426, steelColTag, 42, 426, colMem)
    rotSpringModIK(5436, steelColTag, 43, 436, colMem)
    rotSpringModIK(5446, steelColTag, 44, 446, colMem)

    # beam springs
    # floor 1 right
    rotSpringModIK(5219, steelBeamTag, 21, 219, beamMem)
    rotSpringModIK(5229, steelBeamTag, 22, 229, beamMem)
    rotSpringModIK(5239, steelBeamTag, 23, 239, beamMem)

    # floor 1 left
    rotSpringModIK(5227, steelBeamTag, 22, 227, beamMem)
    rotSpringModIK(5237, steelBeamTag, 23, 237, beamMem)
    rotSpringModIK(5247, steelBeamTag, 24, 247, beamMem)

    # floor 2 right
    rotSpringModIK(5319, steelBeamTag, 31, 319, beamMem)
    rotSpringModIK(5329, steelBeamTag, 32, 329, beamMem)
    rotSpringModIK(5339, steelBeamTag, 33, 339, beamMem)

    # floor 2 left
    rotSpringModIK(5327, steelBeamTag, 32, 327, beamMem)
    rotSpringModIK(5337, steelBeamTag, 33, 337, beamMem)
    rotSpringModIK(5347, steelBeamTag, 34, 347, beamMem)

    # floor 3 right (roof)
    rotSpringModIK(5419, steelRoofBeamTag, 41, 419, beamMem)
    rotSpringModIK(5429, steelRoofBeamTag, 42, 429, beamMem)
    rotSpringModIK(5439, steelRoofBeamTag, 43, 439, beamMem)

    # floor 3 left (roof)
    rotSpringModIK(5427, steelRoofBeamTag, 42, 427, beamMem)
    rotSpringModIK(5437, steelRoofBeamTag, 43, 437, beamMem)
    rotSpringModIK(5447, steelRoofBeamTag, 44, 447, beamMem)

    # geometric transformation for beam-columns
    # command: geomTransf('Type', TransfTag)
    # command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d

    beamTransfTag   = 1
    colTransfTag    = 2

    ops.geomTransf('Linear', beamTransfTag, 0, -1, 0) #beams
    ops.geomTransf('Corotational', colTransfTag, 0, -1, 0) #columns

    # outside of concentrated plasticity zones, use elastic beam columns
    # define the columns
    # first level
    ops.element('elasticBeamColumn', 111, 118, 216, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 112, 128, 226, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 113, 138, 236, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 114, 148, 246, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)

    # second level
    ops.element('elasticBeamColumn', 121, 218, 316, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 122, 228, 326, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 123, 238, 336, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 124, 248, 346, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)

    # third level
    ops.element('elasticBeamColumn', 131, 318, 416, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 132, 328, 426, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 133, 338, 436, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)
    ops.element('elasticBeamColumn', 134, 348, 446, 
        AgCol, Es, Gs, J, IyCol_mod, IzCol_mod, colTransfTag)

    # define the beams
    # first level
    ops.element('elasticBeamColumn', 221, 219, 227, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 222, 229, 237, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 223, 239, 247, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)

    # second level
    ops.element('elasticBeamColumn', 231, 319, 327, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 232, 329, 337, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 233, 339, 347, 
        AgBeam, Es, Gs, J, IyBeam_mod, IzBeam_mod, beamTransfTag)

    # third level
    ops.element('elasticBeamColumn', 241, 419, 427, 
        AgRoofBeam, Es, Gs, J, IyRoofBeam_mod, IzRoofBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 242, 429, 437, 
        AgRoofBeam, Es, Gs, J, IyRoofBeam_mod, IzRoofBeam_mod, beamTransfTag)
    ops.element('elasticBeamColumn', 243, 439, 447, 
        AgRoofBeam, Es, Gs, J, IyRoofBeam_mod, IzRoofBeam_mod, beamTransfTag)

################################################################################
# define leaning column
################################################################################

    # p-Delta columns
    # command: element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag[, '-mass', massPerLength][, '-cMass'])
    # ops.element('elasticBeamColumn', 115, 158, 256, 
    #     ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)
    ops.element('elasticBeamColumn', 115, 15, 256, 
        ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)
    ops.element('elasticBeamColumn', 125, 258, 356, 
        ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)
    ops.element('elasticBeamColumn', 135, 358, 456, 
        ARigid, Es, Gs, J, IRigid, IRigid, colTransfTag)

    # Rotational hinge at leaning column joints via zeroLength elements
    kLC = 1e-9*kip/inch

    # Create the material (spring)
    ops.uniaxialMaterial('Elastic', LCSpringMatTag, kLC)

    # Create moment releases at leaning column ends
    # command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs[, '-doRayleigh', rFlag=0][, '-orient', *vecx, *vecyp])
    # command: equalDOF(retained, constrained, DOF_1, DOF_2)
    def rotLeaningCol(eleID, nodeI, nodeJ):
        ops.element('zeroLength', eleID, nodeI, nodeJ,
            '-mat', elasticMatTag, elasticMatTag, elasticMatTag, 
            elasticMatTag, elasticMatTag, LCSpringMatTag, 
            '-dir', 1, 2, 3, 4, 5, 6, '-orient', 0, 0, 1, 1, 0, 0)           # Create zero length element (spring), rotations allowed about local Z axis
        # ops.equalDOF(nodeI, nodeJ, 1, 2, 3, 4, 6)                                                   # Constrain the translational DOFs and out-of-plane rotations

    # rotLeaningCol(5158, 15, 158)
    rotLeaningCol(5256, 25, 256)
    rotLeaningCol(5258, 25, 258)
    rotLeaningCol(5356, 35, 356)
    rotLeaningCol(5358, 35, 358)
    rotLeaningCol(5456, 45, 456)

################################################################################
# Trusses and diaphragms
################################################################################

    # define leaning columns, all beam sizes
    # truss beams
    # command: element('TrussSection', eleTag, *eleNodes, secTag[, '-rho', rho][, '-cMass', cFlag][, '-doRayleigh', rFlag])
    ops.element('Truss', 314, 14, 15, ARigid, elasticMatTag)
    ops.element('Truss', 324, 24, 25, ARigid, elasticMatTag)
    ops.element('Truss', 334, 34, 35, ARigid, elasticMatTag)
    ops.element('Truss', 344, 44, 45, ARigid, elasticMatTag)

    # define rigid 'diaphragm' at bottom layer
    # command: element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag[, '-mass', massPerLength][, '-cMass'])
    ops.element('elasticBeamColumn', 611, 11, 12, 
        ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)
    ops.element('elasticBeamColumn', 612, 12, 13, 
        ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)
    ops.element('elasticBeamColumn', 613, 13, 14, 
        ARigid, Es, Gs, J, IRigid, IRigid, beamTransfTag)

################################################################################
# define 2-D isolation layer 
################################################################################

    # Isolator parameters
    uy          = 0.00984*inch          # 0.025cm from Scheller & Constantinou

    dSlider1    = 4 *inch               # slider diameters
    dSlider2    = 11*inch
    dSlider3    = 11*inch

    d1      = 10*inch   - dSlider1      # displacement capacities
    d2      = 37.5*inch - dSlider2
    d3      = 37.5*inch - dSlider3

    h1      = 1*inch                    # half-height of sliders
    h2      = 4*inch
    h3      = 4*inch

    L1      = R1 - h1
    L2      = R2 - h2
    L3      = R3 - h3

    uLim    = 2*d1 + d2 + d3 + L1*d3/L3 - L1*d2/L2

    # friction pendulum system
    # kv = EASlider/hSlider
    kv = 6*1000*kip/inch
    ops.uniaxialMaterial('Elastic', fpsMatPTag, kv)
    ops.uniaxialMaterial('Elastic', fpsMatMzTag, 10.0)


    # Define friction model for FP elements
    # command: frictionModel Coulomb tag mu
    ops.frictionModel('Coulomb', frn1ModelTag, mu1)
    ops.frictionModel('Coulomb', frn2ModelTag, mu2)
    ops.frictionModel('Coulomb', frn3ModelTag, mu3)


    # define 2-D isolation layer 
    # command: element TripleFrictionPendulum $eleTag $iNode $jNode $frnTag1 $frnTag2 $frnTag3 $vertMatTag $rotZMatTag $rotXMatTag $rotYMatTag $L1 $L2 $L3 $d1 $d2 $d3 $W $uy $kvt $minFv $tol
    kvt     = 0.01*kv
    minFv   = 1.0
    ops.element('TripleFrictionPendulum', 51, 1, 11, frn1ModelTag, frn2ModelTag, 
        frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
        L1, L2, L3, d1, d2, d3, pOuter, uy, kvt, minFv, 1e-5)
    ops.element('TripleFrictionPendulum', 52, 2, 12, frn1ModelTag, frn2ModelTag, 
        frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
        L1, L2, L3, d1, d2, d3, pInner, uy, kvt, minFv, 1e-5)
    ops.element('TripleFrictionPendulum', 53, 3, 13, frn1ModelTag, frn2ModelTag, 
        frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
        L1, L2, L3, d1, d2, d3, pInner, uy, kvt, minFv, 1e-5)
    ops.element('TripleFrictionPendulum', 54, 4, 14, frn1ModelTag, frn2ModelTag, 
        frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
        L1, L2, L3, d1, d2, d3, pOuter, uy, kvt, minFv, 1e-5)

    # ops.element('TripleFrictionPendulum', 55, 5, 15, frn1ModelTag, frn2ModelTag, 
    #     frn3ModelTag, fpsMatPTag, fpsMatMzTag, fpsMatMzTag, fpsMatMzTag, 
    #     L1, L2, L3, d1, d2, d3, pLc, uy, kvt, minFv, 1e-5)

    # define impact moat as ZeroLengthImpact3D elements
    # https://opensees.berkeley.edu/wiki/index.php/Impact_Material
    # assume slab of base layer is 6 in
    # model half of slab
    VSlab = (6*inch)*(45*ft)*(90*ft)
    pi = 3.14159
    RSlab = (3/4/pi*VSlab)**(1/3)
    
    # assume wall extends 9 ft up to isolation layer and is 12 in thick
    VWall = (12*inch)*(1*ft)*(45*ft)
    RWall = (3/4/pi*VWall)**(1/3)
    
    # concrete slab and wall properties
    Ec = 3645*ksi
    nu_c = 0.2
    h_impact = (1-nu_c**2)/(pi*Ec)
    
    # impact stiffness parameter from Muthukumar, 2006
    khWall = 4/(3*pi*(h_impact + h_impact))*((RSlab*RWall)/(RSlab + RWall))**(0.5)
    e           = 0.7                                                   # coeff of restitution (1.0 = perfectly elastic collision)
    delM        = 0.025*inch                                            # maximum penetration during pounding event, from Hughes paper
    kEffWall    = khWall*((delM)**(0.5))                                # effective stiffness
    a           = 0.1                                                   # yield coefficient
    delY        = a*delM                                                # yield displacement
    nImpact     = 3/2                                                   # Hertz power rule exponent
    EImpact     = khWall*delM**(nImpact+1)*(1 - e**2)/(nImpact+1)       # energy dissipated during impact
    K1          = kEffWall + EImpact/(a*delM**2)                        # initial impact stiffness
    K2          = kEffWall - EImpact/((1-a)*delM**2)                    # secondary impact stiffness

    moatGap     = float(moatGap)                # DmPrime

    ops.uniaxialMaterial('ImpactMaterial', impactMatTag, K1, K2, -delY, -moatGap)
    
    # command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, 
    #   '-dir', *dirs, <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)
    ops.element('zeroLength', 881, 81, 11, '-mat', impactMatTag, elasticMatTag,
            '-dir', 1, 3, '-orient', 1, 0, 0, 0, 1, 0)
    ops.element('zeroLength', 884, 14, 84, '-mat', impactMatTag, elasticMatTag,
            '-dir', 1, 3, '-orient', 1, 0, 0, 0, 1, 0)

# if ran alone, build model and plot
if __name__ == '__main__':
    build()
    print('Model built!')
    
    import opsvis as opsv
    
    opsv.plot_model()
    # fix base for Tfb
    refix(11, "fix")
    refix(12, "fix")
    refix(13, "fix")
    refix(14, "fix")
    # refix(15, "fix")

    nEigenI     = 1;                    # mode i = 1
    nEigenJ     = 2;                    # mode j = 2
    lambdaN     = ops.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
    lambda1     = lambdaN[0];           # eigenvalue mode i = 1
    lambda2     = lambdaN[1];           # eigenvalue mode j = 2
    omega1      = lambda1**(0.5)    # w1 (1st mode circular frequency)
    omega2      = lambda2**(0.5)    # w2 (2nd mode circular frequency)
    Tfb         = 2*3.1415/omega1      # 1st mode period of the structure
    print("Tfb = ", Tfb, " s")          # display the first mode period in the command window

    # unfix base
    refix(11, "unfix")
    refix(12, "unfix")
    refix(13, "unfix")
    refix(14, "unfix")
    # refix(15, "unfix")