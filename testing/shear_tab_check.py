############################################################################
#               Testing for a single MIKPinching spring

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Various test for single HSS brace

# Open issues:  

############################################################################
# UTILITIES
############################################################################
def get_shape(shape_name, member, csv_dir='../resource/'):
    import pandas as pd
    
    if member == 'beam':
        shape_db = pd.read_csv(csv_dir+'beamShapes.csv',
                               index_col=None, header=0)
    elif member == 'column':
        shape_db = pd.read_csv(csv_dir+'colShapes.csv',
                               index_col=None, header=0)
    elif member == 'brace':
        shape_db = pd.read_csv(csv_dir+'braceShapes.csv',
                               index_col=None, header=0)  
    shape = shape_db.loc[shape_db['AISC_Manual_Label'] == shape_name]
    return(shape)

def get_properties(shape):
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

def modified_IK_params(shape, L):
    # reference Lignos & Krawinkler (2011)
    Fy = 50 # ksi
    Es = 29000 # ksi
    
    Zx = float(shape.iloc[0]['Zx'])
    # Sx = float(shape['Sx'])
    Iz = float(shape.iloc[0]['Ix'])
    d = float(shape.iloc[0]['d'])
    htw = float(shape.iloc[0]['h/tw'])
    bftf = float(shape.iloc[0]['bf/2tf'])
    ry = float(shape.iloc[0]['ry'])
    c1 = 25.4
    c2 = 6.895

    # approximate adjustment for isotropic hardening
    My = Fy * Zx * 1.17
    thy = My/(6*Es*Iz/L)
    
    # n is an adjustment to equate stiffness of spring-beam-spring element to 
    # actual stiffness of spring
    n = 10
    Ke = n*My/thy
    # consider using Lb = 0 for beams bc of slab?
    Lb = L
    kappa = 0.4

    if d > 21.0:
        lam = (536*(htw)**(-1.26)*(bftf)**(-0.525)
            *(Lb/ry)**(-0.130)*(c2*Fy/355)**(-0.291))
        thp = (0.318*(htw)**(-0.550)*(bftf)**(-0.345)
            *(Lb/ry)**(-0.0230)*(L/d)**(0.090)*(c1*d/533)**(-0.330)*
            (c2*Fy/355)**(-0.130))
        thpc = (7.50*(htw)**(-0.610)*(bftf)**(-0.710)
            *(Lb/ry)**(-0.110)*(c1*d/533)**(-0.161)*
            (c2*Fy/355)**(-0.320))
    else:
        lam = 495*(htw)**(-1.34)*(bftf)**(-0.595)*(c2*Fy/355)**(-0.360)
        thp = (0.0865*(htw)**(-0.365)*(bftf)**(-0.140)
            *(L/d)**(0.340)*(c1*d/533)**(-0.721)*
            (c2*Fy/355)**(-0.230))
        thpc = (5.63*(htw)**(-0.565)*(bftf)**(-0.800)
            *(c1*d/533)**(-0.280)*(c2*Fy/355)**(-0.430))
        
    thu = 0.2

    return(Ke, My, lam, thp, thpc, kappa, thu)

############################################################################
# Construct column - new MIK
############################################################################

# import OpenSees and libraries
import openseespy.opensees as ops

# remove existing model
ops.wipe()

# units: in, kip, s
# dimensions
inch    = 1.0
ft      = 12.0*inch
sec     = 1.0
g       = 386.4*inch/(sec**2)
kip     = 1.0
ksi     = kip/(inch**2)

L_bay = 30.0 * ft     # ft to in
h_story = 13.0 * ft

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# model gravity masses corresponding to the frame placed on building edge
import numpy as np

# nominal change
L_beam = L_bay
L_col = h_story

selectedCol = get_shape('W24X55', 'beam')


# base node
ops.node(10, 0.0, 0.0, 0.0)
ops.node(510, 0.0, 0.0, 0.0)
ops.fix(10, 1, 1, 1, 1, 1, 1)

# end node
ops.node(20, 0.0, 0.0, L_bay/2)

############################################################################
# Materials 
############################################################################

# General elastic section (non-plastic beam columns, leaning columns)
elastic_mat_tag = 52
torsion_mat_tag = 53
ghost_mat_tag = 54

# Steel material tag
steel_mat_tag = 31
steel_no_fatigue = 33
gp_mat_tag = 32

# Section tags
col_sec_tag = 41
brace_beam_sec_tag = 43
brace_sec_tag = 44
gp_sec_tag = 45

# Integration tags
col_int_tag = 61
brace_beam_int_tag = 63
brace_int_tag = 64

# define material: steel
Es  = 29000*ksi     # initial elastic tangent
nu  = 0.2          # Poisson's ratio
Gs  = Es/(1 + nu) # Torsional stiffness modulus
J   = 1e10          # Set large torsional stiffness

# Frame link (stiff elements)
A_rigid = 1000.0         # define area of truss section
I_rigid = 1e6        # moment of inertia for p-delta columns
ops.uniaxialMaterial('Elastic', elastic_mat_tag, Es)

# define material: Steel02
# command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
Fy  = 50*ksi        # yield strength
b   = 0.003           # hardening ratio
R0 = 15
cR1 = 0.925
cR2 = 0.15

E0 = 0.095
m = 0.95
ops.uniaxialMaterial('Elastic', torsion_mat_tag, J)
ops.uniaxialMaterial('Steel02', steel_no_fatigue, Fy, Es, b, R0, cR1, cR2)
ops.uniaxialMaterial('Fatigue', steel_mat_tag, steel_no_fatigue)


# Modified IK steel
cIK = 1.0
DIK = 1.0
(Ke_beam, My_beam, lam_beam,
 thp_beam, thpc_beam,
 kappa_beam, thu_beam) = modified_IK_params(selectedCol, L_beam)

n_mik = 10

(Ag_beam, Iz_beam, Iy_beam,
 Zx_beam, Sx_beam, d_beam,
 bf_beam, tf_beam, tw_beam) = get_properties(selectedCol)

Iz_beam_mod = Iz_beam*(n_mik+1)/n_mik
Iy_beam_mod = Iy_beam*(n_mik+1)/n_mik
################################################################################
# define column
################################################################################

# Create springs at column and beam ends
# Springs follow Modified Ibarra Krawinkler model
# should note that the parent-spring convention works only if
# material is symmetric
def rot_spring_bilin(eleID, matID, nodeI, nodeJ, mem_tag):
    
    # Create springs at column and beam ends
    # Springs follow Modified Ibarra Krawinkler model
    # have spring local z be global y, then allow rotation around local z
    column_x = [0, 0, 1]
    column_y = [1, 0, 0]
    beam_x = [1, 0, 0]
    beam_y = [0, 0, 1]
    
    # columns
    if mem_tag == 1:
        # Create zero length element (spring), rotations allowed about local z axis
        ops.element('zeroLength', eleID, nodeI, nodeJ,
            '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
            torsion_mat_tag, elastic_mat_tag, matID, 
            '-dir', 1, 2, 3, 4, 5, 6,
            '-orient', *column_x, *column_y,
            '-doRayleigh', 1)           
    # beams
    if mem_tag == 2:
        # Create zero length element (spring), rotations allowed about local z axis
        ops.element('zeroLength', eleID, nodeI, nodeJ,
            '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
            torsion_mat_tag, elastic_mat_tag, matID, 
            '-dir', 1, 2, 3, 4, 5, 6, 
            '-orient', *beam_x, *beam_y,
            '-doRayleigh', 1)

# geometric transformation for beam-columns
# command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
beam_transf_tag   = 1
col_transf_tag    = 2
brace_beam_transf_tag = 3
brace_transf_tag = 4

# column geometry
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(20)
col_x_axis = np.subtract(xyz_j, xyz_i)
vecxy_col = [1, 0, 0] # Use any vector in local x-y, but not local x
vecxz = np.cross(col_x_axis,vecxy_col) # What OpenSees expects
vecxz_col = vecxz / np.sqrt(np.sum(vecxz**2))
ops.geomTransf('PDelta', col_transf_tag, *vecxz_col) # columns

Mmax_st = 0.15*My_beam
# point 1 in Elkady's Pinching4
thy_st = 0.0045
My_st = 0.521*Mmax_st
K0_st = My_st/thy_st 
# point 3 in Elkady's Pinching4
thp_st = 0.075 - 0.0045
Fpp_st = 0.7
Fpn_st = 0.7

lam_st = 0.9

thpc_st = 0.1-0.075
kappa_res_st = 0.901
thu_st = 0.08

ops.uniaxialMaterial('IMKPinching', 333, K0_st, 
                     thp_st, thpc_st, thu_st, My_st, 1/0.521, kappa_res_st,
                     thp_st, thpc_st, thu_st, My_st, 1/0.521, kappa_res_st,
                     lam_st, lam_st, lam_st, lam_st,
                     1.0, 1.0, 1.0, 1.0,
                     DIK, DIK,
                     Fpp_st, Fpn_st)

# ops.uniaxialMaterial('ModIMKPinching', 333, K0_st, a_st, a_st, 
#                       My_st, -My_st, Fpp_st, Fpn_st, a_pinch_st, 
#                       lam_st, lam_st, lam_st, lam_st,
#                       1.0, 1.0, 1.0, 1.0, 
#                       thp_st, thp_st, 
#                       thpc_st, thpc_st, 
#                       kappa_res_st, kappa_res_st, thu_st, thu_st, 
#                       DIK, DIK)

colMem = 1
# rotSpringModIK(51, col_sec_tag, 10, 510, colMem)

rot_spring_bilin(51, 333, 10, 510, colMem)

ops.element('elasticBeamColumn', 61, 510, 20, 
    Ag_beam, Es, Gs, J, Iy_beam_mod, Iz_beam_mod, col_transf_tag)

############################################################################
#              Loading and analysis
############################################################################
monotonic_pattern_tag  = 2
monotonic_series_tag = 1

gm_pattern_tag = 3
gm_series_tag = 4

# ------------------------------
# Loading: axial
# ------------------------------

# create TimeSeries
ops.timeSeries("Linear", monotonic_series_tag)
ops.pattern('Plain', monotonic_pattern_tag, monotonic_series_tag)
ops.load(20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

tol = 1e-8

# ops.system("BandGeneral")   
# ops.test("NormDispIncr", tol, 15)
# ops.numberer("RCM")
# ops.constraints("Plain")
# ops.algorithm("Newton")

ops.test('EnergyIncr', 1.0e-8, 300, 0)
ops.algorithm('KrylovNewton')
ops.system('UmfPack')
ops.numberer("RCM")
ops.constraints("Plain")

node_rot = 'output/node_rotation.out'
spring_moment = 'output/spring_moment.out'

ops.recorder('Node','-node', 510,'-file', node_rot, '-dof', 5, 'disp')
ops.recorder('Element', '-ele', 51, '-file', spring_moment, 'localForce')

ops.analysis("Static")                      # create analysis object

peaks = np.arange(0.0, 5.0, 0.25)
steps = 500
for i, pk in enumerate(peaks):
    du = (-1.0)**i*(peaks[i] / steps)
    ops.integrator('DisplacementControl', 20, 1, du, 1, du, du)
    ops.analyze(steps)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

# ops.analyze(n_steps)
# disp = ops.nodeDisp(201, 1)
# print('Displacement: %.5f' %disp)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

ops.wipe()

############################################################################
# Construct column - new MIK
############################################################################

# import OpenSees and libraries
import openseespy.opensees as ops

# remove existing model
ops.wipe()

# units: in, kip, s
# dimensions
inch    = 1.0
ft      = 12.0*inch
sec     = 1.0
g       = 386.4*inch/(sec**2)
kip     = 1.0
ksi     = kip/(inch**2)

L_bay = 30.0 * ft     # ft to in
h_story = 13.0 * ft

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# model gravity masses corresponding to the frame placed on building edge
import numpy as np

# nominal change
L_beam = L_bay
L_col = h_story

# base node
ops.node(10, 0.0, 0.0, 0.0)
ops.node(510, 0.0, 0.0, 0.0)
ops.fix(10, 1, 1, 1, 1, 1, 1)

# end node
ops.node(20, 0.0, 0.0, h_story)

############################################################################
# Materials 
############################################################################

# General elastic section (non-plastic beam columns, leaning columns)
elastic_mat_tag = 52
torsion_mat_tag = 53
ghost_mat_tag = 54

# Steel material tag
steel_mat_tag = 31
steel_no_fatigue = 33
gp_mat_tag = 32

# Section tags
col_sec_tag = 41
brace_beam_sec_tag = 43
brace_sec_tag = 44
gp_sec_tag = 45

# Integration tags
col_int_tag = 61
brace_beam_int_tag = 63
brace_int_tag = 64

# define material: steel
Es  = 29000*ksi     # initial elastic tangent
nu  = 0.2          # Poisson's ratio
Gs  = Es/(1 + nu) # Torsional stiffness modulus
J   = 1e10          # Set large torsional stiffness

# Frame link (stiff elements)
A_rigid = 1000.0         # define area of truss section
I_rigid = 1e6        # moment of inertia for p-delta columns
ops.uniaxialMaterial('Elastic', elastic_mat_tag, Es)

# define material: Steel02
# command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
Fy  = 50*ksi        # yield strength
b   = 0.003           # hardening ratio
R0 = 15
cR1 = 0.925
cR2 = 0.15

E0 = 0.095
m = 0.95
ops.uniaxialMaterial('Elastic', torsion_mat_tag, J)
ops.uniaxialMaterial('Steel02', steel_no_fatigue, Fy, Es, b, R0, cR1, cR2)
ops.uniaxialMaterial('Fatigue', steel_mat_tag, steel_no_fatigue)


# Modified IK steel
cIK = 1.0
DIK = 1.0
(Ke_beam, My_beam, lam_beam,
 thp_beam, thpc_beam,
 kappa_beam, thu_beam) = modified_IK_params(selectedCol, L_beam)

n_mik = 10

(Ag_beam, Iz_beam, Iy_beam,
 Zx_beam, Sx_beam, d_beam,
 bf_beam, tf_beam, tw_beam) = get_properties(selectedCol)

Iz_beam_mod = Iz_beam*(n_mik+1)/n_mik
Iy_beam_mod = Iy_beam*(n_mik+1)/n_mik
################################################################################
# define column
################################################################################

# Create springs at column and beam ends
# Springs follow Modified Ibarra Krawinkler model
# should note that the parent-spring convention works only if
# material is symmetric
def rot_spring_bilin(eleID, matID, nodeI, nodeJ, mem_tag):
    
    # Create springs at column and beam ends
    # Springs follow Modified Ibarra Krawinkler model
    # have spring local z be global y, then allow rotation around local z
    column_x = [0, 0, 1]
    column_y = [1, 0, 0]
    beam_x = [1, 0, 0]
    beam_y = [0, 0, 1]
    
    # columns
    if mem_tag == 1:
        # Create zero length element (spring), rotations allowed about local z axis
        ops.element('zeroLength', eleID, nodeI, nodeJ,
            '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
            torsion_mat_tag, elastic_mat_tag, matID, 
            '-dir', 1, 2, 3, 4, 5, 6,
            '-orient', *column_x, *column_y,
            '-doRayleigh', 1)           
    # beams
    if mem_tag == 2:
        # Create zero length element (spring), rotations allowed about local z axis
        ops.element('zeroLength', eleID, nodeI, nodeJ,
            '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
            torsion_mat_tag, elastic_mat_tag, matID, 
            '-dir', 1, 2, 3, 4, 5, 6, 
            '-orient', *beam_x, *beam_y,
            '-doRayleigh', 1)

# geometric transformation for beam-columns
# command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
beam_transf_tag   = 1
col_transf_tag    = 2
brace_beam_transf_tag = 3
brace_transf_tag = 4

# column geometry
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(20)
col_x_axis = np.subtract(xyz_j, xyz_i)
vecxy_col = [1, 0, 0] # Use any vector in local x-y, but not local x
vecxz = np.cross(col_x_axis,vecxy_col) # What OpenSees expects
vecxz_col = vecxz / np.sqrt(np.sum(vecxz**2))
ops.geomTransf('PDelta', col_transf_tag, *vecxz_col) # columns

My_st = 0.15*My_beam
# point 1 in Elkady's Pinching4
thy_st = 0.0045
K0_st = (0.521*My_st)/thy_st 
# point 3 in Elkady's Pinching4
thp_st = 0.075 - 0.0045
a_st = (My_st - 0.521*My_st)/(thp_st)
a_pinch_st = 0.75
Fpp_st = 0.4
Fpn_st = 0.4
lam_st = 0.9
thpc_st = 0.1-0.075
kappa_res_st = 0.901
gap_st = 0.08
thu_st = 0.15

# ops.uniaxialMaterial('ModIMKPinching', 333, K0_st, a_st, a_st, 
#                       My_st, -My_st, Fpp_st, Fpn_st, a_pinch_st, 
#                       lam_st, lam_st, lam_st, lam_st,
#                       1.0, 1.0, 1.0, 1.0, 
#                       thp_st, thp_st, 
#                       thpc_st, thpc_st, 
#                       kappa_res_st, kappa_res_st, thu_st, thu_st, 
#                       DIK, DIK)

M_1 = 0.521*My_st
M_2 = 0.967*My_st
M_3 = 1.0*My_st
M_4 = 0.901*My_st
th1 = 0.0045
th2 = 0.0465
th3 = 0.0750
th4 = 0.10
rd_st = 0.57
rf_st = 0.40
uf_st = 0.05

ops.uniaxialMaterial('Pinching4', 332, M_1, th1, M_2, th2, M_3, th3, M_4, th4,
                     -M_1, -th1, -M_2, -th2,- M_3, -th3, -M_4, -th4,
                     rd_st, rf_st, uf_st,  rd_st, rf_st, uf_st,   
                     0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,   
                     0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 'energy')

ops.uniaxialMaterial('MinMax', 333, 332, '-min', -0.08, '-max', 0.08)

colMem = 1
# rotSpringModIK(51, col_sec_tag, 10, 510, colMem)

rot_spring_bilin(51, 333, 10, 510, colMem)

ops.element('elasticBeamColumn', 61, 510, 20, 
    Ag_beam, Es, Gs, J, Iy_beam_mod, Iz_beam_mod, col_transf_tag)

############################################################################
#              Loading and analysis
############################################################################
monotonic_pattern_tag  = 2
monotonic_series_tag = 1

gm_pattern_tag = 3
gm_series_tag = 4

# ------------------------------
# Loading: axial
# ------------------------------

# create TimeSeries
ops.timeSeries("Linear", monotonic_series_tag)
ops.pattern('Plain', monotonic_pattern_tag, monotonic_series_tag)
ops.load(20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

tol = 1e-5

# ops.system("BandGeneral")   
# ops.test("NormDispIncr", tol, 15)
# ops.numberer("RCM")
# ops.constraints("Plain")
# ops.algorithm("Newton")

ops.test('EnergyIncr', 1.0e-5, 300, 0)
ops.algorithm('KrylovNewton')
ops.system('UmfPack')
ops.numberer("RCM")
ops.constraints("Plain")

node_rot_old = 'output/node_rotation_old.out'
spring_moment_old = 'output/spring_moment_old.out'

ops.recorder('Node','-node', 510,'-file', node_rot_old, '-dof', 5, 'disp')
ops.recorder('Element', '-ele', 51, '-file', spring_moment_old, 'localForce')

ops.analysis("Static")                      # create analysis object

peaks = np.arange(0.0, 5.0, 0.25)
steps = 500
for i, pk in enumerate(peaks):
    du = (-1.0)**i*(peaks[i] / steps)
    ops.integrator('DisplacementControl', 20, 1, du, 1, du, du)
    ops.analyze(steps)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

# ops.analyze(n_steps)
# disp = ops.nodeDisp(201, 1)
# print('Displacement: %.5f' %disp)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

ops.wipe()
#%%
############################################################################
#              Plot results
############################################################################

import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')


local_response = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']

spring_res = pd.read_csv(spring_moment, sep=' ', header=None, names=local_response,
                         index_col=False)
spring_rot = pd.read_csv(node_rot, sep=' ', header=None, names=['rotY'])

spring_res_old = pd.read_csv(spring_moment_old, sep=' ', header=None, names=local_response,
                         index_col=False)
spring_rot_old = pd.read_csv(node_rot_old, sep=' ', header=None, names=['rotY'])

# stress strain
fig = plt.figure()
plt.plot(-spring_rot['rotY'], -spring_res['mz']/12, label='IMKPinching')
plt.plot(-spring_rot_old['rotY'], -spring_res_old['mz']/12, label='Pinching4')
plt.title('Cantilever spring moment curvature')
plt.ylabel('Moment (kip-ft)')
plt.xlabel('Rotation (rads)')
plt.legend()
plt.grid(True)

# stress strain
fig = plt.figure()
plt.plot(-spring_rot['rotY'], -spring_res['mz']/12, color='maroon',
         linewidth=1.5, label='new')
# plt.axis('off')
plt.grid(True)