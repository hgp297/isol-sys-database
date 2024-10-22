############################################################################
#               Isolation layer test

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: October 2023

# Description:  Main workfile

# Open issues:  

############################################################################

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')

from db import Database

main_obj = Database(50)

main_obj.design_bearings(filter_designs=True)
main_obj.design_structure(filter_designs=True)

main_obj.scale_gms()

#%% troubleshoot

# # cbf lrb
# run = main_obj.retained_designs.loc[113]

# cbf tfp
# run = main_obj.retained_designs.iloc[0]

# # mf lrb
run = main_obj.retained_designs.loc[702]

# # mf tfp
# run = main_obj.retained_designs.loc[353]

from building import Building

# test build CBF
bldg = Building(run)
bldg.number_nodes()

############################################################################
# Construct bearing
############################################################################

print('=========== Constructing model ===========')
print('Frame type:', bldg.superstructure_system, '|', 
      'Isolator type:', bldg.isolator_system)
print('%d bays, %d stories, D_m = %.2f' % 
      (bldg.num_bays, bldg.num_stories, bldg.D_m))
print('Moat amplification = %.2f | Ry = %.2f' % 
      (bldg.moat_ampli, bldg.RI))
print('Tm = %.2f s | Q = %.2f | k_ratio = %.2f' %
      (bldg.T_m, bldg.Q, bldg.k_ratio))

# model gravity masses corresponding to the frame placed on building edge

# units: in, kip, s
# dimensions
inch    = 1.0
ft      = 12.0*inch
sec     = 1.0
g       = 386.4*inch/(sec**2)
pi = 3.14159
kip     = 1.0
ksi     = kip/(inch**2)

n_bays = bldg.num_bays

############################# construct isolation layer #######################
# nominal change
L_bay = bldg.L_bay*ft
L_beam = bldg.L_bay*ft
L_col = bldg.h_story*ft

w_cases = bldg.all_w_cases
plc_cases = bldg.all_Plc_cases
 
w_floor = w_cases['1.0D+0.5L'] / ft
p_lc = plc_cases['1.0D+0.5L']
# w_floor = bldg.w_fl / ft    # kip/ft to kip/in
# p_lc = bldg.P_lc

# load for isolators vertical
p_outer = sum(w_floor)*L_bay/2
p_inner = sum(w_floor)*L_bay

w_total = w_floor.sum()

# import OpenSees and libraries
import openseespy.opensees as ops

# remove existing model
ops.wipe()

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# model gravity masses corresponding to the frame placed on building edge
import numpy as np
# m_grav_inner = w_floor * L_bay / g
# m_grav_outer = w_floor * L_bay / 2 /g
m_grav_inner = w_total * L_bay / g
m_grav_outer = w_total * L_bay / 2 /g

# divide LC mass over rest of frame
m_lc = sum(p_lc) / g /n_bays

# base nodes
base_nodes = bldg.node_tags['base']
for idx, nd in enumerate(base_nodes):
    ops.node(nd, idx*L_beam, 0.0*ft, -1.0*ft)
    ops.fix(nd, 1, 1, 1, 1, 1, 1)
    
# diaphragm nodes
floor_nodes = bldg.node_tags['diaphragm']
for nd in floor_nodes:
    
    # get multiplier for location from node number
    bay = nd%10
    fl = (nd//10)%10 - 1
    ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
    
    # assign masses, in direction of motion and stiffness
    # DOF list: X, Y, Z, rotX, rotY, rotZ
    if (bay == n_bays) or (bay == 0):
        m_nd = m_grav_outer + m_lc
    else:
        m_nd = m_grav_inner + m_lc
    negligible = 1e-15
    ops.mass(nd, m_nd, m_nd, m_nd,
             negligible, negligible, negligible)
    
    # restrain out of plane motion
    ops.fix(nd, 0, 1, 0, 1, 0, 1)
   
wall_nodes = bldg.node_tags['wall']
ops.node(wall_nodes[0], 0.0*ft, 0.0*ft, 0.0*ft)
ops.node(wall_nodes[1], n_bays*L_beam, 0.0*ft, 0.0*ft)
for nd in wall_nodes:
    ops.fix(nd, 1, 1, 1, 1, 1, 1)
    
# define diaphragm
diaph_id = bldg.elem_ids['diaphragm']
diaph_elems = bldg.elem_tags['diaphragm']

# beam geometry
beam_transf_tag   = 1
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(11)
beam_x_axis = np.subtract(xyz_j, xyz_i)
vecxy_beam = [0, 0, 1] # Use any vector in local x-y, but not local x
vecxz = np.cross(beam_x_axis,vecxy_beam) # What OpenSees expects
vecxz_beam = vecxz / np.sqrt(np.sum(vecxz**2))
ops.geomTransf('PDelta', beam_transf_tag, *vecxz_beam) # beams

# Frame link (stiff elements)
A_rigid = 1000.0         # define area of truss section
I_rigid = 1e6        # moment of inertia for p-delta columns

# define material: steel
Es  = 29000*ksi     # initial elastic tangent
nu  = 0.2          # Poisson's ratio
Gs  = Es/(1 + nu) # Torsional stiffness modulus
J   = 1e10          # Set large torsional stiffness

elastic_mat_tag = 52
ops.uniaxialMaterial('Elastic', elastic_mat_tag, Es)

for elem_tag in diaph_elems:
    i_nd = elem_tag - diaph_id
    j_nd = elem_tag - diaph_id + 1
    ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, beam_transf_tag)
    
# define 2-D isolation layer 
isol_id = bldg.elem_ids['isolator']
base_id = bldg.elem_ids['base']
isol_elems = bldg.elem_tags['isolator']

if bldg.isolator_system == 'TFP':
    
    # TFP system
    # Isolator parameters
    R1 = bldg.R_1
    R2 = bldg.R_2
    uy          = 0.00984*inch      # 0.025cm from Scheller & Constantinou
    dSlider1    = 4 *inch           # slider diameters
    dSlider2    = 11*inch
    d1      = 10*inch   - dSlider1  # displacement capacities
    d2      = 37.5*inch - dSlider2
    h1      = 1*inch                # half-height of sliders
    h2      = 4*inch
    
    L1      = R1 - h1
    L2      = R2 - h2

    # uLim    = 2*d1 + 2*d2 + L1*d2/L2 - L1*d2/L2
    
    # Isolation layer tags
    friction_1_tag = 41
    friction_2_tag = 42
    fps_vert_tag = 44
    fps_rot_tag = 45
    
    # friction pendulum system
    # kv = EASlider/hSlider
    kv = 6*1000*kip/inch
    ops.uniaxialMaterial('Elastic', fps_vert_tag, kv)
    ops.uniaxialMaterial('Elastic', fps_rot_tag, 10.0)


    # Define friction model for FP elements
    # command: frictionModel Coulomb tag mu
    ops.frictionModel('Coulomb', friction_1_tag, bldg.mu_1)
    ops.frictionModel('Coulomb', friction_2_tag, bldg.mu_2)


    # define 2-D isolation layer 
    kvt     = 0.01*kv
    minFv   = 1.0
    
    isol_id = bldg.elem_ids['isolator']
    base_id = bldg.elem_ids['base']
    isol_elems = bldg.elem_tags['isolator']
    for elem_tag in isol_elems:
        i_nd = elem_tag - isol_id
        j_nd = elem_tag - isol_id - base_id + 10
        
        # if top node is furthest left or right, vertical force is outer
        if (j_nd == 0) or (j_nd%10 == n_bays):
            p_vert = p_outer
        else:
            p_vert = p_inner
        ops.element('TripleFrictionPendulum', elem_tag, i_nd, j_nd,
                    friction_1_tag, friction_2_tag, friction_2_tag,
                    fps_vert_tag, fps_rot_tag, fps_rot_tag, fps_rot_tag,
                    L1, L2, L2, d1, d2, d2,
                    p_vert, uy, kvt, minFv, 1e-5)

else:
    # dimensions. Material parameters should not be edited without 
    # modifying design script
    K_bulk = 290.0*ksi
    G_r = 0.060*ksi
    D_inner = bldg.d_lead
    D_outer = bldg.d_bearing - 0.5
    t_shim = 0.13*inch
    t_rubber_whole = bldg.t_r
    n_layers = int(bldg.n_layers)
    t_layer = t_rubber_whole/n_layers

    # calculate yield strength. this assumes design was done correctly
    f_y_Pb = 1.5 # ksi, shear yield strength

    Fy_LRB = f_y_Pb*pi*D_inner**2/4

    alpha = 1.0/bldg.k_ratio
    # Fy_LRB = Q_L/(1 - alpha)

    kc = 10.0
    phi_M = 0.5
    ac = 1.0

    qL_imp = 0.4046256704 # (lbs/in^3) density of lead 
    cL_imp = 0.03076 # (Btu/lb/degF) specific heat of lead at room temp
    kS_imp = 26.0*12.0 # (Btu/(hr*in*F)) thermal conductivity of steel 
    aS_imp = 0.018166036 # (in^2/s) thermal diffusivity of steel 

    sdr = 0.5
    # guesstimate 60% of the entire bearing volume's mass in lead
    # mb = qL_imp*pi*D_outer**2/4*t_rubber_whole*0.6
    mb = 0.0
    cd = 0.0
    tc = 1.0

    tag_1 = 0 # cavitation
    tag_2 = 0 # buckling load variation
    tag_3 = 0 # horiz stiffness variation
    tag_4 = 0 # vertical stiffness variation
    tag_5 = 0 # heat

    addl_params = [0, 0, 1, 1, 0, 0,
                   kc, phi_M, ac, sdr, mb, cd, tc,
                   qL_imp, cL_imp, kS_imp, aS_imp,
                   tag_1, tag_2, tag_3, tag_4, tag_5]

    h = t_rubber_whole + (n_layers-1)*t_shim + tc
    
    for elem_idx, elem_tag in enumerate(isol_elems):
        
        # if it is an unstacked bearing, it will have xx0x
        if (elem_tag//10)%10 == 0:
            i_nd = elem_tag - isol_id
            j_nd = elem_tag - isol_id - base_id + 10
        else:
            bay_pos = elem_tag % 10
            i_nd = base_id + bay_pos
            j_nd = i_nd - base_id + 10
            
        ops.element('LeadRubberX', elem_tag, i_nd, j_nd, Fy_LRB, alpha,
                    G_r, K_bulk, D_inner, D_outer,
                    t_shim, t_layer, n_layers, *addl_params)
    
# define impact moat as ZeroLengthImpact3D elements
# https://opensees.berkeley.edu/wiki/index.php/Impact_Material
# assume slab of base layer is 6 in
# model half of slab
L_bldg = bldg.L_bldg
VSlab = (6*inch)*(L_bldg/2*ft)*(L_bldg*ft)
pi = 3.14159
RSlab = (3/4/pi*VSlab)**(1/3)

# assume wall extends 1 ft up to isolation layer and is 12 in thick
VWall = (12*inch)*(1*ft)*(L_bldg/2*ft)
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

moat_gap = bldg.D_m * 2.0

# Impact material tags
impact_mat_tag = 91

ops.uniaxialMaterial('ImpactMaterial', impact_mat_tag, 
                     K1, K2, -delY, -moat_gap)

# command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, 
#   '-dir', *dirs, <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)
wall_elems = bldg.elem_tags['wall']

# ops.element('zeroLength', wall_elems[0], wall_nodes[0], 10,
#             '-mat', impact_mat_tag,
#             '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
# ops.element('zeroLength', wall_elems[1], int(10+n_bays), wall_nodes[1], 
#             '-mat', impact_mat_tag,
#             '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
    
open('./output/model.out', 'w').close()
ops.printModel('-file', './output/model.out')

############################################################################
#              Loading and analysis
############################################################################

nEigenJ = 1;                    # how many modes to analyze
lambdaN  = ops.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
lambda1 = lambdaN[0];           # eigenvalue mode i = 1
wi = lambda1**(0.5)    # w1 (1st mode circular frequency)
T_1 = 2*3.1415/wi      # 1st mode period of the structure

print("T_1 = %.3f s" % T_1)   

monotonic_pattern_tag  = 2
monotonic_series_tag = 1

grav_pattern_tag = 3
grav_series_tag = 4

# ------------------------------
# Loading: gravity
# ------------------------------

# create TimeSeries
ops.timeSeries("Linear", grav_series_tag)

# create plain load pattern
ops.pattern('Plain', grav_pattern_tag, grav_series_tag)

if bldg.isolator_system == 'TFP':
    # load right above isolation layer to increase stiffness to half-building for TFP
    # line load accounts for Lbay/2 of tributary, we linearly scale
    # to include the remaining portion of Lbldg/2
    ft = 12.0
    w_total = w_floor.sum()
    pOuter = w_total*(L_bay/2)*ft* ((L_bldg - L_bay)/L_bay)
    pInner = w_total*(L_bay)*ft* ((L_bldg - L_bay)/L_bay)

    diaph_nds = bldg.node_tags['diaphragm']
    
    for nd in diaph_nds:
        if (nd%10 == 0) or (nd%10 == n_bays):
            ops.load(nd, 0, 0, -pOuter, 0, 0, 0)
        else:
            ops.load(nd, 0, 0, -pInner, 0, 0, 0)

diaph_nds = bldg.node_tags['diaphragm']
        
# line loads on diaphragm
diaph_elems = bldg.elem_tags['diaphragm']
for elem in diaph_elems:
    w_applied = w_floor[0]
    ops.eleLoad('-ele', elem, '-type', '-beamUniform', 
                -w_applied, 0.0)

nStepGravity = 10  # apply gravity in 10 steps
tol = 1e-5
dGravity = 1/nStepGravity

ops.system("BandGeneral")
ops.test("NormDispIncr", tol, 15)
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("LoadControl", dGravity)
ops.algorithm("Newton")
ops.analysis("Static")
ops.analyze(nStepGravity)

print("Gravity analysis complete!")
ops.loadConst('-time', 0.0)

# ------------------------------
# Recorders
# ------------------------------
ops.wipeAnalysis()
data_dir = './output/'

ops.recorder('Node', '-file', data_dir+'isol_disp.csv', 
             '-time', '-node', *diaph_nds, '-dof', 1, 'disp')
ops.recorder('Node', '-file', data_dir+'isol_base_rxn.csv', 
             '-time', '-node', 
             *base_nodes, '-dof', 1, 'reaction')
ops.recorder('Node', '-file', data_dir+'isol_base_vert.csv', 
             '-time', '-node', 
             *base_nodes, '-dof', 3, 'reaction')


################## static cyclical #################
# create TimeSeries
ops.timeSeries("Linear", monotonic_series_tag)
ops.pattern('Plain', monotonic_pattern_tag, monotonic_series_tag)
ops.load(10, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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

# du = -0.01*inch
# ops.integrator('DisplacementControl', 201, 1, du, 1, du, du)
# max_u = -1.5  # Max displacement
# n_steps = int(round(max_u/du))
# currentDisp = 0.0
ops.analysis("Static")                      # create analysis object

peaks = np.arange(0.0, 30.0, 3.0)
peaks = np.append(peaks, peaks[-1])
steps = 500
for i, pk in enumerate(peaks):
    du = (-1.0)**i*(peaks[i] / steps)
    ops.integrator('DisplacementControl', 10, 1, du, 1, du, du)
    ops.analyze(steps)

################################ transient
# GMDirection = 1  # ground-motion direction

# gm_name = bldg.gm_selected
# scale_factor = bldg.scale_factor
# print('Current ground motion: %s at scale %.2f' % (gm_name, scale_factor))

# ops.constraints('Plain')
# ops.numberer('RCM')
# ops.system('BandGeneral')

# # Convergence Test: tolerance
# tolDynamic          = 1e-5 

# # Convergence Test: maximum number of iterations that will be performed
# maxIterDynamic      = 100

# # Convergence Test: flag used to print information on convergence
# printFlagDynamic    = 0         

# testTypeDynamic     = 'EnergyIncr'
# ops.test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)

# # algorithmTypeDynamic    = 'Broyden'
# # ops.algorithm(algorithmTypeDynamic, 8)
# algorithmTypeDynamic    = 'Newton'
# ops.algorithm(algorithmTypeDynamic)

# # Newmark-integrator gamma parameter (also HHT)
# newmarkGamma = 0.5
# newmarkBeta = 0.25
# ops.integrator('Newmark', newmarkGamma, newmarkBeta)
# ops.analysis('Transient')

# # # TRBDF2 integrator, best with energy
# # ops.integrator('TRBDF2')
# # ops.analysis('Transient')

# #  ---------------------------------    perform Dynamic Ground-Motion Analysis
# # the following commands are unique to the Uniform Earthquake excitation

# ################# ground motion
# # # Uniform EXCITATION: acceleration input
# # gm_dir = '../resource/ground_motions/PEERNGARecords_Unscaled/'
# # inFile = gm_dir + gm_name + '.AT2'
# # outFile = gm_dir + gm_name + '.g3'

# #  # call procedure to convert the ground-motion file
# # from ReadRecord import ReadRecord
# # dt, nPts = ReadRecord(inFile, outFile)
# # g = 386.4
# # GMfatt = g*scale_factor

# # eq_series_tag = 100
# # eq_pattern_tag = 400
# # # time series information
# # ops.timeSeries('Path', eq_series_tag, '-dt', dt, 
# #                '-filePath', outFile, '-factor', GMfatt)     
# # # create uniform excitation
# # ops.pattern('UniformExcitation', eq_pattern_tag, 
# #             GMDirection, '-accel', eq_series_tag)     
# ####################


# ################## cyclical
# dt = 0.01
# dispSeriesTag = 101
# velSeriesTag = 102
# accelSeriesTag = 103

# ops.timeSeries('Path', dispSeriesTag, '-dt', dt, 
#                 '-filePath', './motions/LDDisp.txt', '-factor', 300.0)   
# ops.timeSeries('Path', velSeriesTag, '-dt', dt, 
#                 '-filePath', './motions/LDVel.txt', '-factor', 1.0) 
# ops.timeSeries('Path', accelSeriesTag, '-dt', dt, 
#                 '-filePath', './motions/LDAcc.txt', '-factor', 150.0) 

# # eq_series_tag = 100
# cyclic_pattern_tag = 400

# ops.pattern('UniformExcitation', cyclic_pattern_tag, 
#             GMDirection, '-accel', accelSeriesTag)      
# ####################

# # set up ground-motion-analysis parameters
# sec = 1.0             
# dt_transient = 0.005         
# T_end = 60.0*sec

# import numpy as np
# n_steps = np.floor(T_end/dt_transient)

# # actually perform analysis; returns ok=0 if analysis was successful

# import time
# t0 = time.time()

# ok = ops.analyze(int(n_steps), dt_transient)   
# if ok != 0:
#     ok = 0
#     curr_time = ops.getTime()
#     print("Convergence issues at time: ", curr_time)
#     while (curr_time < T_end) and (ok == 0):
#         curr_time     = ops.getTime()
#         ok          = ops.analyze(1, dt_transient)
#         if ok != 0:
#             print("Trying Newton with Initial Tangent...")
#             ops.algorithm('Newton', '-initial')
#             ok = ops.analyze(1, dt_transient)
#             if ok == 0:
#                 print("That worked. Back to Newton")
#             ops.algorithm(algorithmTypeDynamic)
#         if ok != 0:
#             print("Trying Newton with line search ...")
#             ops.algorithm('NewtonLineSearch')
#             ok = ops.analyze(1, dt_transient)
#             if ok == 0:
#                 print("That worked. Back to Newton")
#             ops.algorithm(algorithmTypeDynamic)
#         if ok != 0:
#             print('Trying Broyden ... ')
#             algorithmTypeDynamic = 'Broyden'
#             ops.algorithm(algorithmTypeDynamic, 8)
#             ok = ops.analyze(1, dt_transient)
#             if ok == 0:
#                 print("That worked. Back to Newton")
#         if ok != 0:
#             print('Trying BFGS ... ')
#             algorithmTypeDynamic = 'BFGS'
#             ops.algorithm(algorithmTypeDynamic)
#             ok = ops.analyze(1, dt_transient)
#             if ok == 0:
#                 print("That worked. Back to Newton")

# t_final = ops.getTime()
# tp = time.time() - t0
# minutes = tp//60
# seconds = tp - 60*minutes
# print('Ground motion done. End time: %.4f s' % t_final)
# print('Analysis time elapsed %dm %ds.' % (minutes, seconds))

ops.wipe()

################################ plot hysteresis ##############################
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
isol_columns = ['time', 'x', 'z', 'rot']

force_columns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 
                'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']
# All hystereses
from bearing import Bearing
isolator = Bearing(run)

col_line = ['column_'+str(col)
               for col in range(0,bldg.num_bays+1)]
just_cols = col_line.copy()
col_line.insert(0, 'time')
isol_disp = pd.read_csv(data_dir+'isol_disp.csv', sep=' ', 
                             header=None, names=col_line)

isol_base_rxn = pd.read_csv(data_dir+'isol_base_rxn.csv', sep=' ', 
                             header=None, names=col_line)

isol_base_vert = pd.read_csv(data_dir+'isol_base_vert.csv', sep=' ', 
                             header=None, names=col_line)

isol_shear = isol_base_rxn[just_cols].sum(axis=1)
isol_axial = isol_base_vert[just_cols].sum(axis=1)
u_bearing, fs_bearing = isolator.get_backbone(mode='building')

if bldg.isolator_system == 'LRB':
    plt.figure()
    plt.plot(isol_disp['column_0'], -isol_shear)
    plt.plot(u_bearing, fs_bearing, linestyle='--')
    plt.title('Isolator hystereses (layer only) (LRB)')
    plt.xlabel('Displ (in)')
    plt.ylabel('Lateral force (kip)')
    plt.grid(True)
else:
    plt.figure()
    plt.plot(isol_disp['column_0'], -isol_shear/isol_axial)
    plt.plot(u_bearing, fs_bearing, linestyle='--')
    plt.title('Isolator hystereses (layer only) (TFP)')
    plt.xlabel('Displ (in)')
    plt.ylabel('V/N')
    plt.grid(True)

zeta_target = bldg.zeta_e
from scipy.spatial import ConvexHull
if bldg.isolator_system == 'LRB':
    loop = np.array([isol_disp['column_0'], -isol_shear]).T
else:
    loop = np.array([isol_disp['column_0'], -isol_shear/isol_axial]).T
hull = ConvexHull(loop)
plt.plot(loop[hull.vertices,0], loop[hull.vertices,1], 'r--', lw=2)

#%%
########################### barebones hysteresis ##############################

col_line = ['column_'+str(col)
               for col in range(0,bldg.num_bays+1)]
just_cols = col_line.copy()
col_line.insert(0, 'time')
isol_disp = pd.read_csv(data_dir+'isol_disp.csv', sep=' ', 
                             header=None, names=col_line)

isol_base_rxn = pd.read_csv(data_dir+'isol_base_rxn.csv', sep=' ', 
                             header=None, names=col_line)

isol_base_vert = pd.read_csv(data_dir+'isol_base_vert.csv', sep=' ', 
                             header=None, names=col_line)

isol_shear = isol_base_rxn[just_cols].sum(axis=1)
isol_axial = isol_base_vert[just_cols].sum(axis=1)
u_bearing, fs_bearing = isolator.get_backbone(mode='building')

if bldg.isolator_system == 'LRB':
    plt.figure()
    plt.plot(-isol_disp['column_0'], isol_shear, color='navy', linewidth=1.5)
    # plt.grid(True)
else:
    plt.figure()
    plt.plot(-isol_disp['column_0'], isol_shear/isol_axial)
    # plt.grid(True)
    
plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
plt.box(False) #remove box

########################### equivalent damping ################################
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

uo_test = np.max(np.abs(isol_disp['column_0']))
io_test = np.argmax(np.abs(isol_disp['column_0']))
if bldg.isolator_system == 'LRB':
    P_cor_test = abs(isol_shear[io_test])
else:
    P_cor_test = abs(isol_shear[io_test]/isol_axial[io_test])
    
Keq_test = P_cor_test/uo_test

uo_theo = np.max(np.abs(u_bearing))
io_theo = np.argmax(np.abs(u_bearing))
P_cor_theo = abs(fs_bearing[io_theo])
Keq_theo = P_cor_theo/uo_theo
Wd_theoretical = PolyArea(u_bearing, fs_bearing)

zeta_theoretical = Wd_theoretical/(2*pi*Keq_theo*uo_theo**2)

Wd_test = hull.volume
zeta_test = Wd_test/(2*pi*Keq_test*uo_test**2)

from numpy import interp

# from ASCE Ch. 17, get damping multiplier
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
# from T_m, zeta_M, S_1
B_m = interp(zeta_test, zetaRef, BmRef)

from gms import get_ST
Sa_tm = get_ST(run, run['T_m'])
disp_test = Sa_tm*g*run['T_m']**2/(4*pi**2*B_m)
print('Expected D_m under loop damping:', disp_test)
########################### conclusions #######################################
# sensitive to load added. perhaps W or Ws is incorrect