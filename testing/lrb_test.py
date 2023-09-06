############################################################################
#               Testing for a single MIK spring

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Various test for single HSS brace

# Open issues:  


############################################################################
# Construct bearing
############################################################################

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

# units: in, kip, s
# dimensions
inch    = 1.0
ft      = 12.0*inch
sec     = 1.0
g       = 386.4*inch/(sec**2)
pi = 3.14159
kip     = 1.0
ksi     = kip/(inch**2)

n_bays = 5
N_LB = 4*n_bays
W_bldg = 10768./N_LB
m = W_bldg/g

# dimensions. Material parameters should not be edited without 
# modifying design script
K_bulk = 290.0*ksi
G_r = 0.060*ksi
D_inner = 7.28*inch
D_outer = 32.76*inch
t_shim = 0.1*inch
t_rubber_whole = 12.35*inch
n_layers = 30
t_layer = t_rubber_whole/n_layers

# calculate yield strength. this assumes design was done correctly
f_y_Pb = 1.5 # ksi, shear yield strength

Q_d = f_y_Pb*pi*D_inner**2/4
k_ratio = 13.868
alpha = 1.0/k_ratio
Fy_LRB = Q_d/(1 - alpha)

kc = 10.0
phi_M = 0.5
ac = 1.0

sdr = 0.5
mb = m
cd = 0.0
tc = 1.0

# qL_imp = 11200
# cL_imp = 3e-1
# kS_imp = 50
# aS_imp = 1.41e-5

qL_imp = 0.4046256704 # (lbs/in^3) density of lead 
cL_imp = 0.03076 # (Btu/lb/degF) specific heat of lead at room temp
kS_imp = 26.0*12.0 # (Btu/(hr*in*F)) thermal conductivity of steel 
aS_imp = 0.018166036 # (in^2/s) thermal diffusivity of steel 

# units: m, N, s, g
# dimensions
'''
g = 9.81
pi = 3.1415
m = 253018/300

K_bulk = 2000e6
G_r = 0.4137e6
D_inner = .229/4.40
D_outer = 1.143/4.40
t_shim = 16.347e-3
alpha = 0.119

tc = 0e-3
n_layers = 20
t_layer = 19.05e-3
t_rubber_whole = n_layers*t_layer

# calculate yield strength. this assumes design was done correctly
Fy_lb = 18e6
TL1 = 20
Q_d = Fy_lb*pi*D_inner**2/4
A = pi/4 * ( (D_outer+tc)**2 - D_inner**2)
k_2 = G_r * A /t_rubber_whole
k_1 = k_2/alpha

# Q = 0.099
# W = 12641.66
# Q_L = Q*W
# k_ratio = 13.868
# alpha = 1.0/k_ratio

Fy_LRB = Q_d/(1 - alpha)

kc = 10.0
phi_M = 0.5
ac = 1.0

sdr = 0.5
mb = m
# TODO: is this really zero?
cd = 0.
qL_imp = 19000
cL_imp = 33
kS_imp = 50
aS_imp = 1.41e-5

'''
h = t_rubber_whole + (n_layers-1)*t_shim + tc

# base node
ops.node(10, 0.0, 0.0, 0.0)
ops.fix(10, 1, 1, 1, 1, 1, 1)

# end node
ops.node(20, 0.0, 0.0, h)
ops.fix(20, 0, 0, 0, 0, 1, 0)

negligible = 1e-5
ops.mass(20, m, m, m,
          negligible, negligible, negligible)

tag_1 = 0 # cavitation
tag_2 = 1 # buckling load variation
tag_3 = 1 # horiz stiffness variation
tag_4 = 0 # vertical stiffness variation
tag_5 = 0 # heat

# Fy_h = 4.43e5
addl_params = [0, 0, 1, 1, 0, 0,
               kc, phi_M, ac, sdr, mb, cd, tc,
               qL_imp, cL_imp, kS_imp, aS_imp,
               tag_1, tag_2, tag_3, tag_4, tag_5]

ops.element('LeadRubberX', 1900, 10, 20, Fy_LRB, alpha,
            G_r, K_bulk, D_inner, D_outer,
            t_shim, t_layer, n_layers, *addl_params)


open('./output/model.out', 'w').close()
ops.printModel('-file', './output/model.out')

############################################################################
#              Loading and analysis
############################################################################
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

# p_applied = W_bldg/n_bays
p_applied = m*g
ops.load(20, 0, 0, -p_applied, 0, 0, 0)

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

node_rxn = 'output/lrb_force.out'
node_disp = 'output/lrb_disp.out'

ops.recorder('Node','-node', 10,'-file', node_rxn, '-dof', 1, 'reaction')
ops.recorder('Node','-node', 20,'-file', node_disp, '-dof', 1, 'disp')
ops.recorder('Node','-node', 20,'-file', 'output/bearing_top.out', '-dof', 1, 3, 'disp')
ops.recorder('Element', '-ele', 1900, '-file', 'output/basic_lrb_force.out', 'basicForce')
ops.recorder('Element', '-ele', 1900, '-file', 'output/basic_lrb_disp.out', 'basicDisplacement')

ops.recorder('Element', '-ele', 1900, '-file', 'output/param.out', 'Parameters')
steps = 500

# ------------------------------
# Loading: cyclic file
# ------------------------------

ops.wipeAnalysis()

dt = 0.01
dispSeriesTag = 101
velSeriesTag = 102
accelSeriesTag = 103

ops.timeSeries('Path', dispSeriesTag, '-dt', dt, 
                '-filePath', './motions/LDDisp.txt', '-factor', 200.0)   
ops.timeSeries('Path', velSeriesTag, '-dt', dt, 
                '-filePath', './motions/LDVel.txt', '-factor', 1.0) 
ops.timeSeries('Path', accelSeriesTag, '-dt', dt, 
                '-filePath', './motions/LDAcc.txt', '-factor', 1.0) 

eq_series_tag = 100
eq_pattern_tag = 400

ops.pattern('MultipleSupport', eq_pattern_tag)
 
ops.groundMotion(eq_series_tag, 'Plain', 
                 '-disp', dispSeriesTag, 
                 '-vel', velSeriesTag, 
                 '-accel', accelSeriesTag)

ops.imposedMotion(20, 1, eq_series_tag)

ops.system('SparseGeneral')
ops.constraints("Transformation")
ops.test('NormDispIncr', 1.0e-5, 20, 0)
algorithmTypeDynamic = 'Newton'
ops.algorithm(algorithmTypeDynamic)
ops.numberer("Plain")



newmarkGamma = 0.5
newmarkBeta = 0.25
ops.integrator('Newmark', newmarkGamma, newmarkBeta)
ops.analysis('Transient')

# set up ground-motion-analysis parameters
sec = 1.0                      

dt_transient = 0.01
T_end = 7470.*dt_transient
n_steps = int(np.floor(T_end/dt_transient))

# actually perform analysis; returns ok=0 if analysis was successful

import time
t0 = time.time()

ok = ops.analyze(n_steps, dt_transient)   
if ok != 0:
    ok = 0
    curr_time = ops.getTime()
    print("Convergence issues at time: ", curr_time)
    while (curr_time < T_end) and (ok == 0):
        curr_time     = ops.getTime()
        ok          = ops.analyze(1, dt_transient)
        if ok != 0:
            print("Trying Newton with Initial Tangent...")
            ops.algorithm('Newton', '-initial')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)
        if ok != 0:
            print("Trying Newton with line search ...")
            ops.algorithm('NewtonLineSearch')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)

t_final = ops.getTime()
tp = time.time() - t0
minutes = tp//60
seconds = tp - 60*minutes
print('Ground motion done. End time: %.4f s' % t_final)
print('Analysis time elapsed %dm %ds.' % (minutes, seconds))

# ------------------------------
# Loading: earthquake
# ------------------------------
'''
ops.wipeAnalysis()
# Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
GMDirection = 1  # ground-motion direction
gm_name = 'RSN3905_TOTTORI_OKY002EW'
scale_factor = 50.0
print('Current ground motion: %s at scale %.2f' % (gm_name, scale_factor))

ops.constraints('Plain')
ops.numberer('RCM')
ops.system('BandGeneral')

# Convergence Test: tolerance
tolDynamic          = 1e-3 

# Convergence Test: maximum number of iterations that will be performed
maxIterDynamic      = 500

# Convergence Test: flag used to print information on convergence
printFlagDynamic    = 0         

testTypeDynamic     = 'NormDispIncr'
ops.test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)

# algorithmTypeDynamic    = 'Broyden'
# ops.algorithm(algorithmTypeDynamic, 8)
algorithmTypeDynamic    = 'Newton'
ops.algorithm(algorithmTypeDynamic)

# Newmark-integrator gamma parameter (also HHT)
newmarkGamma = 0.5
newmarkBeta = 0.25
ops.integrator('Newmark', newmarkGamma, newmarkBeta)
ops.analysis('Transient')

#  ---------------------------------    perform Dynamic Ground-Motion Analysis
# the following commands are unique to the Uniform Earthquake excitation

gm_dir = '../resource/ground_motions/PEERNGARecords_Unscaled/'

# Uniform EXCITATION: acceleration input
inFile = gm_dir + gm_name + '.AT2'
outFile = gm_dir + gm_name + '.g3'

  # call procedure to convert the ground-motion file
from ReadRecord import ReadRecord
dt, nPts = ReadRecord(inFile, outFile)
g = 386.4
GMfatt = g*scale_factor

eq_series_tag = 100
eq_pattern_tag = 400
# time series information
ops.timeSeries('Path', eq_series_tag, '-dt', dt, 
                '-filePath', outFile, '-factor', GMfatt)     
# create uniform excitation
ops.pattern('UniformExcitation', eq_pattern_tag, 
            GMDirection, '-accel', eq_series_tag)          

# set up ground-motion-analysis parameters
sec = 1.0                      
T_end = 60.0*sec


dt_transient = 0.005
n_steps = int(np.floor(T_end/dt_transient))

# actually perform analysis; returns ok=0 if analysis was successful

import time
t0 = time.time()

ok = ops.analyze(n_steps, dt_transient)   
if ok != 0:
    ok = 0
    curr_time = ops.getTime()
    print("Convergence issues at time: ", curr_time)
    while (curr_time < T_end) and (ok == 0):
        curr_time     = ops.getTime()
        ok          = ops.analyze(1, dt_transient)
        if ok != 0:
            print("Trying Newton with Initial Tangent...")
            ops.algorithm('Newton', '-initial')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)
        if ok != 0:
            print("Trying Newton with line search ...")
            ops.algorithm('NewtonLineSearch')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)

t_final = ops.getTime()
tp = time.time() - t0
minutes = tp//60
seconds = tp - 60*minutes
print('Ground motion done. End time: %.4f s' % t_final)
print('Analysis time elapsed %dm %ds.' % (minutes, seconds))

ops.wipe()
'''
#%%
############################################################################
#              Plot results
############################################################################

import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')


# disp plot
disps = pd.read_csv(node_disp, sep=' ', header=None, names=['displacement'])

# disp plot
forces = pd.read_csv(node_rxn, sep=' ', header=None, names=['force'])



# force disp
fig = plt.figure()
plt.plot(disps['displacement'], -forces['force'])
plt.title('Force-displacement recorded at end node')
plt.ylabel('Force (kip)')
plt.xlabel('Displacement (in)')
plt.grid(True)

# # force disp
# fig = plt.figure()
# plt.scatter(disps['displacement'], -forces['force']/1000, 
#             c=np.arange(1, len(disps['displacement'])+1))
# plt.title('Force-displacement recorded at end node')
# plt.ylabel('Force (kN)')
# plt.xlabel('Displacement (m)')
# plt.grid(True)

# cycles
fig = plt.figure()
plt.plot(np.arange(1, len(disps['displacement'])+1)/steps, disps['displacement'])
plt.title('Cyclic history')
plt.ylabel('Displacement history')
plt.xlabel('Cycles')
plt.grid(True)

# disp plot
basic_cols  = ['axial', 'shear', 'element_z_axial', 'torsion', 'moment_y', 'moment_z']
disps = pd.read_csv('output/basic_lrb_disp.out', sep=' ', header=None, names=basic_cols)

# disp plot
forces = pd.read_csv('output/basic_lrb_force.out', sep=' ', header=None, names=basic_cols)

# force disp
fig = plt.figure()
plt.plot(disps['shear'], forces['shear'])
plt.title('Force-displacement (basic in element)')
plt.ylabel('Force (kip)')
plt.xlabel('Displacement (in)')
plt.grid(True)

# param plot
params = pd.read_csv('output/param.out', sep=' ', header=None, 
                     names=['Fcn', "Fcrn", 'Kv', 'Ke', 'dT', 'qY'])

# Cavitation
fig = plt.figure()
plt.plot(np.arange(1, len(params['Fcn'])+1)/steps, 
          forces['axial'] /params['Fcn'])
plt.title('axial (basic in element)/Cavitation force ')
plt.ylabel('Cavitation ratio')
plt.xlabel('Steps')
plt.grid(True)

# buckling
fig = plt.figure()
plt.plot(np.arange(1, len(params['Fcn'])+1)/steps, 
          forces['axial']/params['Fcrn'])
plt.title('axial (basic in element)/Buckling force')
plt.ylabel('Buckling ratio')
plt.xlabel('Steps')
plt.grid(True)

# ke
fig = plt.figure()
plt.plot(np.arange(1, len(params['Fcn'])+1)/steps, 
          params['Ke'])
plt.title('Horizontal stiffness')
plt.ylabel('Ke')
plt.xlabel('Steps')
plt.grid(True)

# strain ratio
fig = plt.figure()
plt.plot(np.arange(1, len(params['Fcn'])+1)/steps, 
          disps['shear']/h*100)
plt.title('Strain ratio')
plt.ylabel('Strain ratio (%)')
plt.xlabel('Steps')
plt.grid(True)

#%% animate

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# top_node = pd.read_csv('output/bearing_top.csv', sep=' ', header=None, names=['x', 'z'])
# history_len = len(top_node['x'])

# dt = 0.0001
# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(autoscale_on=False, xlim=(-50, 50), ylim=(-1, 15))
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# time_template = 'time = %.1fs'
# line, = ax.plot([], [], 'o-', lw=2)
# trace, = ax.plot([], [], '.-', lw=1, ms=2)

# def animate(i):
#     thisx = [0, top_node['x'][i]]
#     thisy = [0, top_node['z'][i]+12.0]
    
#     line.set_data(thisx, thisy)
    
#     time_text.set_text(time_template % (i*dt))
#     return line, trace, time_text

# ani = animation.FuncAnimation(
#     fig, animate, history_len, interval=dt*history_len, blit=True)
# plt.show()