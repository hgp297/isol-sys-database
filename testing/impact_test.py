
# import OpenSees and libraries
import openseespy.opensees as ops

# remove existing model
ops.wipe()

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# wall nodes
wall_nodes = [898, 899]
ops.node(wall_nodes[0], 0.0, 0.0, 0.0)
ops.node(wall_nodes[1], 0.0, 0.0, 0.0)

ops.fix(wall_nodes[0], 1, 1, 1, 1, 1, 1)
    
    
impact_mat_tag = 91
ops.uniaxialMaterial('ImpactMaterial', impact_mat_tag, 
                     10, 1, -.1, -0.5)

ops.element('zeroLength', 8898, wall_nodes[0], wall_nodes[1],
            '-mat', impact_mat_tag,
            '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)

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
ops.load(899, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

tol = 1e-5

ops.recorder('Element', '-file', 'output/impact_forces.csv', 
             '-time', '-ele', 8898, 'basicForce')
ops.recorder('Element', '-file', 'output/impact_disp.csv', 
             '-time', '-ele', 8898, 'basicDeformation')

ops.test('EnergyIncr', 1.0e-5, 300, 0)
ops.algorithm('KrylovNewton')
ops.system('UmfPack')
ops.numberer("RCM")
ops.constraints("Plain")

ops.analysis("Static")                      # create analysis object

import numpy as np
peaks = np.arange(0.0, 1.0, 0.1)
steps = 500

for i, pk in enumerate(peaks):
    du = (-1.0)**i*(pk / steps)
    ops.integrator('DisplacementControl', 899, 1, du, 1, du, du)
    ops.analyze(steps)
    
ops.wipe()