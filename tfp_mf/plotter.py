############################################################################
#               Plotter utility

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: May 2020

# Description:  Plotter utility used to plot csv files in /outputs/

############################################################################

import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']

dataDir = './outputs/'

isolDisp = pd.read_csv(dataDir+'isolDisp.csv', sep=' ', header=None, names=dispColumns)
isolVert = pd.read_csv(dataDir+'isolVert.csv', sep=' ', header=None, names=dispColumns)
isolRot  = pd.read_csv(dataDir+'isolRot.csv', sep=' ', header=None, names=dispColumns)

story1Disp = pd.read_csv(dataDir+'story1Disp.csv', sep=' ', header=None, names=dispColumns)
story2Disp = pd.read_csv(dataDir+'story2Disp.csv', sep=' ', header=None, names=dispColumns)
story3Disp = pd.read_csv(dataDir+'story3Disp.csv', sep=' ', header=None, names=dispColumns)

story0Acc = pd.read_csv(dataDir+'story0Acc.csv', sep=' ', header=None, names=dispColumns)
story1Acc = pd.read_csv(dataDir+'story1Acc.csv', sep=' ', header=None, names=dispColumns)
story2Acc = pd.read_csv(dataDir+'story2Acc.csv', sep=' ', header=None, names=dispColumns)
story3Acc = pd.read_csv(dataDir+'story3Acc.csv', sep=' ', header=None, names=dispColumns)

# forceColumns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']
forceColumns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']

isol1Force = pd.read_csv(dataDir+'isol1Force.csv', sep = ' ', header=None, names=forceColumns)
isol2Force = pd.read_csv(dataDir+'isol2Force.csv', sep = ' ', header=None, names=forceColumns)
isol3Force = pd.read_csv(dataDir+'isol3Force.csv', sep = ' ', header=None, names=forceColumns)
isol4Force = pd.read_csv(dataDir+'isol4Force.csv', sep = ' ', header=None, names=forceColumns)
isolLCForce = pd.read_csv(dataDir+'isolLCForce.csv', sep = ' ', header=None, names=forceColumns)

col1Force = pd.read_csv(dataDir+'colForce1.csv', sep = ' ', header=None, names=forceColumns)
col2Force = pd.read_csv(dataDir+'colForce2.csv', sep = ' ', header=None, names=forceColumns)
col3Force = pd.read_csv(dataDir+'colForce3.csv', sep = ' ', header=None, names=forceColumns)
col4Force = pd.read_csv(dataDir+'colForce4.csv', sep = ' ', header=None, names=forceColumns)

outercolForce = pd.read_csv(dataDir+'colForce1.csv', sep=' ', header=None, names=forceColumns)
innercolForce = pd.read_csv(dataDir+'colForce2.csv', sep=' ', header=None, names=forceColumns)

diaphragmForce1 = pd.read_csv(dataDir+'diaphragmForce1.csv', sep = ' ', header=None, names=forceColumns)
diaphragmForce2 = pd.read_csv(dataDir+'diaphragmForce2.csv', sep = ' ', header=None, names=forceColumns)
diaphragmForce3 = pd.read_csv(dataDir+'diaphragmForce3.csv', sep = ' ', header=None, names=forceColumns)

# impactColumns = ['time', 'iFx', 'iShear', 'iMoment', 'jFx', 'jShear', 'jMoment']
# impactForce1 = pd.read_csv(dataDir+'wallImpactForce1.csv', sep = ' ', header=None, names=impactColumns)
# impactForce2 = pd.read_csv(dataDir+'wallImpactForce2.csv', sep = ' ', header=None, names=impactColumns)


force1Normalize = -isol1Force['iFy']/isol1Force['iFx']
force2Normalize = -isol2Force['iFy']/isol2Force['iFx']
force3Normalize = -isol3Force['iFy']/isol3Force['iFx']
force4Normalize = -isol4Force['iFy']/isol4Force['iFx']
forceLCNormalize = -isolLCForce['iFy']/isolLCForce['iFx']

story1DriftOuter    = (story1Disp['isol1'] - isolDisp['isol1'])/(13*12)
story1DriftInner    = (story1Disp['isol2'] - isolDisp['isol2'])/(13*12)

story2DriftOuter    = (story2Disp['isol1'] - story1Disp['isol1'])/(13*12)
story2DriftInner    = (story2Disp['isol2'] - story1Disp['isol2'])/(13*12)

story3DriftOuter    = (story3Disp['isol1'] - story2Disp['isol1'])/(13*12)
story3DriftInner    = (story3Disp['isol2'] - story2Disp['isol2'])/(13*12)

sumAxial        = isol1Force['iFx'] + isol2Force['iFx'] + isol3Force['iFx'] + isol4Force['iFx']
# equilCheck = impactForce1['iShear']+ diaphragmForce1['iFx'] + isol1Force['jFy'] - outercolForce['iFy']

# wholeLayerShear = isol1Force['iFy'] + isol2Force['iFy'] + isol3Force['iFy'] + isol4Force['iFy'] + isolLCForce['iFy'] - impactForce1['iFx'] + impactForce2['iFx']
# wholeLayerNormalize = wholeLayerShear / (sumAxial + isolLCForce['iFx'])

baseShear = col1Force['iFy'] + col2Force['iFy'] + col3Force['iFy'] + col4Force['iFy']
baseShearNormalize = baseShear/sumAxial

# pushover
fig = plt.figure()
plt.plot(story3Disp['isol1'], baseShearNormalize)
plt.title('Pushover curve')
plt.xlabel('Roof displacement (in)')
plt.ylabel('Normalized base shear')
plt.grid(True)

# isolayer forces
fig = plt.figure()
plt.plot(diaphragmForce1['time'], diaphragmForce1['iFx'])
plt.title('Diaphragm 1 axial')
plt.xlabel('Time (s)')
plt.ylabel('Axial force (kip)')
plt.grid(True)

# fig = plt.figure()
# plt.plot(isol1Force['time'], isol1Force['jFy'])
# plt.title('isolator 1 shear')
# plt.xlabel('Time (s)')
# plt.ylabel('shear force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(outercolForce['time'], outercolForce['iFy'])
# plt.title('column 1 shear')
# plt.xlabel('Time (s)')
# plt.ylabel('shear force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isol1Force['time'], equilCheck)
# plt.title('Equilibrium check at node 1')
# plt.xlabel('Time (s)')
# plt.ylabel('Shear force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(impactForce1['time'], impactForce1['iFx'])
# plt.title('Left wall force')
# plt.xlabel('Time (s)')
# plt.ylabel('Impact force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(impactForce2['time'], impactForce2['iFx'])
# plt.title('Right wall force')
# plt.xlabel('Time (s)')
# plt.ylabel('Impact force (kip)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(diaphragmForce1['time'], diaphragmForce1['iMz'])
# plt.title('Diaphragm 1 moment')
# plt.xlabel('Time (s)')
# plt.ylabel('Moment Z (kip)')
# plt.grid(True)

# # Outer column hysteresis
# fig = plt.figure()
# plt.plot(isolDisp['isol1'], force1Normalize)
# plt.title('Isolator 1 hysteresis')
# plt.xlabel('Displ (in)')
# plt.ylabel('V/N')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol1'], wholeLayerNormalize)
# plt.title('Isolator layer hysteresis')
# plt.xlabel('Displ (in)')
# plt.ylabel('V/N')
# plt.grid(True)

# All hystereses
fig = plt.figure()
plt.plot(isolDisp['isol1'], force1Normalize)
plt.title('Isolator hystereses')
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

plt.plot(isolDisp['isol2'], force2Normalize)
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

plt.plot(isolDisp['isolLC'], forceLCNormalize)
plt.xlabel('Displ (in)')
plt.ylabel('V/N')
plt.grid(True)

# Displacement history
fig = plt.figure()
plt.plot(isolDisp['time'], isolDisp['isol1'])
plt.title('Bearing 1 disp history')
plt.xlabel('Time (s)')
plt.ylabel('Displ (in)')
plt.grid(True)

fig = plt.figure()
plt.plot(story0Acc['time'], story0Acc['isol1'])
plt.title('Isol layer accel history')
plt.xlabel('Time (s)')
plt.ylabel('Accel (in/s2)')
plt.grid(True)

fig = plt.figure()
plt.plot(story1Acc['time'], story1Acc['isol1'])
plt.title('Story 1 accel history')
plt.xlabel('Time (s)')
plt.ylabel('Accel (in/s2)')
plt.grid(True)

# # Displacement history
# fig = plt.figure()
# plt.plot(isolDisp['time'], isolDisp['isol4'])
# plt.title('Bearing 4 disp history')
# plt.xlabel('Time (s)')
# plt.ylabel('Displ (in)')
# plt.grid(True)

# # Vertical displacement
# fig = plt.figure()
# plt.plot(isolDisp['time'], isolVert['isol1'])
# plt.title('Bearing vertical disp history')
# plt.xlabel('Time (s)')
# plt.ylabel('Displ z (in)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isolVert['isol2'])
# plt.xlabel('Time (s)')
# plt.ylabel('Displ z (in)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isolVert['isol3'])
# plt.xlabel('Time (s)')
# plt.ylabel('Displ z (in)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isolVert['isol4'])
# plt.xlabel('Time (s)')
# plt.ylabel('Displ z (in)')
# plt.grid(True)

# Drift history
fig = plt.figure()
plt.plot(isolDisp['time'], story1DriftOuter)
plt.title('Story 1 outer drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)

fig = plt.figure()
plt.plot(isolDisp['time'], story1DriftInner)
plt.title('Story 1 inner drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)

fig = plt.figure()
plt.plot(isolDisp['time'], story3DriftOuter)
plt.title('Story 3 outer drift history')
plt.xlabel('Time (s)')
plt.ylabel('Drift ratio')
plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['time'], story2DriftOuter)
# plt.title('Story 2 outer drift history')
# plt.xlabel('Time (s)')
# plt.ylabel('Drift ratio')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['time'], story3DriftOuter)
# plt.title('Story 3 outer drift history')
# plt.xlabel('Time (s)')
# plt.ylabel('Drift ratio')
# plt.grid(True)

# # Axial force
# fig = plt.figure()
# plt.plot(diaphragmForce1['time'], diaphragmForce1['iFx'])
# plt.title('Diaphragm 1 axial')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (kip)')
# plt.grid(True)


# Rotation history
# fig = plt.figure()
# plt.plot(isolDisp['isol1'], isolRot['isol1'])
# plt.title('Bearing rotation history, outer')
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol2'], isolRot['isol2'])
# plt.title('Bearing rotation history, inner')
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# plt.plot(isolDisp['isol3'], isolRot['isol3'])
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# plt.plot(isolDisp['isol4'], isolRot['isol4'])
# plt.xlabel('Displ x (in)')
# plt.ylabel('Rotation (in/in)')
# plt.grid(True)

# Axial force history

# fig = plt.figure()
# plt.plot(isolDisp['time'], isol1Force['iFx'])
# plt.title('Bearing outer axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isol4Force['iFx'])
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['time'], isol2Force['iFx'])
# plt.title('Bearing inner axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isol3Force['iFx'])
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# plt.plot(isolDisp['time'], isolLCForce['iFx'])
# plt.title('Bearing LC axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['time'], sumAxial)
# plt.title('Total axial force')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# Shear force history
# fig = plt.figure()
# plt.plot(outercolForce['time'], outercolForce['iFy'])
# plt.title('Column shear force, outer')
# plt.xlabel('Time (s)')
# plt.ylabel('Shear force (k)')
# plt.grid(True)


# fig = plt.figure()
# plt.plot(outercolForce['time'], outercolForce['iFx'])
# plt.title('Column axial force, outer')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(innercolForce['time'], innercolForce['iFy'])
# plt.title('Column shear force, inner')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(innercolForce['time'], innercolForce['iFx'])
# plt.title('Column axial force, inner')
# plt.xlabel('Time (s)')
# plt.ylabel('Axial force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol1'], isol1Force['iFy'])
# plt.title('Bearing 1 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol2'], isol2Force['iFy'])
# plt.title('Bearing 2 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol3'], isol3Force['iFy'])
# plt.title('Bearing 3 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(isolDisp['isol4'], isol4Force['iFy'])
# plt.title('Bearing 4 shear force')
# plt.xlabel('Displ (in)')
# plt.ylabel('shear force (k)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(beamRot['rot'], -outerbeamForce['jMz'])
# plt.title('Outer beam moment curvature')
# plt.xlabel('Curvature')
# plt.ylabel('Moment')
# plt.grid(True)