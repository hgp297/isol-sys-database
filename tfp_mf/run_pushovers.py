############################################################################
#               Pushover analyses

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: May 2023

# Description:  Performs pushover analysis of specified structures

# Open issues:  (1) 

############################################################################

# system commands
import os, os.path
import glob
import shutil

############################################################################
#              File management
############################################################################

# remove existing results
# explanation here: https://stackoverflow.com/a/31989328
def remove_thing(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def empty_directory(path):
    for i in glob.glob(os.path.join(path, '*')):
        remove_thing(i)



############################################################################
#              Perform runs
############################################################################
import pandas as pd

def run_pushover(inputStr, pushoverStr, max_drift_ratio=0.1):
    
    empty_directory('pushover')
    
    # read in params
    parCsv  = pd.read_csv(inputStr, index_col=None, header=0)
    param   = dict(zip(parCsv.variable, parCsv.value))
    
    # write the current target design to the main input file
    import csv
    with open('./inputs/bearingInput.csv', 'w',newline='', encoding='utf-8') as file:  
        writer = csv.writer(file)
        writer.writerow(['variable', 'value'])
        for key, value in param.items():
            writer.writerow([key, value])
            
    # build the structure
    import buildModel as bm
    import numpy as np
    
    from openseespy import opensees as ops

    # build model
    bm.build()
    
    # run the pushover
    (w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3) = bm.giveLoads()

    ############################################################################
    #              Clear previous runs
    ############################################################################

    gravSeriesTag   = 1
    gravPatternTag  = 1

    pushoverPatternTag  = 400
    # ------------------------------
    # Loading: gravity
    # ------------------------------

    # create TimeSeries
    ops.timeSeries("Linear", gravSeriesTag)

    # create plain load pattern
    # command:  pattern('Plain', tag, timeSeriesTag)
    # command:  eleLoad('-ele', *eleTags, '-range', eleTag1, eleTag2, '-type', '-beamUniform', Wy, Wz=0.0, Wx=0.0, '-beamPoint', Py, Pz=0.0, xL, Px=0.0, '-beamThermal', *tempPts)
    # command:  load(nodeTag, *loadValues)
    ops.pattern('Plain', gravPatternTag, gravSeriesTag)
    ops.eleLoad('-ele', 611, 612, 613, '-type', '-beamUniform', -w0, 0)
    ops.eleLoad('-ele', 221, 222, 223, '-type', '-beamUniform', -w1, 0)
    ops.eleLoad('-ele', 231, 232, 233, '-type', '-beamUniform', -w2, 0)
    ops.eleLoad('-ele', 241, 242, 243, '-type', '-beamUniform', -w3, 0)

    ops.load(15, 0, 0, -pLc0, 0, 0, 0)
    ops.load(25, 0, 0, -pLc1, 0, 0, 0)
    ops.load(35, 0, 0, -pLc2, 0, 0, 0)
    ops.load(45, 0, 0, -pLc3, 0, 0, 0)

    # load right above isolation layer to increase stiffness to half-building for TFP
    # line load accounts for 15ft of tributary, we add an additional 30ft with the 2x
    ft = 12.0
    pOuter = (w0 + w1 + w2 + w3)*15*ft*2
    pInner = (w0 + w1 + w2 + w3)*30*ft*2

    ops.load(11, 0, 0, -pOuter, 0, 0, 0)
    ops.load(12, 0, 0, -pInner, 0, 0, 0)
    ops.load(13, 0, 0, -pInner, 0, 0, 0)
    ops.load(14, 0, 0, -pOuter, 0, 0, 0)
    
    # ------------------------------
    # Start of analysis generation: gravity
    # ------------------------------

    nStepGravity    = 10                    # apply gravity in 10 steps
    tol             = 1e-5                  # convergence tolerance for test
    dGravity        = 1/nStepGravity        # first load increment

    ops.system("BandGeneral")                   # create system of equation (SOE)
    ops.test("NormDispIncr", tol, 15)           # determine if convergence has been achieved at the end of an iteration step
    ops.numberer("RCM")                         # create DOF numberer
    ops.constraints("Plain")                    # create constraint handler
    ops.integrator("LoadControl", dGravity)     # determine the next time step for an analysis, create integration scheme (steps of 1/10)
    ops.algorithm("Newton")                     # create solution algorithm
    ops.analysis("Static")                      # create analysis object
    ops.analyze(nStepGravity)                   # perform the analysis in N steps

    print("Gravity analysis complete!")

    ops.loadConst('-time', 0.0)
    
    # Set lateral load pattern with a Linear TimeSeries
    ops.pattern('Plain', pushoverPatternTag, gravSeriesTag)

    # Create nodal loads at nodes 3 & 4
    #    nd    FX  FY  FZ MX MY MZ
    ops.load(21, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.load(31, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.load(41, 241.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    #----------------------------------------------------
    # Start of modifications to analysis for push over
    # ----------------------------------------------------
    ops.wipeAnalysis()

    # units: in, kip, s
    # dimensions
    inch    = 1.0
    ft      = 12.0*inch

    hsx     = np.array([13.0*ft, 13.0*ft, 13.0*ft])  
    # Set some parameters
    dU = 0.005  # Displacement increment

    # Change the integration scheme to be displacement control
    #                             node dof init Jd min max
    ops.integrator('DisplacementControl', 41, 1, dU, 1, dU, dU)
    ############################################################################
    #              Recorders
    ############################################################################

    dataDir         = './outputs/pushover/'          # output folder
    #file mkdir $dataDir; # create output folder

    ops.printModel('-file', dataDir+'model.out')
    
    ops.recorder('Node', '-file', dataDir+'ground_rxn.csv', 
                 '-time', '-closeOnWrite', '-node', 
                 81, 1, 2, 3, 4, 84, '-dof', 1, 'reaction')
    
    ops.recorder('Node', '-file', dataDir+'isolDisp.csv', 
                 '-time', '-closeOnWrite', '-node', 
                 11, 12, 13, 14, 15, '-dof', 1, 'disp')
    
    ops.recorder('Node', '-file', dataDir+'story1Disp.csv', 
                 '-time', '-closeOnWrite', '-node', 
                 21, 22, 23, 24, 25, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', dataDir+'story2Disp.csv', 
                 '-time', '-closeOnWrite', '-node', 
                 31, 32, 33, 34, 35, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', dataDir+'story3Disp.csv', 
                 '-time', '-closeOnWrite', '-node', 
                 41, 42, 43, 44, 45, '-dof', 1, 'disp')

    ops.recorder('Element', '-file', dataDir+'isol1Force.csv', 
                 '-time', '-closeOnWrite', '-ele', 51, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol2Force.csv', 
                 '-time', '-closeOnWrite', '-ele', 52, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol3Force.csv', 
                 '-time', '-closeOnWrite', '-ele', 53, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol4Force.csv', 
                 '-time', '-closeOnWrite', '-ele', 54, 'localForce')

    ops.recorder('Element', '-file', dataDir+'colForce1.csv', 
                 '-time', '-closeOnWrite', '-ele', 111, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce2.csv', 
                 '-time', '-closeOnWrite', '-ele', 112, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce3.csv', 
                 '-time', '-closeOnWrite', '-ele', 113, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce4.csv', 
                 '-time', '-closeOnWrite', '-ele', 114, 'localForce')

    # ------------------------------
    # Finally perform the analysis
    # ------------------------------

    # Set some parameters
    maxU = max_drift_ratio*hsx.sum()  # Max displacement
    nSteps = int(round(maxU/dU))
    ok = 0

    # Create the system of equation, a sparse solver with partial pivoting
    ops.system('BandGeneral')

    # Create the constraint handler, the transformation method
    ops.constraints('Plain')

    # Create the DOF numberer, the reverse Cuthill-McKee algorithm
    ops.numberer('RCM')

    ops.test('NormUnbalance', 1.0e-3, 4000)
    ops.algorithm('Newton')
    # Create the analysis object
    ops.analysis('Static')

    ok = ops.analyze(nSteps)
    # for gravity analysis, load control is fine, 0.1 is the load factor increment (http://opensees.berkeley.edu/wiki/index.php/Load_Control)

    testList = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 
                4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 
                6: 'NormUnbalance'}
    algoList = {1:'KrylovNewton', 2: 'SecantNewton' , 
                4: 'RaphsonNewton',5: 'PeriodicNewton', 
                6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}
                
    for i in testList:
        for j in algoList:

            if ok != 0:
                if j < 4:
                    ops.algorithm(algoList[j], '-initial')
                    
                else:
                    ops.algorithm(algoList[j])
                    
                ops.test(testList[i], 1e-3, 1000)
                ok = ops.analyze(nSteps)                            
                print(testList[i], algoList[j], ok)             
                if ok == 0:
                    break
            else:
                continue
            
    print('Pushover complete!')
    
    # read in the output files
    dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']
    forceColumns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 
                    'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']
    
    isol1Force = pd.read_csv(dataDir+'isol1Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol2Force = pd.read_csv(dataDir+'isol2Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol3Force = pd.read_csv(dataDir+'isol3Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol4Force = pd.read_csv(dataDir+'isol4Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    
    story3Disp = pd.read_csv(dataDir+'story3Disp.csv', sep=' ', 
                             header=None, names=dispColumns)
    
    col1Force = pd.read_csv(dataDir+'colForce1.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col2Force = pd.read_csv(dataDir+'colForce2.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col3Force = pd.read_csv(dataDir+'colForce3.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col4Force = pd.read_csv(dataDir+'colForce4.csv', sep = ' ', 
                            header=None, names=forceColumns)
    
    sumAxial = (isol1Force['iFx'] + isol2Force['iFx'] +
                isol3Force['iFx'] + isol4Force['iFx'])
   

    baseShear = (col1Force['iFy'] + col2Force['iFy'] + 
                 col3Force['iFy'] + col4Force['iFy'])
    baseShearNormalize = baseShear/sumAxial
    
    # rewrite the output to pushover
    pushover_df = pd.DataFrame({'roof_disp': story3Disp['isol1'],
                                'base_shear_normalized': baseShearNormalize,
                                'base_shear': baseShear})
    pushover_df.to_csv('./outputs/pushover/'+pushoverStr, index=False)
    
    return()

def plot_pushover(data_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']
    forceColumns = ['time', 'iFx', 'iFy', 'iFz', 'iMx', 'iMy', 'iMz', 
                    'jFx', 'jFy', 'jFz', 'jMx', 'jMy', 'jMz']
    rxn_columns = ['time', 'left_wall', 'isol1', 'isol2', 
                   'isol3', 'isol4', 'right_wall']
    
    ground_reactions = pd.read_csv(data_dir+'ground_rxn.csv', sep=' ', 
                           header=None, names=rxn_columns)
    
    isolDisp = pd.read_csv(data_dir+'isolDisp.csv', sep=' ', 
                           header=None, names=dispColumns)
    
    isol1Force = pd.read_csv(data_dir+'isol1Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol2Force = pd.read_csv(data_dir+'isol2Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol3Force = pd.read_csv(data_dir+'isol3Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    isol4Force = pd.read_csv(data_dir+'isol4Force.csv', sep = ' ', 
                             header=None, names=forceColumns)
    
    story1Disp = pd.read_csv(data_dir+'story1Disp.csv', sep=' ', 
                             header=None, names=dispColumns)
    story2Disp = pd.read_csv(data_dir+'story2Disp.csv', sep=' ', 
                             header=None, names=dispColumns)
    story3Disp = pd.read_csv(data_dir+'story3Disp.csv', sep=' ', 
                             header=None, names=dispColumns)
    
    col1Force = pd.read_csv(data_dir+'colForce1.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col2Force = pd.read_csv(data_dir+'colForce2.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col3Force = pd.read_csv(data_dir+'colForce3.csv', sep = ' ', 
                            header=None, names=forceColumns)
    col4Force = pd.read_csv(data_dir+'colForce4.csv', sep = ' ', 
                            header=None, names=forceColumns)
    
    baseShear = (col1Force['iFy'] + col2Force['iFy'] + 
                  col3Force['iFy'] + col4Force['iFy'])
    
    baseShear_isol = -(isol1Force['iFy'] + isol2Force['iFy'] + 
                  isol4Force['iFy'] + isol3Force['iFy'])
    
    ground_rxn = -(ground_reactions['isol1'] + ground_reactions['isol2']+
                  ground_reactions['isol3'] + ground_reactions['isol4']+
                  ground_reactions['left_wall'] + ground_reactions['right_wall'])
    
    plt.close('all')
    
    plt.figure()
    plt.plot(ground_reactions['time'], ground_rxn)
    plt.title('Load history')
    plt.xlabel('Time step')
    plt.ylabel('Ground horizontal reaction')
    plt.grid(True)
    plt.show()
    
    # base shear vs roof
    plt.figure()
    plt.plot(story3Disp['isol1'], baseShear)
    plt.title('Pushover curve (roof)')
    plt.xlabel('Roof displacement (in)')
    plt.ylabel('Base shear')
    plt.grid(True)
    plt.show()
    
    # base shear vs roof
    bldg_drift = (story3Disp['isol1']-isolDisp['isol1'])/(39*12)
    plt.figure()
    plt.plot(bldg_drift, baseShear)
    plt.title('Pushover curve (superstructure only)')
    plt.xlabel('Building drift (roof - isolation)')
    plt.ylabel('Base shear')
    plt.grid(True)
    plt.show()
    
    # base shear vs isolator
    plt.figure()
    plt.plot(isolDisp['isol1'], baseShear_isol)
    plt.title('Pushover curve (isolator)')
    plt.xlabel('Isolator displacement (in)')
    plt.ylabel('Base shear (isol)')
    plt.grid(True)
    plt.show()
    
    drift_1    = (story1Disp['isol2'] - isolDisp['isol2'])/(13*12)
    drift_2    = (story2Disp['isol2'] - story1Disp['isol2'])/(13*12)
    drift_3    = (story3Disp['isol2'] - story2Disp['isol2'])/(13*12)
    
    # story drifts
    plt.figure()
    plt.plot(drift_1, baseShear)
    plt.plot(drift_2, baseShear)
    plt.plot(drift_3, baseShear)
    plt.title('Drift pushover')
    plt.ylabel('Base shear')
    plt.xlabel('Drift ratio')
    plt.grid(True)
    
    # story drifts
    plt.figure()
    plt.plot(drift_1, ground_rxn)
    plt.plot(drift_2, ground_rxn)
    plt.plot(drift_3, ground_rxn)
    plt.title('Drift pushover (ground reaction)')
    plt.ylabel('Base shear (ground rxn)')
    plt.xlabel('Drift ratio')
    plt.grid(True)
    
    # story drifts
    plt.figure()
    plt.plot(isolDisp['isol1'], drift_1)
    plt.title('Drift comparison')
    plt.xlabel('Isolator displacement (in)')
    plt.ylabel('First story drift')
    plt.grid(True)
    plt.show()
    
#%% main runs

inputString = './inputs/bearingInputVal_baseline.csv'
outputString = 'pushover_addl_baseline.csv'
run_pushover(inputString, outputString)
#%%

# inputString = './inputs/bearingInputVal2_5.csv'
# outputString = 'pushover_addl_2_5.csv'
# run_pushover(inputString, outputString, max_drift_ratio=0.15)

#%%
plot_pushover('./outputs/pushover/')

# inputString = './inputs/bearingInputVal5.csv'
# outputString = 'pushover_addl_5.csv'
# run_pushover(inputString, outputString)

# inputString = './inputs/bearingInputVal2_5.csv'
# outputString = 'pushover_addl_2_5.csv'
# run_pushover(inputString, outputString)