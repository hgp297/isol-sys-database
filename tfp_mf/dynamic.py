############################################################################
#               Earthquake analysis

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: May 2020

# Description:  Script performs dynamic analysis on OpenSeesPy model

# Open issues:  (1) 

############################################################################


# import OpenSees and libraries
from math import floor
import openseespy.opensees as ops

############################################################################
#              Build model
############################################################################

import build as bm

def runGM(gmFilename, gmDefScale, dtTransient, 
          GMDir= "./ground_motions/PEERNGARecords_Unscaled/",
          dataDir='./outputs/'):

    # build model
    bm.build()

    (w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3) = bm.giveLoads()

    ############################################################################
    #              Clear previous runs
    ############################################################################

    gravSeriesTag   = 1
    gravPatternTag  = 1

    eqSeriesTag     = 100
    eqPatternTag    = 400

    ############################################################################
    #              Loading and analysis
    ############################################################################

    from ReadRecord import ReadRecord

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


    ############################################################################
    #                       Eigenvalue Analysis
    ############################################################################

    # fix base for Tfb
    bm.refix(11, "fix")
    bm.refix(12, "fix")
    bm.refix(13, "fix")
    bm.refix(14, "fix")
    # bm.refix(15, "fix")

    nEigenJ     = 3;                    # mode j = 3
    lambdaN     = ops.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
    lambda1     = lambdaN[0];           # eigenvalue mode i = 1
    omega1      = lambda1**(0.5)    # w1 (1st mode circular frequency)
    Tfb         = 2*3.1415/omega1      # 1st mode period of the structure
    print("Tfb = ", Tfb, " s")          # display the first mode period in the command window

    # unfix base
    bm.refix(11, "unfix")
    bm.refix(12, "unfix")
    bm.refix(13, "unfix")
    bm.refix(14, "unfix")
    # bm.refix(15, "unfix")

    # Rayleigh damping to the superstructure only
    regTag      = 80
    # zetaTarget  = 0.05
    # bm.provideSuperDamping(regTag, omega1, zetaTarget)
    bm.provideSuperDamping(regTag, lambdaN, zetai=0.05, zetaj=0.02, modes=[1,3])

    ############################################################################
    #              Recorders
    ############################################################################

    ops.printModel('-file', dataDir+'model.out')
    ops.recorder('Node', '-file', dataDir+'isolDisp.csv', '-time',
        '-closeOnWrite', '-node', 11, 12, 13, 14, 15, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', dataDir+'isolVert.csv', '-time',
        '-closeOnWrite', '-node', 11, 12, 13, 14, 15, '-dof', 3, 'disp')
    ops.recorder('Node', '-file', dataDir+'isolRot.csv', '-time',
        '-closeOnWrite', '-node', 11, 12, 13, 14, 15, '-dof', 5, 'disp')

    ops.recorder('Node', '-file', dataDir+'story1Disp.csv','-time',
        '-closeOnWrite', '-node', 21, 22, 23, 24, 25, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', dataDir+'story2Disp.csv', '-time',
        '-closeOnWrite', '-node', 31, 32, 33, 34, 35, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', dataDir+'story3Disp.csv', '-time',
        '-closeOnWrite', '-node', 41, 42, 43, 44, 45, '-dof', 1, 'disp')

    ops.recorder('Element', '-file', dataDir+'isol1Force.csv',
        '-time', '-closeOnWrite', '-ele', 51, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol2Force.csv',
        '-time', '-closeOnWrite', '-ele', 52, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol3Force.csv',
        '-time', '-closeOnWrite', '-ele', 53, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isol4Force.csv',
        '-time', '-closeOnWrite', '-ele', 54, 'localForce')
    ops.recorder('Element', '-file', dataDir+'isolLCForce.csv',
        '-time', '-closeOnWrite', '-ele', 55, 'localForce')

    ops.recorder('Element', '-file', dataDir+'colForce1.csv',
        '-time', '-closeOnWrite', '-ele', 111, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce2.csv',
        '-time', '-closeOnWrite', '-ele', 112, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce3.csv',
        '-time', '-closeOnWrite', '-ele', 113, 'localForce')
    ops.recorder('Element', '-file', dataDir+'colForce4.csv',
        '-time', '-closeOnWrite', '-ele', 114, 'localForce')

    # ops.recorder('Element', '-file', dataDir+'colForce5.csv', '-time', '-closeOnWrite', '-ele', 121, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'colForce6.csv', '-time', '-closeOnWrite', '-ele', 122, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'colForce7.csv', '-time', '-closeOnWrite', '-ele', 123, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'colForce8.csv', '-time', '-closeOnWrite', '-ele', 124, 'localForce')

    # ops.recorder('Element', '-file', dataDir+'beamForce1.csv', '-time', '-closeOnWrite', '-ele', 221, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'beamForce2.csv', '-time', '-closeOnWrite', '-ele', 222, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'beamForce3.csv', '-time', '-closeOnWrite', '-ele', 223, 'localForce')

    # ops.recorder('Element', '-file', dataDir+'beamForce4.csv', '-time', '-closeOnWrite', '-ele', 231, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'beamForce5.csv', '-time', '-closeOnWrite', '-ele', 232, 'localForce')
    # ops.recorder('Element', '-file', dataDir+'beamForce6.csv', '-time', '-closeOnWrite', '-ele', 233, 'localForce')

    ops.recorder('Element', '-file', dataDir+'diaphragmForce1.csv', 
                 '-time', '-closeOnWrite', '-ele', 611, 'localForce')
    ops.recorder('Element', '-file', dataDir+'diaphragmForce2.csv', 
                 '-time', '-closeOnWrite', '-ele', 612, 'localForce')
    ops.recorder('Element', '-file', dataDir+'diaphragmForce3.csv', 
                 '-time', '-closeOnWrite', '-ele', 613, 'localForce')

    ops.recorder('Element', '-file', dataDir+'impactForceLeft.csv', 
                 '-time', '-closeOnWrite', '-ele', 881, 'basicForce')
    ops.recorder('Element', '-file', dataDir+'impactForceRight.csv', 
                 '-time', '-closeOnWrite', '-ele', 884, 'basicForce')

    ops.recorder('Element', '-file', dataDir+'impactDispLeft.csv', 
                 '-time', '-closeOnWrite', '-ele', 881, 'basicDeformation')
    ops.recorder('Element', '-file', dataDir+'impactDispRight.csv', 
                 '-time', '-closeOnWrite', '-ele', 884, 'basicDeformation')

    ############################################################################
    #                       Dynamic analysis
    ############################################################################

    ops.wipeAnalysis()

    # Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
    GMDirection     = 1                             # ground-motion direction
    GMFile          = gmFilename                    # ground motion file name passed in
    GMFactor        = gmDefScale
    
    print('Current ground motion: ', gmFilename)

    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')

    tolDynamic          = 1e-3      # Convergence Test: tolerance
    maxIterDynamic      = 500        # Convergence Test: maximum number of iterations that will be performed before "failure to converge" is returned
    printFlagDynamic    = 0         # Convergence Test: flag used to print information on convergence (optional)        # 1: print information on each step; 
    testTypeDynamic     = 'NormDispIncr'
    ops.test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)
    # algorithmTypeDynamic    = 'Broyden'
    # ops.algorithm(algorithmTypeDynamic, 8)
    algorithmTypeDynamic    = 'Newton'
    ops.algorithm(algorithmTypeDynamic)
    newmarkGamma    = 0.5           # Newmark-integrator gamma parameter (also HHT)
    newmarkBeta     = 0.25          # Newmark-integrator beta parameter
    ops.integrator('Newmark', newmarkGamma, newmarkBeta)
    ops.analysis('Transient')

    #  ---------------------------------    perform Dynamic Ground-Motion Analysis
    # the following commands are unique to the Uniform Earthquake excitation

    # Uniform EXCITATION: acceleration input
    inFile          = GMDir + GMFile + '.AT2'
    outFile         = GMDir + GMFile + '.g3'        # set variable holding new filename (PEER files have .at2/dt2 extension)

    dt, nPts        = ReadRecord(inFile, outFile)   # call procedure to convert the ground-motion file
    g               = 386.4
    GMfatt          = g*GMFactor                    # data in input file is in g Unifts -- ACCELERATION TH

    ops.timeSeries('Path', eqSeriesTag, '-dt', dt, '-filePath', outFile, '-factor', GMfatt)     # time series information
    ops.pattern('UniformExcitation', eqPatternTag, GMDirection, '-accel', eqSeriesTag)          # create uniform excitation

    # set recorder for absolute acceleration (requires time series defined)
    ops.recorder('Node', '-file', dataDir+'story0Acc.csv',
        '-timeSeries', eqSeriesTag, '-time', '-closeOnWrite',
        '-node', 11, 12, 13, 14, 15, '-dof', 1, 'accel')
    ops.recorder('Node', '-file', dataDir+'story1Acc.csv',
        '-timeSeries', eqSeriesTag, '-time', '-closeOnWrite',
        '-node', 21, 22, 23, 24, 25, '-dof', 1, 'accel')
    ops.recorder('Node', '-file', dataDir+'story2Acc.csv',
        '-timeSeries', eqSeriesTag, '-time', '-closeOnWrite',
        '-node', 31, 32, 33, 34, 35, '-dof', 1, 'accel')
    ops.recorder('Node', '-file', dataDir+'story3Acc.csv',
        '-timeSeries', eqSeriesTag, '-time', '-closeOnWrite',
        '-node', 41, 42, 43, 44, 45, '-dof', 1, 'accel')

    ops.recorder('Node', '-file', dataDir+'story0Vel.csv',
        '-time', '-closeOnWrite', '-node', 11, 12, 13, 14, 15, '-dof', 1, 'vel')
    ops.recorder('Node', '-file', dataDir+'story1Vel.csv',
        '-time', '-closeOnWrite', '-node', 21, 22, 23, 24, 25, '-dof', 1, 'vel')
    ops.recorder('Node', '-file', dataDir+'story2Vel.csv',
        '-time', '-closeOnWrite', '-node', 31, 32, 33, 34, 35, '-dof', 1, 'vel')
    ops.recorder('Node', '-file', dataDir+'story3Vel.csv',
        '-time', '-closeOnWrite', '-node', 41, 42, 43, 44, 45, '-dof', 1, 'vel')

    # set up ground-motion-analysis parameters
    sec             = 1.0                      
    TmaxAnalysis    = 60.0*sec

    Nsteps          = floor(TmaxAnalysis/dtTransient)
    ok              = ops.analyze(Nsteps, dtTransient)   # actually perform analysis; returns ok=0 if analysis was successful

    if ok != 0:
        ok              = 0
        controlTime     = ops.getTime()
        print("Convergence issues at time: ", controlTime)
        while (controlTime < TmaxAnalysis) and (ok == 0):
            controlTime     = ops.getTime()
            ok          = ops.analyze(1, dtTransient)
            if ok != 0:
                print("Trying Newton with Initial Tangent...")
                ops.algorithm('Newton', '-initial')
                ok = ops.analyze(1, dtTransient)
                if ok == 0:
                    print("That worked. Back to Newton")
                ops.algorithm(algorithmTypeDynamic)
            if ok != 0:
                print("Trying Newton with line search ...")
                ops.algorithm('NewtonLineSearch')
                ok = ops.analyze(1, dtTransient)
                if ok == 0:
                    print("That worked. Back to Newton")
                ops.algorithm(algorithmTypeDynamic)


    print('Ground motion done. End time:', ops.getTime())

    ops.wipe()

    return(ok, Tfb, GMFactor)