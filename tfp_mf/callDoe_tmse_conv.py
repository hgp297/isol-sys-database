############################################################################
#               Call DoE function

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: June 2021

# Description:  Script performs DoE by taking point suggestions from MMSE method
# via Matlab's GPML and carries out the OpenSeesPy experiment for the structure

# Open issues:  (1) Hardcoded to check for convergence of gap ratio for 3 values
# of Ry, holding Tm and zeta constant

############################################################################

################################################################################
#              Import and start
################################################################################

import matlab.engine
import numpy as np
import pandas as pd
import runControl
import LHS

eng = matlab.engine.start_matlab()

################################################################################
#              Utility functions
################################################################################

def cleanDat(oldDf):
    # remove excessive scaling
    newDf = oldDf[oldDf.GMScale <= 20]
    
    newDf = newDf.astype({"collapseDrift1": int,
                          "collapseDrift2": int,
                          "collapseDrift3": int,
                          "serviceDrift1": int,
                          "serviceDrift2": int,
                          "serviceDrift3": int,})

    # collapsed
    newDf['collapsed'] = (newDf['collapseDrift1'] | newDf['collapseDrift2']) | \
        newDf['collapseDrift3']
    newDf.loc[newDf.collapsed == 0, 'collapsed'] = -1
    newDf['collapsed'] = newDf['collapsed'].astype(float)
    
    # service lvl
    newDf['serviceFailure'] = (newDf['serviceDrift1'] | newDf['serviceDrift2']) | \
        newDf['serviceDrift3']
    newDf.loc[newDf.serviceFailure == 0, 'serviceFailure'] = -1
    newDf['serviceFailure'] = newDf['serviceFailure'].astype(float)

    # get Bm
    g = 386.4
    zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    BmRef   = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
    newDf['Bm'] = np.interp(newDf['zetaM'], zetaRef, BmRef)

    # expected Dm from design spectrum
    newDf['DesignDm'] = g*newDf['S1']*newDf['Tm']/(4*np.pi**2*newDf['Bm'])

    # expected Dm from ground motion
    newDf['earthquakeDm'] = g*(newDf['GMSTm']/newDf['Bm'])*(newDf['Tm']**2)/(4*np.pi**2)

    # back calculate moatAmpli
    if 'moatAmpli' not in newDf:
        newDf['moatAmpli'] = newDf['moatGap']/newDf['earthquakeDm']

    # gapRatio, recalculated for true rounded up gap
    newDf['gapRatio'] = (newDf['moatGap']*4*np.pi**2)/ \
        (g*(newDf['GMSTm']/newDf['Bm'])*newDf['Tm']**2)

    # TmRatio
    newDf['Ss'] = 2.2815
    newDf['Tshort'] = newDf['S1']/newDf['Ss']
    # newDf['TmRatio'] = newDf['Tm']/newDf['Tshort']

    return(newDf)

################################################################################
#              Main function
################################################################################


def runDoE(probTarget, baseDatStr, batch_size=10, tol=0.05, maxIter=600):

    databasePath = './data/'
    databaseFile = baseDatStr

    unfilteredData = pd.read_csv(databasePath+databaseFile)

    # write filtered data to csv for Matlab
    # make sure to write new cleanDat for the original data set
    # modified definitions of gapRatio and moatAmpli
    filteredData = cleanDat(unfilteredData)
    filteredData.to_csv(databasePath+'isolDatClean_addl_TFP_rand.csv', index=False)

    startFile = 'isolDOEStart_addl_TFP_rand.csv'
    workingFile = 'isolData_addl_TFP_rand.csv'

    inputPath = './inputs/'
    inputFile = 'bearingInput.csv'

    # start from clean data
    # with all parameters calculated, filtered for scale <= 20
    cleanData = pd.read_csv(databasePath+'isolDatClean_addl_TFP_rand.csv')

    # randomly sample points as starting point
    ###########################################
    # CHANGED HERE
    ninit = len(cleanData)
    ###########################################
    workingData = cleanData.sample(n=ninit, random_state=985)
    workingData.to_csv(databasePath+startFile, index=False)
    workingData.to_csv(databasePath+workingFile, index=False)

    # add more points as DoE
    inputVariables, inputValues = LHS.generateInputs(maxIter)

    # initialize new results
    newRes = None

    # for each input sets, write input files
    batch_idx = 1
    batch_no = 1
    
    # get initial gap
    fixTm = 3.0
    fixZeta = 0.15
    nextDouble, hyp_mean, hyp_cov, mu, sd = eng.doe_tmse(databasePath+workingFile, 
                                                 probTarget, nargout=5)
    gapOld1 = eng.gap_conv(databasePath+workingFile, probTarget, hyp_mean, hyp_cov,
                          fixTm, 1.0, fixZeta, mu, sd)
    gapOld2 = eng.gap_conv(databasePath+workingFile, probTarget, hyp_mean, hyp_cov,
                          fixTm, 2.0, fixZeta, mu, sd)
    gapOldHalf = eng.gap_conv(databasePath+workingFile, probTarget, hyp_mean, hyp_cov,
                          fixTm, 0.5, fixZeta, mu, sd)
    
    gapOld = np.array([gapOldHalf, gapOld1, gapOld2])
    
    gapList = np.array([])
    gapList = np.append(gapList, gapOld)
    
    for index, row in enumerate(inputValues):
        print('The run index is ' + str(index) + '.')
        print('The batch index is ' + str(batch_idx) + '.')
        
        if (batch_idx % (batch_size+1) == 0):
            gapNew1 = eng.gap_conv(databasePath+workingFile, probTarget, 
                                  hyp_mean, hyp_cov, fixTm, 1.0, fixZeta, mu, sd)
            gapNew2 = eng.gap_conv(databasePath+workingFile, probTarget, 
                                  hyp_mean, hyp_cov, fixTm, 2.0, fixZeta, mu, sd)
            gapNewHalf = eng.gap_conv(databasePath+workingFile, probTarget, 
                                  hyp_mean, hyp_cov, fixTm, 0.5, fixZeta, mu, sd)
            
            gapNew = np.array([gapNewHalf, gapNew1, gapNew2])
            gapList = np.vstack([gapList, gapNew])
            err = abs(gapOld - gapNew)/gapOld
            tf = err < np.array([tol, tol, tol])
            batch_no += 1
            if np.all(tf):
                print('Convergence reached at. Gap ratio: ' + str(gapNew))
                print('Number of added points: ' + str((batch_idx-1)*(batch_no-1)))
                return(workingData, gapList)
            else:
                pass
            gapOld = gapNew
            batch_idx = 1
            print('Convergence not reached yet. Resetting batch index to 1')
        else:
            pass

        # write LHS half into a dataframe
        lhsVals = [float(val) for val in row]
        inputDf = pd.DataFrame(list(zip(inputVariables, lhsVals)), 
            columns=['variable','value'])

        # get DoE point from GPML and convert to list
        nextDouble, hyp_mean, hyp_cov, mu, sd = eng.doe_tmse(databasePath+workingFile,
                                                     probTarget, nargout=5)
        nextArr = np.array(nextDouble).transpose()
        doeVals = [float(val) for val in nextArr]

        # write DoE half into dataframe
        # currently set x = [gapRatio, TmRatio, zetaM, Ry]
        doeVars = ['gapRatio', 'Tm', 'zetaM', 'RI']
        doeDf = pd.DataFrame(list(zip(doeVars, doeVals)),
            columns=['variable','value'])
        
        # merge both dataframes
        inputDf = inputDf.append(doeDf)

        # write inputDf to csv
        inputDf.to_csv(inputPath+inputFile, index=False)

        # run opsPy
        try:
            runResult = runControl.opsExperiment(inputPath+inputFile)
        except ValueError:
            print('Bearing solver returned negative friction coefficients. Skipping...')
            continue
        except IndexError:
            print('SCWB check failed, no shape exists for design. Skipping...')
            continue

        # skip run if excessively scaled
        if (type(runResult) == bool):
            print('Scale factor exceeded 20.0.')
            continue
        
        # if run succeeded, increase the batch index
        batch_idx += 1

        # clean new df, attach to existing data
        newRes = cleanDat(runResult)
        workingData = workingData.append(newRes, sort=True)
        workingData.to_csv(databasePath+workingFile, index=False)

    print('Did not reach convergence after max runs.')
    return(workingData, gapList)

def write_to_csv(the_list, filename):
    import csv
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(the_list)
################################################################################
#              Main run
################################################################################

probTarget = 0.1
databaseFile = 'addl_TFP_loading_jse_size.csv'
doeData, gapHist = runDoE(probTarget, databaseFile, maxIter=600, tol=0.02)
endFile = 'doe_conv_addl_TFP_rand_10.csv'
gapFile = 'gap_hist_addl_TFP_rand_10.csv'
databasePath = './data/'
doeData.to_csv(databasePath+endFile, index=False)
write_to_csv(list(gapHist), databasePath+gapFile)

probTarget = 0.05
doeData, gapHist = runDoE(probTarget, databaseFile, maxIter=600, tol=0.02)
endFile = 'doe_conv_addl_TFP_rand_5.csv'
gapFile = 'gap_hist_addl_TFP_rand_5.csv'
databasePath = './data/'
doeData.to_csv(databasePath+endFile, index=False)
write_to_csv(list(gapHist), databasePath+gapFile)

probTarget = 0.025
doeData, gapHist = runDoE(probTarget, databaseFile, maxIter=600, tol=0.02)
endFile = 'doe_conv_addl_TFP_rand_2_5.csv'
gapFile = 'gap_hist_addl_TFP_rand_2_5.csv'
databasePath = './data/'
doeData.to_csv(databasePath+endFile, index=False)
write_to_csv(list(gapHist), databasePath+gapFile)









