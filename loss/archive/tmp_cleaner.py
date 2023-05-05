import pandas as pd
import numpy as np

def cleanDat(oldDf, remove_failed_runs=False):
    # remove excessive scaling
    newDf = oldDf[oldDf.GMScale <= 20]
    
    # remove failed runs
    if remove_failed_runs == True:
        newDf = newDf[newDf.runFailed == 0]
    else:
        pass

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

databasePath = './sessionOut/'
databaseFile = 'sessionSummary.csv'

unfilteredData = pd.read_csv(databasePath+databaseFile)

# write filtered data to csv for Matlab
# make sure to write new cleanDat for the original data set
# modified definitions of gapRatio and moatAmpli
filteredData = cleanDat(unfilteredData)
filteredData.to_csv(databasePath+'addl_TFP_loading.csv', index=False)