############################################################################
#               Ground motion selector

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: September 2020

# Description:  Script creates list of viable ground motions and scales from PEER search

# Open issues:  (1) Lengths of sections require specifications
#               (2) Manually specify how many of each EQ you want

############################################################################
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn', ignore SettingWithCopyWarning

############################################################################
#               Section specifiers

# Start: the row with the title of the section (before the headers)
# Length: the number of rows of content, without the headers

############################################################################

def cleanGMs(gmDir, resultsCSV, actualS1, T_fb, T_m,
             summaryStart=33, nSummary=100, scaledStart=144, 
             nScaled=111, unscaledStart=258, nUnscaled=111):

    # # remove all DT2 VT2 files
    # folder                = os.listdir(gmDir)

    # for item in folder:
    #   if item.endswith('.VT2') or item.endswith('.DT2'):
    #       os.remove(os.path.join(gmDir,item))

    # load in sections of the sheet
    summary             = pd.read_csv(gmDir+resultsCSV, 
        skiprows=summaryStart, nrows=nSummary)
    scaledSpectra       = pd.read_csv(gmDir+resultsCSV, 
        skiprows=scaledStart, nrows=nScaled)
    unscaledSpectra     = pd.read_csv(gmDir+resultsCSV, 
        skiprows=unscaledStart, nrows=nUnscaled)

    # Scale both Ss and S1
    # Create spectrum (Ss or S1/T)
    Ss          = 2.2815
    actualSs    = Ss
    Tshort      = actualS1/actualSs
    targetSpectrum  = scaledSpectra[['Period (sec)']]
    targetSpectrum['Target pSa (g)'] = np.where(targetSpectrum['Period (sec)'] < Tshort, 
        actualSs, actualS1/targetSpectrum['Period (sec)'])

    # calculate desired target spectrum average (0.2*Tm, 1.5*Tm)
    tLower                  = T_fb
    tUpper                  = 1.5*T_m

    # geometric mean from Eads et al. (2015)
    targetRange = targetSpectrum[targetSpectrum['Period (sec)'].between(tLower,
        tUpper)]['Target pSa (g)']
    targetAverage = targetRange.prod()**(1/targetRange.size)

    # get the spectrum average for the unscaled GM spectra
    unscaledH1s = unscaledSpectra.filter(regex=("-1 pSa \(g\)$"))       # only concerned about H1 spectra
    unscaledSpectraRange = unscaledH1s[targetSpectrum['Period (sec)'].between(tLower, tUpper)]
    unscaledAverages = unscaledSpectraRange.prod()**(1/len(unscaledSpectraRange.index))

    # determine scale factor to get unscaled to target
    scaleFactorAverage = targetAverage/unscaledAverages
    scaleFactorAverage = scaleFactorAverage.reset_index()
    scaleFactorAverage.columns = ['fullRSN', 'avgSpectrumScaleFactor']

    # rename back to old convention and merge with previous dataframe
    scaleFactorAverage[' Record Sequence Number'] = scaleFactorAverage['fullRSN'].str.extract('(\d+)')
    scaleFactorAverage = scaleFactorAverage.astype({' Record Sequence Number': int})
    summary = pd.merge(summary,
        scaleFactorAverage, 
        on=' Record Sequence Number').drop(columns=['fullRSN'])

    # grab only relevant columns
    summaryNames        = [' Record Sequence Number', 
                           'avgSpectrumScaleFactor',
                           ' Earthquake Name',
                           ' Lowest Useable Frequency (Hz)',
                           ' Horizontal-1 Acc. Filename']
    summarySmol         = summary[summaryNames]

    # Filter by lowest usable frequency
    TMax = tUpper
    freqMin = 1/TMax
    eligFreq = summarySmol[summarySmol[' Lowest Useable Frequency (Hz)'] < freqMin]

    # List unique earthquakes
    uniqEqs = pd.unique(eligFreq[' Earthquake Name'])
    finalGM = None

    # Select earthquakes that are least severely scaled
    for earthquake in uniqEqs:
        matchingEqs = eligFreq[eligFreq[' Earthquake Name'] == earthquake]
        matchingEqs['scaleDifference'] = abs(matchingEqs['avgSpectrumScaleFactor'] - 1.0)
        leastScaled = matchingEqs.sort_values(by=['scaleDifference']).iloc[:3] # take 3 least scaled ones

        if finalGM is None:
            GMHeaders = list(matchingEqs.columns)
            finalGM = pd.DataFrame(columns=GMHeaders)
        
        finalGM = pd.concat([leastScaled,finalGM], sort=False)
        finalGM[' Horizontal-1 Acc. Filename'] = finalGM[' Horizontal-1 Acc. Filename'].str.strip()

    finalGM = finalGM.reset_index()
    finalGM = finalGM.drop(columns=['index', 'scaleDifference'])
    finalGM.columns = ['RSN', 'scaleFactorSpecAvg', 'EQName', 'lowestFreq', 'filename']

    return(finalGM, targetAverage)

def getST(gmDir, resultsCSV, GMFile, scaleFactor, Tquery, 
          summaryStart=33, nSummary=100, unscaledStart=258, nUnscaled=111):

    import re

    # load in sections of the sheet
    summary = pd.read_csv(gmDir+resultsCSV,
        skiprows=summaryStart, nrows=nSummary)
    unscaledSpectra = pd.read_csv(gmDir+resultsCSV,
        skiprows=unscaledStart, nrows=nUnscaled)

    rsn                 = re.search('(\d+)', GMFile).group(1)
    gmUnscaledName      = 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
    gmSpectrum          = unscaledSpectra[['Period (sec)', gmUnscaledName]]
    gmSpectrum.columns  = ['Period', 'Sa']

    SaQueryUnscaled     = np.interp(Tquery, gmSpectrum.Period, gmSpectrum.Sa)
    SaQuery             = scaleFactor*SaQueryUnscaled
    return(SaQuery)



# Specify locations
if __name__ == '__main__':
    summaryStart        = 32
    summaryLength       = 133
    scaledStart         = 176
    scaledLength        = 111
    unscaledStart       = 290
    unscaledLength      = 111
    gmFolder            = './groundMotions/'
    PEERSummary         = 'combinedSearch.csv'
    gmDatabase          = 'testgmList.csv'
    testerS1            = 1.15

    gmDf, specAvg       = cleanGMs(gmFolder, PEERSummary, testerS1, 
                                   summaryStart, summaryLength, scaledStart, 
                                   scaledLength, unscaledStart, unscaledLength)
    gmDf.to_csv(gmFolder+gmDatabase, index=False)