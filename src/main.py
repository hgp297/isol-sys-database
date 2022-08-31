############################################################################
#               Run control

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: August 2020

# Description:  Main script
#               Manages files and writes input files for each run
#               Calls LHS -> design -> buildModel -> eqAnly -> postprocessing
#               Writes results in final csv file

# Open issues:  (1) 

############################################################################

# import OpenSees and libraries
# from openseespy.opensees import *
# import math

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

empty_directory('outputs')

############################################################################
#              Perform runs
############################################################################
import pandas as pd
import LHS
import postprocessing
import eqAnly as eq
import gmSelector
import random

# initialize dataframe as an empty object
resultsDf           = None

# generate LHS input sets
numRuns                         = 300
inputVariables, inputValues     = LHS.generateInputs(numRuns)

# filter GMs, then get ground motion database list
gmPath          = './groundMotions/PEERNGARecords_Unscaled/'
PEERSummary     = 'combinedSearch.csv'
databaseFile    = 'gmList.csv'

# # save GM list used
# gmDatabase        = gmSelector.cleanGMs(gmPath, PEERSummary)
# gmDatabase.to_csv(gmPath+databaseFile, index=False)

# # troubleshooting with list of impact GMs
# gmDatabase        = pd.read_csv('./groundMotions/gmList.csv')

# set seed for reproducibility
random.seed(985)

# for each input sets, write input files
for index, row in enumerate(inputValues):

    print('The run index is ' + str(index) + '.') # run counter

    empty_directory('outputs') # clear run histories

    # write input files as csv columns
    bearingIndex    = pd.DataFrame(inputVariables, columns=['variable']) # relies on ordering from LHS.py
    bearingValue    = pd.DataFrame(row, columns=['value'])

    bearingIndex    = bearingIndex.join(bearingValue)
    param           = dict(zip(bearingIndex.variable, bearingIndex.value))
    bearingIndex.to_csv('./inputs/bearingInput.csv', index=False)

    # scaler for GM needs to go here
    actualS1        = param['S1']
    gmDatabase, specAvg     = gmSelector.cleanGMs(gmPath, PEERSummary, actualS1, 
        32, 133, 176, 111, 290, 111)

    # for each input file, run a random GM in the database
    # with random.randrange(len(gmDatabase.index)) as ind:
    ind           = random.randrange(len(gmDatabase.index))

    filename                = str(gmDatabase['filename'][ind])                  # ground motion name
    filename                = filename.replace('.AT2', '')                      # remove extension from file name
    defFactor               = float(gmDatabase['scaleFactorSpecAvg'][ind])      # scale factor used, either scaleFactorS1 or scaleFactorSpecAvg
    # gmS1                  = float(gmDatabase['scaledSa1'][ind])               # scaled pSa at T = 1s, w.r.t. method above
        
    # move on to next set if bad friction coeffs encountered (handled in superStructDesign)
    try:
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.005) # perform analysis (superStructDesign and buildModel imported within)
    except ValueError:
        print('Bearing solver returned negative friction coefficients. Skipping...')
        continue
    except IndexError:
        print('SCWB check failed, no shape exists for design. Skipping...')
        continue
    if runStatus != 0:
        print('Lowering time step...')
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.001)
    if runStatus != 0:
        print('Lowering time step last time...')
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.0005)
    if runStatus != 0:
        print('Recording run and moving on.')

    resultsHeader, thisRun  = postprocessing.failurePostprocess(filename, scaleFactor, specAvg, runStatus, Tfb) # add run results to holder df

    # if initial run, start the dataframe with headers from postprocessing.py
    if resultsDf is None:
        resultsDf           = pd.DataFrame(columns=resultsHeader)
        
    # add results onto the dataframe
    resultsDf               = pd.concat([thisRun,resultsDf], sort=False)

gmDatabase.to_csv(gmPath+databaseFile, index=False)
resultsDf.to_csv('./sessionOut/sessionSummary.csv', index=False)