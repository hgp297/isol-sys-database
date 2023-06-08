############################################################################
#               Run control: rebuilt

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: May 2023

# Description:  Functions as main caller for Opensees experiments
#               If in "generate" mode: create LHS distribution and then run the
#               experiments until n points are reached. If in "validation" mode
#               run the IDA suite

# Open issues:  (1) 

############################################################################
#              Perform runs
############################################################################
import pandas as pd
import postprocessing
import dynamic as eq
import gm
import random

random.seed(985)
def write_to_csv(the_list, filename):
    import csv
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(the_list)
        
def opsExperiment(inputPath, gmPath='./groundMotions/PEERNGARecords_Unscaled/'):
        
    # filter GMs, then get ground motion database list
    PEERSummary     = 'combinedSearch.csv'

    bearingParams = pd.read_csv(inputPath, 
        index_col=None, header=0)
    param   = dict(zip(bearingParams.variable, bearingParams.value))

    # scaler for GM needs to go here
    S1 = param['S1']
    gmDatabase, specAvg = gm.cleanGMs(gmPath, PEERSummary, S1, 
                                              summaryStart=32, nSummary=133, 
                                              scaledStart=176, nScaled=111, 
                                              unscaledStart=290, nUnscaled=111)

    # for each input file, run a random GM in the database
    # with random.randrange(len(gmDatabase.index)) as ind:
    ind = random.randrange(len(gmDatabase.index))

    filename = str(gmDatabase['filename'][ind])
    filename = filename.replace('.AT2', '')
    defFactor = float(gmDatabase['scaleFactorSpecAvg'][ind])
            
    # move on to next set if bad friction coeffs encountered (handled in superStructDesign)
    if defFactor > 20.0:
        return(False)

    # TODO: if validating hard run, start at .0005 dt
    try:
        # perform analysis (superStructDesign and buildModel imported within)
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.005) 
    except ValueError:
        raise ValueError('Bearing solver returned negative friction coefficients. Skipping...')
    except IndexError:
        raise IndexError('SCWB check failed, no shape exists for design. Skipping...')
    if runStatus != 0:
        print('Lowering time step...')
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.001)
    if runStatus != 0:
        print('Lowering time step last time...')
        runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.0005)
    if runStatus != 0:
        print('Recording run and moving on.')

    # add run results to holder df
    resultsHeader, thisRun  = postprocessing.failurePostprocess(filename, 
        scaleFactor, specAvg, runStatus, Tfb)
        
    # add results onto the dataframe
    resultsDf = pd.DataFrame(thisRun, columns=resultsHeader)

    return(resultsDf)

def generate(num_points=400, inputDir='./inputs/bearingInput.csv',
             output_str='./data/run.csv'):
    import LHS
    # initialize dataframe as an empty object
    resultsDf = None

    # generate LHS input sets
    numRuns = 800
    inputVariables, inputValues = LHS.generateInputs(numRuns)
    
    # for each input sets, write input files
    pt_counter = 0
    print('Starting database generation...')
    for index, row in enumerate(inputValues):
        
        print('The run index is ' + str(index) + '.') # run counter
        print('Converged runs: ' + str(pt_counter) + '.') # run counter
   
        # write input files as csv columns
        bearingIndex = pd.DataFrame(inputVariables, columns=['variable'])
        bearingValue = pd.DataFrame(row, columns=['value'])

        bearingIndex   = bearingIndex.join(bearingValue)
        
        bearingIndex.to_csv(inputDir, index=False)
        
        # run opsPy
        try:
            runResult = opsExperiment(inputDir, 
                                      gmPath='./ground_motions/PEERNGARecords_Unscaled/')
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
        
        pt_counter += 1
            
        # if initial run, start the dataframe with headers from postprocessing.py
        if resultsDf is None:
            resultsDf = runResult
            
        # add results onto the dataframe
        else:
            resultsDf = pd.concat([runResult,resultsDf], sort=False)

        if (pt_counter == num_points):
            break
        
        # saving mechanism
        if (index % 10) == 0:
            resultsDf.to_csv('./data/temp_save.csv', index=False)
    
    resultsDf.to_csv(output_str, index=False)
    
    return(resultsDf)

def run_doe(prob_target, training_set_path, testing_set_path, 
            batch_size=10, error_tol=0.15, maxIter=600, conv_tol=1e-2,
            inputPath = './inputs/', inputFile = 'bearingInput.csv'):

    import numpy as np
    np.random.seed(986)
    random.seed(986)
    from doe import GP
    
    # use a test set to evaluate error metric
    df = pd.read_csv(training_set_path)
    df_test = pd.read_csv(testing_set_path)
    
    
    # collapse as a probability
    from scipy.stats import lognorm
    from math import log, exp
    from scipy.stats import norm
    inv_norm = norm.ppf(0.84)
    beta_drift = 0.25
    # 0.9945 is inverse normCDF of 0.84
    mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) 
    ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
    
    df_test['max_drift'] = df_test[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
    df_test['collapse_prob'] = ln_dist.cdf(df_test['max_drift'])
    test_set = GP(df_test)
    test_set.set_outcome('collapse_prob')
    
    # TODO: change max iter so that it's fixed instead of trial-and-keep
    import LHS
    from postprocessing import cleanDat
    # add more points as DoE
    input_var, input_vals = LHS.generateInputs(maxIter, mode='doe')
    
    rmse = 1.0
    batch_idx = 0
    batch_no = 0
    
    rmse_list = []
    mae_list = []
    
    for index, row in enumerate(input_vals):
        print('The run index is ' + str(index) + '.')
        print('The batch index is ' + str(batch_idx) + '.')
        
        if (batch_idx % (batch_size) == 0):
            
            df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
            df['collapse_prob'] = ln_dist.cdf(df['max_drift'])
            
            mdl = GP(df)
            mdl.set_outcome('collapse_prob')
            mdl.fit_gpr(kernel_name='rbf_ard')
            
            y_hat = mdl.gpr.predict(test_set.X)
            
            print('===== Training model size:', mdl.X.shape[0], '=====')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np
            mse = mean_squared_error(test_set.y, y_hat)
            rmse = mse**0.5
            print('Test set RMSE: %.3f' % rmse)

            mae = mean_absolute_error(test_set.y, y_hat)
            print('Test set MAE: %.3f' % mae)
            
            if len(rmse_list) == 0:
                conv = rmse
            else:
                conv = abs(rmse - rmse_list[-1])/rmse_list[-1]
            
            if rmse < error_tol:
                print('Stopping criterion reached. Ending DoE...')
                print('Number of added points: ' + str((batch_idx)*(batch_no)))
                
                rmse_list.append(rmse)
                write_to_csv(rmse_list, './data/doe/rmse.csv')
                
                mae_list.append(mae)
                write_to_csv(mae_list, './data/doe/mae.csv')
                
                return (df)
            elif conv < conv_tol:
                print('RMSE did not improve beyond convergence tolerance. Ending DoE...')
                print('Number of added points: ' + str((batch_idx)*(batch_no)))
                
                rmse_list.append(rmse)
                write_to_csv(rmse_list, './data/doe/rmse.csv')
                
                mae_list.append(mae)
                write_to_csv(mae_list, './data/doe/mae.csv')
                
                return (df)
            else:
                pass
            batch_idx = 0
            df.to_csv('./data/doe/temp_save.csv', index=False)
            x_next = mdl.doe_rejection_sampler(batch_size, prob_target)
            print('Convergence not reached yet. Resetting batch index to 0...')
            
        # write LHS half into a dataframe
        lhs_vals = [float(val) for val in row]
        inputDf = pd.DataFrame(list(zip(input_var, lhs_vals)), 
            columns=['variable','value'])
        
        doe_vals = [float(val) for val in x_next[batch_idx]]
        doe_vars = ['gapRatio', 'RI', 'Tm', 'zetaM']
        doeDf = pd.DataFrame(list(zip(doe_vars, doe_vals)),
            columns=['variable','value'])
        
        inputDf = inputDf.append(doeDf)
        inputDf.to_csv(inputPath+inputFile, index=False)
        
        # run opsPy
        try:
            runResult = opsExperiment(inputPath+inputFile, 
                                      gmPath='./ground_motions/PEERNGARecords_Unscaled/')
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
        
        # if run is successful and is batch marker, record error metric
        if (batch_idx % (batch_size) == 0):
            rmse_list.append(rmse)
            write_to_csv(rmse_list, './data/doe/rmse.csv')
            
            mae_list.append(mae)
            write_to_csv(mae_list, './data/doe/mae.csv')
        
        batch_idx += 1

        # clean new df, attach to existing data
        newRes = cleanDat(runResult)
        df = df.append(newRes, sort=True)
    
    df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
    df.loc[df['collapsed'] == -1, 'collapsed'] = 0
    df['collapse_prob'] = ln_dist.cdf(df['max_drift'])
    
    print('Did not reach convergence after max runs.')
    return df
    

def validate(inputStr, IDALevel=[1.0, 1.5, 2.0], 
             gmPath='./ground_motions/PEERNGARecords_Unscaled/'):
    
    # initialize dataframe as an empty object
    resultsDf           = None

    # filter GMs, then get ground motion database list
    PEERSummary     = 'combinedSearch.csv'

    # incremental MCE_R levels
    # IDALevel    = np.arange(1.0, 2.50, 0.5).tolist()
    # IDALevel  = np.arange(1.0, 2.5, 0.5).tolist()
    # IDALevel = [1.0, 1.5, 2.0]
    # IDALevel = [3.0]

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

    for lvl in IDALevel:

        print('The IDA level is ' + str(lvl) + '.')

        # scale S1 to match MCE_R level
        actualS1    = param['S1']*lvl

        gmDatabase, specAvg = gm.cleanGMs(gmPath, PEERSummary, actualS1, lvl,
            32, 133, 176, 111, 290, 111)

        # Run eq analysis for 
        for gmIdx in range(len(gmDatabase)):
        # for gmIdx in range(1):

            # ground motion name, extension removed
            filename    = str(gmDatabase['filename'][gmIdx])
            filename    = filename.replace('.AT2', '')

            # scale factor used, either scaleFactorS1 or scaleFactorSpecAvg
            defFactor   = float(gmDatabase['scaleFactorSpecAvg'][gmIdx])

            # skip excessive scaling
            if defFactor > 20.0:
                continue

            # TODO: default time factor for higher IDA levels
            # move on to next set if bad friction coeffs encountered 
            # (handled in superStructDesign)
            try:
                # perform analysis (superStructDesign and buildModel imported within)
                runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.005) 
            except ValueError:
                raise ValueError('Bearing solver returned negative friction coefficients. Skipping...')
                return
            except IndexError:
                raise IndexError('SCWB check failed, no shape exists for design. Skipping...')
                return
            if runStatus != 0:
                print('Lowering time step...')
                runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.001)
            if runStatus != 0:
                print('Lowering time step last time...')
                runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.0005)
            if runStatus != 0:
                print('Recording run and moving on.')


            # add run results to holder df
            resultsHeader, thisRun  = postprocessing.failurePostprocess(filename, 
                scaleFactor, specAvg, runStatus, Tfb, IDALevel=lvl)

            # if initial run, start the dataframe with headers from postprocessing
            if resultsDf is None:
                resultsDf = pd.DataFrame(columns=resultsHeader)
            
            # add results onto the dataframe
            resultsDf = pd.concat([thisRun,resultsDf], sort=False)
            
        # temp save mechanism after every ida level
        resultsDf.to_csv('./data/temp_save.csv', index=False)
        
    return(resultsDf)


#%% generate new data

# output_str = './data/mik_smrf.csv'
# run = generate(400, output_str=output_str)

#%% validate a building (specify design input file)

# inputString = './inputs/bearingInputVal_baseline.csv'
# valDf_base = validate(inputString, IDALevel=[1.0])
# valDf_base.to_csv('./data/validation.csv', index=False)

#%% run doe
# path = './data/mik_smrf.csv'

training_path = './data/training_set.csv'
testing_path = './data/testing_set.csv'

# DOE mechanism: sample from tMSE distribution in batches of 10, target 50% collapse
# Stopping mechanism: if RMSE of collapse prediction < 10% or end of the 600 support points
# or no improvements to the RMSE (<0.1% in RMSE change)
# whichever comes first

doe_df = run_doe(0.5, training_path, testing_path, 
                 error_tol=0.15, conv_tol=0.001, batch_size=10, maxIter=600)
doe_df.to_csv('./data/doe/rmse_doe_set_tol.csv', index=False)
        
# TODO: auto clean
