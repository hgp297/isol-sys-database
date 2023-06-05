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
import eqAnly as eq
import gmSelector
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
    gmDatabase, specAvg = gmSelector.cleanGMs(gmPath, PEERSummary, S1, 
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

def run_doe(prob_target, path, batch_size=10, error_tol=0.15, maxIter=600,
            inputPath = './inputs/', inputFile = 'bearingInput.csv'):

    import numpy as np
    np.random.seed(986)
    random.seed(986)
    
    df = pd.read_csv(path)
    
    
    # fit model with initial n points
    # from doe import GP
    # mdl = GP(df)
    # mdl.set_outcome('collapsed')
    # mdl.fit_gpc(kernel_name='rbf_ard', noisy=True)
    
    import LHS
    from postprocessing import cleanDat
    # add more points as DoE
    input_var, input_vals = LHS.generateInputs(maxIter, mode='doe')
    
    ame = 1.0
    batch_idx = 0
    batch_no = 0
    
    from doe import GP
    ame_list = []
    
    for index, row in enumerate(input_vals):
        print('The run index is ' + str(index) + '.')
        print('The batch index is ' + str(batch_idx) + '.')
        
        if (batch_idx % (batch_size) == 0):
            
            df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
            df.loc[df['collapsed'] == -1, 'collapsed'] = 0
            
            # collapse as a probability
            from scipy.stats import lognorm
            from math import log, exp
            from scipy.stats import norm
            inv_norm = norm.ppf(0.84)
            beta_drift = 0.25
            # 0.9945 is inverse normCDF of 0.84
            mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) 
            ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
            df['collapse_prob'] = ln_dist.cdf(df['max_drift'])
            
            mdl = GP(df)
            mdl.set_outcome('collapse_prob')
            mdl.fit_gpr(kernel_name='rbf_ard')
            
            rsc = mdl.gpr.score(mdl.X, mdl.y)
            print('R-squared :', rsc)
            y_hat = mdl.gpr.predict(mdl.X)
            
            from sklearn.metrics import mean_squared_error
            import numpy as np
            mse = mean_squared_error(mdl.y, y_hat)
            print('Mean squared error: %.3f' % mse)
            
            # ame is average error in predicting collapse risk (expressed in relative %)
            # denominator is 0.001 to prevent extreme outliers
            y_true = np.array(mdl.y).ravel()
            mape = np.abs(y_true-y_hat)/np.maximum(np.abs(y_true), 1e-3)
            ame = np.average(mape, axis=0)
            ame_list.append(ame)
            write_to_csv(ame_list, './data/doe/mean_error.csv')
            print('Average mean error (%%): %.2f' % ame)
            
            if ame < error_tol:
                print('Stopping criterion reached.')
                print('Number of added points: ' + str((batch_idx)*(batch_no)))
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
        
        # if run succeeded, increase the batch index
        batch_idx += 1

        # clean new df, attach to existing data
        newRes = cleanDat(runResult)
        df = df.append(newRes, sort=True)
    
    df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
    df.loc[df['collapsed'] == -1, 'collapsed'] = 0
    
    # collapse as a probability
    from scipy.stats import lognorm
    from math import log, exp
    from scipy.stats import norm
    inv_norm = norm.ppf(0.84)
    beta_drift = 0.25
    # 0.9945 is inverse normCDF of 0.84
    mean_log_drift = exp(log(0.1) - beta_drift*inv_norm) 
    ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)
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

        gmDatabase, specAvg = gmSelector.cleanGMs(gmPath, PEERSummary, actualS1, lvl,
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
# # path = './data/mik_smrf.csv'
# path = './data/doe_init.csv'
# doe_df = run_doe(0.5, path)
# doe_df.to_csv('./data/doe/mik_smrf_doe.csv', index=False)
        
# TODO: auto clean
