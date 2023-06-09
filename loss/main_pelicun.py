############################################################################
#               Loss estimation of TFP database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  Calling function to predict component damage and loss in the 
# initial database

# Open issues:  (1) 

############################################################################

def run_pelicun(database_path, results_path, 
                database_file='run_data.csv', mode='generate'):

    import pandas as pd
    pd.options.display.max_rows = 30
    
    # and import pelicun classes and methods
    from pelicun.base import convert_to_MultiIndex
    from loss import estimate_damage, get_EDP
    
    import warnings
    warnings.filterwarnings('ignore')

    # TODO: integrate data cleaner (check to see if even needed)
    # clean data and add additional variables
    full_isolation_data = pd.read_csv(database_path+database_file)
    
    # write into pelicun style EDP
    edp = get_EDP(full_isolation_data)
    edp.to_csv(database_path+'demand_data.csv', index=True)
    
    # prepare whole set of runs
    
    # load the component configuration
    cmp_marginals = pd.read_csv(database_path+'cmp_marginals.csv', index_col=0)
    
    # Prepare demand data set to match format
    all_demands = pd.read_csv(database_path+'demand_data.csv', 
                              index_col=None,header=None).transpose()
    
    all_demands.columns = all_demands.loc[0]
    all_demands = all_demands.iloc[1:, :]
    all_demands.columns = all_demands.columns.fillna('EDP')
    
    all_demands = all_demands.set_index('EDP', drop=True)
    
    # estimate loss for set
    
    all_losses = []
    loss_cmp_group = []
    col_list = []
    irr_list = []
    
    if mode=='validation':
        IDA_list = []
    
    for run_idx in full_isolation_data.index:
        run_data = full_isolation_data.loc[run_idx]
        
        # TODO: fix EDPs for validation (make suitable for distribution per IDA level)
        # determine what to do for validation
        # validation is currently same as generate
        if mode=='generate':
            raw_demands = all_demands[['Units', str(run_idx)]]
            raw_demands.columns = ['Units', 'Value']
            raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
            raw_demands.index.names = ['type','loc','dir']
        # if mode is validation, treat the dataset as a distribution
        elif mode=='validation':
            raw_demands = all_demands[['Units', str(run_idx)]]
            raw_demands.columns = ['Units', 'Value']
            raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
            raw_demands.index.names = ['type','loc','dir']
        
        print('========================================')
        print('Estimating loss for run index', run_idx)
        
        [cmp, dmg, loss, loss_cmp, agg, 
         collapse_rate, irr_rate] = estimate_damage(raw_demands,
                                                    run_data,
                                                    cmp_marginals,
                                                    mode='generate')
                                                    
        loss_summary = agg.describe([0.1, 0.5, 0.9])
        cost = loss_summary['repair_cost']['50%']
        time_l = loss_summary[('repair_time', 'parallel')]['50%']
        time_u = loss_summary[('repair_time', 'sequential')]['50%']
        
        print('Median repair cost: ', 
              f'${cost:,.2f}')
        print('Median lower bound repair time: ', 
              f'{time_l:,.2f}', 'worker-days')
        print('Median upper bound repair time: ', 
              f'{time_u:,.2f}', 'worker-days')
        print('Collapse frequency: ', 
              f'{collapse_rate:.2%}')
        print('Irreparable RID frequency: ', 
              f'{irr_rate:.2%}')
        print('Replacement frequency: ', 
              f'{collapse_rate+irr_rate:.2%}')
        
        all_losses.append(loss_summary)
        loss_cmp_group.append(loss_cmp)
        col_list.append(collapse_rate)
        irr_list.append(irr_rate)
        
        if mode=='validation':
            IDA_list.append(run_data['IDALevel'])
        
    loss_file = 'loss_estimate_data.csv'
    by_cmp_file = 'loss_estimate_by_groups.csv'
    pd.concat(all_losses).to_csv(results_path+loss_file)
    pd.concat(loss_cmp_group).to_csv(results_path+by_cmp_file)
    
    # flatten data
    loss_df = pd.read_csv(results_path+loss_file, header=[0,1])
    
    loss_header = ['cost_mean', 'cost_std', 'cost_min',
                   'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
                   'time_l_mean', 'time_l_std', 'time_l_min',
                   'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
                   'time_u_mean', 'time_u_std', 'time_u_min',
                   'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']
    
    all_rows = []
    
    for row_idx in range(len(loss_df)):
        if row_idx % 8 == 0:
            # get the block with current run, drop the 'Count'
            run_df = loss_df[row_idx:row_idx+8]
            run_df = run_df.transpose()
            run_df.columns = run_df.iloc[0]
            run_df = run_df.drop(run_df.index[0])
            new_row = pd.concat([run_df.iloc[0], 
                                 run_df.iloc[1], 
                                 run_df.iloc[2]])
            new_row = new_row.drop(new_row.index[0])
            
            all_rows.append(new_row)
            
    loss_df_data = pd.concat(all_rows, axis=1).T
    loss_df_data.columns = loss_header
    
    loss_df_data['collapse_freq'] = col_list
    loss_df_data['irreparable_freq'] = irr_list
    loss_df_data['replacement_freq'] = [x + y for x, y
                                        in zip(col_list, irr_list)]
    
    if mode=='validation':
        loss_df_data['IDA_level'] = IDA_list
    
    
    # clean loss_by_group results
    group_df = pd.read_csv(results_path+by_cmp_file, header=0)
    group_header = ['B_mean', 'B_std', 'B_min',
                   'B_25%', 'B_50%', 'B_75%', 'B_max',
                   'C_mean', 'C_std', 'C_min',
                   'C_25%', 'C_50%', 'C_75%', 'C_max',
                   'D_mean', 'D_std', 'D_min',
                   'D_25%', 'D_50%', 'D_75%', 'D_max',
                   'E_mean', 'E_std', 'E_min',
                   'E_25%', 'E_50%', 'E_75%', 'E_max']
    
    all_rows = []
    
    for row_idx in range(len(group_df)):
        if row_idx % 8 == 0:
            # get the block with current run, drop the 'Count'
            run_df = group_df[row_idx:row_idx+8]
            run_df = run_df.transpose()
            run_df.columns = run_df.iloc[0]
            run_df = run_df.drop(run_df.index[0])
            new_row = pd.concat([run_df.iloc[0], 
                                 run_df.iloc[1], run_df.iloc[2], 
                                 run_df.iloc[3]])
            new_row = new_row.drop(new_row.index[0])
            
            all_rows.append(new_row)
            
    group_df_data = pd.concat(all_rows, axis=1).T
    group_df_data.columns = group_header
    
    all_data = pd.concat([loss_df_data, group_df_data], axis=1)
    all_data.to_csv(results_path+loss_file, index=False)
    
    return(all_data)

#%% main run (analyze training set)

## temporary spyder debugger error hack
import pandas as pd
import collections
collections.Callable = collections.abc.Callable

data_path = './data/tfp_mf_doe/'
res_path = './results/tfp_mf_doe/'
training_data = run_pelicun(data_path, res_path, 
                            database_file='run_data.csv', mode='generate')

#%% validation run


# data_path = './data/tfp_mf_val/'
# res_path = './results/tfp_mf_val/validation/'
# db_file = 'addl_TFP_val.csv'
# validation_input = pd.read_csv(data_path+db_file)
# validation_data = run_pelicun(data_path, res_path, 
#                               database_file=db_file, 
#                               mode='generate')

# # below is taking the mean of median results (over 59 runs)
# validation_data = validation_data.astype('float')
# val_summary = validation_data.describe([0.1, 0.5, 0.9])
# val_input_summary = validation_input.describe([0.5])

#%% baseline validation

# data_path = './data/tfp_mf_val/'
# res_path = './results/tfp_mf_val/baseline/'
# db_file = 'addl_TFP_baseline.csv'
# baseline_input = pd.read_csv(data_path+db_file)
# baseline_data = run_pelicun(data_path, res_path, 
#                             database_file=db_file, 
#                             mode='generate')

# # below is taking the mean of median results (over 59 runs)
# baseline_data = baseline_data.astype('float')
# baseline_summary = baseline_data.describe([0.5])
#%% validation run (full fragility)

# data_path = './data/tfp_mf_val/'
# res_path = './results/tfp_mf_val/validation_full/'
# db_file = 'addl_TFP_val_full.csv'
# validation_input = pd.read_csv(data_path+db_file)
# validation_data = run_pelicun(data_path, res_path, 
#                               database_file=db_file, 
#                               mode='validation')

# # below is taking the mean of median results (over 59 runs)
# validation_data = validation_data.astype('float')
# val_summary = validation_data.describe([0.1, 0.5, 0.9])
# val_input_summary = validation_input.describe([0.5])

#%% baseline run (full fragility)

# data_path = './data/tfp_mf_val/'
# res_path = './results/tfp_mf_val/baseline_full/'
# db_file = 'addl_TFP_baseline_full.csv'
# validation_input = pd.read_csv(data_path+db_file)
# validation_data = run_pelicun(data_path, res_path, 
#                               database_file=db_file, 
#                               mode='validation')

# # below is taking the mean of median results (over 59 runs)
# validation_data = validation_data.astype('float')
# val_summary = validation_data.describe([0.1, 0.5, 0.9])
# val_input_summary = validation_input.describe([0.5])