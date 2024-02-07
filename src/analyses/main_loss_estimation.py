############################################################################
#               Main loss estimation file

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2024

# Description:  Main file which imports the structural database and starts the
# loss estimation

# Open issues:  (1) 

############################################################################


def run_pelicun(database_path, results_path, 
                database_file='run_data.csv', mode='generate'):
    
    import numpy as np
    import pandas as pd
    pd.options.display.max_rows = 30
    
    # import warnings
    # warnings.filterwarnings('ignore')
    
    # and import pelicun classes and methods
    from loss import Loss_Analysis
    
    from pelicun.assessment import Assessment
    # get database
    # initialize, no printing outputs, offset fixed with current components
    PAL = Assessment({
        "PrintLog": False, 
        "Seed": 985,
        "Verbose": False,
        "DemandOffset": {"PFA": 0, "PFV": 0}
    })

    # generate structural components and join with NSCs
    P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')
    
    full_isolation_data = pd.read_csv(database_path+database_file)
    
    # fixed floor usage
    # lab, health, ed, res, office, retail, warehouse, hotel
    fl_usage = [0., 0., 0., 0., 1.0, 0., 0., 0.]
    
    for run_idx, run in full_isolation_data.iterrows():
        
        run_floors = run.num_stories
        run_area = run.L_bldg**2 # sq ft
        
        bldg_usage = [fl_usage]*run_floors
        area_usage = np.array(fl_usage)*run_area
        
        run_loss = Loss_Analysis(run)
        run_loss.nqe_sheets(nqe_dir='../../resource/loss/')
        run_loss.normative_quantity_estimation(bldg_usage, P58_metadata, 
                                               brace_dir='../../resource/')
        run_loss.process_EDP()
        
        print('========================================')
        print('Estimating loss for run index', run_idx)
        
        run_loss.estimate_damage(mode='generate')
        
        # [cmp, dmg, loss, loss_cmp, agg, 
        #  collapse_rate, irr_rate] = estimate_damage(raw_demands,
        #                                             run_data,
        #                                             cmp_marginals,
        #                                             mode='generate')
        
#%% main run
## temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

# run info
data_path = '../../data/'
data_file = 'structural_db_conv.csv'
res_path = './loss_data/'
training_data = run_pelicun(data_path, res_path, 
                            database_file=data_file, mode='generate')