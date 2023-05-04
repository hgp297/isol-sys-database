############################################################################
#               Demand file (pelicun)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: October 2022

# Description:  Utility used to generate demand files (EDP) in the PBE framework

# Open issues:  (1) requires expansion to generalized multifloor
#               (2) check acceleration units

############################################################################



def get_EDP(isol_data):

    EDP_data = isol_data[['accMax0', 'accMax1', 'accMax2', 'accMax3',
                          'velMax0', 'velMax1', 'velMax2', 'velMax3',
        'driftMax1', 'driftMax2', 'driftMax3',
        'accMax0', 'accMax1', 'accMax2', 'accMax3',
        'velMax0', 'velMax1', 'velMax2', 'velMax3',
        'driftMax1', 'driftMax2', 'driftMax3']]

    #type-floor-direction
    EDP_data.columns = ['PFA-1-1', 'PFA-2-1', 'PFA-3-1', 'PFA-4-1',
                        'PFV-1-1', 'PFV-2-1', 'PFV-3-1', 'PFV-4-1',
        'PID-1-1', 'PID-2-1', 'PID-3-1',
        'PFA-1-2', 'PFA-2-2', 'PFA-3-2', 'PFA-4-2',
        'PFV-1-2', 'PFV-2-2', 'PFV-3-2', 'PFV-4-2',
        'PID-1-2', 'PID-2-2', 'PID-3-2']
    # EDP_data.columns = ['PFA-1-1', 'PFA-2-1', 'PFA-3-1', 'PFA-4-1',
    #     'PID-1-1', 'PID-2-1', 'PID-3-1',
    #     'PFA-1-2', 'PFA-2-2', 'PFA-3-2', 'PFA-4-2',
    #     'PID-1-2', 'PID-2-2', 'PID-3-2']
    
    EDP_data.loc['Units'] = ['g','g','g','g',
                             'inps', 'inps', 'inps', 'inps',
                             'rad','rad','rad',
                             'g','g','g','g',
                             'inps', 'inps', 'inps', 'inps',
                             'rad','rad','rad']

    EDP_data["new"] = range(1,len(EDP_data)+1)
    EDP_data.loc[EDP_data.index=='Units', 'new'] = 0
    EDP_data = EDP_data.sort_values("new").drop('new', axis=1)

    return(EDP_data)

# import pandas as pd
# data = pd.read_csv('./pelicun/full_isolation_data.csv', sep=',')
# edp = get_EDP(data)
# edp.to_csv('./pelicun/demand_data.csv', index=True)