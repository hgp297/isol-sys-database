############################################################################
#               System selection

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: August 2022

# Description:  

############################################################################


import numpy as np

def random_system(num):

    # range of desired inputs
    # currently, this is approx'd to match MCER level (no site mod)
    system_dict   = {
        'structural_system' : [1, 4],
        'isolator_system' : [1, 2],
        'num_bays' : [3, 8]
    }

    # create array of limits, then run LHS
    # variable names. IMPORTANT: Ordered by insertion
    param_names = list(system_dict.keys())       
    # variable bounds                            
    param_limits = np.asarray(list(system_dict.values()), dtype=np.float64).T  
    
    lBounds = param_limits[0,]
    uBounds = param_limits[1,]
    
    dimVars = len(system_dict)
    
    for bounds in system_dict.values():
        a = np.random.randint(bounds[0], high=bounds[1]+1, size=num)
        print(a)

    return(param_names)

if __name__ == '__main__':

    names       = random_system(5)
    #print(inputs.shape)