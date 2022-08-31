############################################################################
#               System selection

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: August 2022

# Description:  

############################################################################

from scipy.stats import qmc
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
    param_names      = list(system_dict.keys())                                    # variable names. IMPORTANT: Ordered by insertion
    param_limits     = np.asarray(list(system_dict.values()), dtype=np.float64).T  # variable bounds
    
    lBounds = param_limits[0,]
    uBounds = param_limits[1,]
    
    dimVars = len(system_dict)
    sampler = qmc.LatinHypercube(d=dimVars, seed=985)
    sample = sampler.integers(l_bounds=lBounds, u_bounds=uBounds,
                              n=num,endpoint=True) 
    param_set = sample.integers(l_bounds=lBounds,
        u_bounds=uBounds, n=num)

    return(param_names, param_set)

if __name__ == '__main__':

    names, inputs       = random_system(50)
    print(inputs.shape)