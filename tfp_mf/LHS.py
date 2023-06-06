############################################################################
#               Latin Hypercube sampling

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: August 2020

# Description:  Script generates input parameter space based on LHS, given bounds

# Open issues:  (1) Seek nondimensionalized variables

############################################################################

from scipy.stats import qmc
import numpy as np

def generateInputs(num, mode='generate'):

    # range of desired inputs
    # currently, this is approx'd to match MCER level (no site mod)
    if mode=='doe':
        inputDict 	= {
    		'S1' 			: [0.8, 1.3],
    		'mu1' 			: [0.01, 0.05],
    		'R1' 			: [15, 45],
    	}
    else:
        inputDict   = {
            'S1'            : [0.8, 1.3],
            'Tm'            : [2.5, 4.0],
            'zetaM'         : [0.10, 0.20],
            'mu1'           : [0.01, 0.05],
            'R1'            : [15, 45],
            'moatAmpli'       : [0.3, 1.8],
            'RI'            : [0.5, 2.0]
        }

    # create array of limits, then run LHS
    paramNames      = list(inputDict.keys())
    paramLimits     = np.asarray(list(inputDict.values()), 
                                 dtype=np.float64).T  # variable bounds
    
    lBounds = paramLimits[0,]
    uBounds = paramLimits[1,]
    
    dimVars = len(inputDict)
    sampler = qmc.LatinHypercube(d=dimVars, seed=985)
    sample = sampler.random(n=num)
    
    paramSet = qmc.scale(sample, lBounds, uBounds)

    return(paramNames, paramSet)

if __name__ == '__main__':

    names, inputs       = generateInputs(50)
    print(inputs.shape)