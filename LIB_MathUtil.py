'''
Some utilities for computation
'''

import numpy as np
import pandas as pd
import math

# Return a mean with highest/lowest data thrown out
# Data is an numpy array
# Trim=10 means throw out the top 10% and bottom 10% of values
def meanTrimmed(data, trim=10):
    if trim > 0:
        lowerBound = np.percentile(data, trim)
        upperBound = np.percentile(data, (100-trim))
    
        return np.mean( data[ ( data >= lowerBound) & ( data <= upperBound ) ] )
    else:
        return np.mean( data )

def trimOutliers( data, trim=10):
    if type(data) is not pd.core.series.Series and type(data) is not np.array:
        raise Exception( "Wrong data type passed to trimOutliers (%s)" % type(data) )
                         
    lowerBound = np.percentile(data, trim)
    upperBound = np.percentile(data, (100-trim))
    
    return data[ ( data >= lowerBound) & ( data <= upperBound ) ]

### Retruend weighted std 
#   From http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
#   values and weights should be numpy nd arrays with the same shape

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return  math.sqrt(variance)