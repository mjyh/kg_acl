'''
Calculate features for some devices and create a scatterplot of two of them
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import LIB_Database as db
import LIB_MathUtil as ut

from time import time

numDevices = 20;
deviceIDs = db.getDeviceIDs()
deviceIDs = deviceIDs[:numDevices]
print deviceIDs
resultColumns = [ "mean1", "mean2", "std1", "std2", "p90_1", "p90_2", 'p10_1', 'p10_2' ];
data = pd.DataFrame( index = deviceIDs, columns = resultColumns)

for deviceID in deviceIDs:
    init = time(); print "Calculating for device %s" % deviceID
    rawData = db.getRawDataWhere("train", deviceID, [ "T", 'X' ] )
    
    series = rawData['T']
    series = series - series.shift(1) #compute diffs
    series = series[ 1:: ]

    halfway = series.shape[0]/2
    
    series1 = series[:halfway]
    series1 = ut.trimOutliers( series1, 10 )
    #print series1
    
    series2 = series[halfway+1:]
    series2 = ut.trimOutliers( series2, 10 )
    
    tdiff_mean1 = series1.mean()
    tdiff_mean2 = series2.mean()

    std1 = series1.std()
    std2 = series2.std()
    
    p90_1 = np.percentile( series1, 90 )
    p90_2 = np.percentile( series2, 90 )
    
    p10_1 = np.percentile( series1, 10 )
    p10_2 = np.percentile( series2, 10 )
    
    data.at[ deviceID, 'mean1' ] = tdiff_mean1
    data.at[ deviceID, 'mean2' ] = tdiff_mean2
    data.at[ deviceID, 'std1' ] = std1
    data.at[ deviceID, 'std2' ] = std2
    data.at[ deviceID, 'p90_1' ] = p90_1
    data.at[ deviceID, 'p90_2' ] = p90_2
    data.at[ deviceID, 'p10_1' ] = p10_1
    data.at[ deviceID, 'p10_2' ] = p10_2
    
    print "DONE! %.1f s\n" % ((time()-init))
    
#plt.plot( data[ "Mean1"], data[ "Mean2" ], 'ro', [ 100, 300 ], [100,300])
print data
#data = data[ data['X'] < 1e7 ]
#data = data[ data['Y'] < 1e7 ]

X1 = data[ "p90_1"]
Y1 = data[ "p90_2" ]

X2 = data[ "mean2"]
Y2 = data[ "std2" ]

plt.plot( X1, Y1, 'ro')
#plt.plot( X1, Y1, 'ro', X2, Y2, 'bo' )

#plt.title( 'Average time between samples' )
#plt.xlabel( 'Mean from first sequence' )
#plt.ylabel( 'Mean from second sequence')

for label, x, y in zip(deviceIDs, X1, Y1 ):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

#for label, x, y in zip(deviceIDs, X2, Y2 ):
#    plt.annotate(
#        label, 
#        xy = (x, y), xytext = (-20, 20),
#        textcoords = 'offset points', ha = 'right', va = 'bottom',
#        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
plt.show()