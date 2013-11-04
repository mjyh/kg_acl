'''
For a particular device, takes samples, computes
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import LIB_Database as db
import LIB_MathUtil as ut

from time import time

deviceID = '68'
linesPerSection = 300
sections = 4
series = db.getRawDataWhere("train", deviceID, [ "T" ] )['T']

series = series - series.shift(1) #compute diffs
series = series[ 1:: ]
#series = ut.trimOutliers(series, 10)
#series = series[ (series < 10000) ]
#series = series[ (series > 1000) & (series < 100000)  ]


#numElements = series.shape[0]

allNumZeros = []
allMeans = []
allStds = []
allSeries = []
currentStart = 1
currentEnd = linesPerSection
for currentSectionNum in range(0, sections):
    currentSeries = series[ currentStart:currentEnd ]
    currentSeries = ut.trimOutliers( currentSeries, 10 )
    currentStart = currentStart + linesPerSection
    currentEnd = currentEnd + linesPerSection
    
    plt.subplot( "22%d" % (currentSectionNum+1) )
    hist = plt.hist(currentSeries, 80)
    plt.title( "%s.%d" % (deviceID, currentSectionNum) )
    
    allNumZeros.append( sum( currentSeries<1 ) )
    allMeans.append( currentSeries.mean() )
    allStds.append( currentSeries.std() )
   
    
plt.show()

    
    
print allNumZeros
print allMeans
print allStds


#for i in range(1,1000):
#    print series.iloc[i]


