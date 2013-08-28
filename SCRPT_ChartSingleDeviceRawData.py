'''
Create a histogram of certain raw data for a device
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import LIB_Database as db
import LIB_MathUtil as ut

from time import time

deviceID = '8'

series = db.getRawDataWhere("train", deviceID, [ "T" ] )['T']

series = series - series.shift(1) #compute diffs
series = series[ 1:: ]
#series = ut.trimOutliers(series, 5)
series = series[ (series >= 0) & (series < 1000) ]
#series = series[ (series > 1000) & (series < 100000)  ]

numZeroes = sum(series[series==1])
numElements = series.shape[0]


allSeries = []

hist = plt.hist(series, 100)
print "%d zeroes, %d elements" % ( numZeroes, numElements )
plt.title( deviceID )
plt.show()


print hist