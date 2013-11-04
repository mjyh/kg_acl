# -*- coding: utf-8 -*-
"""

Run CV testing for Random Forest classifier
"""

import numpy as np
import pandas as pd
import sklearn as sklearn


import gc

from time import time
from sklearn.ensemble import RandomForestClassifier


import LIB_Features as feat
import LIB_CV as cv

    

### Start of main script, some initialization

loadData = 1 # load features or compute them

cv.setupSimilarDevices() # prep the list of similar devices

### Get the features

if loadData:

    X = np.load( 'RF_X.npy' )
    y = np.load( 'RF_y.npy' )
    y = y[:,0]
    
else:
    
    X, y = feat.buildRFTrainingData()
    X[ 46409 ][4] = 1.57 #fix a nan

### Perform N-fold cross validation

n_CV = 5

### Get a shuffled version of indices

n_samples = X.shape[0]
shuffled_indices =cv.make_shuffled_indices( n_samples )
X = X[ shuffled_indices ]
y = y[ shuffled_indices ]

CVResultsList = []

### Perform N-fold CV


for cur_CV in range( 0, n_CV ):
    rf = None  
    
    print "Running CV batch %d of %d" % ( cur_CV + 1, n_CV )
    
    ### Set up training and cv data
    init = time(); print "Splitting training/CV set"    
    
    ind_train, ind_cv = cv.splitTrainCVIndices( n_samples, cur_CV, n_CV )
    deviceIDs = np.sort( np.unique( y[ ind_train ] ) )
    print "%d devices in training set" % ( deviceIDs.shape[0] ) # check in case we didn't get all devices in the training set
    print "Done! %.2f s\n" % ( time() - init )    
    
    ### Build CV questions
    init = time(); print "Setting up CV questions"    
    questionData, questionIDs = cv.getRFCVQuestions( X[ ind_cv ], y[ ind_cv ] )
    print "Done! %.2f s\n" % ( time() - init )    
     
    # Fit model
    init = time(); print "Fitting model"    
    rf = RandomForestClassifier( n_estimators=50 )
    rf.fit( X[ind_train], y[ind_train] )
    print "Done! %.2f s\n" % ( time() - init )   
   
    # Run against CV questions
    init = time(); print "Calculating predictions"    
    CVResult = cv.getRFCVResults( rf, X[ind_cv], y[ind_cv], deviceIDs, questionIDs, questionData )
    print "Done! %.2f s\n" % ( time() - init ) 
    
    # Add to results
    CVResultsList.append( CVResult )

    gc.collect()
    
avgAUC = 0

for resultDict in CVResultsList:
        
        avgAUC = avgAUC + resultDict[ 'AUC' ]

avgAUC = avgAUC / n_CV

print CVResultsList
