# -*- coding: utf-8 -*-
"""
Run logistic regression for various values of C (regularization parameter)

Use C = .1 to 1
"""


import numpy as np
import pandas as pd
import sklearn as sklearn

import gc

from time import time
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


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

### Normalize Data

X = np.delete( X, [8], axis=1 )
X = np.array( feat.normalizeAllData( pd.DataFrame(X) ) )

### Perform N-fold cross validation

#n_CV = 5

### Get a shuffled version of indices

n_samples = X.shape[0]
shuffled_indices =cv.make_shuffled_indices( n_samples )
X = X[ shuffled_indices ]
y = y[ shuffled_indices ]

### Perform N-fold CV

C_array = np.array( [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30 ] )
CVResultsList = []
results = []

for currentC in C_array:
    model = None  

    print "Running C = %f" % ( currentC )
    
    ### Set up training and cv data
    init = time(); print "Splitting training/CV set"    
    
    ind_train, ind_cv = cv.simpleSplitCVIndices( n_samples )
    deviceIDs = np.sort( np.unique( y[ ind_train ] ) )
    print "%d devices in training set" % ( deviceIDs.shape[0] ) # check in case we didn't get all devices in the training set
    print "Done! %.2f s\n" % ( time() - init )    
    
    ### Build CV questions
    init = time(); print "Setting up CV questions"    
    questionData, questionIDs = cv.getRFCVQuestions( X[ ind_cv ], y[ ind_cv ] )
    print "Done! %.2f s\n" % ( time() - init )    
     
    # Fit logistic regression model
    init = time(); print "Fitting model"    
    model = LogisticRegression( penalty='l1', C=currentC ).fit( X[ind_train], y[ind_train] )
    print "Done! %.2f s\n" % ( time() - init )   
   
    # Run against CV questions
    init = time(); print "Calculating predictions"    
    CVResult = cv.getCVResults( model, X[ind_cv], y[ind_cv], deviceIDs, questionIDs, questionData )
    print "Done! %.2f s\n" % ( time() - init ) 
    
    # Add to results
    CVResultsList.append( CVResult )
    print "AUC for C = %f is %f" % ( currentC, CVResult[ 'AUC' ] )

    gc.collect()
    

for resultDict in CVResultsList:
    results.append( resultDict[ 'AUC' ])


print results