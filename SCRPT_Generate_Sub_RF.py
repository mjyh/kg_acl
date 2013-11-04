# -*- coding: utf-8 -*-
"""
Generate a submission file using random forest from sklearn

This builds a random forest from all the training data, then computes and writes
a submission for the test data
"""

import numpy as np
import pandas as pd

import gc

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import LIB_Features as feat
import LIB_Database as db

import CONST_Accel as Const

from time import time

loadData = 1
gc.collect()

### Get the features
if loadData:

    X = np.load( 'RF_X.npy' )
    y = np.load( 'RF_y.npy' )
    
    X_test = np.load( 'RF_Xtest.npy' )    
    
else:
    
    X, y = feat.buildRFTrainingData()

    X_test = feat.buildRFTestData()

 
### Create the model and fit it
 
init = time(); print "Fitting model"    
rf = RandomForestClassifier( n_estimators=40 )
rf.fit( X, y )
print "Done! %.2f s\n" % ( time() - init )    

#init = time(); print "Scoring model"   
#rf.score( X, y)
#print "Done! %.2f s\n" % ( time() - init )    

init = time(); print "Running predictions"   
z = rf.predict( X_test )
print "Done! %.2f s\n" % ( time() - init )    

#rf.predict_proba( X_test )



### Loading some data

init = time(); print "Loading data for generating submission" 
deviceIDs = db.getDeviceIDs()
questionData = db.getQuestionData()
questionIDs = db.getQuestionIDs()

print "Done! %.2f s\n" % ( time() - init )    

n_questions = questionIDs.__len__()

### Initialize result data frame, it contains [ QuestionId, IsTrue ]
result = pd.DataFrame( index = questionIDs, columns = [ 'IsTrue' ] )

ind = 0
init = time()

### For each question, get the model score
for questionID in questionIDs:
    
    if (ind+1) % 1000 == 0:
        print "   %d of %d questions done, %.1f s so far" % ( ind+1, n_questions, time()-init )
            
    quizDevice = questionData.at[ questionID, 'QuizDevice' ]
    predictCol = deviceIDs.index( quizDevice )
    
    result.at[ questionID, 'IsTrue' ] = rf.predict_proba( X_test[ind] )[0][predictCol]  
    
    ind = ind + 1

result.to_csv( 'RFSub.csv', index_label = 'QuestionId' )