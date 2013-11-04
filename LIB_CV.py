# -*- coding: utf-8 -*-
"""
Files for bulding CV data
"""

import numpy as np
import numpy.random as rand
import pandas as pd
import sklearn as sklearn

from time import time

import CFG_SimilarDevice as SimDev

def setupSimilarDevices():
    for deviceID, similarDevices in SimDev.devices_clusters.iteritems():
        if deviceID in similarDevices:
            similarDevices.remove( deviceID )
        
### Returns a similar device to the given one, will not return the same deviceID back
def getRandomSimilarDevice( deviceID ):
    similarDevices = SimDev.devices_clusters[ deviceID ]
    index = int( np.floor( rand.uniform( 0, len( similarDevices ) ) ) )
    return similarDevices[ index ]
    
### Shuffles data around
#   given n, returns list of nubmers 1-n in random order

def make_shuffled_indices( n_samples ):
    indices = np.arange(n_samples)
    np.random.shuffle( indices )
    return indices

### 60% train, 40% CV
def simpleSplitCVIndices( n_samples ):
    indexCutoff = np.round( n_samples * .6 ).astype(int)
    
    ind_train = np.arange( 0, indexCutoff )
    ind_cv = np.arange( indexCutoff+1, n_samples )
    
    return ind_train, ind_cv


### Divides samples into testing and training sections. Will select a contiguous block of samples for CV
#   The block depends on how many n-fold CV you are running
#   Example: ( cur_CV = 0, n_CV = 3 ) selects the first 33% of data as CV, the latter 66% as training
#   
#   n_samples: number of samples
#   n_CV: number of CVs to be run total, n-fold CV
#   cur_CV: an integer from 0 to n_CV - 1, represent which iteration of n-fold CV you are currently on
#
#   Returns ( training indices, CV indices ) 

def splitTrainCVIndices( n_samples, cur_CV, n_CV ):
    
    # split data into chunks according to n-fold
    indexCutoffs = np.round( n_samples * np.linspace( 0, 1.0, n_CV + 1 ) ).astype(int)
    
    #get the CV indices
    cv_start = indexCutoffs[ cur_CV ]
    cv_end = indexCutoffs[ cur_CV + 1]
    
    ind_cv = np.arange( cv_start, cv_end )
    
    #get the training indices, in general there are two sections: before and after the CV indices
    train1_start = 0
    train1_end = cv_start
    
    ind_train1 = np.arange( train1_start, train1_end  )
    
    train2_start = cv_end
    train2_end = n_samples
    
    ind_train2 = np.arange( train2_start, train2_end )
    
    # concatenate training indices
    ind_train = np.concatenate( ( ind_train1, ind_train2 ) )
    
    # return results
    return ind_train, ind_cv
    
    
### Test a random forest against a set of CV data. Returns a dictionary of CV test results
#
#   Inputs    
#   
#   rfModel: random forest model 
#   X_cv, y_cv : numpy array
#   deviceIDs, questionIDs:
    
def getCVResults( model, X_cv, y_cv, deviceIDs, questionIDs, questionData ):
    
    n_questions = y_cv.shape[0]

    question_ind = 0
    
    init = time()
    
    prediction_probs = np.zeros( n_questions )
    prediction_device = np.zeros( n_questions )
    
    
    for questionID in questionIDs:
        
        if (question_ind + 1) % 1000 == 0:
            print "   %d of %d questions done, %.1f s so far" % ( question_ind+1, n_questions, time()-init )
                
        quizDevice = questionData.at[ questionID, 'QuizDevice' ]
        
        if quizDevice in deviceIDs:
            predictCol = np.where( deviceIDs == quizDevice )[0][0]
            prediction_probs[ question_ind ] = model.predict_proba( X_cv[question_ind] )[0][predictCol]
        else:
            prediction_probs[ question_ind ] = 0
        
        question_ind = question_ind + 1
    
    prediction_device = model.predict(X_cv)
    
    AUC = sklearn.metrics.roc_auc_score( prediction_device == y_cv, prediction_probs )
      
    result = { 'AUC': AUC }
    
    return result

### Given X and y data, builds questions
#   returns a DataFrame <questionData> and a list <questionIDs>
#
#

def getRFCVQuestions( X_cv, y_cv ):
   
    n_questions = X_cv.shape[0]
    questionIDs = np.arange( 1, n_questions+1)
    
    nextAnswer = True
    
    questionData = pd.DataFrame( index = questionIDs, columns = [ 'QuizDevice' ] )
    
    for ind in range( 0, n_questions ):
        
        ### Choose True or False
        answer = nextAnswer
        nextAnswer = not nextAnswer
        
        questionID = ind + 1
        
        deviceID = y_cv[ind]
        
        if answer:
            quizDevice = deviceID
        else:
            quizDevice = getRandomSimilarDevice( deviceID )
        
        questionData.at[ questionID, 'QuizDevice' ] = quizDevice
        
    return ( questionData, questionIDs  )
    