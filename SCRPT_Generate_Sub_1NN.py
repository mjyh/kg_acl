'''
Script used to generate a submission for the accelerometer Kaggle competition

USAGE: SCRPT_GenerateSubmission.py <a>

a: 1 to write an output file, 0 otherwise
'''

import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial
import scipy.stats

import CONST_Accel as Const
import LIB_Database as db
import LIB_Features as ft

from time import time

### Parameters

# Calculate/retrieve from database certain data, or load previously saved calculations
loadTrain = 1
loadTest = 1
loadQuestion = 1

# Whether to save intermediate results to .csv
saveTrain = 0
saveTest = 0
saveQuestion = 0

# write results to output file
saveSubmission = 0

printResults = 0

# What features will we use to classify points
features = [ "X_mean", "Y_mean", "Z_mean", "Tdiff_mean" ] 

### Load data: features for training and test data, and questions to answer

# Training data
if loadTrain:
    init = time(); print "Loading training features from .csv"
    trainingData = pd.DataFrame.from_csv( Const.TrainFeaturesFile )
    print "DONE! %.1f s\n" % ((time()-init))
else:
    init = time(); print "Calculating training features"
    trainingData = ft.buildTrainingDataPerDevice( features )  
    print "DONE! %.1f s\n" % ((time()-init))

# Test data
if loadTest:
    init = time(); print "Loading testing features from .csv"
    testData = pd.DataFrame.from_csv( Const.TestFeaturesFile )
    print "DONE! %.1f s\n" % ((time()-init))
else:
    init = time(); print "Populating testing features"
    testData = ft.buildTestDataPerDevice( features )
    print "DONE! %.1f s\n" % ((time()-init))

# Question data
if loadQuestion:
    init = time(); print "Loading testing features from .csv"
    questionData = pd.DataFrame.from_csv( Const.QuestionFile )
    print "DONE! %.1f s\n" % ((time()-init))
else:
    init = time(); print "Populating question data"
    questionData = db.getQuestionData()
    print "DONE! %.1f s\n" % ((time()-init))

# Save data so we don't have to always recompute features
if saveTrain:
    init = time(); print "saving training features"
    trainingData.to_csv( Const.TrainFeaturesFile, index_label = "Device" )
    print "DONE! %.1f s\n" % ((time()-init))

if saveTest:
    init = time(); print "saving test features"
    testData.to_csv( Const.TestFeaturesFile, index_label = "SequenceId" )
    print "DONE! %.1f s\n" % ((time()-init))

if saveQuestion:
    init = time(); print "saving question data"
    questionData.to_csv( Const.QuestionFile, index_label = "QuestionId" )
    print "DONE! %.1f s\n" % ((time()-init))


### Normalize the data
# Each row of the training or test data is a point in 4d space (Tdiff_mean, X_mean, Y_mean, Z_mean)
# The next step is to compute the distance between each training/test point

trainingData, testData = ft.normalizeData( trainingData, testData )


### Some setup for the Nearest Neighbor algorithm

init = time(); print "Getting various ids"
deviceIDs = db.getDeviceIDs()
sequenceIDs = db.getSequenceIDs()
questionIDs = db.getQuestionIDs()
print "DONE! %.1f s\n" % ((time()-init))

def indexToDeviceID( ind ):
    return deviceIDs[ ind ]

numDevices = trainingData.shape[0]
numSequences = testData.shape[0]
numQuestions = questionData.shape[0]


### Calculate euclidean squared distances, don't need to square root it because only rankings matter

init = time(); print "Computing squared distances"
euclideanSq = sp.spatial.distance.cdist( testData, trainingData, 'sqeuclidean' )
euclideanSq = pd.DataFrame( euclideanSq, index = sequenceIDs, columns = deviceIDs )
print "DONE! %.1f s\n" % ((time()-init))


### Now for each sequenceID, get a list of all devices, sorted by distance

# ranks is indexed by SequenceId. Iterating through the columns of a given index provides deviceIDs, in increasing order of distance from the test sequence point

init = time(); print "Sorting ranks"
ranks = np.argsort( euclideanSq, axis = 1 )
print "DONE! %.1f s\n" % ((time()-init))

init = time(); print "Converting sort results to device IDs"
ranks = ranks.applymap( indexToDeviceID )
print "DONE! %.1f s\n" % ((time()-init))

### Build the answers

# For each sequence, ranks.loc[ SequenceId ] gives a Series containing the devices closest to that sequence
init = time(); print "buliding answers"

answers = pd.DataFrame( index = questionIDs, columns = [ 'IsTrue' ] )

numDone = 0
for questionID in questionIDs:
    
    if numDone % 10000 == 0:
        print "%d of %d done" % ( numDone, numQuestions )
    
    questionDeviceID = questionData.at[ questionID, 'QuizDevice' ]
    sequenceID = questionData.at[ questionID, 'SequenceId' ]
    ranksForSequence = ranks.loc[ sequenceID ]
    
    # iterate through all known devices, in ascending order of distance to the test point, until we find the QuizDevice
    # score is the negative of the rank of the QuizDevice, eg, if QuizDevice's location is the closest to the sequence's location, score = -1
    # higher scores (less negative) correspond with a higher chance of the QuizDevice being the actual device behind the test sequence
    score = -1
    for currentDeviceID in ranksForSequence:
        if currentDeviceID == questionDeviceID:
            break
        score = score - 1
        
    answers.at[ questionID, 'IsTrue' ] = score
    
    numDone = numDone + 1
    
print "DONE! %.1f s\n" % ((time()-init))

### Save answers
if saveSubmission:
    init = time(); print "saving answers"
    answers.to_csv( Const.SubmissionFile, index_label = 'QuestionId' )
    print "DONE! %.1f s\n" % ((time()-init))

### For debugging small examples

if printResults:
    print "training Data:\n %s\n" % trainingData
    print "testData:\n %s\n" % testData
    print "squared distances:\n %s\n" % euclideanSq
    print "device ID ranks:\n %s\n" % ranks
    print "answers:\n%s\n" % answers

