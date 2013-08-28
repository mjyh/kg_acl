'''
Functions for turning raw accelerometer data into features
'''

import pandas as pd
import numpy as np

import LIB_MathUtil as ut
import LIB_Database as db

from time import time

def buildTrainingDataPerDevice( features ):
    return buildFeaturesDatPerDevicea( "train", features )

def buildTestDataperDevice( features ):
    return buildFeaturesDataPerDevice( "test", features )


# Returns a DataFrame indexed by DeviceID/SequenceID, with columns of features
# table is "train" or "test", for training or testing data
# features are which features to calculate
def buildFeaturesDataPerDevice( table, features ):
    
    # Different config values based on whether we're building features from testing or training data
    if table == "train":
        IDs = db.getDeviceIDs()
        idsPerReport = 1
    else:
        IDs = db.getSequenceIDs()
        idsPerReport = 100
    
    # Set up return value
    result = pd.DataFrame( index = IDs, columns = features ) #training data

    numDone = 0.0
    numIDs = IDs.__len__()
    
    init = time()
    
    ### For each id, compute the features
    for ID in IDs:
        
        # Report on progress every so often
        if numDone % idsPerReport == 0:
            print "    Populating features for id %s.. %d of %d done, %.1f s so far" % ( ID, numDone, numIDs, time()-init )
        
        rawDataForID = db.getRawDataWhere(table, ID, [ "X", "Y", "Z", "T" ] )
        
        # Populate each feature
        for feature in features:
            
            if feature in [ "X_mean", "Y_mean", "Z_mean" ]: #input the trimmed mean
                
                data = rawDataForID[ feature[0] ]
                #data = db.getRawDataWhere(table, ID, feature[0] ) #X_mean gets data from a column named 'X', etc.
                data = np.array(data) #convert to numpy array
                result.at[ ID,  feature ] = ut.meanTrimmed( data )
                
            elif feature in [ "Tdiff_mean" ]: 
                
                #Get a list of timestamps, calculate the diff with a lag of 1, then calculate the trimmed mean
                data = rawDataForID[ 'T' ]
                
                data = data - data.shift(1) #compute diffs
                data = data[ 1:: ] #remove the first element because it had no previous time
                result.at[ ID, feature ] = ut.meanTrimmed( data )
            else:
                raise Exception( "unable to generate feature %s for device %s" % ( feature, ID ) )
        
        
        #and another id complete
        numDone = numDone + 1
    
    return result


# Normalize the training and test data
# trainingData and testData are DataFrames of features, indexed by DeviceID or SequenceId
def normalizeData( trainingData, testData):

    ### Normalize the data
    
    means = trainingData.mean()
    stds = trainingData.std()
    
    if pd.isnull(means).any(0):
        print "Error: Nan value in feature means for id %s" % id
    
    if pd.isnull(stds).any(0):
        print "Error: Nan value in feature stds for id %s" % id
    
    trainingData = ( trainingData - means ) / stds
    testData = ( testData - means ) / stds
    
    return trainingData, testData