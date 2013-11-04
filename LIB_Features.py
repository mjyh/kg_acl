'''
Functions for turning raw accelerometer data into features
'''

import pandas as pd
import numpy as np

import LIB_MathUtil as ut
import LIB_Database as db
import LIB_CV as cv

from time import time

samplesPerSeq = 300   

### Returns a list of features for a 300-sample long sequence

def buildFeaturesFromSequence( rawDataForSeq ):
    
    xDataForSequence = []
    
    ### Tdiff stats

    #Get a list of timestamps, calculate the diff with a lag of 1, then calculate the trimmed mean
    t_data = rawDataForSeq[ 'T' ]
    
    t_data = t_data - t_data.shift(1) #compute diffs
    t_data = t_data[ 1:: ] #remove the first element because it had no previous time
    
    tDiff_mean = ut.meanTrimmed( t_data )
    tDiff_std = t_data.std()
    #tDiff_n_zeroes = sum( data[ data < 1] )
    tDiff_pt_90 = np.percentile( t_data, 90 )
    tDiff_pt_10 = np.percentile( t_data, 10)
    
    xDataForSequence.append( tDiff_mean )
    xDataForSequence.append( tDiff_std )
    xDataForSequence.append( tDiff_pt_90 )
    xDataForSequence.append( tDiff_pt_10 )
    
    ### X, Y, Z means, X/Y/Z_diff means
    
    for axis in [ "X", "Y", "Z" ]: #input the trimmed mean 
        
        axis_data = rawDataForSeq[ axis ]
        axis_diffs = axis_data - axis_data.shift(1)
        axis_diffs = axis_diffs[ 1:: ]
        
        axis_data = np.array(axis_data) #convert to numpy array
        
        axis_meanWeighted = np.average( axis_data[ :-1 ], weights = t_data )
        axis_stdWeighted =  ut.weighted_std( axis_data[ :-1 ], t_data )
        axis_diff_meanWeighted = np.average( axis_diffs, weights = t_data )
        
        xDataForSequence.append( axis_meanWeighted )
        xDataForSequence.append( axis_stdWeighted )
        xDataForSequence.append( axis_diff_meanWeighted )
    
    return xDataForSequence
    
### Returns features for test data
#   for each sequence, calculates features
#   returns np ndarray

def buildTestDataFeatures():
    
    idsPerReport = 100
    
    IDs = db.getSequenceIDs()
    
    # Set up return value
    X = []

    n_ids_done = 0
    numIDs = IDs.__len__()
    
    init = time()
    
    ### For each id, compute the features
    for ID in IDs:
        
        # Report on progress every so often
        if n_ids_done % idsPerReport == 0:
            print "    Populating features for id %s.. %d of %d done, %.1f s so far" % ( ID, n_ids_done, numIDs, time()-init )
        
        ### Get all raw data
        rawDataForSequence = db.getRawDataWhere( "test", ID, [ "X", "Y", "Z", "T" ] )
    
        featuresForSequence = buildFeaturesFromSequence( rawDataForSequence )
    
        X.append( featuresForSequence )
            
        n_ids_done = n_ids_done + 1
        
    return np.array( X )
    
    
### Builds features for the training data. 
#   The raw training data must be split into 300-sample sequences, dach sequence is transformed into a set of features
#
#    
#   Returns a tuple of (X, y), X is a np ndarray, y is a 1-dimensional ndarray
def buildTrainingDataFeaturesAnswers():
    idsPerReport = 1
    
    IDs = db.getDeviceIDs()
    
    # Set up return value
    X = None
    y = None

    n_ids_done = 0
    numIDs = IDs.__len__()
    
    init = time()
    
    ### For each id, compute the features
    for ID in IDs:
        
        xDataForID = []
        yDataForID = []
        
        # Report on progress every so often
        if n_ids_done % idsPerReport == 0:
            print "    Populating features for id %s.. %d of %d done, %.1f s so far" % ( ID, n_ids_done, numIDs, time()-init )
        
        ### Get all raw data
        rawDataForID = db.getRawDataWhere( "train", ID, [ "X", "Y", "Z", "T" ] )
        
        n_samples = rawDataForID.shape[0]
        n_sequences = int( np.floor( n_samples / samplesPerSeq ) )
        
        for cur_sequence in range( 0, n_sequences ):
            
            startInd = 1 + cur_sequence * samplesPerSeq
            endInd = ( cur_sequence + 1 ) * samplesPerSeq
                        
            rawDataForSeq = rawDataForID[ startInd:endInd + 1]
            
            xDataForSequence = buildFeaturesFromSequence( rawDataForSeq )

            xDataForID.append( xDataForSequence )
            yDataForID.append([ ID ] )
            
            #and another id complete
        n_ids_done = n_ids_done + 1
    
        xArrayForID = np.array( xDataForID )
        yArrayForID = np.array( yDataForID )
    
        if X is None:
            X = xArrayForID
        else: 
            X = np.vstack( ( X, xArrayForID) )
        
        if y is None:
            y = yArrayForID
        else:
            y = np.vstack( ( y, yArrayForID ) )
        
    return ( X, y )
    
    
### Build per-device training features for 1 NN method

def buildTrainingDataPerDevice( features ):
    return buildFeaturesDataPerDevice( "train", features )

### build per-device test features for the 1 NN method
def buildTestDataPerDevice( features ):
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

#### Normalize the training and test data
# X is a dataframe
# trainingData and testData are DataFrames of features, indexed by DeviceID or SequenceId
def normalizeAllData( X ):

    ### Normalize the data
    
    means = X.mean()
    stds = X.std()
    
    if pd.isnull(means).any(0):
        print "Error: Nan value in feature means for id %s" % id
    
    if pd.isnull(stds).any(0):
        print "Error: Nan value in feature stds for id %s" % id
    
    result = ( X - means ) / stds
    
    return result
    
# Normalize the training and test data
# trainingData and testData are DataFrames of features, indexed by DeviceID or SequenceId
def normalizeTrainTestData( trainingData, testData):

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