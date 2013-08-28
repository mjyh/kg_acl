'''

Create the CV test questions and answers, put them into the Accel database

'''


import numpy as np
import pandas as pd
import numpy.random as rand

from time import time

import CONST_Accel as Const
import CFG_SimilarDevice as SimDev
import LIB_Database as db
import pandas.io.sql

def setupSimilarDevices():
    for deviceID, similarDevices in SimDev.devices_clusters.iteritems():
        if deviceID in similarDevices:
            similarDevices.remove( deviceID )
        
### Returns a similar device to the given one, will not return the same deviceID back
def getRandomSimilarDevice( deviceID ):
    similarDevices = SimDev.devices_clusters[ deviceID ]
    index = int( np.floor( rand.uniform( 0, len( similarDevices ) ) ) )
    return similarDevices[ index ]
    
###

setupSimilarDevices()

### Consts

samplesPerSequence = 300

### Whether to actually save in database

saveToDB = False

### For each device, get the raw data, use last 40% for CV, split into sequences of 300, and assign quiz devices

deviceIDs = db.getDeviceIDs()
currentQuestionId = 1
currentSequenceID = 300000
nextAnswer = True

for deviceID in deviceIDs:
    init = time(); print "Generating info for device %d" % deviceID
    
    rawDataForDevice = db.getRawDataWhere("train", deviceID, [ "T", "X", "Y", "Z" ] )
    
    numEntries = rawDataForDevice.shape[0]
    
    trainingCutoffIndex = np.round( numEntries * .6 )
    CVStartIndex = trainingCutoffIndex + 1
    numSequences = int( np.floor( ( numEntries - CVStartIndex + 1)  / samplesPerSequence ) )
    
    CVEndIndex = CVStartIndex + numSequences*samplesPerSequence - 1
    
    CVRawData = rawDataForDevice[ CVStartIndex:(CVEndIndex + 1) ] #note, (CVEndIndex + 1)th element isn't included
    
    CVRawData[ 'SequenceId' ] = 0
    ### get db cursor
    cur = db.getDBCursor()
    con = db.getDBCon()
    
    ### set up question_cv entries for this device
    currentStartIndex = 1
    currentEndIndex = samplesPerSequence
    for sequenceIndex in range( 0, numSequences ):
        ### Choose True or False
        answer = nextAnswer
        nextAnswer = not nextAnswer
        
        if answer:
            quizDevice = deviceID
        else:
            quizDevice = getRandomSimilarDevice( deviceID )
        
        ### put question into question_cv table
        #query = "INSERT INTO questions_cv VALUES( %d, %d, %d, %d )" % (currentQuestionId, currentSequenceID, quizDevice, deviceID)
        
        #con.execute( query )
        
        ### set up test_cv entries for this device
        CVRawData[ 'SequenceId' ][ currentStartIndex:currentEndIndex + 1 ] = currentSequenceID
        
        
        
        currentQuestionId = currentQuestionId + 1
        currentSequenceID = currentSequenceID + 1
        currentStartIndex = currentStartIndex + samplesPerSequence
        currentEndIndex = currentEndIndex + samplesPerSequence
    pd.io.sql.write_frame(CVRawData, 'test_cv', con, if_exists='append')
    #con.commit()
    
    print "DONE! %.1f s\n" % ((time()-init))
    