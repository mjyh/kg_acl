'''
Functions for accessing the raw accelerometer data in the sqlite database
'''

import pandas as pd
import sqlite3 as lite

import CONST_Accel as Const

from time import time

def getDBCon():
    con = lite.connect( Const.AccelDB )
    return con

#return a sqlite cursor object pointing to the main database
def getDBCursor():
    con = getDBCon()
    cur = con.cursor()
    return cur

#Returns a list of device IDs (from the training data)
def getDeviceIDs():
    result = pd.Series.from_csv(Const.DeviceIDFile)
    result = result.tolist()
    result = [int(x) for x in result]
    return result
    #return getIDs( "train", "Device" )

#Returns a list of sequence IDs (from the testing data)
def getSequenceIDs():
    return getIDs( "test", "SequenceID" )

#Returns a list of question IDs (from the question data)
def getQuestionIDs():
    return getIDs( "questions", "QuestionID" )

#Return a distinct list, given a table and a column
def getIDs( table, column ):
    cur = getDBCursor()

    query = "SELECT DISTINCT %s FROM %s" % ( column, table )
       
    cur.execute( query )
    rows = cur.fetchall()

    ids = [ row[0] for row in rows ]
    
    return ids

# Fetches a DataFrame containing training or test data for a particular device and column(s)
# table should be test or train
# columns can be "T", "X", "Y" or "Z", or a list of multiple columns
def getRawDataWhere( table, id, columns ):
    
    
    if type(columns) is str:
        columns = [ columns ]
        
    cur = getDBCursor()
    
    if table == "train":
        IDColumnName = "Device"
    else: #table == "test"
        IDColumnName = "SequenceID"
    
    columnsString = ",".join( columns )
    
    #eg, SELECT X FROM train WHERE Device='7'
    query = "SELECT %s FROM %s WHERE %s='%s'" % ( columnsString, table, IDColumnName, id )
    
    cur.execute( query )
    rows = cur.fetchall()
  
    result = pd.DataFrame( rows, columns = columns )
    
    return result

# Returns a DataFrame containing the question information needed for the test
def getQuestionData( ):
    cur = getDBCursor()
    
    query = "SELECT QuestionId, SequenceId, QuizDevice FROM questions" 
    
    cur.execute( query )
    rows = cur.fetchall()

    # DataFrame index is QuestionId
    index = [ row[0] for row in rows ]
    
    # DataFrame values are SequenceId and QuizDevice
    data = [ ( row[1], row[2] ) for row in rows ]
    
    result = pd.DataFrame( data, index = index, columns = [ 'SequenceId', 'QuizDevice' ] )
    
    return result