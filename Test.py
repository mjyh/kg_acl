'''
Created on Aug 28, 2013

@author: Mike
'''

import CONST_Accel as Const
import pandas as pd
import pandas.io.sql
import LIB_Database as db
from time import time

df = pd.DataFrame( [ {'T': 12, 'X': 1.5, 'Y': 2.4, 'Z':4.5, 'SequenceId': 14 } ] )

con = db.getDBCon()

#pandas.io.sql.write_frame(df, 'test_cv', con, if_exists='append')
#init = time(); print "starting to read"
#df = pd.read_csv(Const.TrainDataFile)

#print "DONE! %.1f s\n" % ((time()-init))