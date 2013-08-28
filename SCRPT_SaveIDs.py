'''
Get Device IDs and save them into a CSV
'''


import pandas as pd
import numpy as np


import CONST_Accel as Const
import LIB_Database as db


x = db.getDeviceIDs()
#df = pd.Series( x )
#df.to_csv(Const.DeviceIDFile, index="index", index_label = 1)

#x = db.getSequenceIDs()
#df = pd.Series( x )
#df.to_csv(Const.SequenceIDFile, index="index", index_label = 1)