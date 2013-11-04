# -*- coding: utf-8 -*-
"""
Create X and y  
"""
import numpy as np
import pandas as pd

import LIB_Features as feat


# Build the data
X, y = feat.buildTrainingDataFeaturesAnswers()
#X[ 46409 ][4] = 0 #fix a nan

X_test = feat.buildTestDataFeatures()


# Save it

#np.save( 'RF_X.npy', X )
#np.save( 'RF_y.npy', y )
#np.save( 'RF_Xtest.npy', X_test )