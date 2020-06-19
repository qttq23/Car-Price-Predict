
# https://www.geeksforgeeks.org/command-line-arguments-in-python/
import sys
# total arguments 
nargs = len(sys.argv) 
if(nargs < 3):
	print('not enough arguments')
	exit()

filename_ytest = sys.argv[1]
filename_ypred = sys.argv[2]


###############################

import numpy as np
import pandas as pd
# read data from file
print('--> reading csv...')
ytest = pd.read_csv(filename_ytest)
ypred = pd.read_csv(filename_ypred)

print(ytest)
print(ypred)

ytest = ytest['price'].values
ypred = ypred['0'].values

print(ytest)
print(ypred)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2_score2
 

print('--> calculate score, rmse')
print(r2_score2(ytest, ypred))
print(mean_squared_error(ytest, ypred)**0.5)

