

# https://www.geeksforgeeks.org/command-line-arguments-in-python/
import sys
# total arguments 
nargs = len(sys.argv) 
if(nargs < 4):
	print('not enough arguments')
	exit()

filename_model = sys.argv[1]
filename_xtest = sys.argv[2]
filename_ypredict = sys.argv[3]





#https://thispointer.com/pandas-loop-or-iterate-over-all-or-certain-columns-of-a-dataframe/#:~:text=DataFrame.iteritems(),and%20column%20contents%20as%20series.&text=As%20there%20were%203%20columns%20so%203%20tuples%20were%20returned%20during%20iteration.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# read data from file
print('--> reading csv...')
dataX = pd.read_csv(filename_xtest)


print(dataX.shape)
dataX.fillna(value=0, inplace=True) 
X = dataX[[
'color', 'manufacturer', 'model', 'odometer', 'year', 'engineType', 'photos',

'transmission', 'engineFuel', 'engineCapacity',
'bodyType','drivetrain',

'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 
'feature_6', 'feature_7', 'feature_8', 'feature_9', ]]






# load dictionary
filename_dictionary = 'dictionary'
print('--> loading dictionary...')
import pickle
mydict = pickle.load(open(filename_dictionary, 'rb'))
# print(mydict)


print('--> transforming data...')
# apply to data
X['color'].replace(
	to_replace=mydict['color'], 
	inplace=True,
	regex=True)

# X['manu_model'] = new_manu_model
X['manufacturer'].replace(
	to_replace=mydict['manu_model'], 
	inplace=True,
	regex=True)

X['model'].replace(
	to_replace=mydict['model'], 
	inplace=True,
	regex=True)

X['engineType'].replace(
	to_replace=mydict['engineType'], 
	inplace=True,
	regex=True)

X['odometer'] = -X['odometer']


X['transmission'].replace(
	to_replace=mydict['transmission'], 
	inplace=True,
	regex=True)

X['engineFuel'].replace(
	to_replace=mydict['engineFuel'], 
	inplace=True,
	regex=True)


X['bodyType'].replace(
	to_replace=mydict['bodyType'], 
	inplace=True,
	regex=True)

X['drivetrain'].replace(
	to_replace=mydict['drivetrain'], 
	inplace=True,
	regex=True)



sum_feature = []
for index, row in X.iterrows():
	keyName = sum([row['feature_0'] 
				, row['feature_1'] 
				, row['feature_2'] 
				, row['feature_3'] 
				, row['feature_4'] 
				, row['feature_5'] 
				, row['feature_6'] 
				, row['feature_7'] 
				, row['feature_8'] 
				, row['feature_9']]) 
	sum_feature.append(keyName)
X['sum_feature'] = sum_feature

sum_engine = []
for index, row in X.iterrows():
	keyName = sum([row['engineFuel'],
 				row['engineType'], 
 				row['engineCapacity'],
 				row['bodyType'],
 				row['drivetrain']
 				]) 
	sum_engine.append(keyName)
X['sum_engine'] = sum_engine




X = X[[
 'manufacturer',
 'model',
 'transmission', 
 'color',
 'odometer', 
 'year', 

 'engineFuel',
 'engineType', 
 'engineCapacity',
 'bodyType',
 'drivetrain',
 # 'sum_engine',
 
 'photos',
 'sum_feature'
]].values


print('--> data after transforming')
print(X)
print("---------------")



# use simple linear regression
#  Implementation using scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2_score2
 
# Cannot use Rank 1 matrix in scikit learn



# load the model from disk
print('--> loading model...\n')
import pickle
loaded_model = pickle.load(open(filename_model, 'rb'))


# Y Prediction
print('--> predicting...')
y_pred = loaded_model.predict(X)


print('--> calculate score, rmse')
# Calculating R2 Score
# print(r2_score2(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred)**0.5)

print('--> parameters and predict result:')
print(loaded_model.coef_)
# print(y_test)
print(y_pred)

print('--> save prediction to csv file')
pd.DataFrame(y_pred).to_csv(filename_ypredict)
