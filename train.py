

# https://www.geeksforgeeks.org/command-line-arguments-in-python/
import sys
# total arguments 
nargs = len(sys.argv) 
if(nargs < 4):
	print('not enough arguments')
	exit()

filename_xtrain = sys.argv[1]
filename_ytrain = sys.argv[2]
filename_model = sys.argv[3]





#https://thispointer.com/pandas-loop-or-iterate-over-all-or-certain-columns-of-a-dataframe/#:~:text=DataFrame.iteritems(),and%20column%20contents%20as%20series.&text=As%20there%20were%203%20columns%20so%203%20tuples%20were%20returned%20during%20iteration.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# read data from file
print('--> reading csv...')
dataX = pd.read_csv(filename_xtrain)
dataY = pd.read_csv(filename_ytrain)


print(dataX.shape)
print(dataY.shape)

dataX.fillna(value=0, inplace=True) 
# dataX = dataX.head(2470)
# dataY = dataY.iloc[2370:2390]


# make table X + Y
X = dataX[[
'color', 'manufacturer', 'model', 'odometer', 'year', 'engineType', 'photos',

'transmission', 'engineFuel', 'engineCapacity',
'bodyType','drivetrain',

'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 
'feature_6', 'feature_7', 'feature_8', 'feature_9', ]]

Y = dataY['price'].values
X['price'] = Y


# print(X)
# print(Y)
# print("---------------")


# dictionary to save catagory-value
mydict = {
	'color':{},
	'manufacturer':{},
	'model':{}, 
	'manu_model': {},
	'odometer':{},
	'year':{},
	'engineType':{},
	'photos':{},

	'transmission':{},
	'engineFuel':{},
	'engineCapacity':{},
	'bodyType':{},
	'drivetrain':{}
}
# 	".*[a-zA-Z].*": float("nan")





print('--> calculating data...')
# calculate mean for each column
X_color_mean = X.groupby(['color'], as_index=False).mean().sort_values(by='price', ascending=True)
X_manufacturer_mean = X.groupby(['manufacturer'], as_index=False).mean().sort_values(by='price', ascending=True)
X_model_mean = X.groupby(['model'], as_index=False).mean().sort_values(by='price', ascending=True)
# X_manufacturer_mean = X.groupby(['manufacturer', 'model'], as_index=False).mean().sort_values(by='price', ascending=True)

X_engineType_mean = X.groupby(['engineType'], as_index=False).mean().sort_values(by='price', ascending=True)

X_transmission_mean = X.groupby(['transmission'], as_index=False).mean().sort_values(by='price', ascending=True)
X_engineFuel_mean = X.groupby(['engineFuel'], as_index=False).mean().sort_values(by='price', ascending=True)
X_bodyType_mean = X.groupby(['bodyType'], as_index=False).mean().sort_values(by='price', ascending=True)
X_drivetrain_mean = X.groupby(['drivetrain'], as_index=False).mean().sort_values(by='price', ascending=True)

## mode
# X_color_mean = X.groupby(['color'], as_index=False).agg(lambda x:x.value_counts().index[0])


# save to dictionary
count = 1
for index, row in X_color_mean.iterrows():
    mydict['color'][row['color']] = count
    count+=1

count = 1
# new_manu_model = []
for index, row in X_manufacturer_mean.iterrows():
	keyName = row['manufacturer']
	mydict['manu_model'][keyName] = count
	count+=1

# for index, row in X.iterrows():
# 	keyName = row['manufacturer']
# 	new_manu_model.append(keyName)


count = 1
for index, row in X_model_mean.iterrows():
    mydict['model'][row['model']] = count
    count+=1

count = 1
for index, row in X_engineType_mean.iterrows():
    mydict['engineType'][row['engineType']] = count
    count+=1


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




count = 1
for index, row in X_transmission_mean.iterrows():
    mydict['transmission'][row['transmission']] = count
    count+=1

count = 1
for index, row in X_engineFuel_mean.iterrows():
    mydict['engineFuel'][row['engineFuel']] = count
    count+=1


count = 1
for index, row in X_bodyType_mean.iterrows():
    mydict['bodyType'][row['bodyType']] = count
    count+=1

count = 1
for index, row in X_drivetrain_mean.iterrows():
    mydict['drivetrain'][row['drivetrain']] = count
    count+=1


# save and load dictionary
filename_dictionary = 'dictionary'
print('--> saving dictionary to file...')
import pickle
pickle.dump(mydict, open(filename_dictionary, 'wb'))

print('--> loading dictionary...')
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

X['sum_feature'] = sum_feature


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



# print(X)
# print(Y)
# print("---------------")


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

Y = Y

print('--> data after transforming')
print(X)
print(Y)
print("---------------")


print('--> split train data')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# use simple linear regression
#  Implementation using scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2_score2
 
# Cannot use Rank 1 matrix in scikit learn


# Creating Model
reg = LinearRegression()

print('--> training...')
# Fitting training data
reg = reg.fit(x_train, y_train)



print('--> saving model...\n')
# save the model to disk
import pickle
pickle.dump(reg, open(filename_model, 'wb'))

print('--> loading model...\n')
# load the model from disk
loaded_model = pickle.load(open(filename_model, 'rb'))


print('--> predicting...')
# Y Prediction
y_pred = loaded_model.predict(x_test)
 

print('--> calculate score, rmse')
# Calculating R2 Score
print(r2_score2(y_test, y_pred))
print(mean_squared_error(y_test, y_pred)**0.5)

print('--> parameters and predict result:')
print(loaded_model.coef_)
print(y_test)
print(y_pred)
