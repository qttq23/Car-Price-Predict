

# requirements:
1. anaconda3 installed
2. libraries: numpy, pandas, matplotlib, PIL, sklearn, pickle, seaborn

# run:
1. open anaconda3 and navigate to this folder
2. type:
python train.py <path_to_Xtrain> <path_to_Ytrain> <path_to_save_model>
python test.py <path_to_model> <path_to_Xtest> <path_to_save_Ypredict>

eg:
python train.py res\X_train.csv res\Y_train.csv model
python test.py model res\X_train.csv output\y_pred.csv

# evaluate:
python evaluate.py <path_to_Ytest> <path_to_Ypredict>
python evaluate.py res\Y_train.csv output\y_pred.csv


# Car price prediction project
## Linear Regression
