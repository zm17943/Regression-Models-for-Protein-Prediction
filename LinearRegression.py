import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
import heapq


file_location = '/Users/zhangzhang/nima/all_data.dat'
df = pd.read_csv(file_location)
df = df.loc[df['File Name'].str.contains("PBMC")]


# Patients' ids
sample_id = ["727", "730", "740", "742", "746", "765", "772", "779", "787"]
num_patients = 9

'''
Except for the patient id in the parameter, all patients' data are integrated 
into a training matrix where each sample has 329 features and 103 protein labels.
'''
def train_data(df, id):
	X = []
	Y = []
	for each_id in sample_id:
		if each_id == id: continue
		attributes = df.loc[df['File Name'].str.contains(each_id)]
		features = attributes.iloc[: , 1 : 329]
		labels = attributes.iloc[: , 329 : 432]
		for feature in features.values.tolist():
			X.append(feature)
		for label in labels.values.tolist():
			Y.append(label)
	X = np.mat(X)
	Y = np.mat(Y)
	return X, Y


'''
Data of the patient id in the parameter are integrated into a testing matrix 
where each sample has 329 features and 103 protein labels.
'''
def test_data(df, id):
	X = []
	Y = []
	for each_id in sample_id:
		if each_id != id: continue
		attributes = df.loc[df['File Name'].str.contains(each_id)]
		features = attributes.iloc[: , 1 : 329]
		labels = attributes.iloc[: , 329 : 432]
		for feature in features.values.tolist():
			X.append(feature)
		for label in labels.values.tolist():
			Y.append(label)
	X = np.mat(X)
	Y = np.mat(Y)
	return X, Y


loss = 0                                           # MSE loss for all predictors
predictor = [0] * 103                              # MSE loss for each protein predictor


# 9 rounds of "leave-one-out" evaluations
for each_sample_id in sample_id:

	X_train, Y_train = train_data(df, each_sample_id)
	X_test, Y_test = test_data(df, each_sample_id)

	reg = LinearRegression().fit(X_train, Y_train)     # Multiregressor

	y_hat = reg.predict(X_test)

	for i in range(len(X_test)):
		loss = loss + np.sqrt(np.sum(np.square(Y_test[i] - y_hat[i])))
		for j in range(103):
			predictor[j] += math.sqrt((np.array(Y_test)[i][j] - np.array(y_hat)[i][j]) * (np.array(Y_test)[i][j] - np.array(y_hat)[i][j]))


# Average the results
loss = loss / num_patients                         
predictor = np.array(predictor) / num_patients      

# The top five protein predictors and corrsponding loss
idx = np.argpartition(predictor, 5)
print(idx)
print(predictor[idx[:5]])


