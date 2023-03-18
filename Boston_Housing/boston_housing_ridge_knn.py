"""
Here using Ridge and KNN
to get house price forcast model
This file demonstrates how to analyze boston
housing dataset. and will upload the
results to kaggle.com and compete with people
in class!
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, metrics, model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_predict
from sklearn.linear_model import Ridge

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'
SUBMISSION_EXAMPLE = 'boston_housing/submission_example.csv'


def main():

	# data cleaning
	train_data = data_preprocessing(TRAIN_FILE, mode='Train')
	test_data = data_preprocessing(TEST_FILE, mode='Test')

	knn(train_data, test_data)
	ridge(train_data, test_data)


def ridge(train_data, test_data):

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.2)

	normalizer = preprocessing.MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)
	x_val = normalizer.transform(x_val)
	test_data = normalizer.transform(test_data)

	poly_phi = preprocessing.PolynomialFeatures(degree=2)
	x_train = poly_phi.fit_transform(x_train)
	x_val = poly_phi.transform(x_val)
	test_data = poly_phi.transform(test_data)

	ridge = Ridge(alpha=0.0000001)
	ridge.fit(x_train, y_train)
	cv = cross_val_score(ridge, x_train, y_train, cv=10)
	x_prediction = ridge.predict(x_train)
	val_prediction = ridge.predict(x_val)
	print('RMS_train_Error_ridge: ', metrics.mean_squared_error(x_prediction, y_train) ** 0.5)
	print('RMS_val_Error_ridge: ', metrics.mean_squared_error(val_prediction, y_val) ** 0.5)
	print(ridge.coef_)

	test_prediction = ridge.predict(test_data)
	out_file(test_prediction, 'ridge.csv')


def knn(train_data, test_data):

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.3)

	normalizer = preprocessing.MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)
	x_val = normalizer.transform(x_val)
	test_data = normalizer.transform(test_data)

	model = KNeighborsRegressor()
	parameters = {'n_neighbors':range(1, 10), 'leaf_size': range(1, 5)}

	gs = GridSearchCV(model, parameters, scoring='r2', cv=5)
	gs.fit(x_train, y_train)
	print(gs.best_params_, gs.best_score_)
	predict_train = gs.predict(x_train)
	print('RMS_train_Error_knn: ', metrics.mean_squared_error(predict_train, y_train) ** 0.5)
	predict_val = gs.predict(x_val)
	print('RMS_val_Error_knn: ', metrics.mean_squared_error(predict_val, y_val) ** 0.5)

	predictions_test = np.around(gs.predict(test_data), decimals=13)
	out_file(predictions_test, 'knn.csv')


def data_preprocessing(filename, mode='Train'):

	# Will get: ['ID','crim','zn','indus', 'chas', 'nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
	data = pd.read_csv(filename)

	if mode == 'Train':
		features = ['crim', 'indus', 'nox','rm','age','dis','rad', 'ptratio','black','lstat','medv']
		data = data[features]
	else:
		features = ['crim', 'indus', 'nox','rm','age','dis','rad', 'ptratio','black','lstat']
		data = data[features]

	return data


def out_file(predictions, filename):
	data = pd.read_csv(TEST_FILE)
	id = data.ID

	print('\n=============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(id)):
			out.write(str(id[i])+','+str(predictions[i])+'\n')
	print('================================================')


if __name__ == '__main__':
	main()
