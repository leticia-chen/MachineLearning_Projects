"""
Here using Support Linear Regression
to get house price forcast model
This file demonstrates how to analyze boston
housing dataset. and will upload the
results to kaggle.com and compete with people
in class!
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, metrics, model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_predict

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'
SUBMISSION_EXAMPLE = 'boston_housing/submission_example.csv'


def main():

	# data cleaning
	train_data = data_preprocessing(TRAIN_FILE, mode='Train')
	test_data = data_preprocessing(TEST_FILE, mode='Test')

	linear_regression(train_data, test_data)
	linear_regression_cv(train_data, test_data)


def linear_regression(train_data, test_data):

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.3)

	normalizer = preprocessing.MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)
	x_val = normalizer.transform(x_val)
	test_data = normalizer.transform(test_data)

	poly_phi = preprocessing.PolynomialFeatures(degree=2)
	x_train = poly_phi.fit_transform(x_train)
	x_val = poly_phi.transform(x_val)
	test_data = poly_phi.transform(test_data)

	# model: linear regression
	linear = linear_model.LinearRegression(n_jobs=1)
	linear.fit(x_train, y_train)

	predictions_train = linear.predict(x_train)
	print('RMS_train_Error_linear: ', metrics.mean_squared_error(predictions_train, y_train) ** 0.5)

	# validation data Error:
	predict_val = linear.predict(x_val)
	print('RMS_validation_Error_linear: ', metrics.mean_squared_error(predict_val, y_val) ** 0.5)

	prediction_test = np.around(linear.predict(test_data), decimals=13)
	out_file(prediction_test, 'linear_regression_degree2.csv')


def linear_regression_cv(train_data, test_data):

	"""
	Using cross-validation
	"""

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.3)

	normalizer = preprocessing.MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)
	x_val = normalizer.transform(x_val)
	test_data = normalizer.transform(test_data)

	poly_phi = preprocessing.PolynomialFeatures(degree=2)
	x_train = poly_phi.fit_transform(x_train)
	x_val = poly_phi.transform(x_val)
	test_data = poly_phi.transform(test_data)

	linear = linear_model.LinearRegression(n_jobs=1)
	parameters = {'n_jobs': range(1, 10)}

	g_search = GridSearchCV(linear, parameters, scoring='r2', cv=10)
	g_search.fit(x_train, y_train)

	print('best_score_**: ', g_search.best_score_)
	print('best_parameters_**: ', g_search.best_params_)

	predictions_train = g_search.predict(x_train)
	print('RMS_train_Error_linear_cv: ', metrics.mean_squared_error(predictions_train, y_train) ** 0.5)

	predict_val = g_search.predict(x_val)
	print('RMS_validation_Error_linear: ', metrics.mean_squared_error(predict_val, y_val) ** 0.5)

	prediction_test = np.around(g_search.predict(test_data), decimals=13)
	out_file(prediction_test, 'linear_regression_cv.csv')


def data_preprocessing(filename, mode='Train'):

	# Will get: ['ID','crim','zn','indus', 'chas', 'nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
	data = pd.read_csv(filename)
	# print(data)
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
