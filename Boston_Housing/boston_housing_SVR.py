"""
Here try Support Vector Machines
to get house price forcast model
This file demonstrates how to analyze boston
housing dataset. and will upload the
results to kaggle.com and compete with people
in class!
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, metrics, model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'
SUBMISSION_EXAMPLE = 'boston_housing/submission_example.csv'


def main():

	# data cleaning
	train_data = data_preprocessing(TRAIN_FILE, mode='Train')
	test_data = data_preprocessing(TEST_FILE, mode='Test')

	svr(train_data, test_data)
	svr_all_cv(train_data, test_data)


def svr(train_data, test_data):
	"""
	The datas are divided into 70/30 for train and validation data
	"""

	svr_train_data, svr_val_data = model_selection.train_test_split(train_data, test_size=0.3)
	# get true label
	y_train_svr = svr_train_data.pop('medv')
	y_val_svr = svr_val_data.pop('medv')

	normalizer = preprocessing.MinMaxScaler()
	svr_train_data = normalizer.fit_transform(svr_train_data)
	svr_val_data = normalizer.transform(svr_val_data)
	test_data = normalizer.transform(test_data)

	svr = SVR(kernel='rbf', C=200, epsilon=1.0, gamma=1.0)
	svr.fit(svr_train_data, y_train_svr)
	svr_train_predictions = svr.predict(svr_train_data)
	print('RMS_train_Error_SVR: ', metrics.mean_squared_error(svr_train_predictions, y_train_svr) ** 0.5)

	svr_val_predictions = svr.predict(svr_val_data)
	print('RMS_val_Error_SVR: ', metrics.mean_squared_error(svr_val_predictions, y_val_svr) ** 0.5)

	predictions_test_svr = np.around(svr.predict(test_data), decimals=13)
	out_file(predictions_test_svr, 'svr.csv')


def svr_all_cv(train_data, test_data):

	"""
	Using whole data to be train data
	"""

	y_train = train_data.medv
	x_train = train_data.drop('medv', axis=1, inplace=False)

	normalizer = MinMaxScaler()
	x_train = normalizer.fit_transform(x_train)
	test_data = normalizer.transform(test_data)

	poly = PolynomialFeatures(degree=1)
	x_train = poly.fit_transform(x_train)
	test_data = poly.transform(test_data)

	svr_cv = SVR(kernel='rbf', C=280, epsilon=1.0, gamma=0.8)
	svr_cv.fit(x_train, y_train)
	# In this case, the result of cross_val_score doesn't make difference
	svr_cvs = cross_val_score(svr_cv, x_train, y_train, cv=10)
	svr_cv_predict = svr_cv.predict(x_train)
	print('RMS_train_Error_SVR_cv: ', metrics.mean_squared_error(svr_cv_predict, y_train) ** 0.5)

	predictions_test_svr_cv = np.around(svr_cv.predict(test_data), decimals=13)
	out_file(predictions_test_svr_cv, 'svr_all_cv.csv')


def data_preprocessing(filename, mode='Train'):

	# Will get: ['ID','crim','zn','indus', 'chas', 'nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
	data = pd.read_csv(filename)

	if mode == 'Train':
		features = ['crim', 'indus', 'nox','rm','age','dis','rad', 'tax', 'ptratio','black','lstat','medv']
		data = data[features]
	else:
		features = ['crim', 'indus', 'nox','rm','age','dis','rad', 'tax', 'ptratio','black','lstat']
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
