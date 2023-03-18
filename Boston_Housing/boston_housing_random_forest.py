"""
Here using Random Forest
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'
SUBMISSION_EXAMPLE = 'boston_housing/submission_example.csv'


def main():

	# data cleaning
	train_data = data_preprocessing(TRAIN_FILE, mode='Train')
	test_data = data_preprocessing(TEST_FILE, mode='Test')

	random_forest(train_data, test_data)
	random_forest_cv(train_data, test_data)


def random_forest(train_data, test_data):

	train_data = train_data.drop('ptratio', axis=1, inplace=False)
	test_data = test_data.drop('ptratio', axis=1, inplace=False)

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.3)

	poly = preprocessing.PolynomialFeatures(degree=1)
	x_train = poly.fit_transform(x_train)
	x_val = poly.transform(x_val)
	test_data = poly.transform(test_data)

	model = RandomForestRegressor()

	parameters = {'max_depth': range(3, 5), 'min_samples_leaf': range(2, 4),
				  'max_leaf_nodes': range(9, 13)}

	gs = GridSearchCV(model, parameters, scoring='r2', cv=6)
	gs.fit(x_train, y_train)

	print('best_score: ', gs.best_score_)
	print('best_parameters: ', gs.best_params_)

	forest = RandomForestRegressor(max_depth=4, min_samples_leaf=2, max_leaf_nodes=11)
	forest.fit(x_train, y_train)
	train_error_forest = forest.predict(x_train)
	print('RMS_train_Error_forest: ', metrics.mean_squared_error(train_error_forest, y_train) ** 0.5)

	val_error_forest = forest.predict(x_val)
	print('RMS_val_Error_forest: ', metrics.mean_squared_error(val_error_forest, y_val) ** 0.5)

	prediction_test_forest = np.around(forest.predict(test_data), decimals=13)
	out_file(prediction_test_forest, 'random_forest.csv')


def random_forest_cv(train_data, test_data):

	"""
	Using cross validation
	"""

	train_data = train_data.drop('ptratio', axis=1, inplace=False)
	test_data = test_data.drop('ptratio', axis=1, inplace=False)

	y_train_forest = train_data.medv
	forest_train_data = train_data.drop('medv', axis=1, inplace=False)

	poly = PolynomialFeatures(degree=1)
	forest_train_data = poly.fit_transform(forest_train_data)
	test_data = poly.transform(test_data)

	model = RandomForestRegressor()

	parameters = {'max_depth': range(3, 5), 'min_samples_leaf': range(2, 4),
				  'max_leaf_nodes': range(9, 13)}

	gs = GridSearchCV(model, parameters, scoring='r2', cv=6)
	gs.fit(forest_train_data, y_train_forest)
	predictions = gs.predict(forest_train_data)
	print('RMS_train_Error_forest_cv_*: ', metrics.mean_squared_error(predictions, y_train_forest) ** 0.5)

	print('best_score: ', gs.best_score_)
	print('best_parameters: ', gs.best_params_)

	forest = RandomForestRegressor(max_depth=4, min_samples_leaf=2, max_leaf_nodes=11)
	forest.fit(forest_train_data, y_train_forest)
	score = cross_val_score(forest, forest_train_data, y_train_forest, cv=6, scoring='r2')
	print('score: ', score.mean())
	train_error_forest = forest.predict(forest_train_data)
	print('RMS_train_Error_forest_cv_**: ', metrics.mean_squared_error(train_error_forest, y_train_forest) ** 0.5)

	prediction_test_forest = np.around(forest.predict(test_data), decimals=13)
	out_file(prediction_test_forest, 'random_forest_cv.csv')


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
