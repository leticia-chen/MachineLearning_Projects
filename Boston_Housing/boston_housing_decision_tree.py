"""
Here using Decision Tree
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
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'
SUBMISSION_EXAMPLE = 'boston_housing/submission_example.csv'


def main():

	# data cleaning
	train_data = data_preprocessing(TRAIN_FILE, mode='Train')
	test_data = data_preprocessing(TEST_FILE, mode='Test')

	get_graphviz(train_data)

	decision_tree(train_data, test_data)
	decision_tree_cv(train_data, test_data)


def get_graphviz(train_data):

	"""
	This function is to get graphviz through
	In order to know which features will be better for traning
	"""

	price = train_data.medv
	names = ['crim','zn','indus', 'chas', 'nox','rm','age','dis','rad','tax','ptratio','black','lstat']
	features = train_data.drop('medv', axis=1, inplace=False)
	print(len(features))

	d_tree = DecisionTreeRegressor(max_depth=6)
	d_tree.fit(features, price)
	tree.export_graphviz(d_tree, feature_names=names, out_file='tree')


def decision_tree(train_data, test_data):

	price = train_data.medv
	features = train_data.drop('medv', axis=1, inplace=False)

	x_train, x_val, y_train, y_val = model_selection.train_test_split(features, price, test_size=0.3)

	model = DecisionTreeRegressor()
	parameters = {'max_depth': range(2, 6), 'max_leaf_nodes': range(19, 27), 'min_samples_leaf': range(2, 5)}

	# g_search = GridSearchCV(model, parameters, scoring='r2', cv=6)
	# g_search.fit(x_train, y_train)

	# print('best_score: ', g_search.best_score_)
	# print('best_parameters: ', g_search.best_params_)

	d_tree = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=19, min_samples_leaf=4)
	d_tree.fit(x_train, y_train)
	train_error_tree = d_tree.predict(x_train)
	print('RMS_train_Error_tree: ', metrics.mean_squared_error(train_error_tree, y_train) ** 0.5)

	val_error_tree = d_tree.predict(x_val)
	print('RMS_val_Error_tree: ', metrics.mean_squared_error(val_error_tree, y_val) ** 0.5)

	prediction_test_tree = np.around(d_tree.predict(test_data), decimals=13)
	out_file(prediction_test_tree, 'decision_tree.csv')


def decision_tree_cv(train_data, test_data):

	"""
	Using cross validation
	"""

	y_train = train_data.medv
	x_train = train_data.drop('medv', axis=1, inplace=False)

	model = DecisionTreeRegressor()
	parameters = {'max_depth':range(2, 7), 'max_leaf_nodes':range(18, 27), 'min_samples_leaf':range(2, 5)}

	g_search = GridSearchCV(model, parameters, scoring='r2', cv=6)
	g_search.fit(x_train, y_train)

	print('best_score: ', g_search.best_score_)
	print('best_parameters: ', g_search.best_params_)

	d_tree = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=20, min_samples_leaf=4)
	d_tree.fit(x_train, y_train)

	predic_train = d_tree.predict(x_train)
	print('RMS_train_Error_tree_cv: ', metrics.mean_squared_error(predic_train, y_train) ** 0.5)

	predic_test = np.around(d_tree.predict(test_data), decimals=13)
	out_file(predic_test, 'decision_tree_cv.csv')


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
