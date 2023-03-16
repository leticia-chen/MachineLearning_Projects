"""
File: titanic_level2.py
Name: Leticia Chen
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

nan_cache = defaultdict(float)


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""

	data = pd.read_csv(filename)
	cache = defaultdict(float)

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	# print(data.count())

	if mode == 'Train':
		# To select key that are interested
		key_need = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		# Dictionary now is with only 8 keys
		data = data[key_need]
		# Moving missing data row
		data.dropna(inplace=True)
		# Get and remove labels data
		labels = data.pop('Survived')
		# Keep average of Age and Fare´s into nan_cache dictionary
		nan_cache['Age'] = round(data['Age'].mean(), 3)
		nan_cache['Fare'] = round(data['Fare'].mean(), 3)
		# print(nan_cache.Age)
		return data, labels

	elif mode == 'Test':
		key_need = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data = data[key_need]
		# Fill in the NaN cells by the values in nan_cache to make it consistent
		data['Age'].fillna(nan_cache['Age'], inplace=True)
		data['Fare'].fillna(nan_cache['Fare'], inplace=True)
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""

	if feature == 'Sex':
		# One hot encoding for a new category Male
		data['Sex_1'] = 0  # default=0
		data.loc[data.Sex == 1, 'Sex_1'] = 1
		# One hot encoding for a new category Female
		data['Sex_0'] = 0
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		# No need Sex anymore!
		data.pop('Sex')
	elif feature == 'Pclass':
		# One hot encoding for a new category FirstClass
		data['Pclass_0'] = 0
		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		# One hot encoding for a new category SecondClass
		data['Pclass_1'] = 0
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		# One hot encoding for a new category ThirdClass
		data['Pclass_2'] = 0
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		# No need Pclass anymore!
		data.pop('Pclass')
	else:
		data['Embarked_0'] = 0
		data.loc[data.Embarked == 0, 'Embarked_0'] = 1
		data['Embarked_1'] = 0
		data.loc[data.Embarked == 1, 'Embarked_1'] = 1
		data['Embarked_2'] = 0
		data.loc[data.Embarked == 2, 'Embarked_2'] = 1
		data.pop('Embarked')

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""

	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)

	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> 0.8019662921348315
	TODO: real accuracy on degree2 -> 0.8370786516853933
	TODO: real accuracy on degree3 -> 0.8764044943820225
	"""
	train_data, Y = data_preprocess(TRAIN_FILE, mode='Train')
	test_data = data_preprocess(TEST_FILE, mode='Test')
	train_data = one_hot_encoding(train_data, 'Sex')
	train_data = one_hot_encoding(train_data, 'Pclass')
	train_data = one_hot_encoding(train_data, 'Embarked')

	# To do standardization to get X´s miu and sigma value which will be keep in standerdizer(object)
	standerdizer = preprocessing.StandardScaler()
	X_train = standerdizer.fit_transform(train_data)

	# To define model h
	h = linear_model.LogisticRegression(max_iter=10000)

	# This is polynomial feature
	poly_phi = preprocessing.PolynomialFeatures(degree=1)
	# 让直线根据几个特征而弯几弯
	X_train = poly_phi.fit_transform(X_train)

	# Output of classifier is weight
	classifier = h.fit(X_train, Y)
	# weight*X to compare with Y
	acc = classifier.score(X_train, Y)
	print(acc)


if __name__ == '__main__':
	main()
