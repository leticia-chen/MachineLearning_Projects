"""
File: titanic_level1.py
Name: Leticia Chen
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
from collections import defaultdict
from statistics import mean
from util import *
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	filename = TRAIN_FILE if mode == 'Train'else TEST_FILE
	data = defaultdict(list)
	with open(filename, 'r') as f:
		first_line = True
		for line in f:
			line = line.strip()
			if first_line:
				first_line = False
			else:
				lst = line.split(',')
				# The line will be ignore if Age value or Embarked value is missing data in Train mode
				if not lst[6] or not lst[len(lst) - 1]:
					continue
				data['PassengerId'].append(int(lst[0]))
				if mode == 'Train':
					data['Survived'].append(int(lst[1]))
					start = 2
				else:
					start = 1
				for i in range(len(lst)):
					if i == start:
						data['Pclass'].append(int(lst[i]))
					elif i == start + 1:
						data['Name'].append(lst[i])
					elif i == start + 3:
						data['Sex'].append(1) if lst[i] == 'male' else data['Sex'].append(0)
					elif i == start + 4:
						data['Age'].append(float(lst[i])) if lst[i] != '' else data['Age'].append(29.642)
					elif i == start + 5:
						data['SibSp'].append(int(lst[i]))
					elif i == start + 6:
						data['Parch'].append(int(lst[i]))
					elif i == start + 7:
						data['Ticket'].append(lst[i])
					elif i == start + 8:
						data['Fare'].append(float(lst[i])) if lst[i] != '' else data['Fare'].append(34.567)
					elif i == start + 9:
						data['Cabin'].append(lst[i])
					else:
						if i == start + 10:
							if lst[i] == 'S':
								data['Embarked'].append(0)
							elif lst[i] == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)

	# For Training data using shole ignore below column data
	if mode == 'Train' or mode == 'Test':
		data.pop('PassengerId')
		data.pop('Name')
		data.pop('Ticket')
		data.pop('Cabin')

	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	index = []
	lst = []

	# feature 是需要做 one hot encoding 的 key 的名称
	for key, value in data.items():
		# Elements(value) in a feature(key) its minimum number will be 0
		if feature == key:
			lst = start_zero(value)
		# To get how many unique element of value (ex:[0, 1, 2], means 3 different elements in value)
		for num in lst:
			if num not in index:
				index.append(num)

	# Built new necessary key-value in data dictionary, default value is 0
	# 本来放在 for key, value 底下，但错误讯息说 change dict size during interaction
	for m in range(len(index)):
		data[f'{feature}_{m}'] = [0]*len(lst)

	# Each feature represent its true value
	for i in range(len(lst)):
		# 新key的编号，如果0号，而旧的value值也是0，则新value的值会是 1
		for j in index:
			if lst[i] == j:
				data[f'{feature}_{lst[i]}'][i] = 1

	data.pop(feature)
	return data


def start_zero(lst):
	"""
	To let feature data minimum number starting from zero
	:param lst: list, value of feature
	:return: list, feature minmum number will be 0
	"""
	ele = min(lst)
	# gap = ele - 0
	new_lst = []
	for i in range(len(lst)):
		new_num = lst[i]-ele
		new_lst.append(new_num)
	return new_lst


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key, value in data.items():
		big = max(value)
		small = min(value)
		if big > 1:
			for i in range(len(value)):
				data[key][i] = (value[i]-small)/(big-small)
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())							# 取keys出来，装入list; ex: ['Pclass', 'Age',...]
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0						# 每个feature的weight的初始值为0 {'Age':0,'SibSp':0,'Parch':0,...}
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0			# {'AgeAge':0, 'AgeSibSp':0, 'AgeFare':0...}
	# Step 2 : Start training
	for epoch in range(num_epochs):
		for num in range(len(labels)):					# To run 712 data lines
			if degree == 1:
				# Step 3: Feature extract
				feature_vector_d1 = featureExtractor(inputs, num, degree)
				# Step 4: Weights updating
				k = dotProduct(feature_vector_d1, weights)
				h = 1/(1+math.exp(-k))
				increment(weights, (- alpha * (h - labels[num])), feature_vector_d1)
			else:
				# Step 3: Feature extract
				feature_vector_d2 = featureExtractor(inputs, num, degree)
				# Step 4: Weights updating
				k = dotProduct(feature_vector_d2, weights)
				h = 1 / (1 + math.exp(-k))
				increment(weights, (- alpha * (h - labels[num])), feature_vector_d2)

	return weights


def featureExtractor(inputs, num, degree):
	"""
	To get only one line feature_vector:
	for example->{''Age':0.2,'SibSp':0,'Parch':1,...'AgeAge':0.068, 'AgeSibSp':0, 'AgeFare':0.34}
	:param inputs: dict[str, list], key is the column name, value is its data
	:param num: int, index of labels
	:param degree: int, degree of polynomial features
	:return: feature vector: dict[str, float], key is the column name, value is its data of one line
	"""
	feature_vector_d1 = defaultdict(float)
	if degree == 1:
		for key, value in inputs.items():
			# To extract only one line value of each column, ex: {'Age':0.2,'SibSp':0,'Parch':1,...}
			feature_vector_d1[key] = value[num]
		return feature_vector_d1

	feature_vector_d2 = defaultdict(float)
	keys = list(inputs.keys())
	if degree == 2:
		for key, value in inputs.items():
			feature_vector_d2[key] = value[num]

		# To extract only one line´s data value into a list [0.2, 0, 1, ...]
		values = list(feature_vector_d2.values())
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				# Here will be feature element * feature element {'AgeAge':0.068, 'AgeSibSp':0, 'AgeFare':0.34}
				feature_vector_d2[keys[i] + keys[j]] = values[i]*values[j]
		# Will return {''Age':0.2,'SibSp':0,'Parch':1,...'AgeAge':0.068, 'AgeSibSp':0, 'AgeFare':0.34}
		return feature_vector_d2
