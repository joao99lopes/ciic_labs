from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ClassificationReport
import pandas as pd
import os

import lab6_aux

dir_path = os.path.join(os.getcwd(), 'lab6-p1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

dataset = pd.read_csv(file_path, sep=',', index_col=[0,1])

df = lab6_aux.pre_processing(dataset)

cols = [col for col in df.columns if col not in ['Persons','AboveLimit']]

data = df[cols]
target = df['Persons']
target_bin = df['AboveLimit']


###############
# Above Limit #
###############

data_temp, data_test, target_temp, target_test = train_test_split(data, target_bin, test_size = 0.20, random_state = 10)
data_train, data_validation, target_train, target_validation = train_test_split(data_temp, target_temp, test_size = 0.25, random_state = 10)

clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,2))
pred = clf.fit(data_train.values,target_train.values).predict(data_validation.values)
print("PROB BIN:",accuracy_score(target_validation,pred))
visualizer = ClassificationReport(clf,classes=["Under limit","Above limit"])
visualizer.fit(data_train.values, target_train.values)
visualizer.score(data_validation.values, target_validation.values)
g = visualizer.poof()


#####################
# Amount of Persons #
#####################

data_temp, data_test, target_temp, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)
data_train, data_validation, target_train, target_validation = train_test_split(data_temp, target_temp, test_size = 0.25, random_state = 10)

# lbfgs
clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,1), max_iter=256)
pred = clf.fit(data_train.values,target_train.values).predict(data_validation.values)
print("PROB lbfgs:",accuracy_score(target_validation,pred))
visualizer = ClassificationReport(clf)
visualizer.fit(data_train.values, target_train.values) # Fit the training data to the
visualizer.score(data_validation.values, target_validation.values)
g = visualizer.poof()

