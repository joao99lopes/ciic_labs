from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

import lab6_aux

dir_path = os.path.join(os.getcwd(), 'lab6-p1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

dataset = pd.read_csv(file_path, sep=',', index_col=[0,1])

df = lab6_aux.pre_processing(dataset)

cols = [col for col in df.columns if col not in ['Persons']]

data = df[cols]
target = df['Persons']

data_temp, data_test, target_temp, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)
data_train, data_validation, target_train, target_validation = train_test_split(data_temp, target_temp, test_size = 0.16, random_state = 10)

# LinearSVC
from sklearn.svm import LinearSVC
svc_model  = LinearSVC(random_state=0)
pred = svc_model.fit(data_train,target_train).predict(data_validation)
print("LinearSVC accuracy : ",accuracy_score(target_validation, pred,normalize = True))

# Naive-Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
pred = gnb.fit(data_train,target_train).predict(data_validation)
print("Naive-Bayes accuracy : ",accuracy_score(target_validation, pred, normalize = True))

# K-Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
pred = neigh.fit(data_train,target_train).predict(data_validation)
print ("KNeighbors accuracy score : ",accuracy_score(target_validation,pred))

#####
# Performance Comparison
###
from yellowbrick.classifier import ClassificationReport

# GaussianNB
visualizer = ClassificationReport(gnb,classes=["0 Persons","1 Persons","2 Persons","3 Persons"])
visualizer.fit(data_train,target_train)
visualizer.score(data_validation,target_validation)
g = visualizer.poof()

# LinearSVC
visualizer = ClassificationReport(svc_model,classes=["0 Persons","1 Persons","2 Persons","3 Persons"])
visualizer.fit(data_train,target_train)
visualizer.score(data_validation,target_validation)
g = visualizer.poof()

# KNeighborsClassifier
visualizer = ClassificationReport(neigh,classes=["0 Persons","1 Persons","2 Persons","3 Persons"])
visualizer.fit(data_train,target_train)
visualizer.score(data_validation,target_validation)
g = visualizer.poof()

