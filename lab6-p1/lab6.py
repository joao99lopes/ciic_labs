from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ClassificationReport
import pandas as pd
import pickle
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
filename = 'exercise_1_model.sav'

# DATA VALIDATION
#clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,2))
#pred = clf.fit(data_train.values,target_train.values).predict(data_validation.values)
#print("PROB BIN:",accuracy_score(target_validation,pred))
#visualizer = ClassificationReport(clf,classes=["Under limit","Above limit"])
#visualizer.fit(data_train.values, target_train.values)
#visualizer.score(data_validation.values, target_validation.values)
#g = visualizer.poof()


clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,2))
clf_fit = clf.fit(data_train.values,target_train.values)
#pred = clf_fit.predict(data_test.values)
pickle.dump(clf_fit, open(filename, 'wb'))
#print("PROB A:",accuracy_score(target_test,pred))
#print("MACRO_PRECISION A:",precision_score(target_test,pred,average='macro'))
#print("MACRO_RECALL A:",recall_score(target_test,pred,average='macro'))
#print("MACRO_F1 A:",f1_score(target_test,pred,average='macro'))
#visualizer = ClassificationReport(clf,classes=["Under limit","Above limit"])
#visualizer.fit(data_train.values, target_train.values)
#visualizer.score(data_test.values, target_test.values)
#g = visualizer.poof()


#####################
# Amount of Persons #
#####################

data_temp, data_test, target_temp, target_test = train_test_split(data, target, test_size = 0.20, random_state = 10)
data_train, data_validation, target_train, target_validation = train_test_split(data_temp, target_temp, test_size = 0.25, random_state = 10)
filename = 'exercise_2_model.sav'
# DATA VALIDATION
#clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,1), max_iter=256)
#pred = clf.fit(data_train.values,target_train.values).predict(data_validation.values)
#print("PROB B:",accuracy_score(target_validation,pred))
#print("MACRO_PRECISION B:",precision_score(target_test,pred,average='macro'))
#print("MACRO_RECALL B:",recall_score(target_test,pred,average='macro'))
#print("MACRO_F1 B:",f1_score(target_test,pred,average='macro'))
#visualizer = ClassificationReport(clf)
#visualizer.fit(data_train.values, target_train.values) # Fit the training data to the
#visualizer.score(data_validation.values, target_validation.values)
#g = visualizer.poof()

clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(4,1), max_iter=256)
clf_fit = clf.fit(data_train.values,target_train.values)
#pred = clf_fit.predict(data_test.values)
pickle.dump(clf_fit, open(filename, 'wb'))
#print("PROB B:",accuracy_score(target_test,pred))
#print("MACRO_PRECISION B:",precision_score(target_test,pred,average='macro'))
#print("MACRO_RECALL B:",recall_score(target_test,pred,average='macro'))
#print("MACRO_F1 B:",f1_score(target_test,pred,average='macro'))
#visualizer = ClassificationReport(clf)
#visualizer.fit(data_train.values, target_train.values) # Fit the training data to the
#visualizer.score(data_test.values, target_test.values)
#g = visualizer.poof()

'''
res = {}
for i in range(1,6):
    for j in range(1,6):
        target_feature='AboveLimit'
        if target_feature == "AboveLimit":
            clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(i,j))
        else:
            clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1, solver='lbfgs', hidden_layer_sizes=(i,j), max_iter=400)
            
        pred = clf.fit(data.values,target.values).predict(data.values)
        print(target_feature,"PROB:",accuracy_score(target,pred))
        print(target_feature,"MACRO_PRECISION:",precision_score(target,pred,average='macro'))
        print(target_feature,"MACRO_RECALL:",recall_score(target,pred,average='macro'))
        print(target_feature,"MACRO_F1:",f1_score(target,pred,average='macro'))
        visualizer = ClassificationReport(clf)
        if target_feature == "AboveLimit":
            visualizer = ClassificationReport(clf,classes=["Under limit","Above limit"])
        visualizer.fit(data.values, target.values)
        visualizer.score(data.values, target.values)
        g = visualizer.poof()
'''