from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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

svc_model  = LinearSVC(random_state=0)

pred = svc_model.fit(data_train,target_train).predict(data_validation)
print("LinearSVC accuracy : ",accuracy_score(target_validation, pred,normalize = True))
