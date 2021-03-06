from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



dir_path = os.path.join(os.getcwd(), 'lab4', 'Lab4_DataSets')
file_name = 'WA_Fn-UseC_-Sales-Win-Loss.csv'
file_path = os.path.join(dir_path, file_name)

sales_data = pd.read_csv(file_path, sep=',')

le = preprocessing.LabelEncoder()

#convert the categorical columns into numeric
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])


cols = [col for col in sales_data.columns if col not in['Opportunity Number','Opportunity Result']]
data = sales_data[cols]

target = sales_data['Opportunity Result']

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, random_state=10)

svc_model = LinearSVC(random_state=0)

pred = svc_model.fit(data_train, target_train).predict(data_test)

print("LinearSVC accuracy : ",accuracy_score(target_test, pred,normalize = True))
