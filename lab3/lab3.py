import lab3_aux
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.join(os.getcwd(), 'lab3', 'Lab3_DataSets')
file_name = 'EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv'
file_path = os.path.join(dir_path, file_name)

custom_date_parser = lambda x: lab3_aux.parse_date(x)

columns = ['Time (UTC)', 'Open', 'High', 'Low', 'Close', 'Volume']
#columns = ['Open', 'High', 'Low', 'Close', 'Volume']

df = pd.read_csv(file_path, sep=',', index_col=[0], parse_dates=['Time (UTC)'], date_parser=custom_date_parser)
#df = pd.read_csv(file_path, sep=',', usecols=columns, index_col=0, on_bad_lines='skip', parse_dates=True, infer_datetime_format=True, date_parser=custom_date_parser)
#print(df)
df1 = lab3_aux.drop_invalid_lines(df)

#lab3_aux.print_dataframe(df1)
#df1.set_index('Time (UTC)')

print(lab3_aux.sigma(df1))

#x = df1['Time (UTC)']
#y = df1[['Open','High','Low','Close','Volume']]

df1.plot()
#plt.ylabel('teste')
plt.show()

