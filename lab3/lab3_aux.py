from datetime import datetime
from sqlite3 import Timestamp
from time import process_time_ns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

'''
        print(i,"time: {}".format(dataframe['Time (UTC)'][i]))
        print(i,"open: {}".format(dataframe['Open'][i]))
        print(i,"high: {}".format(dataframe['High'][i]))
        print(i,"low: {}".format(dataframe['Low'][i]))
        print(i,"close: {}".format(dataframe['Close'][i]))
        print(i,"volume: {}\n".format(dataframe['Volume'][i]))
'''        

def drop_invalid_lines(dataframe):
    df = dataframe
    for i in range(len(df)):
        drop = False
        try:            
#            date = datetime.strptime(df['Time (UTC)'][i], "%Y.%m.%d %H:%M:%S")
#            if isinstance(df['Time (UTC)'], str):
#                drop = True
#                print("ERROR: Time (UTC):",i)
            if not drop and not isinstance(df['Open'][i], np.float64):
                drop = True
                print("ERROR: Open:",i)
            if not drop and not isinstance(df['High'][i], np.float64):
                drop = True
                print("ERROR: High:",i)
            if not drop and not isinstance(df['Low'][i], np.float64):
                drop = True
                print("ERROR: Low:",i)
            if not drop and not isinstance(df['Close'][i], np.float64):
                drop = True
                print("ERROR: Close:",i)
            if not drop and not isinstance(eval(df['Volume'][i]), (float, int)):
                drop = True
                print("ERROR: Volume:",i)

            if drop:
                print("\napaguei linha",i)
                df.drop(labels=[i,], axis=0, inplace=True)
        except Exception as e:
#            print("ERROR: 'drop_invalid_lines' in line {}.\nThe following line will be removed:\n{}\n\nCause of the error:\n{}".format(i, df.loc[df.index[i]], e))
            print(e)
            df.drop(labels=[df.index[i-1],], axis=0, inplace=True)
    
#    print("\n\nDF\n",df)
    return df


def parse_date(time_col):
    date = time_col.split('.')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2][:2])
    startTime = datetime(year,month,day,0,0)
    return startTime

