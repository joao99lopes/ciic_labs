from datetime import datetime
from sqlite3 import Timestamp
import math
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
            if not drop and not isinstance(df['Open'][i], np.float64):
                df.loc[df.index[i], "Open"] = 0.0
        except Exception as e:
            df.loc[df.index[i], "Open"] = 0.0
        try:            
            if not drop and not isinstance(df['High'][i], np.float64):
                df.loc[df.index[i], "High"] = 0.0
        except Exception as e:
            df.loc[df.index[i], "High"] = 0.0
        try:            
            if not drop and not isinstance(df['Low'][i], np.float64):
                df.loc[df.index[i], "Low"] = 0.0
        except Exception as e:
            df.loc[df.index[i], "Low"] = 0.0
        try:            
            if not drop and not isinstance(df['Close'][i], np.float64):
                df.loc[df.index[i], "Close"] = 0.0
        except Exception as e:
            df.loc[df.index[i], "Close"] = 0.0
        try:            
            if not drop and not isinstance(eval(df['Volume'][i]), (float,int)):
                df.loc[df.index[i], "Volume"] = 0.0
            else:
                df.loc[df.index[i], "Volume"] = eval(df['Volume'][i])
        except Exception as e:
            df.loc[df.index[i], "Volume"] = 0.0
    return df


def parse_date(time_col):
    date = time_col.split('.')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2][:2])
    startTime = datetime(year,month,day,0,0)
    return startTime


def print_dataframe(df):
    for i in range(len(df)):
        print(df.loc[df.index[i]])
        
        
def sigma(df):
    aux = {
        'Open':[],
        'High':[],
        'Low':[],
        'Close':[],
        'Volume':[]
    }
    res = {}
    for i in range(len(df)):
        for key in aux.keys():
            aux[key].append(((df[key][i] - df.mean(axis='index')[key])**2))
    for key in aux.keys():
        res[key] = math.sqrt(sum(aux[key])/len(aux[key]))
    
    return res