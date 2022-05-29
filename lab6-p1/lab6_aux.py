from cmath import nan
from functools import total_ordering
from math import isnan
from xmlrpc.client import boolean
import pandas as pd
from datetime import date, time
import numpy as np
import matplotlib.pyplot as plt

col_types = {
    "S1Temp": np.float64,
    "S2Temp": np.float64,
    "S3Temp": np.float64,
    "S1Light": np.int64,
    "S2Light": np.int64,
    "S3Light": np.int64,
    "CO2": np.int64,
    "PIR1": bool,    
    "PIR2": bool,
    "Persons": np.int64    
}

col_quartiles = {}


def preprocessing(dataframe):
    df = dataframe
    populate_quartiles(df)
#    for col in col_quartiles.keys():
#        print(col,"\n",col_quartiles[col])
    df = check_missing_values(dataframe=df)
    df = remove_noise(df)
    populate_quartiles(df)
    
    df = clean_outliers(df)
    for col in df.columns:
        df[col].plot()
        plt.ylabel(col)
#        plt.show()
    return df



def check_missing_values(dataframe):
    # fill missing values with last valid value
    dataframe.fillna(method='ffill', inplace=True)
    return dataframe


def remove_noise(dataframe):
    df = dataframe
    row_index=0    
    while (row_index < len(df)):
        for col in col_types.keys():
            # if a value isn't valid (wrong type) removes row
            if not isinstance(df[col][row_index].__class__, col_types[col].__class__):
                print("Error in row",row_index,"in col",col,"with value",df[col][row_index],"\ntype should be", col_types[col])
                df = df.drop([row_index])
            # if a value is considered noise (negative value) its row is removed
            if is_noise(df, row_index, col):
                df.drop([df.index[row_index]],inplace=True)
                row_index-=1
        row_index+=1
    return df 


def clean_outliers(dataframe):
    df = dataframe
    row_index = 0
    outlier_count = {}
    outlier_count_total = 0
    cols = [col for col in df.columns if col in["S1Temp","S3Temp","S1Light","S3Light"]]
    
    print("PRE-len",len(df))
    while row_index < len(df):
        for col in cols:
            if is_outlier(df, row_index, col):
                if (col not in outlier_count.keys()):
                    outlier_count[col] = 0
                outlier_count[col] +=1
                outlier_count_total +=1
#                df.drop([df.index[row_index]],inplace=True)
#                row_index-=1
        row_index+=1
    print("POS-len",len(df))
    print("outliers:\n{}\nTOTAL: {}".format(outlier_count,outlier_count_total))

    return df


###
# AUX functions
##

def is_noise(dataframe, row_index, col_type):
    if "PIR" in col_type and dataframe[col_type][row_index] not in (0,1):
        return True
    elif dataframe[col_type][row_index] < 0:
        return True
    return False


def is_outlier(dataframe, row_index, col):
    value = dataframe[col][row_index]
    q1 = col_quartiles[col]["q1"]
    q3 = col_quartiles[col]["q3"]
    outlier_limitation = 1.5
    lower_limit = q1/outlier_limitation
    upper_limit = q3/(1/outlier_limitation)

    if value < lower_limit or value > upper_limit:
        print("Outlier detected col {} value {} upper {} lower {}".format(col,value,upper_limit,lower_limit))
        return True
    return False


def populate_quartiles(dataframe):
    for col in dataframe.columns:
        tmp = {}
        tmp["min"] = dataframe[col].min()
        tmp["q1"] = dataframe[col].quantile(0.25)
        tmp["median"] = dataframe[col].quantile(0.5)
        tmp["q3"] = dataframe[col].quantile(0.75)
        tmp["max"] = dataframe[col].max()
        tmp["mean"] = dataframe[col].mean()
        tmp["std"] = dataframe[col].std()
        col_quartiles[col] = tmp        
    return