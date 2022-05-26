from cmath import nan
from math import isnan
from xmlrpc.client import boolean
import pandas as pd
from datetime import date, time
import numpy as np

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

quartile_default = {
    "min": 0,
    "q1": 0,
    "mean": 0,
    "q3": 0,
    "max": 0,
}



def check_missing_values(dataframe, row_index):
    df = dataframe
#    print(df)
    for col in dataframe.columns:
        # if a value isn't valid (wrong type) removes row
        if not isinstance(df[col][row_index].__class__, col_types[col].__class__):
            print("Error in row",row_index,"in col",col,"with value",df[col][row_index],"\ntype should be", col_types[col])
            df = df.drop([row_index])
        # if a value is missing, inserts last value
        if isnan(df[col][row_index]):
            print("Row",row_index,"had empty value in col", col)
            print(df[col][row_index],"---------------------------")
#            df.iloc[row_index,col] = df.iloc[row_index-1,col]
#            df.at[row_index,col]=df[col][row_index-1]
#            df.loc[row_index,[col]] = df[col][row_index-1]

            df[col].replace([df[col][row_index]], [df[col][row_index-1]], inplace=True)
#            print(df.iloc[[row_index-1,row_index]],"\n\nlen: {}\n".format(len(df[col])))
    return df


def remove_noise(dataframe, row_index):
    df = dataframe
#    print(df)
    for col in col_types.keys():
        # if a value isn't valid (wrong type) removes row
        if not isinstance(df[col][row_index].__class__, col_types[col].__class__):
            print("Error in row",row_index,"in col",col,"with value",df[col][row_index],"\ntype should be", col_types[col])
            df = df.drop([row_index])
        # if a value is considered noise
        # CHECK IF THIS IS THE CORRECT WAY TO FIND NOISE
        if is_noise(df, row_index, col):
            # do code
            break
    return df 


def preprocessing(dataframe):
    df = dataframe
#    populate_quartiles(df)
    print("'{}'".format(df['S2Temp'][4571]))
    print(df.iloc[[4571]])

    for i in range(len(df)):
        df = check_missing_values(dataframe=df, row_index=i)
    return df


def is_noise(dataframe, row_index, col_type):
    if "PIR" in col_type and dataframe[col_type][row_index] not in (0,1):
        return True
    elif dataframe[col_type][row_index] < 0:
        return True
    return False


def is_outlier(value, col):
    #1.5*(q3-q1)
    return

def populate_quartiles(dataframe):
    for col in dataframe.columns:
        aux = quartile_default
        aux["min"] = min(dataframe[col])
        aux["q1"] = dataframe[col].quantile(0.25)
        aux["mean"] = dataframe[col].quantile(0.5)
        aux["q3"] = dataframe[col].quantile(0.75)
        aux["max"] = max(dataframe[col])
        col_quartiles[col] = aux
        print(col, aux)
    return