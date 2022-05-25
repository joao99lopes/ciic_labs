from xmlrpc.client import boolean
import pandas as pd
from datetime import date, time
import numpy as np

col_types = {
    "Date": date,
    "Time": time,
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



def check_missing_values(dataframe, row):
    df = dataframe
    print(df)
    for col in col_types.keys():
        # if a value isn't valid (wrong type) removes row
        if not isinstance(df[col][row].__class__, col_types[col].__class__):
            print("Error in row",row,"in col",col,"with value",df[col][i],"\ntype should be", col_types[col])
            df = df.drop([row])
        # if a value is missing, inserts last value
        if isinstance(df[col][row], str) and not(df[col][row] and df[col][row].strip()) or df[col][i] == None:
            print("Row",row,"had empty value in col", col)
            df[col] = df[col].replace([df[col][row]],df[col][row-1])
    return df


def remove_noise(dataframe, row):
    df = dataframe
    print(df)
    for col in col_types.keys():
        # if a value isn't valid (wrong type) removes row
        if not isinstance(df[col][row].__class__, col_types[col].__class__):
            print("Error in row",row,"in col",col,"with value",df[col][i],"\ntype should be", col_types[col])
            df = df.drop([row])
        # if a value is considered noise
        # CHECK IF THIS IS THE CORRECT WAY TO FIND NOISE
        if is_noise(df[col][row], col):
            break
    return df 


def preprocessing(dataframe):
    df = dataframe
    for i in range(len(df)):
        df = check_missing_values(dataframe=df, row=i)
    return df


def is_noise(row_content, col_type):
    if "PIR" in col and row_content not in (0,1):
        return True
    
    return False


def is_outlier(value, col):
    return

def populate_quartiles(df):
    for col in col_types.keys():
        col_quartiles[col] = quartile_default
    return