from cProfile import label
from cmath import nan
from functools import total_ordering
from math import isnan
from xmlrpc.client import boolean
import pandas as pd
from datetime import date, time
import numpy as np
import matplotlib.pyplot as plt
import collections


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


def pre_processing(dataframe):
    df = check_missing_values(dataframe)
    df = remove_noise(df)
    populate_quartiles(df)
    df = clean_outliers(df)
    df = min_max_normalization(df)
    return df


def add_binary_result(dataframe):
    above_limit = []
    for row in range(len(dataframe)):
        if dataframe['Persons'][row] > 2:
            above_limit.append(1)
        else:
            above_limit.append(0)
    dataframe["AboveLimit"] = above_limit
    return dataframe
####################
# NOME DA CAIXINHA #
####################

def check_missing_values(dataframe):
    # fill missing values with last valid value
    dataframe.fillna(method='ffill', inplace=True)
    return dataframe


def remove_noise(dataframe):
    df = dataframe
    row_index=0    
    while (row_index < len(df)):
        for col in col_types.keys():
            # if a value is considered noise (negative value) its row is removed
            if is_noise(df, row_index, col):
                df.drop([df.index[row_index]],inplace=True)
                row_index-=1
        row_index+=1
    return df 


def clean_outliers(dataframe):
    df = dataframe
    row_index = 0
    outlier_count = 0
    outlier_rows = []
    cols = ["S1Temp","S1Light","S3Light"]
    initlen = len(df)
    last_valid_value = {}
    while row_index < len(df):
        for col in cols:
            if is_outlier(df, row_index, col):
                df.drop([df.index[row_index]],inplace=True)
                if row_index not in outlier_rows:
                    outlier_rows.append(row_index)
                outlier_count +=1
                populate_quartiles(df)
                row_index-=1
            else:
                last_valid_value[col] = df[col][row_index]
        row_index+=1
    print("outliers: {} total: {}".format(outlier_count,len(outlier_rows)))
    return df


def z_score_normalization(dataframe):
    df = dataframe
    cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons"]]
    for row in range(len(df)):
        for col in cols:
            # z_score = (x-mean)/std
            z_score_value = (df[col][row] - col_quartiles[col]["mean"])/col_quartiles[col]["std"]
            df.iloc[row, df.columns.get_loc(col)] = z_score_value
    return df

def min_max_normalization(dataframe):
    df = dataframe
    cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons"]]
    for row in range(len(df)):
        for col in cols:
            # min_max = (x-min)/(max-min)
            min_max = (df[col][row] - col_quartiles[col]["min"])/(col_quartiles[col]["max"] - col_quartiles[col]["min"])
            df.iloc[row, df.columns.get_loc(col)] = min_max
    return df



#################
# AUX functions #
#################

def is_noise(dataframe, row_index, col_type):
    # if a value isn't valid (wrong type) removes row
    if not isinstance(dataframe[col_type][row_index].__class__, col_types[col_type].__class__):
        print("Noise detected! Cause: invalid type in row {} col {}".format(row_index,col_type))
        return True
    # if PIR is not a binary value
    elif "PIR" in col_type and dataframe[col_type][row_index] not in (0,1):
        print("Noise detected! Cause: invalid PIR value in row {} ".format(row_index))
        return True
    # if a value is negative
    elif dataframe[col_type][row_index] < 0:
        print("Noise detected! Cause: negative value in row {} col {}".format(row_index,col_type))
        return True
    # if movement is detected and the room is empty (ghostbusters!)
    elif "PIR" in col_type and dataframe[col_type][row_index] == 1 and dataframe["Persons"][row_index] == 0:
        print("Noise detected! Cause: movement detected in empty room in row {} ".format(row_index))
        return True
    return False


def is_outlier(dataframe, row_index, col):
    value = dataframe[col][row_index]
    q1 = col_quartiles[col]["q1"]
    q3 = col_quartiles[col]["q3"]
    outlier_limitation = 20
    lower_limit = q1/outlier_limitation
    upper_limit = q3/(1/outlier_limitation) # multiplying was raising an error
    if value < lower_limit or value > upper_limit:
        print("Outlier detected row {} col {} value {} upper {} lower {}".format(row_index,col,value,upper_limit,lower_limit))
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
        tmp["std"] = dataframe[col].std()   # standart deviation
        col_quartiles[col] = tmp        
    return


def draw_density_graph(dataframe,col):
    res = {}
    for row in range(len(dataframe)):
        if dataframe[col][row] not in res.keys():
            res[dataframe[col][row]] = 0
        res[dataframe[col][row]] += 1
    od = collections.OrderedDict(sorted(res.items()))
    
    plt.plot(od.keys(),od.values())
    plt.ylabel(col)
    plt.show()
            

def draw_graph(dataframe, split=False):
    if split:
        for col in dataframe.columns:
            dataframe[col].plot()
            plt.ylabel(col)
            plt.show()
    else:
        dataframe.plot()
        plt.show()