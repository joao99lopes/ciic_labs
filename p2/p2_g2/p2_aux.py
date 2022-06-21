from datetime import datetime
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import collections

# type of each feature in the dataframe 
col_types = {
    "Date": type(datetime.date),
    "Time": type(datetime.time),
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

# globally stores the quartiles per feature in the given dataframe
col_quartiles = {}


def pre_processing(dataframe, normalization=True):
    """Receives a dataframe, 
    fills its empty values, 
    removes rows with values that are considered "noisy" or "outliers", 
    normalize its values using min-max normalization,
    insert a binary collumn ('1' if there are more people in a room than the allowed amount) 
    and finally returns the processed dataframe.

    Args:
        dataframe (DataFrame): dataframe of the imported .csv file

    Returns:
        DataFrame: returns the processed dataframe 
    """
    df = convert_date_and_time(dataframe[dataframe.columns])
    df = check_missing_values(df)
  #  draw_graph(df,True)
    pre =len(df)
    df = remove_noise(df)
    noise =len(df)
    populate_quartiles(df)
    df = clean_outliers(df)
#    draw_graph(df,True)
    df = add_binary_result(df)
    if (normalization):
        df = min_max_normalization(df)
#    draw_graph(df,True)
    return df

def convert_date_and_time(dataframe):
    df = dataframe
    for row_index, row in df.iterrows():
        df.loc[row_index, "Time"] = datetime.strptime(df["Time"][row_index], "%H:%M:%S").time()
        df.loc[row_index, "Date"] = datetime.strptime(df["Date"][row_index], "%d/%m/%Y").date()
    return df

def add_binary_result(dataframe):

    above_limit = []

    for row_index, row in dataframe.iterrows():
        if dataframe['Persons'][row_index] > 2:
            above_limit.append(1)
        else:
            above_limit.append(0)
    dataframe["AboveLimit"] = above_limit
    return dataframe


def add_fuzzy_features(dataframe):
    lights = []
    acceleration = []
    time = []
    index = 1
    for row_index, row in dataframe.iterrows():
        
        lights.append(max(0, (dataframe["S1Light"][row_index] + dataframe["S2Light"][row_index] + dataframe["S3Light"][row_index])))

        if index <= 200:
            acceleration.append((dataframe["CO2"][row_index] - dataframe["CO2"][0])/index)
        else:
            aux = (dataframe["CO2"][row_index] - dataframe.iloc[[row_index-200],[dataframe.columns.get_loc("CO2")]])/200
            acceleration.append(aux.iat[0,0])
        index += 1
        
        time.append(dataframe["Time"][row_index].hour + dataframe["Time"][row_index].minute/60)

    dataframe["Lights"] = lights
    dataframe["CO2Acceleration"] = normalize_co2_acceleration(acceleration)
    dataframe["FloatTime"] = time
#    populate_quartiles(dataframe)

    return dataframe

def normalize_co2_acceleration(co2_acceleration):
    res = []    
    min_co2 = min(co2_acceleration)
    max_co2 = max(co2_acceleration)
    for i in range(len(co2_acceleration)):
        if (co2_acceleration[i] > 0):
            res.append(co2_acceleration[i]/abs(max_co2))
        else:
            res.append(co2_acceleration[i]/abs(min_co2))
    return res
#################
# AUX FUNCTIONS #
#################

def check_missing_values(dataframe):
    # fill missing values with last valid value
    dataframe.fillna(method='ffill', inplace=True)
    return dataframe


def remove_noise(dataframe):
    df = dataframe
    for row_index, row in dataframe.iterrows():
        for col in col_types.keys():
            # if a value is considered noise (negative value) its row is removed
            if is_noise(df, row_index, col):
                df=df.drop([row_index])
                break
    return df 


def clean_outliers(dataframe):
    df = dataframe
    outlier_count = 0
    outlier_rows = []
    cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons","Time","Date"]]
    last_valid_value = {}
    for row_index, row in dataframe.iterrows():
        for col in cols:
            if is_outlier(df, row_index, col):
                df = df.drop([row_index])
                if row_index not in outlier_rows:
                    outlier_rows.append(row_index)
                outlier_count +=1
                populate_quartiles(df)
                break
            else:
                last_valid_value[col] = df[col][row_index]
    return df


def min_max_normalization(dataframe):
    df = dataframe
    cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons","Time", "Date", "LightsOn", "AboveLimit"]]
    for row_index, row in dataframe.iterrows():
        for col in cols:
            # min_max = (x-min)/(max-min)
            min_max = (df[col][row_index] - col_quartiles[col]["min"])/(col_quartiles[col]["max"] - col_quartiles[col]["min"])
            df.loc[row_index, col] = min_max
    return df


def is_noise(dataframe, row_index, col_type):
    # if a value isn't valid (wrong type) removes row
    if not isinstance(dataframe[col_type][row_index].__class__, col_types[col_type].__class__):
        return True
    if col_type in ["Date","Time"]:
        return False
    # if PIR is not a binary value
    elif "PIR" in col_type and dataframe[col_type][row_index] not in (0,1):
        return True
    # if a value is negative
    elif dataframe[col_type][row_index] < 0:
        return True
    # if movement is detected and the room is empty
    elif "PIR" in col_type and dataframe[col_type][row_index] == 1 and dataframe["Persons"][row_index] == 0:
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
        return True
    return False


def populate_quartiles(dataframe):
    cols = [col for col in dataframe.columns if col not in ["Date","Time","Persons","LightsOn"]]
    for col in cols:
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