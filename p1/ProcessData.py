from copyreg import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from yellowbrick.classifier import ClassificationReport
import pandas as pd
import numpy as np
import pickle
class ProcessDataFrame:
    
    def __init__(self, filepath):
        self.get_col_types()
        self.col_quartiles = {}
        self.filepath = filepath
        self.dataframe = self.get_dataframe_from_filepath()
        self.pre_process_dataframe()
        self.test_dataframe("AboveLimit")
        self.test_dataframe("Persons")
    
    
    def get_col_types(self):
        # type of each feature in the dataframe 
        self.col_types = {
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
        return
    
    def populate_quartiles(self):
        for col in self.dataframe.columns:
            tmp = {}
            tmp["min"] = self.dataframe[col].min()
            tmp["q1"] = self.dataframe[col].quantile(0.25)
            tmp["median"] = self.dataframe[col].quantile(0.5)
            tmp["q3"] = self.dataframe[col].quantile(0.75)
            tmp["max"] = self.dataframe[col].max()
            tmp["mean"] = self.dataframe[col].mean()
            tmp["std"] = self.dataframe[col].std()   # standart deviation
            self.col_quartiles[col] = tmp        
        return
      
        
    def get_dataframe_from_filepath(self):
        dataframe = pd.read_csv(self.filepath, sep=',', index_col=[0,1])
        return dataframe
    
    
    def pre_process_dataframe(self):
        self.check_missing_values()
        self.remove_noise()
        self.populate_quartiles()
        self.clean_outliers()
        self.min_max_normalization()
        self.add_binary_result()
        return
    
    
    def check_missing_values(self):
        # fill missing values with last valid value
        self.dataframe.fillna(method='ffill', inplace=True)
        return
    
    
    def remove_noise(self):
        df = self.dataframe
        row_index=0    
        while (row_index < len(df)):
            for col in self.col_types.keys():
                # if a value is considered noise (negative value) its row is removed
                if self.is_noise(df, row_index, col):
                    df.drop([df.index[row_index]],inplace=True)
                    row_index-=1
            row_index+=1
        self.dataframe = df
        return 

    def is_noise(self, dataframe, row_index, col_type):
        # if a value isn't valid (wrong type) removes row
        if not isinstance(dataframe[col_type][row_index].__class__, self.col_types[col_type].__class__):
            return True
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


    def clean_outliers(self):
        df = self.dataframe
        row_index = 0
        cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons"]]
        last_valid_value = {}
        while row_index < len(df):
            for col in cols:
                if self.is_outlier(df, row_index, col):
                    df.drop([df.index[row_index]],inplace=True)
                    self.populate_quartiles()
                    row_index-=1
                else:
                    last_valid_value[col] = df[col][row_index]
            row_index+=1
        self.dataframe = df
        return
    
    def is_outlier(self, dataframe, row_index, col):
        value = dataframe[col][row_index]
        q1 = self.col_quartiles[col]["q1"]
        q3 = self.col_quartiles[col]["q3"]
        outlier_limitation = 20
        lower_limit = q1/outlier_limitation
        upper_limit = q3/(1/outlier_limitation) # multiplying was raising an error
        if value < lower_limit or value > upper_limit:
            return True
        return False


    def min_max_normalization(self):
        df = self.dataframe
        cols = [col for col in df.columns if col not in ["PIR1","PIR2","Persons"]]
        for row in range(len(df)):
            for col in cols:
                # min_max = (x-min)/(max-min)
                x = df[col][row]
                min = self.col_quartiles[col]["min"]
                max = self.col_quartiles[col]["max"]
                min_max = (x - min)/(max - min)
                df.iloc[row, df.columns.get_loc(col)] = min_max
        self.dataframe = df
        return


    def add_binary_result(self):
        above_limit = []
        for row in range(len(self.dataframe)):
            above_limit.append(int(self.dataframe['Persons'][row] > 2))
        self.dataframe["AboveLimit"] = above_limit        
        return
    
    
    def test_dataframe(self,target_feature):
        
        cols = [col for col in self.dataframe.columns if col not in ['Persons','AboveLimit']]
        data = self.dataframe[cols]
        target = self.dataframe[target_feature]
        if target_feature == "AboveLimit":
            classes = ["Under Limit","Above Limit"]
            clf = pickle.load(open('exercise_1_model.sav', 'rb'))
            print("\n################")
            print("#  Exercise 1  #")
            print("################")
        else:
            classes = ["0 Persons","1 Persons","2 Persons","3 Persons"]
            clf = pickle.load(open('exercise_2_model.sav', 'rb'))
            print("\n################")
            print("#  Exercise 2  #")
            print("################")
        
        pred = clf.predict(data.values)
        
        print("Precision")
        score = precision_score(target,pred,average=None)
        for i in range(len(score)):
            print("\t{}:".format(classes[i]), score[i])
        print("\tmacro:",precision_score(target,pred,average='macro'))
        
        print("Recall")
        score = recall_score(target,pred,average=None)
        for i in range(len(score)):
            print("\t{}:".format(classes[i]), score[i])
        print("\tmacro:",recall_score(target,pred,average='macro'))
        
        print("F1")
        print("\tmacro:",f1_score(target,pred,average='macro'))
        
        visualizer = ClassificationReport(clf,classes=classes)
        visualizer.fit(data.values, target.values)
        visualizer.score(data.values, target.values)
        g = visualizer.poof()
