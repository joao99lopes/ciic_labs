from ProcessData import ProcessDataFrame

import os
dir_path = os.path.join(os.getcwd(), 'lab6-p1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

def TestMe(filename):
    p = ProcessDataFrame(filename)

TestMe(file_path)