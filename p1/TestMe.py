from ProcessData import ProcessDataFrame
import os
import sys

dir_path = os.path.join(os.getcwd())
file_name = sys.argv[1]
file_path = os.path.join(dir_path, file_name)

def TestMe(filename):
    p = ProcessDataFrame(filename)

TestMe(file_path)