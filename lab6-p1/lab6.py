import pandas as pd
import os

import lab6_aux

dir_path = os.path.join(os.getcwd(), 'lab6-p1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

dataset = pd.read_csv(file_path, sep=',')

lab6_aux.check_missing_values(dataset)