import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os
import pandas as pd
import lab11_aux

dir_path = os.path.join(os.getcwd(), 'lab6-p1.1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

dataset = pd.read_csv(file_path, sep=',')

df = lab11_aux.pre_processing(dataset)

x_light1 = np.sort(df["S1Light"].to_numpy())
x_light2 = np.sort(df["S2Light"].to_numpy())
x_light3 = np.sort(df["S3Light"].to_numpy())
#x_CO2_increase = df["CO2Acceleration"]

#       [ INICIO , MEIO , FINAL ]
l1_norm = (100 - x_light1.min())/(x_light1.max() - x_light1.min())
l2_norm = (100 - x_light2.min())/(x_light2.max() - x_light2.min())
l3_norm = (200 - x_light3.min())/(x_light3.max() - x_light3.min())

light1_off = fuzz.trimf(x_light1, [0, 0, l1_norm])  
light1_on = fuzz.trimf(x_light1, [l1_norm, 1, 1])

light2_off = fuzz.trimf(x_light2, [0, 0, l2_norm])  
light2_on = fuzz.trimf(x_light2, [l2_norm, 1, 1])  

light3_off = fuzz.trimf(x_light3, [0, 0, l3_norm])  
light3_on = fuzz.trimf(x_light3, [l3_norm, 1, 1])  


fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_light1, light1_off, 'r', linewidth=1.5, label='Off')
ax0.plot(x_light1, light1_on, 'g', linewidth=1.5, label='On')
ax0.set_title('Light 1')
ax0.legend()

ax1.plot(x_light2, light2_off, 'r', linewidth=1.5, label='Off')
ax1.plot(x_light2, light2_on, 'g', linewidth=1.5, label='On')
ax1.set_title('Light 2')
ax1.legend()

ax2.plot(x_light3, light3_off, 'r', linewidth=1.5, label='Off')
ax2.plot(x_light3, light3_off, 'g', linewidth=1.5, label='On')
ax2.set_title('Light 3')
ax2.legend()


for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()