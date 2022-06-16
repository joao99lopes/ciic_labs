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

df = lab11_aux.pre_processing(dataset, normalization=False)
fuzzy_df = lab11_aux.add_fuzzy_features(df)

x_light1 = np.sort(df["S1Light"].to_numpy())
x_light2 = np.sort(df["S2Light"].to_numpy())
x_light3 = np.sort(df["S3Light"].to_numpy())
x_CO2_increase = np.sort(df["CO2Acceleration"].to_numpy())

#       [ INICIO , MEIO , FINAL ]
#l1_norm = 0.20449897750511248
#l2_norm = 0.1937984496124031
#l3_norm = 0.35714285714285715
l1_norm = 100
l2_norm = 100
l3_norm = 200

print("l1",l1_norm)
print("l2",l2_norm)
print("l3",l3_norm)
light1_off = fuzz.trimf(x_light1, [x_light1.min(), x_light1.min(), l1_norm])  
light1_on = fuzz.trimf(x_light1, [l1_norm, x_light1.max(), x_light1.max()])

light2_off = fuzz.trimf(x_light2, [x_light2.min(), x_light2.min(), l2_norm])  
light2_on = fuzz.trimf(x_light2, [l2_norm, x_light2.max(), x_light2.max()])  

light3_off = fuzz.trimf(x_light3, [x_light3.min(), x_light3.min(), l3_norm])  
light3_on = fuzz.trimf(x_light3, [l3_norm, x_light3.max(), x_light3.max()])  

print("MININININININ", x_CO2_increase.min())


co2_decrease_fast = fuzz.trimf(x_CO2_increase, [x_CO2_increase.min(), x_CO2_increase.min(), -2])
co2_decrease = fuzz.trimf(x_CO2_increase, [-2, -1, -0.5])
co2_normal = fuzz.trimf(x_CO2_increase, [x_CO2_increase.min(), x_CO2_increase.min(), -0.5])
co2_increase = fuzz.trimf(x_CO2_increase, [x_CO2_increase.min(), x_CO2_increase.min(), -0.5])
co2_increase_fast = fuzz.trimf(x_CO2_increase, [x_CO2_increase.min(), x_CO2_increase.min(), -0.5])



fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

ax0.plot(x_light1, light1_off, 'r', linewidth=1.5, label='Off')
ax0.plot(x_light1, light1_on, 'g', linewidth=1.5, label='On')
ax0.set_title('Light 1')
ax0.legend()

ax1.plot(x_light2, light2_off, 'r', linewidth=1.5, label='Off')
ax1.plot(x_light2, light2_on, 'g', linewidth=1.5, label='On')
ax1.set_title('Light 2')
ax1.legend()

ax2.plot(x_light3, light3_off, 'r', linewidth=1.5, label='Off')
ax2.plot(x_light3, light3_on, 'g', linewidth=1.5, label='On')
ax2.set_title('Light 3')
ax2.legend()

ax3.plot(x_CO2_increase, co2_decrease_fast, 'r', linewidth=1.5, label='Decreasing Fast')
ax3.plot(x_CO2_increase, co2_decrease, 'r', linewidth=1.5, label='Decreasing')
ax3.plot(x_CO2_increase, co2_normal, 'r', linewidth=1.5, label='Normal')
ax3.plot(x_CO2_increase, co2_increase, 'r', linewidth=1.5, label='Increasing')
ax3.plot(x_CO2_increase, co2_increase_fast, 'r', linewidth=1.5, label='Increasing Fast')
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()