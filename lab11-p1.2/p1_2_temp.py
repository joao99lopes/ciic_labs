import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os
import pandas as pd
from skfuzzy import control as ctrl

import lab11_aux

dir_path = os.path.join(os.getcwd(), 'lab6-p1.1')
file_name = 'Lab6Dataset.csv'
file_path = os.path.join(dir_path, file_name)

dataset = pd.read_csv(file_path, sep=',')

df = lab11_aux.pre_processing(dataset, normalization=False)
fuzzy_df = lab11_aux.add_fuzzy_features(df)
cols = ["Lights", "FloatTime", "CO2Acceleration"]
fuzzy_data = fuzzy_df[cols]


print((fuzzy_df))

x_lights = np.sort(fuzzy_data["Lights"].to_numpy())
x_time = np.sort(fuzzy_df["FloatTime"].to_numpy())
x_CO2_var = np.sort(fuzzy_data["CO2Acceleration"].to_numpy())

lights_0 = fuzz.trimf(x_lights, [0, 0, 150])  
lights_1 = fuzz.trimf(x_lights, [0, 150, 300])  
lights_2 = fuzz.trimf(x_lights, [150, 300, 450])  
lights_3 = fuzz.trimf(x_lights, [300, 450, 600])  

print("L1", np.average(np.sort(fuzzy_df.loc[fuzzy_df["S1Light"] > 100]["S1Light"].to_numpy())))
print("L2", (df["S2Light"].to_numpy().max()))
print("L3", (df["S3Light"].to_numpy().max()))

print("MININININININ", x_lights.min())
print("MAXAXAXAXAXAX", x_lights.max())


"""

co2_decrease_fast = fuzz.trimf(x_CO2_var, [-1, -1, -0.5])
co2_decrease = fuzz.trimf(x_CO2_var, [-1, -0.5, 0])
co2_stable = fuzz.trimf(x_CO2_var, [-0.3, 0, 0.3])
co2_increase = fuzz.trimf(x_CO2_var, [0, 0.5, 1])
co2_increase_fast = fuzz.trimf(x_CO2_var, [0.5, 1, 1])


light_1 = ctrl.Antecedent(np.arange(0, 2, 1), 'light_1')
light_2 = ctrl.Antecedent(np.arange(0, 2, 1), 'light_2')
light_3 = ctrl.Antecedent(np.arange(0, 2, 1), 'light_3')
daytime = ctrl.Antecedent(['Day', 'Night'], 'daytime')
co2_acceleration = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'co2_acceleration')
output = ctrl.Consequent(["Under Limit", "Above Limit"], 'output')

light_1["off"] = light1_off
light_1["on"] = light1_on
light_2["off"] = light2_off
light_2["on"] = light2_on
light_3["off"] = light3_off
light_3["on"] = light3_on

daytime.automf(names=["Day","Night"])
"""
"""
n_lights["0"] = 0
n_lights["1"] = 1
n_lights["2"] = 2
n_lights["3"] = 3

daytime["Day"] = "Day"
daytime["Night"] = "Night"
"""
"""
output["Decreasing Fast"] = co2_decrease_fast
output["Decreasing"] = co2_decrease
output["Stable"] = co2_stable
output["Increasing"] = co2_increase
output["Increasing Fast"] = co2_increase_fast

rule1 = ctrl.Rule(n_lights["3"], consequent=output["Above Limit"], label="rule 1")
rule2 = ctrl.Rule((n_lights["0"] | n_lights["1"] | n_lights["2"]) & daytime["Night"], consequent=output["Under Limit"], label="rule 2")
rule3 = ctrl.Rule((n_lights["0"] | n_lights["1"] | n_lights["2"]) & daytime["Day"] & co2_acceleration["Stable"], consequent=output["Under Limit"], label="rule 3")

system = ctrl.ControlSystem(rules=[rule1, rule2, rule3])
room_control = ctrl.ControlSystemSimulation(system)

room_control.input["NLights"] = 3
room_control.input["Daytime"] = "Day"
room_control.input["CO2Acceleration"] = 0.4

room_control.compute()
"""
'''
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

ax3.plot(x_CO2_var, co2_decrease_fast, 'r', linewidth=1.5, label='Decreasing Fast')
ax3.plot(x_CO2_var, co2_decrease, 'm', linewidth=1.5, label='Decreasing')
ax3.plot(x_CO2_var, co2_stable, 'g', linewidth=1.5, label='Stable')
ax3.plot(x_CO2_var, co2_increase, 'b', linewidth=1.5, label='Increasing')
ax3.plot(x_CO2_var, co2_increase_fast, 'c', linewidth=1.5, label='Increasing Fast')
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

'''
