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
cols = ["Lights", "FloatTime", "CO2Acceleration", "S3Light"]
fuzzy_data = fuzzy_df[cols]


print((fuzzy_df))

x_light3 = np.sort(fuzzy_data["S3Light"].to_numpy())
x_lights = np.sort(fuzzy_data["Lights"].to_numpy())
x_time = np.sort(fuzzy_df["FloatTime"].to_numpy())
x_CO2_var = np.sort(fuzzy_data["CO2Acceleration"].to_numpy())

lights = ctrl.Antecedent(np.arange(0, 1500, 1), 'light')
daytime = ctrl.Antecedent(np.arange(0, 24, 1), 'daytime')
weather = ctrl.Antecedent(np.arange(0, 500, 1), 'weather')
output = ctrl.Consequent(np.arange(0, 2, 1), 'output')

weather_cloudy = fuzz.trimf(weather.universe, [0, 0, 170])
weather_sunny = fuzz.trimf(weather.universe, [170, 500, 500])

time_day = fuzz.trimf(daytime.universe, [7, 13, 19])
time_before_morning = fuzz.trimf(daytime.universe, [0, 0, 7])
time_evening = fuzz.trimf(daytime.universe, [19, 24, 24])

lights_under_700 = fuzz.trimf(lights.universe, [0, 0, 700])  
lights_over_700 = fuzz.trimf(lights.universe, [700, 1500, 1500])  
lights_under_1000 = fuzz.trimf(lights.universe, [0, 0, 1000])  
lights_over_1000 = fuzz.trimf(lights.universe, [1000, 1500, 1500])  

lights["night less than three"] = lights_under_700
lights["night more than three"] = lights_over_700
lights["cloudy less than three"] = lights_under_700
lights["cloudy more than three"] = lights_over_700
lights["sunny less than three"] = lights_under_1000
lights["sunny more than three"] = lights_over_1000

daytime["day"] = time_day
daytime["before morning"] = time_before_morning
daytime["evening"] = time_evening

weather["sunny"] = weather_sunny
weather["cloudy"] = weather_cloudy

output.automf(names=["under limit", "above limit"])
#output["under limit"] = 0
#output["above limit"] = 1

#rule_weather_sunny = 
rule1 = ctrl.Rule(antecedent=((daytime["before morning"] | daytime["evening"]) & lights["night less than three"]), consequent=output["under limit"], label="rule 1")
rule2 = ctrl.Rule(antecedent=((daytime["before morning"] | daytime["evening"]) & lights["night more than three"]), consequent=output["above limit"], label="rule 2")
rule3 = ctrl.Rule(antecedent=(daytime["day"] & weather["sunny"] & lights["sunny less than three"]), consequent=output["under limit"], label="rule 3")
rule4 = ctrl.Rule(antecedent=(daytime["day"] & weather["sunny"] & lights["sunny more than three"]), consequent=output["above limit"], label="rule 4")
rule5 = ctrl.Rule(antecedent=(daytime["day"] & weather["cloudy"] & lights["cloudy less than three"]), consequent=output["under limit"], label="rule 5")
rule6 = ctrl.Rule(antecedent=(daytime["day"] & weather["cloudy"] & lights["cloudy more than three"]), consequent=output["above limit"], label="rule 6")

system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5, rule6])
room_control = ctrl.ControlSystemSimulation(system, flush_after_run = len(fuzzy_df))


z = []
for row_index, row in fuzzy_df.iterrows():
    room_control.input["light"] = x_lights[row_index]
    room_control.input["daytime"] = x_time[row_index]
    room_control.input["weather"] = x_light3[row_index]
    room_control.compute()
    z.append(room_control.output["output"])
    print(room_control.output["output"], fuzzy_df["AboveLimit"][row_index], fuzzy_df["Persons"][row_index])
"""

tmp=[]
for row_index, row in fuzzy_df.iterrows():
    if fuzzy_df["Date"][row_index] not in tmp:
        tmp.append(fuzzy_df["Date"][row_index])
#        print(fuzzy_df)
        aux_df = fuzzy_df.loc[(fuzzy_df["Date"]==fuzzy_df["Date"][row_index]) & (fuzzy_df["FloatTime"] > 7) & (fuzzy_df["FloatTime"] < 19) & (fuzzy_df["Persons"] == 3) & (fuzzy_df["Lights"] > 700)]
        aux_df_weather = fuzzy_df.loc[(fuzzy_df["Date"]==fuzzy_df["Date"][row_index]) & (fuzzy_df["CO2"] < 400)]
        l1max = aux_df["S1Light"].max() 
        l2max = aux_df["S2Light"].max() 
        l3max = aux_df["S3Light"].max() 
        lmax = aux_df["Lights"].max() 

#        print(aux_df)
        weather="CLOUDY"
        if aux_df_weather["S3Light"].max() > 160:
            weather="SUNNY"
            
        l1mean = aux_df["S1Light"].mean() 
        l2mean = aux_df["S2Light"].mean() 
        l3mean = aux_df["S3Light"].mean()
        lmean = aux_df["Lights"].mean()
        print("Day:",fuzzy_df["Date"][row_index],"SIZEEE", len(aux_df))
        print("Weather",weather)
        print("\tL1\tMax: {}\tMean: {}".format(l1max,l1mean))        
        print("\tL2\tMax: {}\tMean: {}".format(l2max,l2mean))        
        print("\tL3\tMax: {}\tMean: {}".format(l3max,l3mean))        
        print("\tSUM\tMax: {}\tMean: {}".format(lmax,lmean))        



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
