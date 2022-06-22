import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os
import pandas as pd
from skfuzzy import control as ctrl
import sys
import p2_aux

dir_path = os.path.join(os.getcwd())
file_name = sys.argv[1]
file_path = os.path.join(dir_path, file_name)


dataset = pd.read_csv(file_path, sep=',')

df = p2_aux.pre_processing(dataset, normalization=False)

cols = [col for col in df.columns if col not in ["Time", "Date", "S1Temp", "S2Temp", "S3Temp", "PIR1", "PIR2", "Persons", "AboveLimit"]]
fuzzy_df = df[cols]

# Lights 1 & 2
light_1 = ctrl.Antecedent(np.arange(0, 551, 1), 'light_1')
light_2 = ctrl.Antecedent(np.arange(0, 551, 1), 'light_2')
lights_12 = ctrl.Consequent(np.arange(0, 3, 1), 'lights_12')

#NORMAL LIGHTS input
light_1["off"] = fuzz.trimf(light_1.universe, [0, 0, 110])
light_1["on"] = fuzz.trimf(light_1.universe, [100, 550, 550])
light_2["off"] = fuzz.trimf(light_2.universe, [0, 0, 110])
light_2["on"] = fuzz.trimf(light_2.universe, [100, 550, 550])

lights_12["0"] = fuzz.trimf(lights_12.universe, [0, 0, 0])
lights_12["1"] = fuzz.trimf(lights_12.universe, [1, 1, 1])
lights_12["2"] = fuzz.trimf(lights_12.universe, [2, 2, 2])

rule1 = ctrl.Rule(antecedent=((light_1["off"] & light_2["off"])), consequent=lights_12["0"], label="rule 1")
rule2 = ctrl.Rule(antecedent=((light_1["on"] & light_2["off"]) | (light_1["off"] & light_2["on"])), consequent=lights_12["1"], label="rule 2")
rule3 = ctrl.Rule(antecedent=((light_1["on"] & light_2["on"])), consequent=lights_12["2"], label="rule 3")

lights12_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
lights12 = ctrl.ControlSystemSimulation(lights12_ctrl)


lights_12_list = {}
for row_index, row in fuzzy_df.iterrows():
    lights12.input['light_1'] = fuzzy_df["S1Light"][row_index]
    lights12.input['light_2'] = fuzzy_df["S2Light"][row_index]
    lights12.compute()
    lights_12_list[row_index] = lights12.output['lights_12']

# weather
light_3 = ctrl.Antecedent(np.arange(0, 601, 1), 'light_3')
co2 = ctrl.Antecedent(np.arange(0, 1401, 5), 'co2')
weather = ctrl.Consequent(np.arange(0, 2, 1), 'weather')

#S3LIGHT df and CO2 input for weather
light_3["low (w)"] = fuzz.trimf(light_3.universe, [0, 0, 170])
light_3["high (w)"] = fuzz.trimf(light_3.universe, [160, 600, 600])
co2["lab empty"] = fuzz.trimf(co2.universe, [0, 0, 400])
co2["lab not empty"] = fuzz.trimf(co2.universe, [360, 1400, 1400])

weather["cloudy"] = fuzz.trimf(weather.universe, [0, 0, 0])
weather["sunny"] = fuzz.trimf(weather.universe, [1, 1, 1])

rule4 = ctrl.Rule(antecedent=((light_3["low (w)"] & co2["lab empty"])), consequent=weather["cloudy"], label="rule 4")
rule5 = ctrl.Rule(antecedent=((light_3["high (w)"] & co2["lab empty"])), consequent=weather["sunny"], label="rule 5")
rule6 = ctrl.Rule(antecedent=((light_3["low (w)"] & co2["lab not empty"])), consequent=weather["cloudy"], label="rule 6")
rule7 = ctrl.Rule(antecedent=((light_3["high (w)"] & co2["lab not empty"])), consequent=weather["sunny"], label="rule 7")

weather_ctrl = ctrl.ControlSystem([rule4, rule5, rule6, rule7])
weather_final = ctrl.ControlSystemSimulation(weather_ctrl)

weather_list = {}
for row_index, row in fuzzy_df.iterrows():
    weather_final.input['light_3'] = fuzzy_df["S3Light"][row_index]
    weather_final.input['co2'] = fuzzy_df["CO2"][row_index]
    weather_final.compute()
    weather_list[row_index] = weather_final.output['weather']

#S3LIGHT df and Weather input for Light3 final
light_3["low"] = fuzz.trimf(light_3.universe, [0, 0, 200])
light_3["medium"] = fuzz.trimf(light_3.universe, [190, 310, 310])
light_3["high"] = fuzz.trimf(light_3.universe, [300, 600, 600])
weather_input = ctrl.Antecedent(np.arange(min(weather_list.values()), max(weather_list.values()), 1), 'weather_input')
weather_input["cloudy"] = [0]
weather_input["sunny"] = [1]

light3_f = ctrl.Consequent(np.arange(0, 2, 1), 'light3_f')
light3_f["0"] = fuzz.trimf(light3_f.universe, [0, 0, 0])
light3_f["1"] = fuzz.trimf(light3_f.universe, [1, 1, 1])

#Lights3_final
rule8 = ctrl.Rule(antecedent=light_3["low"], consequent=light3_f["0"], label="rule 8")
rule9 = ctrl.Rule(antecedent=(light_3["medium"] & weather_input["cloudy"]), consequent=light3_f["1"], label="rule 9")
rule10 = ctrl.Rule(antecedent=(light_3["medium"] & weather_input["sunny"]), consequent=light3_f["0"], label="rule 10")
rule11 = ctrl.Rule(antecedent=light_3["high"], consequent=light3_f["1"], label="rule 11")

light3_ctrl = ctrl.ControlSystem([rule8, rule9, rule10, rule11])
light3_final = ctrl.ControlSystemSimulation(light3_ctrl)

light3_list = {}
for row_index, row in fuzzy_df.iterrows():
    light3_final.input['light_3'] = fuzzy_df["S3Light"][row_index]
    light3_final.input['weather_input'] = round(weather_list[row_index])
    light3_final.compute()
    light3_list[row_index] = light3_final.output['light3_f']

#light3_final.input['light_3'] = 350
#light3_final.input['weather_input'] = 1
#light3_final.compute()
#light3_f.view(light3_final)
#input()
#Total_Lights

lights_12_input = ctrl.Antecedent(np.arange(0, 3, 1), 'lights_12_input')
light_3_input = ctrl.Antecedent(np.arange(0, 2, 1), 'light_3_input')
total_lights = ctrl.Consequent(np.arange(0, 4, 1), 'total_lights')

lights_12_input["0"] = fuzz.trimf(lights_12_input.universe, [0, 0, 0])
lights_12_input["1"] = fuzz.trimf(lights_12_input.universe, [1, 1, 1])
lights_12_input["2"] = fuzz.trimf(lights_12_input.universe, [2, 2, 2])
light_3_input["0"] = fuzz.trimf(light_3_input.universe, [0, 0, 0])
light_3_input["1"] = fuzz.trimf(light_3_input.universe, [1, 1, 1])

total_lights["0"] = fuzz.trimf(total_lights.universe, [0, 0, 0])
total_lights["1"] = fuzz.trimf(total_lights.universe, [1, 1, 1])
total_lights["2"] = fuzz.trimf(total_lights.universe, [2, 2, 2])
total_lights["3"] = fuzz.trimf(total_lights.universe, [3, 3, 3])

rule12 = ctrl.Rule(antecedent=lights_12_input["0"] & light_3_input["0"], consequent=total_lights["0"], label="rule 12")
rule13 = ctrl.Rule(antecedent=(lights_12_input["1"] & light_3_input["0"]) | (lights_12_input["0"] & light_3_input["1"]), consequent=total_lights["1"], label="rule 13")
rule14 = ctrl.Rule(antecedent=(lights_12_input["2"] & light_3_input["0"]) | (lights_12_input["1"] & light_3_input["1"]), consequent=total_lights["2"], label="rule 14")
rule15 = ctrl.Rule(antecedent=lights_12_input["2"] & light_3_input["1"], consequent=total_lights["3"], label="rule 15")

total_lights_ctrl = ctrl.ControlSystem([rule12, rule13, rule14, rule15])
total_lights_final = ctrl.ControlSystemSimulation(total_lights_ctrl)

total_lights_list = {}
for row_index, row in fuzzy_df.iterrows():
    total_lights_final.input['lights_12_input'] = lights_12_list[row_index]
    total_lights_final.input['light_3_input'] = light3_list[row_index]
    total_lights_final.compute()
    total_lights_list[row_index] = total_lights_final.output['total_lights']

#total_lights_final.input['lights_12_input'] = 0
#total_lights_final.input['light_3_input'] = 0
#total_lights_final.compute()
#total_lights.view(sim=total_lights_final)
#input()

#final rules
total_lights_input = ctrl.Antecedent(np.arange(0, 4, 1), "total_lights_input")
room_overcrowded = ctrl.Consequent(np.arange(0, 2, 1), "room_overcrowded")

total_lights_input["less than three"] = fuzz.trimf(total_lights.universe, [0, 0, 2])
total_lights_input["three"] = fuzz.trimf(total_lights.universe, [2, 3, 3])

room_overcrowded["0"] = fuzz.trimf(light_3_input.universe, [0, 0, 0])
room_overcrowded["1"] = fuzz.trimf(light_3_input.universe, [1, 1, 1])

rule16 = ctrl.Rule(antecedent=total_lights_input["less than three"], consequent=room_overcrowded["0"], label="rule 16")
rule17 = ctrl.Rule(antecedent=total_lights_input["three"], consequent=room_overcrowded["1"], label="rule 17")

room_ctrl = ctrl.ControlSystem([rule16, rule17])
room_final = ctrl.ControlSystemSimulation(room_ctrl)

room_overcrowded_list = {}
for row_index, row in fuzzy_df.iterrows():
    room_final.input['total_lights_input'] = total_lights_list[row_index]
    room_final.compute()
    room_overcrowded_list[row_index] = room_final.output['room_overcrowded']

res = []
for i in total_lights_list.keys():
    res.append((room_overcrowded_list[i]))
#    print(room_overcrowded_list[i])
#room_overcrowded.view(sim=room_final)

#print(max(res))