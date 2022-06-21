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

light_1 = ctrl.Antecedent(np.arange(0, 551, 1), 'light_1')
light_2 = ctrl.Antecedent(np.arange(0, 551, 1), 'light_2')
light_3 = ctrl.Antecedent(np.arange(0, 601, 1), 'light_3')
co2 = ctrl.Antecedent(np.arange(0, 1401, 5), 'co2')
lights_12 = ctrl.Consequent(np.arange(0, 3, 1), 'lights_12')
weather = ctrl.Consequent(np.arange(0, 2, 1), 'weather')
light3_f = ctrl.Consequent(np.arange(0, 2, 1), 'light3_f')

#NORMAL LIGHTS input
light_1["off"] = fuzz.trimf(light_1.universe, [0, 0, 250])
light_1["on"] = fuzz.trimf(light_1.universe, [240, 550, 550])
light_2["off"] = fuzz.trimf(light_2.universe, [0, 0, 250])
light_2["on"] = fuzz.trimf(light_2.universe, [240, 550, 550])

#S3LIGHT df and CO2 input for weather
light_3["low (w)"] = fuzz.trimf(light_3.universe, [0, 0, 170])
light_3["high (w)"] = fuzz.trimf(light_3.universe, [160, 600, 600])
co2["lab empty"] = fuzz.trimf(co2.universe, [0, 0, 400])
co2["lab not empty"] = fuzz.trimf(co2.universe, [360, 1400, 1400])

#S3LIGHT df and Weather input for Light3 final
light_3["low"] = fuzz.trimf(light_3.universe, [0, 0, 200])
light_3["medium"] = fuzz.trimf(light_3.universe, [190, 310, 310])
light_3["high"] = fuzz.trimf(light_3.universe, [300, 600, 600])

#outputs
lights_12["0"] = fuzz.trimf(lights_12.universe, [0, 0, 0])
lights_12["1"] = fuzz.trimf(lights_12.universe, [1, 1, 1])
lights_12["2"] = fuzz.trimf(lights_12.universe, [2, 2, 2])
weather["cloudy"] = fuzz.trimf(weather.universe, [0, 0, 0])
weather["sunny"] = fuzz.trimf(weather.universe, [1, 1, 1])
light3_f["0"] = fuzz.trimf(light3_f.universe, [0, 0, 0])
light3_f["1"] = fuzz.trimf(light3_f.universe, [1, 1, 1])

#lights12
rule1 = ctrl.Rule(antecedent=((light_1["off"] & light_2["off"])), consequent=lights_12["0"], label="rule 1")
rule2 = ctrl.Rule(antecedent=((light_1["on"] & light_2["off"]) | (light_1["off"] & light_2["on"])), consequent=lights_12["1"], label="rule 2")
rule3 = ctrl.Rule(antecedent=((light_1["on"] & light_2["on"])), consequent=lights_12["2"], label="rule 3")

lights12_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
lights12 = ctrl.ControlSystemSimulation(lights12_ctrl)

lights12.input['light_1'] = 245
lights12.input['light_2'] = 400

lights12.compute()

print(lights12.output['lights_12'])
lights_12.view(sim=lights12)
input()

#weather
rule4 = ctrl.Rule(antecedent=((light_3["low (w)"] & co2["lab empty"])), consequent=weather["cloudy"], label="rule 4")
rule5 = ctrl.Rule(antecedent=((light_3["high (w)"] & co2["lab empty"])), consequent=weather["sunny"], label="rule 5")

weather_ctrl = ctrl.ControlSystem([rule4, rule5])
weather_final = ctrl.ControlSystemSimulation(weather_ctrl)

weather_final.input['light_3'] = 150
weather_final.input['co2'] = 300

weather_final.compute()

print(weather_final.output['weather'])
weather.view(sim=weather_final)
input()

#Lights3_final
rule6 = ctrl.Rule(antecedent=((light_3["low"] & weather_final.output["weather"] == "cloudy") | (light_3["low"] & weather["sunny"])), consequent=light3_f["off"], label="rule 6")
rule7 = ctrl.Rule(antecedent=((light_3["medium"] & weather_final["cloudy"])), consequent=light3_f["on"], label="rule 7")
rule8 = ctrl.Rule(antecedent=((light_3["medium"] & weather_final["sunny"])), consequent=light3_f["off"], label="rule 8")
rule9 = ctrl.Rule(antecedent=((light_3["high"] & weather_final["cloudy"]) | (light_3["high"] & co2["sunny"])), consequent=light3_f["1"], label="rule 9")

light3_ctrl = ctrl.ControlSystem([rule6, rule7, rule8, rule9])
light3_final = ctrl.ControlSystemSimulation(light3_ctrl)

light3_final.input['light_3'] = 150

light3_final.compute()

print(light3_final.output['weather'])
light3_f.view(sim=light3_final)
input()

#Total_Lights