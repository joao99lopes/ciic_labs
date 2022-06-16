import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_core = np.arange(0, 101, 10)
x_clock = np.arange(0, 4.5, 0.5)
x_fan = np.arange(0, 6001, 1000)

core_cold = fuzz.trimf(x_core, [0, 0, 50])
core_warm = fuzz.trimf(x_core, [0, 50, 100])
core_hot = fuzz.trimf(x_core, [50, 100, 100])

clock_low = fuzz.trimf(x_clock, [0, 0, 1.5])
clock_normal = fuzz.trimf(x_clock, [0.5 , 2, 3.5])
clock_turbo = fuzz.trimf(x_clock, [2.5, 4, 4])

fan_slow = fuzz.trimf(x_fan, [0, 0, 2000])
fan_normal = fuzz.trimf(x_fan, [1000, 3000, 5000]) 
fan_fast = fuzz.trimf(x_fan, [4000, 6000, 6000])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_core, core_cold, 'b', linewidth=1.5, label='Cold')
ax0.plot(x_core, core_warm, 'g', linewidth=1.5, label='Warm')
ax0.plot(x_core, core_hot, 'r', linewidth=1.5, label='Hot')
ax0.set_title('Core Temp')
ax0.legend()

ax1.plot(x_clock, clock_low, 'b', linewidth=1.5, label='Low')
ax1.plot(x_clock, clock_normal, 'g', linewidth=1.5, label='Normal')
ax1.plot(x_clock, clock_turbo, 'r', linewidth=1.5, label='Turbo')
ax1.set_title('Clock Speed')
ax1.legend()

ax2.plot(x_fan, fan_slow, 'b', linewidth=1.5, label='Slow')
ax2.plot(x_fan, fan_normal, 'g', linewidth=1.5, label='Normal')
ax2.plot(x_fan, fan_fast, 'r', linewidth=1.5, label='Fast')
ax2.set_title('Fan Speed')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()