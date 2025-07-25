import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=10)
plt.rc('mathtext', fontset='cm')

energy = pd.read_csv("data/energy.csv")

fig, ax = plt.subplots(3, 1, figsize=(4, 5))
ax1, ax2, ax3 = ax
plt.subplots_adjust(wspace=0, hspace=0.2, 
                    left=0.2,top=.95, right=0.95, bottom=0.10)

num_lines = len(energy.columns) // 3

line_colors = ["red", "blue", "lime", "darkorange", "darkviolet"]

t = energy["MCS"]

for i in range(num_lines):
    
    ax1.plot(t,energy[f'ft{i}'], color=line_colors[i])
    ax2.plot(t,energy[f'st{i}'], color=line_colors[i])
    ax3.plot(t,energy[f'e{i}'], color=line_colors[i])


ax1.set_ylabel("First term")
ax2.set_ylabel("Second term")

ax1.set_xticks([])
ax1.set_xticklabels([])

ax2.set_xticks([])
ax2.set_xticklabels([])

ax3.set_ylabel("Energy")
ax3.set_xlabel("Changes")

# ax1.set_xscale("log")
# ax2.set_xscale("log")
# ax3.set_xscale("log")

plt.savefig("plots/energy.png", dpi=300)
