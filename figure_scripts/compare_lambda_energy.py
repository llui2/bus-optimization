# figure_scripts/compare_lambda_energy.py

import os
import matplotlib.pyplot as plt
import pandas as pd

# Carpeta amb els resultats
folder = "results/lambda"
lambda_vals = [3, 4, 5, 6, 7]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 7))
plt.subplots_adjust(hspace=0.4)

colors = ["red", "blue", "green", "orange", "purple"]

for i, lmb in enumerate(lambda_vals):
    file_path = os.path.join(folder, f"energy_lambda_{lmb}.csv")
    if not os.path.exists(file_path):
        print(f"No trobat: {file_path}")
        continue

    df = pd.read_csv(file_path)

    label = f"$\\lambda = {lmb}$"
    ax1.plot(df["ft0"], label=label, color=colors[i])
    ax2.plot(df["st0"], label=label, color=colors[i])
    ax3.plot(df["e0"], label=label, color=colors[i])

# Títols i llegendes
ax1.set_title("Primer terme: servei")
ax2.set_title("Segon terme: cost")
ax3.set_title("Energia total")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Iteració")
    ax.set_ylabel("Valor")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("plots/compare_lambda_energy.png", dpi=300)
print("Gràfica guardada a plots/compare_lambda_energy.png")

