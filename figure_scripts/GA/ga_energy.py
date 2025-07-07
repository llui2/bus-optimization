import pandas as pd
import matplotlib.pyplot as plt
import os

# Llegeix el fitxer amb l’energia del GA
df = pd.read_csv("data/energy_ga.csv")

# Crear carpeta si no existeix
os.makedirs("plots/GA", exist_ok=True)

# Dibuixa la gràfica
plt.figure(figsize=(10, 6))
plt.plot(df["generation"], df["energy"], label="GA", color="red")
plt.xlabel("Generació")
plt.ylabel("Energia")
plt.title("Evolució de l'energia (GA)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/GA/ga_energy.png", dpi=300)
plt.show()
