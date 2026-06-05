import os
import pandas as pd
import numpy as np

# carpeta on estÃ  aquest script
script_dir = os.path.dirname(__file__)

# arrel del projecte bus-optimization
project_dir = os.path.dirname(os.path.dirname(script_dir))

# rutes absolutes
input_path = os.path.join(project_dir, "data", "bus_network", "od_matrix.csv")
output_path = os.path.join(project_dir, "data", "bus_network", "od_matrix_fixed.csv")

print("INPUT PATH:", input_path)
print("OUTPUT PATH:", output_path)

# carregar dades
df = pd.read_csv(input_path)

# obtenir tots els nodes Ãºnics
nodes = sorted(set(df["src"]).union(set(df["dst"])))

# crear matriu buida
n = len(nodes)
node_to_idx = {node: i for i, node in enumerate(nodes)}
D = np.zeros((n, n))

# omplir matriu
for _, row in df.iterrows():
    i = node_to_idx[row["src"]]
    j = node_to_idx[row["dst"]]
    D[i, j] = row["value"]
    D[j, i] = row["value"]

# convertir a DataFrame
matrix_df = pd.DataFrame(D, index=nodes, columns=nodes)

# guardar
matrix_df.to_csv(output_path)

print("OD convertida guardada a:", output_path)
print("Shape:", matrix_df.shape)
print("Existeix?", os.path.exists(output_path))
