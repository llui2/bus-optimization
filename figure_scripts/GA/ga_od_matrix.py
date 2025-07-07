import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.lines import Line2D

# Carrega la matriu OD
od_df = pd.read_csv("data/bus_network/od_matrix.csv")

# Crea graf dirigit
G = nx.DiGraph()

# Afegeix nodes (parades de bus)
nodes = sorted(set(od_df["src"]).union(set(od_df["dst"])))
G.add_nodes_from(nodes)

# Afegeix arestes amb pes segons la demanda
for _, row in od_df.iterrows():
    src, dst, demand = int(row["src"]), int(row["dst"]), row["value"]
    if demand > 0:
        G.add_edge(src, dst, weight=demand)

# 🔧 Carrega posicions reals dels nodes del graf viari
road_nodes_df = pd.read_csv("data/road_network/nodes.csv")
positions = {int(row["node"]): (row["x"], row["y"]) for _, row in road_nodes_df.iterrows() if int(row["node"]) in G.nodes}

# Normalitza pesos per al gruix de línia
weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(weights)
norm_weights = [3 * w / max_weight for w in weights]

# Densitat suau (com a city.py)
d = 20
x = [i + 0.5 for i in range(d)]
y = [i + 0.5 for i in range(d)]
X, Y = np.meshgrid(x, y)
sigma = d / 12
center = d / 2
gaussian = np.exp(-((X - center)**2 + (Y - center)**2) / (2 * sigma**2))

# --- Plot ---
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, gaussian, cmap="Reds", alpha=0.2)

nx.draw_networkx_nodes(G, positions, node_color='white', edgecolors='black', node_size=300)
nx.draw_networkx_edges(G, positions, edge_color='brown', alpha=0.3, width=norm_weights)
nx.draw_networkx_labels(G, positions, font_size=6)

# Llegenda manual
legend_elements = [
    Line2D([0], [0], color='brown', lw=0.5, label='Demanda baixa'),
    Line2D([0], [0], color='brown', lw=2, label='Demanda mitjana'),
    Line2D([0], [0], color='brown', lw=4, label='Demanda alta')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.axis("off")
os.makedirs("plots/GA", exist_ok=True)
plt.tight_layout()
plt.savefig("plots/GA/ga_od_graph.png", dpi=300)
plt.show()
