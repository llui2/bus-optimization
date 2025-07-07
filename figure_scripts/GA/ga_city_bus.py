import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os

# Carrega nodes i arestes
nodes_df = pd.read_csv("data/road_network/nodes.csv")
edges_df = pd.read_csv("data/road_network/edges.csv")

# Crear graf
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(int(row["node"]), pos=(row["x"], row["y"]))
for _, row in edges_df.iterrows():
    G.add_edge(int(row["src"]), int(row["dst"]), cost=row.get("cost", 1))

pos = nx.get_node_attributes(G, "pos")

# Carrega de parades de bus
bus_stops_df = pd.read_csv("data/bus_network/nodes.csv")
bus_stop_nodes = bus_stops_df["node"].tolist()

# Carrega línia de bus (GA)
lines_df = pd.read_csv("data/bus_network/lines.csv", header=None)
line_nodes = lines_df.iloc[0].dropna().astype(int).tolist()

# Generar camins reals entre parades consecutives
ga_path_edges = []
for i in range(len(line_nodes) - 1):
    try:
        path = nx.shortest_path(G, source=line_nodes[i], target=line_nodes[i+1], weight="cost")
        ga_path_edges.extend([(path[j], path[j+1]) for j in range(len(path)-1)])
    except nx.NetworkXNoPath:
        print(f"No path between {line_nodes[i]} and {line_nodes[i+1]}")

# --- Densitat de població com a fons ---
d = 20
x = np.linspace(0, d-1, 100)
y = np.linspace(0, d-1, 100)
X, Y = np.meshgrid(x, y)
sigma = d / 12
center_x, center_y = d / 2, d / 2
gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

# --- Plot ---
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, gaussian, cmap="Reds", alpha=0.4)  # densitat

# Graf base
nx.draw(G, pos, node_size=5, edge_color='gray', with_labels=False)

# Parades de bus
nx.draw_networkx_nodes(G, pos, nodelist=bus_stop_nodes, node_color='orange', node_size=30, label="Parades de bus")

# Línia de bus (camins reals)
nx.draw_networkx_edges(G, pos, edgelist=ga_path_edges, edge_color='red', width=2, label="Línia GA")
nx.draw_networkx_nodes(G, pos, nodelist=line_nodes, node_color='red', node_size=40)

# Etiquetes de nodes
labels = {node: str(node) for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=5)

plt.legend()
plt.axis("off")
plt.tight_layout()

# Crear carpeta si no existeix
os.makedirs("plots/GA", exist_ok=True)
plt.savefig("plots/GA/ga_city_bus.png", dpi=300)
plt.show()
