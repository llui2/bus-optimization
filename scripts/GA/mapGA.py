import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import subprocess

print("Executant GeneticOptimizer.py...")
subprocess.run(["python", "../GA2/GeneticOptimizer.py"], check=True)

with open("../../results/best_line_GA.txt", "r") as f:
    BEST_LINE = list(map(int, f.read().strip().split(",")))
#BEST_LINE=[151, 169, 187, 247, 229, 211, 271, 253]
print("Millor línia trobada:", BEST_LINE)

ROAD_NODES_PATH = "../../data/road_network/nodes.csv"
ROAD_EDGES_PATH = "../../data/road_network/edges.csv"
nodes_df = pd.read_csv(ROAD_NODES_PATH)
edges_df = pd.read_csv(ROAD_EDGES_PATH)

G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(row["node"], pos=(row["x"], row["y"]))
for _, row in edges_df.iterrows():
    G.add_edge(row["src"], row["dst"])

pos = nx.get_node_attributes(G, 'pos')

# Dibuix del mapa
plt.figure(figsize=(10, 10))

# Afegim una capa de densitat de població (gaussiana centrada)
x_coords = nodes_df["x"]
y_coords = nodes_df["y"]
xmin, xmax = x_coords.min(), x_coords.max()
ymin, ymax = y_coords.min(), y_coords.max()
res = 300  # resolució

x_grid = np.linspace(xmin, xmax, res)
y_grid = np.linspace(ymin, ymax, res)
X, Y = np.meshgrid(x_grid, y_grid)

x0 = (xmin + xmax) / 2
y0 = (ymin + ymax) / 2
sigma = (xmax - xmin) / 8 # per canviar la mida del cercle de densitat

Z = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='Reds', alpha=0.3)

# llegim les parades disponibles
bus_stops_df = pd.read_csv("../../data/bus_network/nodes.csv")
NB = list(bus_stops_df["node"])  # llista de parades disponibles

# Xarxa viària
nx.draw(G, pos, node_size=10, edge_color='lightgray', with_labels=False)
nx.draw_networkx_nodes(G, pos, nodelist=NB, node_color='green', node_size=25)

# Dibuixem la millor linia
real_edges = []
for i in range(len(BEST_LINE) - 1):
    path = nx.shortest_path(G, source=BEST_LINE[i], target=BEST_LINE[i+1], weight="cost")
    real_edges += list(zip(path[:-1], path[1:]))

nx.draw_networkx_nodes(G, pos, nodelist=BEST_LINE, node_color='red', node_size=30)
nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color='red', width=2)

# Afegim l'etiqueta de l'ordre a cada parada
labels = {node: str(i+1) for i, node in enumerate(BEST_LINE)}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')

plt.title("Línia de bus optimitzada per GA", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("../../plots/GA/mapa_linea_GA.png", dpi=300)
plt.show()
