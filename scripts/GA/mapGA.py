import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import subprocess

print("Executant GeneticOptimizer.py...")
subprocess.run(["python", "../GA/GeneticOptimizer.py"], check=True)

with open("../../results/best_line_GA.txt", "r") as f:
    BEST_LINE = list(map(int, f.read().strip().split(",")))
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
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("../../plots/GA/mapa_linea_GA.png", dpi=300)
plt.show()
