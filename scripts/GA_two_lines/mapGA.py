import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from od_generation import generate_field

print("Executant od_matrix.py...")
subprocess.run(["python", "../GA/od_matrix.py"], check=True)

print("Executant GeneticOptimizer.py...")
subprocess.run(["python", "../GA/GeneticOptimizer.py"], check=True)


with open("../../results/best_line_GA.txt", "r") as f:
    lines = f.readlines()

line1 = list(map(int, lines[0].replace("LINE1:", "").strip().split(",")))
line2 = list(map(int, lines[1].replace("LINE2:", "").strip().split(",")))
bus_lines = [line1, line2]


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

# Dibuixem el camp de densitat com a fons
X, Y, field = generate_field(d=20)
plt.imshow(
    field,
    extent=[0, 20, 0, 20],
    origin="lower",
    cmap="YlOrRd",
    alpha=0.4  # Transparència per veure el graf per sobre
)

# llegim les parades disponibles
bus_stops_df = pd.read_csv("../../data/bus_network/nodes.csv")
NB = list(bus_stops_df["node"])  # llista de parades disponibles

# Xarxa viària
nx.draw(G, pos, node_size=10, edge_color='lightgray', with_labels=False)
nx.draw_networkx_nodes(G, pos, nodelist=NB, node_color='green', node_size=25)

colors = ["blue", "green"]
real_edges_all = []

for idx, line in enumerate(bus_lines):
    real_edges = []
    for i in range(len(line) - 1):
        path = nx.shortest_path(G, source=line[i], target=line[i+1], weight="cost")
        real_edges += list(zip(path[:-1], path[1:]))

    nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color=colors[idx % len(colors)], width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=line, node_color=colors[idx % len(colors)], node_size=30)

    # Etiquetes amb l'ordre de parades per a cada línia
    labels = {node: f"{i+1}" for i, node in enumerate(line)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')


plt.title("Dues línies de bus optimitzades per GA", fontsize=14)
plt.axis("off")
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("../../plots/GA/mapa_dues_linies_GA.png", dpi=300)
plt.show()
