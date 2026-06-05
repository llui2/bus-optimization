import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def generate_field(d):
    x = np.linspace(0, d-1, 100)
    y = np.linspace(0, d-1, 100)
    X, Y = np.meshgrid(x, y)
    sigma = d / 12
    center_x, center_y = d / 2, d / 2
    gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    field = np.round(gaussian, 2)
    return X, Y, field

# Carrega de nodes i arestes
nodes_df = pd.read_csv("data/road_network/nodes.csv")
edges_df = pd.read_csv("data/road_network/edges.csv")

# Crear graf
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(int(row["node"]), pos=(row["x"], row["y"]))
for _, row in edges_df.iterrows():
    G.add_edge(int(row["src"]), int(row["dst"]))

# Posicions
pos = nx.get_node_attributes(G, "pos")

# Carrega de parades de bus
bus_stops_df = pd.read_csv("data/bus_network/nodes.csv")
bus_stop_nodes = bus_stops_df["node"].tolist()

# Genera i ploteja el camp de densitat (field)
d = 20  # ha de coincidir amb la mida del grid
X, Y, field = generate_field(d)

plt.figure(figsize=(10, 10))

# Dibuixa el camp de població (com a fons)
plt.contourf(X, Y, field, alpha=0.4, cmap='Reds')

# Dibuixa el graf de carreteres
nx.draw(G, pos, node_size=5, edge_color='gray', with_labels=False)

# Dibuixar parades en vermell
nx.draw_networkx_nodes(G, pos, nodelist=bus_stop_nodes, node_color='red', node_size=30)

# Afegir etiquetes dels nodes (numeració)
labels = {node: str(node) for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=6)

plt.axis("off")
plt.tight_layout()
plt.savefig("plots/GA/ga_city.png", dpi=300)
plt.show()
