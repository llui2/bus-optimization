import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

road_nodes = pd.read_csv("data/road_network/nodes.csv")
road_edges = pd.read_csv("data/road_network/edges.csv")

bus_nodes = pd.read_csv("data/bus_network/nodes.csv")

# Create a road graph
G = nx.Graph()

# Create a bus graph
G_B = nx.Graph()

for _, row in road_nodes.iterrows():

    node = int(row['node'])
    G.add_node(node, pos=(row['x'], row['y']))
    
    if node in bus_nodes['node'].values:
        G_B.add_node(node, pos=(row['x'], row['y']))

for _, row in road_edges.iterrows():
    G.add_edge(row['src'], row['dst'], weight=row['cost'])

pos = nx.get_node_attributes(G, 'pos')

# Read od_matrix.dat as i j value
# od_matrix = pd.read_csv("data/bus_network/od_matrix.dat", sep=" ", header=None)
# od_matrix.columns = ["i", "j", "value"]

od_matrix = pd.read_csv("data/bus_network/od_matrix.csv")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plt.subplots_adjust(wspace=0, hspace=0, left=0,
                        top=1, right=1, bottom=0)

OD = nx.Graph()

# Add nodes corresponding to the bus stops
for node in G_B.nodes:
    OD.add_node(node, pos=G_B.nodes[node]['pos'])

# Add edges corresponding to the OD matrix
for _, row in od_matrix.iterrows():
    i = int(row['src'])
    j = int(row['dst'])
    value = row['value']
    if i in OD.nodes and j in OD.nodes:
        OD.add_edge(i, j, weight=value)

edge_widths = [OD.edges[edge]['weight'] for edge in OD.edges]
edge_widths = np.array(edge_widths)
edge_widths = (edge_widths - edge_widths.min()) / (edge_widths.max() - edge_widths.min())

from matplotlib import colormaps
cmap = colormaps["Reds"]
edge_colors = [cmap(edge_width) for edge_width in edge_widths]

pos = nx.get_node_attributes(OD, 'pos')

nodesOD = nx.draw_networkx_nodes(OD, pos=pos, ax=ax, node_size=40, node_color="white",
                                  edgecolors="silver", alpha=1)
sorted_edges = sorted(OD.edges(data=True), key=lambda x: x[2]['weight'])

for edge in sorted_edges:
    i, j, data = edge
    weight = data['weight']
    color = cmap(weight)

    nx.draw_networkx_edges(OD, pos=pos, ax=ax, edgelist=[(i, j)], width=2, edge_color=[color], alpha=0.4)

nodesOD.set_zorder(2)

ax.set_aspect('equal')
ax.axis("off")

plt.savefig("plots/od_matrix.png", dpi=300)
# os.system("open plots/od_matrix.png")
