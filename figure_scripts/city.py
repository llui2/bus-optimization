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

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plt.subplots_adjust(wspace=0, hspace=0, left=0, top=1, right=1, bottom=0)

nodesG = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=10, node_color="silver", alpha=1)
edgesG = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="silver", width=4, alpha=1)
nodesG.set_zorder(1)
edgesG.set_zorder(1)

labels = nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=3, font_color="black")

nodesG_B = nx.draw_networkx_nodes(G_B, pos=pos, ax=ax, node_size=40, node_color="white",
                                  edgecolors="silver", alpha=1)
nodesG_B.set_zorder(2)

# Add field
field = pd.read_csv("data/bus_network/field.csv")
x_unique = np.sort(field['x'].unique())
y_unique = np.sort(field['y'].unique())
X, Y = np.meshgrid(x_unique, y_unique)
field = field.pivot(index='y', columns='x', values='field').values
contour = ax.contourf(X, Y, field, cmap="Reds", alpha=0.8)
contour.set_zorder(0)


ax.set_aspect('equal')
ax.axis("off")

plt.savefig("plots/city.png", dpi=300)
# os.system("open plots/city.png")




