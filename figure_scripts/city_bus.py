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

# labels = nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=4, font_color="black")

nodesG_B = nx.draw_networkx_nodes(G_B, pos=pos, ax=ax, node_size=40, node_color="white",
                                  edgecolors="silver", alpha=1)
nodesG_B.set_zorder(2)

# Add field
field = pd.read_csv("data/bus_network/field.csv")
x_unique = np.sort(field['x'].unique())
y_unique = np.sort(field['y'].unique())
X, Y = np.meshgrid(x_unique, y_unique)
field = field.pivot(index='y', columns='x', values='field').values
contour = ax.contourf(X, Y, field, cmap="Reds", alpha=0.5)
contour.set_zorder(0)

# Add bus lines
node_size = 20
line_width = 2.5
line_colors = ["red", "blue", "lime", "darkorange", "darkviolet"]

bus_lines = pd.read_csv("data/bus_network/lines.csv", sep=" ", header=None)

# Plot bus stops shared by two or more lines
for i in range(len(bus_lines)):
    for j in range(i+1, len(bus_lines)):
        
        shared_nodes = set(bus_lines.loc[i, 0].split(",")).intersection(set(bus_lines.loc[j, 0].split(",")))
        shared_nodes = [int(x) for x in shared_nodes]
        shared_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=shared_nodes, ax=ax, node_size=70, node_color="white", edgecolors="black", linewidths=1)
        shared_nodes.set_zorder(2)

# Plot bus lines
for i, row in bus_lines.iterrows():

    N_L = list(bus_lines.loc[i, 0].split(","))
    N_L = [int(x) for x in N_L]

    color = line_colors[i]

    G_L = nx.Graph()
    G_L.add_nodes_from(N_L)
    nodes = nx.draw_networkx_nodes(G_L, pos=pos, ax=ax, node_size=node_size, node_color=color)
    nodes.set_zorder(3)

    # Plot path
    last_edges = None
    for i in range(len(N_L)-1):
        G_allowed = G.copy()

        # Avoid going backwards
        if last_edges is not None:
            for edge in last_edges:
                G_allowed.remove_edge(*edge)

        try:
            path = min(nx.shortest_path(G_allowed, source=N_L[i], target=N_L[i+1], weight="cost"), nx.shortest_path(G_allowed, source=N_L[i+1], target=N_L[i], weight="cost"))
        except nx.NetworkXNoPath:
            for edge in last_edges:
                G_allowed.add_edge(*edge)
            path = min(nx.shortest_path(G_allowed, source=N_L[i], target=N_L[i+1], weight="cost"), nx.shortest_path(G_allowed, source=N_L[i+1], target=N_L[i], weight="cost"))

        edges_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        nodes_l = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, ax=ax, node_size=2, node_color=color, alpha=1)
        nodes_l.set_zorder(3)
        edges_l = nx.draw_networkx_edges(G, pos=pos, edgelist=edges_path, ax=ax, edge_color=color, width=line_width, alpha=1)
        edges_l.set_zorder(3)

        if len(path) > 1:
            last_edges = edges_path


ax.set_aspect('equal')
ax.axis("off")

plt.savefig("plots/city_bus.png", dpi=300)
# os.system("open plots/city_bus.png")




