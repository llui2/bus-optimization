import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from od_generation import OD_matrix, generate_field
import os

if __name__ == "__main__":
    d = 20
    shift = 0.2
    seed = 42

    # Carreguem nodes i posicions
    nodes_df = pd.read_csv("../../data/road_network/nodes.csv")
    edges_df = pd.read_csv("../../data/road_network/edges.csv")
    pos = {int(row["node"]): (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    G = nx.Graph()
    for node in pos:
        G.add_node(node, pos=pos[node])
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["src"]), int(row["dst"]))

    bus_df = pd.read_csv("../../data/bus_network/nodes.csv")
    NB = list(bus_df["node"])

    D, node_to_bus_index = OD_matrix(seed, NB, pos, d)
    X, Y, field = generate_field(d)

    # Visualització
    plt.figure(figsize=(12, 12))

    # Fons: densitat
    plt.imshow(field, extent=(0, d, 0, d), origin="lower", cmap="Reds", alpha=0.3)

    # Xarxa viària
    nx.draw_networkx_edges(G, pos, alpha=0.05)

    # Nodes (parades bus)
    bus_pos = {node: pos[node] for node in NB}
    nx.draw_networkx_nodes(G, pos=bus_pos, nodelist=NB,
                            node_color='white', edgecolors='black', linewidths=1.5, node_size=300)
    nx.draw_networkx_labels(G, pos=bus_pos, labels={n: n for n in NB}, font_size=8)

    # Arcs OD
    for i in range(len(NB)):
        for j in range(i + 1, len(NB)):
            if D[i, j] > 0:
                ni, nj = NB[i], NB[j]
                xi, yi = pos[ni]
                xj, yj = pos[nj]
                value = D[i, j]
                alpha = min(0.1 + value / 30, 1)
                lw = max(0.5, value / 2)
                plt.plot([xi, xj], [yi, yj], color='darkred', alpha=alpha, linewidth=lw)

    # Llegenda personalitzada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', lw=0.5, alpha=0.3, label='Demanda baixa'),
        Line2D([0], [0], color='darkred', lw=2, alpha=0.6, label='Demanda mitjana'),
        Line2D([0], [0], color='darkred', lw=4, alpha=0.9, label='Demanda alta'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.axis('off')
    plt.title("Demanda OD respecte la densitat de població (GA)", fontsize=14)

    os.makedirs("../../plots", exist_ok=True)
    plt.savefig("../../plots/GA/OD_matrix_GA_estil.png", bbox_inches='tight', dpi=300)
    
    #plt.show()
