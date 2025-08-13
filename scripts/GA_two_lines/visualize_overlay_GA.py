import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from od_generation import OD_matrix, generate_field
import os

def superposa_linia_bus(G, pos, OD, NB, field, bus_lines, save_path="output.png"):
    fig, ax = plt.subplots(figsize=(10, 10))

    d = int(np.sqrt(len(G)))
    ax.imshow(field, cmap="Reds", extent=[0, d, 0, d], alpha=0.4, origin='lower')

    max_OD = np.max(OD)
    for i in range(len(NB)):
        for j in range(len(NB)):
            if i != j and OD[i, j] > 0:
                ni, nj = NB[i], NB[j]
                x1, y1 = pos[ni]
                x2, y2 = pos[nj]
                alpha = OD[i, j] / max_OD
                ax.plot([x1, x2], [y1, y2], color='darkred', alpha=alpha*0.8, linewidth=alpha*3)

    # Parades
    for n in NB:
        x, y = pos[n]
        ax.scatter(x, y, color="white", edgecolor="black", zorder=5, s=80)
        ax.text(x, y, str(n), fontsize=8, ha="center", va="center", zorder=6)

    # Línies optimitzades
    colors = ["blue", "green"]  # Pots afegir més si vols suportar més línies
    for idx, line in enumerate(bus_lines):
        for i in range(len(line)-1):
            u, v = line[i], line[i+1]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=colors[idx % len(colors)], linewidth=3, zorder=7)

    ax.set_title("Dues línies de bus optimitzades sobre demanda OD i densitat", fontsize=14)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    d = 20
    shift = 0.2
    seed = 42

    # Carreguem grafs i posicions
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

    # Aquesta línia s’ha de llegir des d’un fitxer o copiar directament
    # best_line = [211, 229, 169, 145, 127, 187, 247, 289] lambda = 0,5
    # best_line = [337, 367, 349, 253, 187, 169, 229, 211] lambda = 0,2
    #best_line = [211, 229, 169, 187, 205, 325, 349, 289] # lambda = 0,8

    with open("../../results/best_line_GA.txt", "r") as f:
        lines = f.readlines()

    line1 = list(map(int, lines[0].replace("LINE1:", "").strip().split(",")))
    line2 = list(map(int, lines[1].replace("LINE2:", "").strip().split(",")))

    best_line_real = [line1, line2]


    # Crear carpeta si no existeix
    os.makedirs("../../plots/GA", exist_ok=True)

    # Superposar línia sobre mapa OD
    superposa_linia_bus(
        G, pos, D, NB, field,
        best_line_real,
        save_path="../../plots/GA/OD_matrix_GA_overlay.png"
    )

    print("Figura guardada a: plots/GA/OD_matrix_GA_overlay.png")
