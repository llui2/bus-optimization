import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Graph

# Si cal (ruta on tens dot.exe)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# --- DADES (7 nodes, simètrica, diagonal 0) ---
labels = ["A", "B", "C", "D", "E", "F", "G"]
N = len(labels)

# Matriu OD simètrica "fàcil d'explicar"
D = np.array([
    [0, 12,  0,  6,  0, 18,  0],
    [12, 0, 25,  0,  7,  0, 10],
    [0, 25, 0, 20,  0,  5,  0],
    [6,  0, 20, 0, 14,  0,  8],
    [0,  7,  0, 14, 0, 22,  0],
    [18, 0,  5,  0, 22, 0, 16],
    [0, 10,  0,  8,  0, 16, 0]
], dtype=float)

# --- 1) HEATMAP ---
plt.figure(figsize=(7, 6))
ax = sns.heatmap(
    D,
    annot=True,
    fmt=".0f",
    cmap="Reds",
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Demanda"},
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_title("Matriu Origen–Destí (OD) simètrica")
ax.set_xlabel("Destí")
ax.set_ylabel("Origen")
plt.tight_layout()
plt.savefig("results/od_heatmap.png", dpi=300)
plt.close()

# --- 2) GRAF (mateixa OD) ---
g = Graph("OD_graph", engine="neato")
g.attr(overlap="false", splines="true")
g.attr("node", shape="circle", style="filled", fillcolor="white", fontsize="14")

max_w = D.max() if D.max() > 0 else 1

for name in labels:
    g.node(name)

for i in range(N):
    for j in range(i + 1, N):   # i<j perquè és simètrica
        w = D[i, j]
        if w > 0:
            penwidth = 0.8 + 6.0 * (w / max_w)
            g.edge(labels[i], labels[j],
                   label=str(int(w)),
                   penwidth=f"{penwidth:.2f}",
                   color="#b30000")

g.render("results/od_graph", format="png", cleanup=True)
g.render("results/od_graph", format="pdf", cleanup=True)

print("Generat: od_heatmap.png, od_graph.png i od_graph.pdf")