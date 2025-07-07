import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import random
import math as m
import numpy as np
import networkx as nx
from GeneticAlgorithmOneLine import GeneticAlgorithmOneLine
from GA_one_line_bus.DataHandlerGA import DataHandlerGA


# -----------------------------
# FUNCIONS AUTÒNOMES PEL GA
# -----------------------------

def euclidean_distance(a, b):
    return m.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def road_network(seed, d, shift):
    random.seed(seed)
    np.random.seed(seed)

    G = nx.grid_2d_graph(d, d, periodic=False)
    G = nx.convert_node_labels_to_integers(G)

    # Calculem posicions amb desplaçament
    pos = {}
    for node in G.nodes():
        x = node % d
        y = node // d
        pos[node] = (
            x + random.uniform(-shift, shift),
            y + random.uniform(-shift, shift)
        )

    # Assignem les posicions com a atribut del node
    for node in G.nodes():
        G.nodes[node]["pos"] = pos[node]

    # Assignem cost a les arestes com a distància euclidiana
    for u, v in G.edges():
        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]
        G.edges[u, v]["cost"] = euclidean_distance(p1, p2)

    return G, pos


def bus_stops(G):
    N = len(G.nodes())
    N_B = list(G.nodes)[1:N:6]
    return N_B

def generate_field(d):
    x = np.linspace(0, d-1, 100)
    y = np.linspace(0, d-1, 100)
    X, Y = np.meshgrid(x, y)
    sigma = d / 12
    center_x, center_y = d / 2, d / 2
    gaussian1 = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    field = np.round(gaussian1, 2)
    return X, Y, field

def OD_matrix(seed, N_B, pos, d):
    random.seed(seed)
    np.random.seed(seed)
    _, _, field = generate_field(d)
    field_values = []
    for node in N_B:
        x, y = pos[node]
        i = int(x * 100 / d)
        j = int(y * 100 / d)
        field_values.append(field[j, i])
    field_values = np.array(field_values)
    D = np.zeros((len(N_B), len(N_B)), dtype=float)
    for i in range(len(N_B)):
        for j in range(i+1, len(N_B)):
            if i != j:
                D[i, j] = max(field_values[j] * 10, field_values[i] * 10) * random.choice([0, 0, 0, 1])
                D[j, i] = D[i, j]
    D = np.round(D, 2)
    node_to_bus_index = {node: idx for idx, node in enumerate(N_B)}
    return D, node_to_bus_index

def bus_stop_cost(G, N_B):
    C = np.zeros((len(N_B), len(N_B)))
    all_pairs_cost = dict(nx.all_pairs_dijkstra_path_length(G, weight="cost"))
    for i in range(len(N_B)):
        for j in range(i+1, len(N_B)):
            C[i, j] = all_pairs_cost[N_B[i]][N_B[j]]
            C[j, i] = C[i, j]
    return C

def save_bus_lines_csv(lines, file):
    with open(file, "w") as f:
        for line in lines:
            f.write(",".join(str(node) for node in line) + "\n")

# -----------------------------
# EXECUCIÓ DEL GA
# -----------------------------

def run_GA(seed=123, d=20, shift=0.2, L=8, lambdaa=0.4, max_iter=50, pop_size=20, mutation_rate=0.1):
    print(f"\nEntrenant línia amb GA: λ = {lambdaa}, població = {pop_size}, generacions = {max_iter}\n")
    random.seed(seed)
    G, pos = road_network(seed, d, shift)
    N_B = bus_stops(G)
    D, node_to_bus_index = OD_matrix(seed, N_B, pos, d)
    C = bus_stop_cost(G, N_B)

    ga = GeneticAlgorithmOneLine(
        G=G,
        NB=N_B,
        D=D,
        C=C,
        lambda_=lambdaa,
        pop_size=pop_size,
        generations=max_iter,
        mutation_rate=mutation_rate
    )

    best_line = ga.evolve()

    save_bus_lines_csv([best_line], "data/bus_network/lines.csv")

    dh = DataHandlerGA(sheet_name="GA")

    line_bus_only = [n for n in best_line if n in N_B]
    final_energy, avg_travel_distance, population_total, central_coverage_pct, converged_iter = dh.compute_metrics(
        line_nodes=line_bus_only,
        obj=[ga.fitness(best_line)],
        pos=pos,
        node_to_bus_index=node_to_bus_index,
        D=D,
        G=G,
        d=d
    )

    metrics = dh.prepare_metrics_dict(
        final_energy=final_energy,
        avg_travel_distance=avg_travel_distance,
        population_total=population_total,
        central_coverage_pct=central_coverage_pct,
        converged_iter=converged_iter,
        seed=seed,
        L=L,
        T_0=None,
        lambdaa=lambdaa,
        max_iter=max_iter
    )
    dh.export_to_excel(metrics)
    print("Execució completa amb GA.\n")

if __name__ == "__main__":
    run_GA(lambdaa=0.4, max_iter=50, pop_size=20)

