import random
import math as m
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from tqdm import tqdm
import warnings

# Suppress warning
warnings.filterwarnings("ignore", message="Couldn't reach some vertices")

# --------------------------------------------
# --------------------------------------------
#  ROAD NETWORK


def road_network(seed, d, shift):
    """
    Generates a grid graph with d x d nodes and random shifts in node positions.
    Each edge has a cost attribute that is the euclidean distance between the nodes.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Create a grid graph
    G = nx.grid_2d_graph(d, d, periodic=False)

    # Randomly remove edges
    edges = list(G.edges())
    random.shuffle(edges)
    for edge in edges:
        G.remove_edge(*edge)
        if nx.is_connected(G) == False:
            G.add_edge(*edge)
            break

    # Random shifts
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    G = nx.convert_node_labels_to_integers(G)
    pos = {i: (x+random.uniform(0, shift), y+random.uniform(0, shift))
           for i, ((x, y), _) in enumerate(pos.items())}

    # Add pos attribute to nodes
    for i, node in enumerate(G.nodes):
        G.nodes[node]["pos"] = pos[i]

    # Add cost attribute to edges
    for edge in G.edges:
        G.edges[edge]["cost"] = np.linalg.norm(
            np.array(G.nodes[edge[0]]["pos"]) - np.array(G.nodes[edge[1]]["pos"]))

    # Save nodes and edges to CSV files
    save_road_network_csv(
        G, pos, "data/road_network/nodes.csv", "data/road_network/edges.csv")

    return G, pos


def save_road_network_csv(G, pos, nodes_file, edges_file):
    """
    Saves the road network nodes and edges to CSV files.
    """

    # Save nodes and positions to CSV file
    with open(nodes_file, "w") as f:
        f.write("node,x,y\n")
        for node in G.nodes():
            x, y = pos[node]
            f.write(f"{node},{x},{y}\n")

    # Save edges and costs to CSV file
    with open(edges_file, "w") as f:
        f.write("src,dst,cost\n")
        for src, dst in G.edges():
            f.write(f"{src},{dst},{G.edges[src, dst]['cost']}\n")

# --------------------------------------------
# --------------------------------------------
# BUS STOPS


def bus_stops(G):
    """
    Randomly selects a subset of nodes to be bus stops.
    """

    # Select bus stops
    N = len(G.nodes())
    N_B = list(G.nodes)[1:N:6]

    # Save bus stops to a CSV file
    save_bus_stops_csv(N_B, "data/bus_network/nodes.csv")

    return N_B


def save_bus_stops_csv(N_B, file):
    """
    Saves the bus stops to a CSV file.
    """

    # Save bus stops to CSV file
    with open(file, "w") as f:
        f.write("node\n")
        for node in N_B:
            f.write(f"{node}\n")

# --------------------------------------------
# --------------------------------------------
# DEMAND FIELD


def generate_field(d):
    """
    Generates a 2D scalar field with two Gaussian distributions over the grid.
    """

    # Create a grid
    x = np.linspace(0, d-1, 100)
    y = np.linspace(0, d-1, 100)
    X, Y = np.meshgrid(x, y)

    # Create a field
    sigma = d / 20
    center_x, center_y = d / 2, d / 2  # d / 4, d / 4
    gaussian1 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    center_x, center_y = 3 * d / 4, 3 * d / 4
    gaussian2 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    center_x, center_y = d / 4, 3 * d / 4
    gaussian3 = np.exp(-((X - center_x)**2 + (Y - center_y)
                       ** 2) / (2 * sigma**2))
    field = gaussian1  # + gaussian2 #+ gaussian3

    # squares field
    # field = np.zeros((100, 100))
    # field[20:40, 20:40] = 1
    # field[60:80, 60:80] = 1

    # Create a field
    # sigma = 10
    # center_x, center_y = d / 2, d / 2
    # gaussian1 = np.exp(-((X - center_x)**2 + (Y - center_y)
    #                    ** 2) / (2 * sigma**2))
    # field = gaussian1

    # Round field values
    field = np.round(field, 2)

    # Save field values to a CSV file
    save_bus_stop_field_value_csv(X, Y, field, "data/bus_network/field.csv")

    return X, Y, field


def save_bus_stop_field_value_csv(X, Y, field, file):
    """
    Saves the field values at the bus stops to a CSV file.
    """

    # Save field values to CSV file
    with open(file, "w") as f:
        f.write("x,y,field\n")
        for i in range(len(X)):
            for j in range(len(Y)):
                f.write(f"{X[i, j]},{Y[i, j]},{field[i, j]}\n")

# --------------------------------------------
# --------------------------------------------
# ORIGIN-DESTINATION MATRIX


def OD_matrix(seed, N_B, pos, d):
    """
    Generates an origin-destination matrix for the bus stops based on a field.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Generate field
    _, _, field = generate_field(d)

    # Compute field values at bus stops
    field_values = []
    for node in N_B:
        x, y = pos[node]
        i = int(x * 100 / d)
        j = int(y * 100 / d)
        field_values.append(field[j, i])
    field_values = np.array(field_values)

    # Generate OD matrix
    # D = np.zeros((len(N_B), len(N_B)), dtype=float)
    # for i in range(len(N_B)):
    #     for j in range(len(N_B)):
    #         if i != j:
    #             D[i, j] = field_values[i] * field_values[j] * 10

    D = np.zeros((len(N_B), len(N_B)), dtype=float)
    for i in range(len(N_B)):
        for j in range(i+1, len(N_B)):
            if i != j:
                D[i, j] = max(field_values[j] * 10, field_values[i]
                              * 10) * random.choice([0 for _ in range(10)] + [1])
                D[j, i] = D[i, j]

    # Round OD matrix values
    D = np.round(D, 2)

    # Mapping from node to index in the OD matrix
    node_to_bus_index = {node: idx for idx, node in enumerate(N_B)}

    # Save OD matrix to a CSV file
    save_OD_matrix_csv(D, N_B, "data/bus_network/od_matrix.csv")

    return D, node_to_bus_index


def save_OD_matrix_csv(D, N_B, file):
    """
    Saves the origin-destination matrix to a CSV file.
    """

    # Save OD matrix to CSV file
    with open(file, "w") as f:
        f.write("src,dst,value\n")
        for i in range(len(N_B)):
            for j in range(len(N_B)):
                if i != j:
                    f.write(f"{N_B[i]},{N_B[j]},{D[i, j]}\n")

# --------------------------------------------
# --------------------------------------------
# INITIAL BUS ROUTE


def initial_bus_route(N_B, L):
    """
    Randomly selects L bus stops to be the initial bus route.
    """

    # Select initial bus route
    N_L = random.sample(N_B, L)

    return N_L


def save_bus_lines_csv(lines, file):
    """
    Saves the bus lines to a CSV file.
    """

    # Save bus lines to CSV file
    with open(file, "w") as f:
        for line in lines:
            f.write(",".join(str(node) for node in line) + "\n")

# --------------------------------------------
# --------------------------------------------
# BUS ROUTES COST


def bus_stop_cost(G, N_B):
    """
    Path cost between bus stops in the road network.
    """

    # Compute cost between bus stops
    C = np.zeros((len(N_B), len(N_B)))
    all_pairs_cost = dict(nx.all_pairs_dijkstra_path_length(G, weight="cost"))
    for i in range(len(N_B)):
        for j in range(i+1, len(N_B)):
            C[i, j] = all_pairs_cost[N_B[i]][N_B[j]]
            C[j, i] = C[i, j]

    return C

# --------------------------------------------
# --------------------------------------------
# LINES OBJECTIVE FUNCTION


# def lines_energies_old(D, lines, node_to_bus_index, C, lambdaa):
#     """
#     Objective function for bus route optimization using igraph.
#     """

#     # Create graph
#     G_B = ig.Graph()

#     nodes = list(set([node for line in lines for node in line]))
#     node_index = {node: idx for idx, node in enumerate(nodes)}
#     G_B.add_vertices(len(nodes))

#     edges = []
#     costs = []
#     for line in lines:
#         edge = [(min(node_index[line[i]], node_index[line[i+1]]),
#                  max(node_index[line[i]], node_index[line[i+1]])) for i in range(len(line)-1)]
#         edges += edge
#         G_B.add_edges(edge)
#         cost = [C[node_to_bus_index[line[i]], node_to_bus_index[line[i+1]]]
#                 for i in range(len(line)-1)]
#         G_B.es["cost"] = [float(c) for c in cost]
#         costs += cost
#     # Count repeated edges
#     n_L = [edges.count(edge) for edge in G_B.get_edgelist()]

#     # Compute passenger shares
#     # p_e = \sum_{i \neq j} D_{ij} \frac{\sigma_{ij}(e)}{\sigma_{ij}} / n_L(e) / length(path(i,j))
#     num_nodes = len(nodes)
#     p_e_term = [0] * len(edges)
#     node_bus_indices = [node_to_bus_index[node] for node in nodes]
#     for source in range(num_nodes):
#         for target in range(num_nodes):
#             if source != target:
#                 # paths = G_B.get_shortest_paths(source, to=target, weights=G_B.es["cost"], output="epath")
#                 paths = G_B.get_shortest_paths(
#                     source, to=target, weights=G_B.es["cost"], output="epath")
#                 num_edges_path = len(paths[0]) if len(paths) > 0 else 1
#                 if num_edges_path > 0:
#                     D_value = D[node_bus_indices[source],
#                                 node_bus_indices[target]] / num_edges_path
#                 for path in paths:
#                     for edge in path:
#                         p_e_term[edge] += float(D_value / n_L[edge])

#     # Initialize list of objective functions and indices
#     obj = [0] * len(lines)
#     fstt = [0] * len(lines)
#     sndt = [0] * len(lines)
#     # Compute objective function for each line
#     for i, line in enumerate(lines):
#         line_edges = [(min(node_index[line[i]], node_index[line[i+1]]),
#                        max(node_index[line[i]], node_index[line[i+1]])) for i in range(len(line)-1)]
#         # Compute first term
#         sum1 = sum([p_e_term[G_B.get_eid(edge[0], edge[1])]
#                    for edge in line_edges])
#         # second_term = sum([G_B.es[G_B.get_eid(edge[0], edge[1])]["cost"] for edge in line_edges])
#         sum2 = sum([costs[G_B.get_eid(edge[0], edge[1])]
#                    for edge in line_edges])
#         fstt[i] = - sum1
#         sndt[i] = lambdaa * sum2
#         obj[i] = - sum1 + lambdaa * sum2

#     return obj, fstt, sndt


# def lines_energies_prev(D, lines, node_to_bus_index, C, lambdaa):
#     """
#     Objective function for bus route optimization using igraph.
#     """

#     # Create graph
#     G_B = ig.Graph()

#     # Add vertices to the graph
#     from itertools import chain
#     nodes = list(set(chain.from_iterable(lines)))
#     node_index = {node: idx for idx, node in enumerate(nodes)}
#     G_B.add_vertices(len(nodes))

#     # Add edges to the graph with costs and degeneration
#     from collections import defaultdict
#     edge_cost_dict = {}
#     edge_count = defaultdict(int)
#     for line in lines:
#         for i in range(len(line) - 1):
#             u = node_index[line[i]]
#             v = node_index[line[i+1]]
#             edge = (min(u, v), max(u, v))
#             bus_u = node_to_bus_index[line[i]]
#             bus_v = node_to_bus_index[line[i+1]]
#             cost = C[bus_u, bus_v]
#             if edge not in edge_cost_dict:
#                 edge_cost_dict[edge] = cost
#             edge_count[edge] += 1
#     edges = list(edge_cost_dict.keys())
#     costs = [edge_cost_dict[edge] for edge in edges]
#     nL = [edge_count[edge] for edge in edges]
#     G_B.add_edges(edges)
#     G_B.es["cost"] = [float(c) for c in costs]

#     num_nodes = len(nodes)
#     p_e_term = [0.0] * len(edges)
#     node_bus_indices = [node_to_bus_index[node]
#                         for node in nodes]  # Precompute node -> bus index once
#     weights = G_B.es["cost"]  # cache attribute lookup

#     # Compute passenger shares
#     for source in range(num_nodes):
#         bus_src = node_bus_indices[source]
#         for target in range(num_nodes):
#             if source == target:
#                 continue
#             bus_tgt = node_bus_indices[target]
#             # Compute shortest path (edge-path)
#             paths = G_B.get_shortest_paths(
#                 source, to=target, weights=weights, output="epath")
#             if not paths or not paths[0]:  # no path found
#                 continue
#             path_edges = paths[0]
#             num_edges = len(path_edges)
#             D_val = D[bus_src, bus_tgt]
#             weight_per_edge = D_val / num_edges
#             for edge in path_edges:
#                 p_e_term[edge] += weight_per_edge / nL[edge]

#     # Compute objective function for each line
#     obj = [0.0] * len(lines)
#     fstt = [0.0] * len(lines)
#     sndt = [0.0] * len(lines)
#     for i, line in enumerate(lines):
#         line_edges = [(min(node_index[line[j]], node_index[line[j+1]]),
#                        max(node_index[line[j]], node_index[line[j+1]])) for j in range(len(line) - 1)]
#         edge_ids = [G_B.get_eid(u, v) for u, v in line_edges]
#         sum1 = sum(p_e_term[eid] for eid in edge_ids)
#         sum2 = sum(costs[eid] for eid in edge_ids)
#         fstt[i] = -sum1
#         sndt[i] = sum2
#         obj[i] = fstt[i] + lambdaa * sndt[i]

#     return obj, fstt, sndt

from itertools import chain
from collections import defaultdict

def lines_energies(D, lines, node_to_bus_index, C, lambdaa):
    """
    Objective function for bus route optimization using igraph.
    """
    # Create graph
    G_B = ig.Graph()
    
    # Flatten and deduplicate nodes
    nodes = list(set(chain.from_iterable(lines)))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    G_B.add_vertices(len(nodes))
    
    # Build edge data 
    edge_cost_dict = {}
    edge_count = defaultdict(int)
    
    # Process all lines at once to build edge data
    for line in lines:
        for i in range(len(line) - 1):
            u, v = node_index[line[i]], node_index[line[i+1]]
            edge = (min(u, v), max(u, v))  # Undirected edge
            bus_u, bus_v = node_to_bus_index[line[i]], node_to_bus_index[line[i+1]]
            
            if edge not in edge_cost_dict:
                edge_cost_dict[edge] = C[bus_u, bus_v]
            edge_count[edge] += 1
    
    # Add all edges
    edges = list(edge_cost_dict.keys())
    costs = [edge_cost_dict[edge] for edge in edges]
    nL = [edge_count[edge] for edge in edges]
    
    G_B.add_edges(edges)
    G_B.es["cost"] = [float(c) for c in costs]
    
    # Precompute all data needed for passenger share calculation
    num_nodes = len(nodes)
    p_e_term = np.zeros(len(edges))
    node_bus_indices = [node_to_bus_index[node] for node in nodes]
    weights = G_B.es["cost"]
    
    # Compute passenger shares for all origin-destination pairs
    for source in range(num_nodes):
        bus_src = node_bus_indices[source]
        
        # Need to process all targets since demand could be asymmetric
        for target in range(num_nodes):
            if source == target:
                continue
                
            bus_tgt = node_bus_indices[target]
            D_val = D[bus_src, bus_tgt]
            
            if D_val == 0:  # Skip if no demand between these stops
                continue
                
            # Compute shortest path
            paths = G_B.get_shortest_paths(
                source, to=target, weights=weights, output="epath")
            if not paths or not paths[0]:
                continue
                
            path_edges = paths[0]
            num_edges = len(path_edges)
            weight_per_edge = D_val / num_edges
            
            # Update p_e_term for all edges in this path
            for edge in path_edges:
                p_e_term[edge] += weight_per_edge / nL[edge]
    
    # Compute objective function for each line
    obj = [0.0] * len(lines)
    fstt = [0.0] * len(lines)
    sndt = [0.0] * len(lines)
    
    for i, line in enumerate(lines):
        # Calculate edges in this line
        line_edges = []
        for j in range(len(line) - 1):
            u, v = node_index[line[j]], node_index[line[j+1]]
            line_edges.append((min(u, v), max(u, v)))
            
        # Look up edge IDs
        edge_ids = [G_B.get_eid(*edge) for edge in line_edges]
        
        # Calculate objective components
        sum1 = sum(p_e_term[eid] for eid in edge_ids)
        sum2 = sum(costs[eid] for eid in edge_ids)
        
        fstt[i] = -sum1  # Demand term
        sndt[i] = sum2   # Cost term
        obj[i] = fstt[i] + lambdaa * sndt[i]
    
    return obj, fstt, sndt


# --------------------------------------------
# --------------------------------------------
# SIMULATED ANNEALING FOR MULTI BUS ROUTES OPTIMIZATION


def lines_simulated_annealing(D, N_B, lines, node_to_bus_index, C, max_iter, T_0, lambdaa):
    """
    Simulated annealing algorithm for the bus route optimization.
    """

    random.seed(None)
    np.random.seed(None)

    # Initial temperature
    T = T_0

    # Initial objective function values
    H_old, ft_old, st_old = lines_energies(
        D, lines, node_to_bus_index, C, lambdaa)

    # Generate file to store energy values
    file_name = "data/energy.csv"
    with open(file_name, "w") as f:
        f.write(
            ",".join(f"ft{k},st{k},e{k}" for k in range(len(H_old))) + "\n")
        f.write(",".join(str(ft) + "," + str(st) + "," + str(obj)
                for ft, st, obj in zip(ft_old, st_old, H_old)) + "\n")

    j = -1
    for i in tqdm(range(int(max_iter)), desc="SA optimization", ncols=100):
    # for i in range(int(max_iter)):

        if j == len(lines)-1:
            j = 0
        else:
            j += 1

        N_L = lines[j]

        N_L_new = N_L.copy()

        n_i = random.choice(list(set(N_L)))
        n_j = random.choice(list(set(N_B) - {n_i}))

        N_L_new[N_L_new.index(n_i)] = n_j
        if n_j in N_L:
            N_L_new[N_L_new.index(n_j)] = n_i

        new_lines = lines.copy()
        new_lines[j] = N_L_new

        H_new, ft_new, st_new = lines_energies(
            D, new_lines, node_to_bus_index, C, lambdaa)

        delta_H = H_new[j] - H_old[j]

        try:
            if random.random() < min(1, m.exp(-delta_H / T)):

                lines = new_lines
                H_old = H_new

                with open(file_name, "a") as f:
                    f.write(",".join(str(ft) + "," + str(st) + "," + str(obj)
                            for ft, st, obj in zip(ft_new, st_new, H_new)) + "\n")

        except OverflowError:
            pass

        T = T_0 * (1 - i / max_iter)

    return lines, H_old

# --------------------------------------------
# --------------------------------------------
# GLOBAL COVERED DEMAND


def global_covered_demand(D, lines, node_to_bus_index, C):
    """
    Computes the global coverage demand for the bus routes.
    """

    # Create graph
    G_B = ig.Graph()

    from itertools import chain
    nodes = list(set(chain.from_iterable(lines)))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    G_B.add_vertices(len(nodes))

    from collections import defaultdict

    edge_cost_dict = {}
    edge_count = defaultdict(int)

    for line in lines:
        for i in range(len(line) - 1):
            u = node_index[line[i]]
            v = node_index[line[i+1]]
            edge = (min(u, v), max(u, v))
            bus_u = node_to_bus_index[line[i]]
            bus_v = node_to_bus_index[line[i+1]]
            cost = C[bus_u, bus_v]

            if edge not in edge_cost_dict:
                edge_cost_dict[edge] = cost

            edge_count[edge] += 1

    edges = list(edge_cost_dict.keys())
    costs = [edge_cost_dict[edge] for edge in edges]
    nL = [edge_count[edge] for edge in edges]

    G_B.add_edges(edges)
    G_B.es["cost"] = [float(c) for c in costs]

    num_nodes = len(nodes)
    p_e_term = [0.0] * len(edges)
    node_bus_indices = [node_to_bus_index[node]
                        for node in nodes]  # Precompute node -> bus index once
    D_covered = 0.0
    weights = G_B.es["cost"]  # cache attribute lookup
    for source in range(num_nodes):
        bus_src = node_bus_indices[source]
        for target in range(num_nodes):
            if source == target:
                continue
            bus_tgt = node_bus_indices[target]
            # Compute shortest path (edge-path)
            paths = G_B.get_shortest_paths(
                source, to=target, weights=weights, output="epath")
            if not paths or not paths[0]:  # no path found
                continue
            path_edges = paths[0]
            num_edges = len(path_edges)
            D_val = D[bus_src, bus_tgt]
            D_covered += D_val
            weight_per_edge = D_val / num_edges
            for edge in path_edges:
                p_e_term[edge] += weight_per_edge / nL[edge]

    a = round(D_covered, 2)
    b = round(sum([p_e_term[i] * nL[i] for i in range(len(p_e_term))]), 2)
    if a != b:
        print("Not Satisfied")

    return np.round(D_covered / np.sum(D) * 100, 2)


# --------------------------------------------
# --------------------------------------------
# GLOBAL TRAVEL COST
def global_travel_cost(C, lines, node_to_bus_index):
    """
    Computes the global travel cost for the bus routes.
    """
    # Initialize travel cost
    travel_cost = 0

    # Compute travel cost
    for line in lines:
        for i in range(len(line)-1):
                travel_cost += C[node_to_bus_index[line[i]],
                                 node_to_bus_index[line[i+1]]]

    travel_cost = np.round(travel_cost, 2)

    return -travel_cost
# --------------------------------------------