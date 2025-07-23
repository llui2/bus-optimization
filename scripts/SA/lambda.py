# Study the covered demand and travel cost of the bus lines with different values of lambda
import numpy as np

from informal import (
    road_network,
    bus_stops,
    OD_matrix,
    bus_stop_cost,
    initial_bus_route,  
    lines_simulated_annealing,
    global_covered_demand,
    global_travel_cost
    )

seed, d, shift = 10, 20, 0.2
max_iter, T_0 = 1e4, 3

lambdaas = np.arange(0, 1, 0.05)

G, pos = road_network(seed, d, shift)
N_B = bus_stops(G)
D, node_to_bus_index = OD_matrix(seed, N_B, pos, d)
C = bus_stop_cost(G, N_B)

num_lines, L = 2, 8

file_name = "data/lambda/results.csv"
with open(file_name, "w") as f:
    f.write(f"seed,d,shift,max_iter,T_0,lambda,num_lines,L,covered_demand,travel_cost\n")

    for lambdaa in lambdaas:

        for _ in range(10):

            print(f"lambda = {lambdaa} | iter = {_} \r", end="")

            lines = [initial_bus_route(N_B, L) for _ in range(num_lines)]
            lines, obj = lines_simulated_annealing(D, N_B, lines, node_to_bus_index, C, max_iter, T_0, lambdaa)
            
            covered_demand = global_covered_demand(D, lines, node_to_bus_index, C)
            travel_cost = global_travel_cost(C, lines, node_to_bus_index)

            f.write(f"{seed},{d},{shift},{max_iter},{T_0},{lambdaa},{num_lines},{L},{covered_demand},{travel_cost}\n")


