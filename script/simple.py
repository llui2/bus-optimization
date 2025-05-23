from informal import (
    road_network,
    bus_stops,
    OD_matrix,
    bus_stop_cost,
    initial_bus_route,  
    lines_simulated_annealing,
    lines_energies,
    global_covered_demand,
    global_travel_cost,
    save_bus_lines_csv
)

seed = 10  # random.randint(0, 1000)
d = 20
shift = 0.2

# FIXAT
# d = 20, shift = 0.2, num_lines = 2, L = 8

# NO FIXAT
# lambda, max_iter = 1e3, T_0 = 3

G, pos = road_network(seed, d, shift)

N_B = bus_stops(G)

D, node_to_bus_index = OD_matrix(seed, N_B, pos, d)

C = bus_stop_cost(G, N_B)

num_lines = 2
L = 8
lines = [initial_bus_route(N_B, L) for _ in range(num_lines)]

lambdaa = 2# 0.2 * L * num_lines * 1

print("\n lambda = ", lambdaa, "\n")

obj = lines_energies(D, lines, node_to_bus_index, C, lambdaa)

max_iter = 1e5
T_0 = 3

lines, obj = lines_simulated_annealing(
    D, N_B, lines, node_to_bus_index, C, max_iter, T_0, lambdaa)

# Compute global covered demand
coverage_demand = global_covered_demand(D, lines, node_to_bus_index, C)
print("\n Coverage demand: ", coverage_demand, " %")

# Compute global travel cost
travel_cost = global_travel_cost(C, lines, node_to_bus_index)
print("\n Travel cost: ", travel_cost, " u")

save_bus_lines_csv(lines, "data/bus_network/lines.csv")
