import informal 

seed = 10  # random.randint(0, 1000)
d = 20
shift = 0.2

# FIXAT
# d = 20, shift = 0.2, num_lines = 2, L = 8

# NO FIXAT
# lambda, max_iter = 1e3, T_0 = 3

G, pos = informal.road_network(seed, d, shift)

N_B = informal.bus_stops(G)

D, node_to_bus_index = informal.OD_matrix(seed, N_B, pos, d)

C = informal.bus_stop_cost(G, N_B)

num_lines = 2
L = 8
lines = [informal.initial_bus_route(N_B, L) for _ in range(num_lines)]

lambdaa = 2# 0.2 * L * num_lines * 1

print("\n lambda = ", lambdaa, "\n")

obj = informal.lines_energies(D, lines, node_to_bus_index, C, lambdaa)

max_iter = 1e5
T_0 = 3

lines, obj = informal.lines_simulated_annealing(
    D, N_B, lines, node_to_bus_index, C, max_iter, T_0, lambdaa)

# Compute global covered demand
coverage_demand = informal.global_covered_demand(D, lines, node_to_bus_index, C)
print("\n Coverage demand: ", coverage_demand, " %")

# Compute global travel cost
travel_cost = informal.global_travel_cost(C, lines, node_to_bus_index)
print("\n Travel cost: ", travel_cost, " u")

informal.save_bus_lines_csv(lines, "data/bus_network/lines.csv")
