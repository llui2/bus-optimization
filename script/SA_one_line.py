import random
import os
import numpy as np
from tqdm import tqdm

from informal import (
    #euclidean_distance,
    road_network,
    bus_stops,
    OD_matrix,
    bus_stop_cost,
    initial_bus_route,
    save_bus_lines_csv,
    lines_energies,
    lines_simulated_annealing,
)

class SimulatedAnnealing:
    def __init__(self, seed, d=20, shift=0.2, L=8):
        self.seed = seed
        self.d = d
        self.shift = shift
        self.L = L

        self.G, self.pos = road_network(seed, d, shift)
        self.N_B = bus_stops(self.G)
        self.D, self.node_to_bus_index = OD_matrix(seed, self.N_B, self.pos, d)
        self.C = bus_stop_cost(self.G, self.N_B)

        self.initial_line = initial_bus_route(self.N_B, L)
        self.lines = [self.initial_line]
        self.optimized_lines = None
        self.obj = None

    def run(self, max_iter=1000, T_0=3, lambdaa=None):
        if lambdaa is None:
            lambdaa = 1 * self.L

        print(f"\nEntrenant una línia amb λ = {lambdaa}, T_0 = {T_0}, iteracions = {max_iter}\n")
        obj, ft, st = lines_energies(
            self.D, self.lines, self.node_to_bus_index, self.C, lambdaa
        )

        optimized_lines, obj = lines_simulated_annealing(
            self.D, self.N_B, self.lines, self.node_to_bus_index, self.C, max_iter, T_0, lambdaa
        )

        self.optimized_lines = optimized_lines
        self.obj = obj

        save_bus_lines_csv(
            optimized_lines, os.path.join("data/bus_network", "lines.csv")
        )

        print("Entrenament complet. Línia optimitzada desada a lines.csv\n")

    def get_result(self):
        return {
            "line": self.optimized_lines[0],
            "energy": self.obj[0],
            "network": self.G,
            "positions": self.pos,
            "N_B": self.N_B,
            "D": self.D,
            "C": self.C,
            "node_to_bus_index": self.node_to_bus_index,
        }