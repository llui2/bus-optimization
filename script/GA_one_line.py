# Entrenar una única línia de bus amb un enfocament evolutiu
# Cada "individu" és una línia (una seqüència de parades)
# L’objectiu és maximitzar la demanda coberta i minimitzar la distància total

# estudi en profunditat

import random
import numpy as np
import os
from informal import (
    road_network, bus_stops, OD_matrix, bus_stop_cost,
    save_bus_lines_csv, line_energy, euclidean_distance  # Canvi important aquí
)

class GeneticAlgorithmOneLine:
    def __init__(self, seed, d=20, shift=0.2, L=8, pop_size=20):
        self.seed = seed
        self.d = d
        self.shift = shift
        self.L = L
        self.pop_size = pop_size
        
        self.G, self.pos = road_network(seed, d, shift)
        self.N_B = bus_stops(self.G)
        self.D, self.node_to_bus_index = OD_matrix(seed, self.N_B, self.pos, d)
        self.C = bus_stop_cost(self.G, self.N_B)

        self.population = []
        self.best_individual = None
        self.best_score = float('inf')

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            individual = random.sample(self.N_B, self.L)
            self.population.append(individual)

    def evaluate(self, individual, lambdaa):
        return line_energy(self.D, individual, self.node_to_bus_index, self.C, lambdaa)  # una sola línia

    def select_parents(self, fitnesses):
        sorted_indices = np.argsort(fitnesses)
        return [self.population[i] for i in sorted_indices[:2]]  # els 2 millors

    def crossover(self, parent1, parent2):
        cut = random.randint(1, self.L - 2)
        child = parent1[:cut] + [node for node in parent2 if node not in parent1[:cut]]
        return child[:self.L]

    def mutate(self, individual):
        if random.random() < 0.2:
            i = random.randint(0, self.L - 1)
            new_node = random.choice([n for n in self.N_B if n not in individual])
            individual[i] = new_node
        return individual

    def run(self, generations=50, lambdaa=8):
        print(f"\nEntrenant una línia amb GA: lambda = {lambdaa}, generacions = {generations}\n")
        random.seed(self.seed)
        self.initialize_population()

        for gen in range(generations):
            fitnesses = [self.evaluate(ind, lambdaa) for ind in self.population]

            best_gen_score = min(fitnesses)
            if best_gen_score < self.best_score:
                self.best_score = best_gen_score
                self.best_individual = self.population[np.argmin(fitnesses)]

            new_population = []
            parents = self.select_parents(fitnesses)
            while len(new_population) < self.pop_size:
                child = self.crossover(*parents)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

        save_bus_lines_csv([self.best_individual], os.path.join("data/bus_network", "lines.csv"))
        print("Entrenament complet. Línia GA desada a lines.csv\n")

    def get_result(self):
        return {
            "line": self.best_individual,
            "energy": self.best_score,
            "network": self.G,
            "positions": self.pos,
            "N_B": self.N_B,
            "D": self.D,
            "C": self.C,
            "node_to_bus_index": self.node_to_bus_index
        }
