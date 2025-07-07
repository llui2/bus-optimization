import networkx as nx
import random
import csv
import os

import numpy as np

class GeneticAlgorithmOneLine:
    def __init__(self, G, NB, D, C, lambda_, pop_size=50, generations=100, mutation_rate=0.1):
        self.G = G              # Graf de carreteres
        self.NB = NB            # Parades de bus
        self.D = D              # Matriu OD
        self.C = C              # Cost entre parades
        # guarda les parelles útils (origin, dest, demanda)
        self.od_pairs = []
        B = len(self.NB)
        for i in range(B):
            for j in range(B):
                if i != j and D[i][j] > 0:
                    origin = self.NB[i]
                    dest = self.NB[j]
                    self.od_pairs.append((origin, dest, D[i][j]))

        self.lambda_ = lambda_  # Paràmetre d'equilibri
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.precompute_shortest_paths()

    '''
    Cada individu és una llista ordenada de L parades aleatòries (sense repeticions).
    Longitud desitjada d’una línia és L = 10
    '''
    def initialize_population(self, L=10):
        population = []

        # 1. Calcula valors del camp de demanda
        pos = nx.get_node_attributes(self.G, "pos")
        d = int(len(pos) ** 0.5)
        _, _, field = self.generate_field(d)

        field_values = []
        for node in self.NB:
            x, y = pos[node]
            i = int(x * 100 / d)
            j = int(y * 100 / d)
            field_values.append(field[j, i])

        # 2. Normalitza com a probabilitat
        total = sum(field_values)
        probs = [v / total for v in field_values]

        # 3. Genera individus amb start/end segons demanda
        while len(population) < self.pop_size:
            start = random.choices(self.NB, weights=probs)[0]
            end = random.choices(self.NB, weights=probs)[0]
            if start != end and nx.has_path(self.G, start, end):
                path = nx.shortest_path(self.G, source=start, target=end, weight='cost')
                if len(path) >= L:
                    trimmed = path[:L]
                    population.append(trimmed)

        return population

    def generate_field(self, d):
        x = np.linspace(0, d-1, 100)
        y = np.linspace(0, d-1, 100)
        X, Y = np.meshgrid(x, y)
        sigma = d / 12
        center_x, center_y = d / 2, d / 2
        gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        return X, Y, gaussian

    '''
    Fórmula de H(G_L)
    '''
    def fitness(self, line):
        """
        Fitness basada en la nova fórmula d'energia:
        H(G_L) = -(1 - λ) * ∑(p_e - c_e)
        """
        edges = self.calculate_edges(line)
        total = 0
        for e in edges:
            pe = self.calculate_passenger_share(e)
            ce = self.calculate_cost(e)
            total += (pe - ce)
        # Penalitza girs: 0.5 punts per gir (ajustar si cal)
        num_turns = self.count_turns(line)
        penalty = 2.0 * num_turns
        return -(1 - self.lambda_) * total + penalty

    '''
    Selecció per torneig -> es pot fer amb ruleta (mirar-ho més endavant)
    '''
    def select_parents(self, population, fitnesses):
        parents = random.choices(population, weights=fitnesses, k=2)
        return parents[0], parents[1]

    '''
    Creuament parcial (tipo PMX)
    '''
    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 1. Copiem la subcadena del primer pare
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        # 2. Omplim els buits amb els gens del pare 2 que no estan al fill
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                # Salta gens del pare2 ja presents al fill
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1

        return child


    '''
    Intercanviem dues parades aleatòriament.
    '''
    def mutate(self, line):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(line)), 2)
            line[i], line[j] = line[j], line[i]
        return line

    '''
    Donada una llista ordenada de parades (line),
    aquesta funció et retorna la llista d’arestes consecutives (parelles de parades).
    '''
    def calculate_edges(self, line):
        edges = []
        for i in range(len(line) - 1):
            u = line[i]
            v = line[i + 1]
            edges.append((u, v))
        return edges
    
    def calculate_cost(self, edge):
        u, v = edge
        path = nx.shortest_path(self.G, source=u, target=v, weight='weight')
        cost = 0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            cost += self.G[a][b]['cost']
        return cost

    '''
    Formula de p_e
    '''
    def precompute_shortest_paths(self):
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path(self.G, weight="cost"))

    def calculate_passenger_share(self, edge):
        u, v = edge
        total_pe = 0

        for origin, dest, demand in self.od_pairs:
            try:
                path = self.shortest_paths[origin][dest]
                if len(path) < 2:
                    continue
                if self._edge_in_path(edge, path):
                    total_pe += demand / (len(path) - 1)
            except KeyError:
                continue

        return total_pe

    def _edge_in_path(self, edge, path):
        u, v = edge
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) == (u, v) or (path[i], path[i + 1]) == (v, u):
                return True
        return False

    '''
    Actua com el main del GA
    '''
    def evolve(self):
        self.energy_by_generation = []

        population = self.initialize_population()
        for gen in range(self.generations):
            print(f"Generació {gen+1}/{self.generations}")
            fitnesses = [self.fitness(ind) for ind in population]

            # 🧠 Elitisme: guarda el millor individu
            best_idx = fitnesses.index(max(fitnesses))
            best_individual = population[best_idx]
            best_fitness = fitnesses[best_idx]

            new_population = [best_individual]  # conserva el millor

            while len(new_population) < self.pop_size:
                attempts = 0
                while attempts < 10:
                    p1, p2 = self.select_parents(population, fitnesses)
                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    if self.is_valid_line(child):
                        new_population.append(child)
                        break
                    attempts += 1
                else:
                    # fallback
                    new_population.append(random.choice(population))

            population = new_population
            self.energy_by_generation.append(best_fitness)

        # Guarda l’energia en CSV
        with open("data/energy_ga.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "energy"])
            for gen_idx, energy in enumerate(self.energy_by_generation):
                writer.writerow([gen_idx, energy])

        # Retorna el millor individu final
        final_fitnesses = [self.fitness(ind) for ind in population]
        best_index = final_fitnesses.index(max(final_fitnesses))
        return population[best_index]

    
    def is_valid_line(self, line):
        # Comprova que hi ha camí entre cada parella consecutiva
        for i in range(len(line) - 1):
            u, v = line[i], line[i + 1]
            if not nx.has_path(self.G, u, v):
                return False
        return True

    def count_turns(self, line):
        turns = 0
        for i in range(1, len(line) - 1):
            u = line[i - 1]
            v = line[i]
            w = line[i + 1]

            # Direccions: vector entre nodes (en posició 2D)
            pu = self.G.nodes[u]["pos"]
            pv = self.G.nodes[v]["pos"]
            pw = self.G.nodes[w]["pos"]

            # vectors direccionals
            vec1 = (pv[0] - pu[0], pv[1] - pu[1])
            vec2 = (pw[0] - pv[0], pw[1] - pv[1])

            # si canvia la direcció (no són colineals), compta com a gir
            if vec1 != vec2:
                turns += 1

        return turns

