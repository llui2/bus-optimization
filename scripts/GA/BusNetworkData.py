import random
import numpy as np
from BusNetworkData import BusNetworkData
import networkx as nx
import pandas as pd
import copy
from collections import Counter
#################################################
#       Classe principal Genetic Algorithm      #
#################################################
class GeneticOptimizer:
    # 1. --- INITIAL POPULATION ---
    def __init__(self, NB, L, n_population, OD, C, seed=None):
        self.NB = NB
        self.L = L
        self.n_population = n_population
        self.OD = OD
        self.C = C
        self.seed = seed
        self.population = self._generate_initial_population()

    def _generate_initial_population(self):
        """
        Genera la població inicial de línies de bus candidates per a l'algorisme genètic.

        Cada individu de la població representa una possible línia de bus, formada per una
        seqüència ordenada de parades seleccionades del conjunt total de parades disponibles (NB).

        Si s'han proporcionat pesos de demanda (`demand_weights`), les parades es seleccionen
        ponderadament, de manera que les parades situades en zones de més densitat tenen més
        probabilitat d'aparèixer.

        Returns:
            list[list[int]]: Llista d'individus. Cada individu és una llista ordenada de parades.
        """

        if self.seed is not None:
            random.seed(self.seed)
        
        population = []
        for _ in range(self.n_population):
            individual = random.sample(self.NB, k=self.L)
            population.append(individual)
        return population

    def print_population(self):
        for i, individual in enumerate(self.population):
            print(f"Individu {i+1}: {individual}")
    
    # 2. --- FITNESS FUNCTION ---
    def fitness(self, individual, lambda_):

        """
        Avalua la qualitat d'una línia de bus representada per una seqüència de parades.

        L’energia de la línia es calcula com:
        E = −(1 − λ) · p_e + λ · c_e

        Args:
        individual (list[int]): Seqüència ordenada de parades que formen una línia de bus.
        lambda_ (float): Paràmetre de pes entre cost i servei (0 ≤ λ ≤ 1).

        Returns:
            float: Valor de fitness (com més alt, millor)
        """
        total_pe = 0
        total_ce = 0

        G_line = nx.Graph()
        
        # 1. Construir el graf de la línia
        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            G_line.add_edge(u, v)

        # 2. Per cada aresta, calculem pe i ce
        for u, v in G_line.edges():
            # c_e = 1 (fins que tinguem distàncies reals)
            try:
                i_idx = self.NB.index(u)
                j_idx = self.NB.index(v)
                c_e = self.C[i_idx, j_idx]
            except ValueError:
                c_e = np.inf  # try-catch per si alguna parada no està definida

            total_ce += c_e

            # p_e: suma de passatgers a l'aresta
            pe = 0
            for i in individual:
                for j in individual:
                    if i != j:
                        try:
                            path = nx.shortest_path(G_line, source=i, target=j)
                            if (u in path) and (v in path):
                                pe += self.OD.loc[i, j] / len(path)  # repartim la demanda
                        except:
                            continue
            total_pe += pe

        # 3. Fórmula de l'energia (H)
        energy = -(1 - lambda_) * total_pe + lambda_ * total_ce
        return -energy

    # 3. --- SELECTION ---
    def select_parents_tournament(self, lambda_, k=3):

        """
        Selecciona els pares per creuar mitjançant el mètode de torneig.

        L'individu amb el millor fitness és seleccionat com a pare. Això es repeteix fins a obtenir tants pares
        com individus a la població.

        Args:
            lambda_ (float): Paràmetre de pes entre servei i cost per calcular el fitness.
            k (int, optional): Mida del torneig (nombre d'individus que competeixen). Per defecte 3.

        Returns:
            list[list[int]]: Llista de pares seleccionats per a la següent generació.
        """

        selected = []
        fitnesses = [self.fitness(ind, lambda_) for ind in self.population]

        for _ in range(len(self.population)):
            # Escollim k individus aleatoris per competir
            candidates = random.sample(list(zip(self.population, fitnesses)), k)

            # Triem el millor d'aquests (fitness més alt)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def select_elite(self, lambda_, n_elite):

        """
        Selecciona els millors individus de la població segons el seu fitness (elitisme).

        Ordena tota la població per valor de fitness (de millor a pitjor) i retorna els millors.
        D'aquesta manera assegura que les millors solucions es preservin en la següent generació sense modificar-se.

        Args:
            lambda_ (float): Paràmetre de pes entre servei i cost per calcular el fitness.
            n_elite (int): Nombre d’individus d’elit a conservar.

        Returns:
            list[list[int]]: Llista amb els millors individus de la població.
        """

        fitnesses = [self.fitness(ind, lambda_) for ind in self.population] #  list comprehension - calcula per a cada individu el seu fitness - llista amb els valors fitness
        sorted_population = [ind for _, ind in sorted(zip(fitnesses, self.population), reverse=True)] # ordena els valors - llista ordenada dels valors fitness
        return sorted_population[:n_elite] # slicing operation - es guarda el # d'individus seleccionat per paràmetres

    # 4. --- CROSSOVER ---
    def crossover_OX(self, parent1, parent2):

        """
        Aplica el mètode de Order Crossover (OX) entre dos pares.

        Funcionament:
        1. Se selecciona un segment aleatori del parent1 i es copia directament al fill.
        2. Es recorre el parent2 i es completen les posicions buides del fill amb els valors
        que no estiguin presents i mantenint l’ordre.

        Args:
            parent1 (list[int]): Primer progenitor (línia de bus representada com a seqüència de parades).
            parent2 (list[int]): Segon progenitor.

        Returns:
            list[int]: Nou individu (fill) generat a partir de la combinació dels dos pares.
        """

        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 1. Segment del parent1
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        
        # 2. Completar amb parent2 (evitant duplicats)
        p2_index = 0
        for i in range(size):
            if child[i] is not None:
                continue
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
        
        return child

    # 4. --- MUTACIÓ ---
    def mutate(self, individual, prob=0.2):

        """
        Aplica una mutació aleatòria sobre un individu amb una certa probabilitat (20% -> s'ha d'afinar)

        El mètode de mutació és l’intercanvi de dues parades aleatòries dins la línia.
        Aquesta operació manté la validesa de l’individu, canvia l’ordre. Ajuda a evitar l’estancament evolutiu.

        Args:
            individual (list[int]): Línia de bus (seqüència de parades) que serà mutada.
            prob (float, optional): Probabilitat d'aplicar la mutació. Per defecte 0.2 (20%).

        Returns:
            list[int]: Nova línia (mutada o igual a l’original si no hi ha hagut mutació).
        """

        mutated = copy.deepcopy(individual)
        if random.random() < prob:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    # ho adjuntem tot
    def evolve_one_generation(self, lambda_, n_elite=1, k=3):

        """
        Es combina l'elitisme i el torneig per obtenir resultats més precisos
        
        1. Elitisme
        2. Selecció de pares per torneig
        3. Crossover i mutació
        4. Formació de la nova població

        Args:
            lambda_ (float): Paràmetre de pes entre servei i cost a la funció de fitness.
            n_elite (int, optional): Nombre d’individus d’elit a conservar. Per defecte 1.
            k (int, optional): Mida del torneig per la selecció. Per defecte 3.
        """

        # 1. Elitisme
        elite = self.select_elite(lambda_, n_elite)

        # 2. Selecció de pares via torneig
        parents = self.select_parents_tournament(lambda_, k)

        # 3. Generar fills amb crossover OX
        children = []
        while len(children) < (self.n_population - n_elite):
            p1, p2 = random.sample(parents, 2)
            #child = self.crossover_OX(p1, p2)
            #children.append(child)
            child = self.crossover_OX(p1, p2)
            child = self.mutate(child, prob=0.2)
            children.append(child)

        # 4. Nova població = elit + fills
        self.population = elite + children

if __name__ == "__main__":

    # Paràmetres fixats
    d = 20
    shift = 0.2
    L = 8           # longitud línia de bus - nombre total de parades que ha de tenir cada individu
    n_population = 5
    seed = 42
    lambda_ = 0.5   # Valor de compromís entre servei i cost
    n_generations = 50

    data = BusNetworkData(
        nodes_path="../../data/bus_network/nodes.csv",
        od_path="../../data/bus_network/od_matrix.csv"
    )
    NB = data.get_available_stops()

    edges_df = pd.read_csv("../../data/road_network/edges.csv")     # Carreguem les arestes del graf viàri
    nodes_df = pd.read_csv("../../data/road_network/nodes.csv")     # Carreguem nodes i posicions
    pos = {int(row["node"]): (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    # Per crear el graf
    G = nx.Graph()
    for node, (x, y) in pos.items():    # afegim els nodes
        G.add_node(node, pos=(x, y))

    for _, row in edges_df.iterrows():  # afegim les arestes
        G.add_edge(int(row["src"]), int(row["dst"]), cost=row["cost"])

    # Calculem la matriu de costos reals entre parades
    C = np.zeros((len(NB), len(NB)))    # matriu quadrada de zeros
    all_costs = dict(nx.all_pairs_dijkstra_path_length(G, weight="cost"))   # Dijkstra per calcular el cost mínim entre tots els parells - diccionari de diccionaris

    for i, ni in enumerate(NB):
        for j, nj in enumerate(NB):
            if ni in all_costs and nj in all_costs[ni]:     # en cas que hi hagi un camí, es guarda al diccionari
                C[i, j] = all_costs[ni][nj]
            else:
                C[i, j] = np.inf

    optimizer = GeneticOptimizer(NB, L, n_population, data.OD, C, seed)

    print("Població seleccionada (NB):")
    optimizer.print_population()

    all_stops = [stop for individual in optimizer.population for stop in individual]
    most_common = Counter(all_stops).most_common(10)

    for gen in range(n_generations):
        optimizer.evolve_one_generation(lambda_, n_elite=2, k=3)

        # Seguiment del fitness per generació
        fitnesses = [optimizer.fitness(ind, lambda_) for ind in optimizer.population]
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        
        print(f"Gen {gen+1:3}: Best = {best:.3f} | Avg = {avg:.3f}")
        
        progress = (gen + 1) / n_generations * 100
        print(f"Progressió: {progress:.1f}% completat", end="\r")

    #for i, individual in enumerate(optimizer.population):
    #    fitness_val = optimizer.fitness(individual, lambda_)
    #best_fitness = optimizer.fitness(best_individual, lambda_)
    #print(f"Fitness final: {best_fitness:.3f}")

    best_individual = max(optimizer.population, key=lambda ind: optimizer.fitness(ind, lambda_))

    with open("../../results/best_line_GA.txt", "w") as f:
        f.write(",".join(map(str, best_individual)))
