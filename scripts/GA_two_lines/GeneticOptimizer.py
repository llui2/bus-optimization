import random
import numpy as np
import networkx as nx
import pandas as pd
import copy

from collections import Counter

from BusNetworkData import BusNetworkData
from od_generation import OD_matrix

def total_route_length(self, lines):
    total_length = 0
    for line in lines:
        for i in range(len(line) - 1):
            u = self.NB.index(line[i])
            v = self.NB.index(line[i + 1])
            total_length += self.C[u, v]
    return total_length



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
            #individual = random.sample(self.NB, k=self.L)
            #population.append(individual)
            shuffled = self.NB.copy()

            random.shuffle(shuffled)
            line1 = shuffled[:self.L]

            random.shuffle(shuffled)
            line2 = shuffled[:self.L]

            population.append([line1, line2])

        ## Individu de prova: dues línies idèntiques (mateixa ruta)
        # → Serveix per verificar si la penalització de solapament funciona
        #line = self.NB[:self.L]  # Ex: [101, 102, 103, 104]
        #individual = [line, line.copy()]  # Dues línies iguals

        #for _ in range(self.n_population):
        #    population.append(copy.deepcopy(individual))

        return population

    def print_population(self):
        for i, individual in enumerate(self.population):
            print(f"Individu {i+1}: {individual}")
    
    # 2.1. Afegir penalització pel fitness (solapament)
    def penalitzacio_solapament(self, line1, line2, alpha=50):
        """
        Penalitza si dues línies comparteixen nodes (parades).

        Args:
            line1, line2 (list[int]): dues línies de bus
            alpha (float): força de la penalització

        Returns:
            float: valor a restar al fitness (penalització positiva)
        """
        n_nodes = len(set(line1).union(set(line2)))
        n_shared = len(set(line1).intersection(set(line2)))
        ratio = n_shared / n_nodes  # percentatge de solapament

        penalitzacio = alpha * ratio
        print(f"  -> Solapament: {n_shared}/{n_nodes} nodes ({ratio:.2f}) -> Penalització: {penalitzacio:.2f}")
        return alpha * ratio

    def penalitzacio_longitud(self, individual, beta=0.5):
        """
        Penalitza individus amb línies massa llargues.

        Args:
            individual (list[list[int]]): dues línies de bus
            beta (float): paràmetre de pes per penalitzar la longitud

        Returns:
            float: valor positiu a sumar com a penalització
        """
        length = self.total_route_length(individual)
        penal = beta * length
        print(f"  -> Longitud total: {length:.2f} → Penalització longitud: {penal:.2f}")
        return penal

    def total_route_length(self, individual):
        """
        Calcula la llargada total (física) d’un individu (dues línies de bus)
        basada en la suma dels costos entre parades consecutives.

        Args:
            individual (list[list[int]]): [line1, line2]

        Returns:
            float: Llargada total de les dues línies
        """
        total = 0
        for line in individual:
            for i in range(len(line) - 1):
                try:
                    i_idx = self.NB.index(line[i])
                    j_idx = self.NB.index(line[i+1])
                    cost = self.C[i_idx, j_idx]
                except ValueError:
                    cost = np.inf
                total += cost
        return total

    def bonus_nodes_alts(self, individual, gamma=30):
        """
        Dona un bonus si les línies passen per parades amb alta demanda.

        Args:
            individual (list[list[int]]): Dues línies de bus.
            gamma (float): Valor de pes del bonus.

        Returns:
            float: Valor negatiu (bonus) a restar a l'energia.
        """
        high_demand_nodes = set()

        # 1. Detectem nodes d'alta demanda (per exemple, top 5% dels valors OD totals per node)
        node_scores = {node: 0 for node in self.NB}
        for i, ni in enumerate(self.NB):
            node_scores[ni] = sum(self.OD[i, :]) + sum(self.OD[:, i])  # entrada + sortida

        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = max(1, len(sorted_nodes) // 20)  # top 5%
        high_demand_nodes = {node for node, _ in sorted_nodes[:top_k]}

        # 2. Comptem quants nodes d'alta demanda apareixen
        all_stops = set(individual[0] + individual[1])
        n_hits = len(high_demand_nodes.intersection(all_stops))
        bonus = -gamma * n_hits  # negatiu perquè és un "bonus"

        print(f"  -> {n_hits} nodes d’alta demanda coberts → Bonus: {bonus:.2f}")
        return bonus


    # 2.2 --- FITNESS FUNCTION ---
    def fitness(self, individual, lambda_):

        """
        Avalua la qualitat d'una línia de bus representada per una seqüència de parades.

        L’energia de la línia es calculas com:
        E = −(1 − λ) · p_e + λ · c_e

        Args:
        individual (list[int]): Seqüència ordenada de parades que formen una línia de bus.
        lambda_ (float): Paràmetre de pes entre cost i servei (0 ≤ λ ≤ 1).

        Returns:
            float: Valor de fitness (com més alt, millor)
        """

        line1, line2 = individual

        G1 = nx.Graph()
        for i in range(len(line1) - 1):
            G1.add_edge(line1[i], line1[i + 1])

        G2 = nx.Graph()
        for i in range(len(line2) - 1):
            G2.add_edge(line2[i], line2[i + 1])

        # Mapa de quantes línies passen per cada aresta
        edge_count = Counter()
        for u, v in G1.edges():
            edge_count[frozenset((u, v))] += 1
        for u, v in G2.edges():
            edge_count[frozenset((u, v))] += 1

        total_pe = 0
        total_ce = 0

        # Crear graf combinat per rutes OD
        G_combined = nx.Graph()
        G_combined.add_edges_from(G1.edges())
        G_combined.add_edges_from(G2.edges())

        #G_line = nx.Graph()
        
        # 1. Construir el graf de la línia
        #for i in range(len(individual) - 1):
        #    u, v = individual[i], individual[i+1]
        #    G_line.add_edge(u, v)

        # 2. Per cada aresta, calculem pe i ce
        for edge in G_combined.edges():
            u, v = edge
            try:
                i_idx = self.NB.index(u)
                j_idx = self.NB.index(v)
                c_e = self.C[i_idx, j_idx]
            except ValueError:
                c_e = np.inf  # try-catch per si alguna parada no està definida

            total_ce += c_e

            # p_e: suma de passatgers a l'aresta
            pe = 0
            for i in self.NB:
                for j in self.NB:
                    if i != j:
                        try:
                            path = nx.shortest_path(G_combined, source=i, target=j)
                            #edges_in_path = list(zip(path[:-1], path[1:]))
                            edges_in_path = [frozenset((path[k], path[k + 1])) for k in range(len(path) - 1)]
                            #if (u, v) in edges_in_path or (v, u) in edges_in_path:
                            if frozenset((u, v)) in edges_in_path:
                                i_idx = self.NB.index(i)
                                j_idx = self.NB.index(j)
                                demand = self.OD[i_idx, j_idx]
                                #pe += self.OD[i_idx, j_idx] / len(path)
                                n_lines = edge_count[frozenset((u, v))]
                                pe += demand / len(path) / n_lines
                        except:
                            continue
            total_pe += pe

        # 3. Fórmula de l'energia (H)
        energy = -(1 - lambda_) * total_pe + lambda_ * total_ce

        penalitzacio = self.penalitzacio_solapament(line1, line2)
        penal_long = self.penalitzacio_longitud(individual, beta=0.1)
        bonus_demand = self.bonus_nodes_alts(individual, gamma=10)

        energy += penalitzacio
        energy += penal_long
        energy += bonus_demand

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
        def ox(line1, line2):
            size = len(line1)
            start, end = sorted(random.sample(range(size), 2))
        
            # 1. Segment del parent1
            child = [None] * size
            child[start:end+1] = line1[start:end+1]
        
            # 2. Completar amb parent2 (evitant duplicats)
            p2_index = 0
            for i in range(size):
                if child[i] is not None:
                    continue
                # Busquem un valor de parent2 que no estigui al child
                while p2_index < size and line2[p2_index] in child:
                    p2_index += 1
                if p2_index < size:
                    child[i] = line2[p2_index]
            assert len(set(child)) == len(child), "Error: la línia resultant té parades duplicades"
            return child
            
        c1_line1 = ox(parent1[0], parent2[0])
        c1_line2 = ox(parent1[1], parent2[1])
        return [c1_line1, c1_line2]

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

        def mutate_line(line):
            line = copy.deepcopy(line)
            if random.random() < prob:
                i, j = random.sample(range(len(line)), 2)
                line[i], line[j] = line[j], line[i]
            return line

        mutated1 = mutate_line(individual[0])
        mutated2 = mutate_line(individual[1])
        return [mutated1, mutated2]

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
    #L = 4
    n_population = 30
    seed = 42
    lambda_ = 0.5   # Valor de compromís entre servei i cost
    n_generations = 50

    # Carreguem nodes i posicions
    nodes_df = pd.read_csv("../../data/road_network/nodes.csv")
    pos = {int(row["node"]): (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    # Carreguem les parades de bus
    bus_stops_df = pd.read_csv("../../data/bus_network/nodes.csv")
    #NB = list(bus_stops_df["node"])
    import random
    NB = random.sample(list(bus_stops_df["node"]), 20)

    # Generem la matriu OD basada en la densitat i la desem al fitxer
    D, node_to_bus_index = OD_matrix(seed, NB, pos, d, save_files=True)

    # Debug: veure quins valors té la matriu OD cap al centre
    central_node = NB[len(NB)//2]  # aproximació
    print(f"\nDemanda cap a la parada central ({central_node}):")
    for i, origen in enumerate(NB):
        print(f"De {origen} → {central_node}: D = {D[i, NB.index(central_node)]}")

    edges_df = pd.read_csv("../../data/road_network/edges.csv")     # Carreguem les arestes del graf viàri
    nodes_df = pd.read_csv("../../data/road_network/nodes.csv")     # Carreguem nodes i posicions
    pos = {int(row["node"]): (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    # per guardar el seguiment de GA
    log_path = "../../results/log_GA.txt"
    with open(log_path, "w") as log_file:
        log_file.write("Inici execució GA\n")
        log_file.write(f"Paràmetres: L={L}, n_population={n_population}, lambda={lambda_}, generations={n_generations}\n\n")

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
    #debug_log_path = "../../results/log_GA_detailed.txt"
    optimizer = GeneticOptimizer(NB, L, n_population, D, C, seed)

    print("Població seleccionada (NB):")
    optimizer.print_population()

    #all_stops = [stop for individual in optimizer.population for stop in individual]
    all_stops = [stop for individual in optimizer.population for line in individual for stop in line]
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

    best_individual = max(optimizer.population, key=lambda ind: optimizer.fitness(ind, lambda_))

    with open("../../results/best_line_GA.txt", "w") as f:
        #f.write(",".join(map(str, best_individual)))
        f.write("LINE1: " + ",".join(map(str, best_individual[0])) + "\n")
        f.write("LINE2: " + ",".join(map(str, best_individual[1])) + "\n")
