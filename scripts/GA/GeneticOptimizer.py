import numpy as np
import networkx as nx
import pandas as pd

from collections import Counter

from od_generation import OD_matrix
from tests.seed.seed_GA import seed_GA


#################################################
#       Classe principal Genetic Algorithm      #
#################################################
class GeneticOptimizer:
    # 1. --- INITIAL POPULATION ---
    def __init__(self, NB, L, n_population, OD, C, seed=24, debug_log=None, rng=None):
        self.NB = NB
        self.L = L
        self.n_population = n_population
        self.OD = OD
        self.C = C
        self.seed = seed
        self.debug_log = debug_log
        
        self.rng = rng if rng is not None else seed_GA(seed)    # si no ens passem, creem amb la seed del GA
        self.node_to_idx = {n: i for i, n in enumerate(self.NB)}
        self.population = self._generate_initial_population()
        self._build_nearest(k=5)


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
        return [self.rng.sample(self.NB, k=self.L) for _ in range(self.n_population)]

    def print_population(self):
        for i, individual in enumerate(self.population):
            print(f"Individu {i+1}: {individual}")

    # Mapa de veïns més propers (per cost C) de cada parada
    def _build_nearest(self, k=5):
        self._nearest = {}
        for i, n in enumerate(self.NB):
            order = np.argsort(self.C[i])
            # Evitem el mateix node i ens quedem amb els k més propers
            #self._nearest[n] = [self.NB[j] for j in order if j != i][:k]
            self._nearest[n] = [self.NB[j] for j in order if j != i and np.isfinite(self.C[i, j])][:k]

    # 2. --- FITNESS FUNCTION ---
    def fitness(self, individual, lambda_):

        """
        Avalua la qualitat d'una línia de bus representada per una seqüència de parades.

        L’energia de la línia es calcula com:
        E = (1 − λ) · p_e - λ · c_e

        - servei (p_e): demanda OD entre parades de la línia, repartida per arestes del camí
                        amb 1 / l_ij (on l_ij = # d'arestes entre i i j dins la línia → per línia simple: j - i).
        - cost (c_e): suma de costos C entre parades consecutives de la línia.

        Args:
        individual (list[int]): Seqüència ordenada de parades que formen una línia de bus.
        lambda_ (float): Paràmetre de pes entre cost i servei (0 ≤ λ ≤ 1).

        Returns:
            float: Valor de fitness (com més alt, millor)
        """
        # Índexs a NB per a cada parada de la línia (evita NB.index)
        idx = [self.node_to_idx[n] for n in individual]
        L = len(idx)

        # 1) Cost: suma de costos entre consecutives
        total_ce = 0.0
        for a in range(L - 1):
            total_ce += self.C[idx[a], idx[a+1]]

        # 2) Servei: repartim cada OD(i,j) a les arestes a..j-1 amb pes 1/(j-i)
        #    equivalent a la definició de pe del teu model per una línia (nL(e)=1).
        edge_pe = [0.0] * (L - 1)

        for i in range(L):
            for j in range(i + 1, L):        # <- només parelles i<j
                #dij = self.OD[idx[i], idx[j]]
                dij = self.OD[idx[i]][idx[j]]

                if dij <= 0.0:
                    continue
                share = dij / (j - i)  # repartiment 1/l_ij
                #k = i
                #while k < j:
                #    edge_pe[k] += share
                #    k += 1
                #edge_pe[i:j] += dij / (j - i)   # repartiment 1/l_ij
                for k in range(i, j):
                    edge_pe[k] += share

        total_pe = sum(edge_pe)
        return (1.0 - lambda_) * total_pe - lambda_ * total_ce # és el fitness que volem maximitzar


    # 3. --- SELECTION ---
    def select_parents_tournament(self, zipped_pop_fits, k):

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

        for _ in range(len(zipped_pop_fits)):
            # Escollim k individus aleatoris per competir
            candidates = self.rng.sample(zipped_pop_fits, k)

            # Triem el millor d'aquests (fitness més alt)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def select_elite(self, zipped_pop_fits, n_elite):

        """
        Selecciona els millors individus de la població segons el seu fitness (elitisme).

        Ordena tota la població per valor de fitness (de millor a pitjor) i retorna els millors.
        D'aquesta manera assegura que les millors solucions es preservin en la següent generació sense modificar-se.

        Args:
            zipped_pop_fits: llista de tuples (individu, fitness).
            n_elite (int): Nombre d’individus d’elit a conservar.

        Returns:
            list[list[int]]: Llista amb els millors individus de la població.
        """
        sorted_pairs = sorted(zipped_pop_fits, key=lambda x: x[1], reverse=True)
        return [ind for ind, _ in sorted_pairs[:n_elite]]


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
        start, end = sorted(self.rng.sample(range(size), 2))
        
        # 1. Segment del parent1
        child = [None] * size
        child[start:end+1] = parent1[start:end+1] # es copia el segment del primer fill
        
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
    def mutate(self, individual, prob, p_adj, p_2opt):

        """
        Aplica una mutació aleatòria sobre un individu amb una certa probabilitat (20% -> s'ha d'afinar)

        El mètode de mutació és l’intercanvi de dues parades aleatòries dins la línia.
        Aquesta operació manté la validesa de l’individu, canvia l’ordre. Ajuda a evitar l’estancament evolutiu.

        Args:
            individual (list[int]): Línia de bus (seqüència de parades) que serà mutada.
            prob (float, optional): Probabilitat d'aplicar la mutació. Per defecte 0.2 (20%).

        Returns:
            list[int]: Nova línia (mutada o igual a l’original si no hi ha hagut mutació).


        NOU:
        Mutació híbrida centrada en moviments locals:
            - p_adj: intercanvi de parades adjacents (suau, només reordena)
            - p_2opt: 2-opt curt (inverteix un segment petit → suavitza ziga-zagues)
            - p_neighbor: reemplaça una parada per un veí proper (canvi local del node)
            
            NOTA: p_adj + p_2opt + p_neighbor = 1.0
        """
        # si individu < prob toca mutar. Cas contrari es retorna la línia de bus actual
        if self.rng.random() >= prob:
            return individual

        # procés de mutació:
        child = list(individual) # còpia
        L = len(child)
        random_aux = self.rng.random() # s'obté un valor aleatòri nou

        s = p_adj + p_2opt + max(0.0, 1.0 - (p_adj + p_2opt))
        p_adj, p_2opt = p_adj/s, p_2opt/s

        # tractament amb random_aux amb distribució de probabilitat condicional, són mutuament excloents (suma de probabilitat ha de sumar 1)

        # 1) Adjacent-swap: intercanvia dues parades contigües (p_adj)
        if random_aux < p_adj and L >= 2:
            i = self.rng.randrange(L - 1)
            child[i], child[i+1] = child[i+1], child[i]
            return child

        # 2) 2-opt curt: inverteix un trosset.
        if random_aux < p_adj + p_2opt and L >= 3: # p_neighbor = p_adj + p_2opt
            # s'escullen dos punts aleatòris on a < b
            a, b = sorted(self.rng.sample(range(L), 2))

            # es limita la llargada del segment per evitar el zig-zag
            if b - a > 3: b = a + 3
            child[a:b+1] = reversed(child[a:b+1]) # inverteix el tram dels dos punts escollits
            return child

        # 3) Reemplaç per veí proper (local i sense duplicats)
        pos = self.rng.randrange(L)
        old = child[pos]
        present = set(child)
        cand = [c for c in self._nearest.get(old, []) if c not in present] # per consultar la llista de veïns més propers del node

        # si hi ha nodes propers
        if cand:
            # nodes veïns a la seqüència (si existeixen)
            prevn = child[pos-1] if pos > 0 else None
            nextn = child[pos+1] if pos < L-1 else None

            # indices (evitem lookups repetits)
            prev_idx = self.node_to_idx[prevn] if prevn is not None else None
            next_idx = self.node_to_idx[nextn] if nextn is not None else None

            def two_edge_cost(x):
                xi = self.node_to_idx[x]
                c = 0.0
                if prev_idx is not None:
                    c += self.C[prev_idx, xi]
                if next_idx is not None:
                    c += self.C[xi, next_idx]
                return c

            base = two_edge_cost(old)

            # ordena candidats pel seu cost local (menor = millor)
            cand.sort(key=two_edge_cost)

            # per no ser massa determinista, tria entre els 3 millors
            top = cand[:min(3, len(cand))]
            pick = self.rng.choice(top)

            # ACCEPTA només si no empitjora (pots relaxar amb un epsilon)
            if two_edge_cost(pick) <= base:        # o <= base * 1.02 amb 2% de tolerància
                child[pos] = pick
                return child
            # si empitjora, cau al fallback de sota

            # el 70% escull el més proper al node
            # el 30% es repartirà entre les altres parades
            #new = cand[0] if self.rng.random() < 0.7 else self.rng.choice(cand[:min(3, len(cand))])
            #child[pos] = new
            #return child

        # si no hi ha nodes propers o la probabilitat no suma 1: Fallback -> s'aplica adjacent-swap
        if L >= 2:
            i = pos if pos < L-1 else pos-1
            child[i], child[i+1] = child[i+1], child[i]
        return child


    # ho adjuntem tot
    def evolve_one_generation(self, lambda_, n_elite=1, k=10): # provar els valors 5 o 10 || els k=10 es podria posar al main

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
        # 0. Fitness de tota la població (pre-compute)
        fits = [self.fitness(ind, lambda_) for ind in self.population]
        zipped = list(zip(self.population, fits))

        # 1. Elitisme
        elite = self.select_elite(zipped, n_elite)

        # 2. Selecció de pares via torneig
        parents = self.select_parents_tournament(zipped, k)

        # 3. Generar fills amb crossover OX + mutació
        children = []
        need = self.n_population - n_elite

        while len(children) < need:
            p1, p2 = self.rng.sample(parents, 2)
            child = self.crossover_OX(p1, p2)
            child = self.mutate(child, prob=0.2, p_adj=0.5, p_2opt=0.45) # la probabilitat es podria posar al main
            children.append(child)

        # 4. Nova població = elit + fills
        self.population = elite + children

if __name__ == "__main__":

    # Paràmetres fixats
    d = 20
    shift = 0.2
    L = 8           # longitud línia de bus - nombre total de parades que ha de tenir cada individu
    n_population = 50
    # seed = 42
    seed = 999
    lambda_ = 0.8   # Valor de compromís entre servei i cost
    n_generations = 100 # amb 1000 iteracions dona el mateix resultat

    ## variables a posar aqui:
        # prob, k

    # Carreguem nodes i posicions
    nodes_df = pd.read_csv("../../data/road_network/nodes.csv")
    pos = {int(row["node"]): (row["x"], row["y"]) for _, row in nodes_df.iterrows()}

    # Carreguem les parades de bus
    bus_stops_df = pd.read_csv("../../data/bus_network/nodes.csv")
    NB = list(bus_stops_df["node"])

    # Generem la matriu OD basada en la densitat i la desem al fitxer
    D, node_to_bus_index = OD_matrix(seed, NB, pos, d, save_files=True)

    # Debug: veure quins valors té la matriu OD cap al centre
    #central_node = NB[len(NB)//2]  # aproximació
    #print(f"\nDemanda cap a la parada central ({central_node}):")
    #for i, origen in enumerate(NB):
    #    print(f"De {origen} → {central_node}: D = {D[i, NB.index(central_node)]}")



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
    debug_log_path = "../../results/log_GA_detailed.txt"
    optimizer = GeneticOptimizer(NB, L, n_population, D, C, seed, debug_log=debug_log_path)

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

    best_individual = max(optimizer.population, key=lambda ind: optimizer.fitness(ind, lambda_))

    with open("../../results/best_line_GA.txt", "w") as f:
        f.write(",".join(map(str, best_individual)))