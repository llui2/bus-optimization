# script/line_by_line_GA.py

from GA_one_line import GeneticAlgorithmOneLine
from DataHandler_SA import DataHandler

# -----------------------------
# PARÀMETRES D'ENTRENAMENT
# -----------------------------

seed = 456
d = 20
L = 8
T_0 = 3  # No s’utilitza en GA, però per comparar millor ho mantenim
lambdaa = 8
max_iter = 1000  # El considerem com a nombre de generacions
generations = 50
pop_size = 20

# -----------------------------
# EXECUCIÓ AMB ALGORISME GENÈTIC
# -----------------------------

ga = GeneticAlgorithmOneLine(seed=seed, d=d, L=L, pop_size=pop_size)
ga.run(generations=generations, lambdaa=lambdaa)
result = ga.get_result()

# -----------------------------
# CÀLCUL I EXPORTACIÓ DE MÈTRIQUES
# -----------------------------

dh = DataHandler(sheet_name="GA")  # 📌 Guardem a la fulla “GA”
final_energy, avg_travel_distance, population_total, central_coverage_pct, converged_iter = dh.compute_metrics(
    line_nodes=result["line"],
    obj=[result["energy"]],
    pos=result["positions"],
    node_to_bus_index=result["node_to_bus_index"],
    D=result["D"],
    G=result["network"],
    d=d
)

metrics = dh.prepare_metrics_dict(
    final_energy=final_energy,
    avg_travel_distance=avg_travel_distance,
    population_total=population_total,
    central_coverage_pct=central_coverage_pct,
    converged_iter=converged_iter,
    seed=seed,
    L=L,
    T_0=T_0,
    lambdaa=lambdaa,
    max_iter=max_iter
)

dh.export_to_excel(metrics)

print("Execució amb GA completa. Mètriques registrades a la fulla 'GA'.")
