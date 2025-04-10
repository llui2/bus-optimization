# script/line_by_line.py

from SA_one_line import SimulatedAnnealing
from DataHandler import DataHandler

# -----------------------------
# PARÀMETRES D'ENTRENAMENT
# -----------------------------

seed = 123
d = 20
L = 8 # longitud
T_0 = 3
lambdaa = 8
max_iter = 1000

# -----------------------------
# ENTRENAMENT AMB SIMULATED ANNEALING
# -----------------------------

sa = SimulatedAnnealing(seed=seed, d=d, L=L)
sa.run(max_iter=max_iter, T_0=T_0, lambdaa=lambdaa)
result = sa.get_result()

# -----------------------------
# CÀLCUL I EXPORTACIÓ DE MÈTRIQUES
# -----------------------------

dh = DataHandler()
final_energy, avg_travel_distance, population_total, central_coverage_pct, converged_iter = dh.compute_metrics(
    line_nodes=result["line"],
    obj=[result["energy"]],
    pos=result["positions"],
    node_to_bus_index=result["node_to_bus_index"],
    D=result["D"],
    G=result["network"],
    d=d
)
# script/line_by_line.py

from SA_one_line import SimulatedAnnealing
from DataHandler import DataHandler

# -----------------------------
# PARÀMETRES D'ENTRENAMENT
# -----------------------------

# Define parameters as a dictionary for better organization and readability
params = {
    "seed": 123,
    "d": 20,
    "L": 8,
    "T_0": 3,
    "lambdaa": 8,
    "max_iter": 1000
}

# -----------------------------
# ENTRENAMENT AMB SIMULATED ANNEALING
# -----------------------------

# Initialize SimulatedAnnealing with the defined parameters
sa = SimulatedAnnealing(**params)
sa.run(max_iter=params["max_iter"], T_0=params["T_0"], lambdaa=params["lambdaa"])
result = sa.get_result()

# -----------------------------
# CÀLCUL I EXPORTACIÓ DE MÈTRIQUES
# -----------------------------

# Initialize DataHandler
dh = DataHandler()

# Compute metrics using the result from SimulatedAnnealing
final_energy, avg_travel_distance, population_total, central_coverage_pct, converged_iter = dh.compute_metrics(
    line_nodes=result["line"],
    obj=[result["energy"]],
    pos=result["positions"],
    node_to_bus_index=result["node_to_bus_index"],
    D=result["D"],
    G=result["network"],
    d=params["d"]
)

# Prepare metrics dictionary for export
metrics = dh.prepare_metrics_dict(
    final_energy=final_energy,
    avg_travel_distance=avg_travel_distance,
    population_total=population_total,
    central_coverage_pct=central_coverage_pct,
    converged_iter=converged_iter,
    **params  # Unpack parameters dictionary
)

# Export metrics to Excel
dh.export_to_excel(metrics)

# Print completion message
print("Execució completa. Mètriques registrades a l’Excel.\n")
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

print("Execució completa. Mètriques registrades a l’Excel.\n")