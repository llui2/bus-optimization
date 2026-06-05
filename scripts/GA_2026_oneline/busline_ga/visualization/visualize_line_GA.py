from __future__ import annotations

import argparse
import os
import random
import matplotlib.pyplot as plt

from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.core.network_model import load_network_model
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.config.od_scenarios import resolve_od_case_path


def jittered_polyline(u, v, rng, amp=0.08, kinks=2):
    x1, y1 = u
    x2, y2 = v

    pts = [(x1, y1)]
    for t in range(1, kinks + 1):
        alpha = t / (kinks + 1)

        xb = x1 + alpha * (x2 - x1)
        yb = y1 + alpha * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        px, py = -dy, dx

        j_perp = rng.uniform(-amp, amp)
        j_free_x = rng.uniform(-amp, amp) * 0.25
        j_free_y = rng.uniform(-amp, amp) * 0.25

        xj = xb + px * j_perp + j_free_x
        yj = yb + py * j_perp + j_free_y
        pts.append((xj, yj))

    pts.append((x2, y2))
    return pts


def normalize_edge(u, v):
    return (u, v) if u <= v else (v, u)


def aggregate_stop_demand(network) -> dict[int, float]:
    demand_by_stop: dict[int, float] = {}

    for stop in network.bus_stops:
        total_demand = 0.0
        for other_stop in network.bus_stops:
            if other_stop == stop:
                continue
            total_demand += float(network.get_od_value(stop, other_stop))
            total_demand += float(network.get_od_value(other_stop, stop))

        demand_by_stop[stop] = total_demand

    return demand_by_stop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a visualitzar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viÃ ria")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), args.od_case, args.map_case)
    results_dir = os.path.join(case_results_dir, "line_visualization")
    os.makedirs(results_dir, exist_ok=True)

    road_nodes_path, road_edges_path = resolve_map_case_paths(str(PROJECT_ROOT), args.map_case, verbose=True)
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), args.od_case)

    print("Carregant dades...")
    print("Cas OD =", args.od_case)
    print("Cas mapa =", args.map_case)
    network = load_network_model(
        road_nodes_path,
        road_edges_path,
        bus_stops_path,
        od_matrix_path
    )

    objective = ObjectiveFunction(network)
    objective.precompute_edge_costs()
    objective.precompute_edge_service()

    lambda_ = 0.5

    ga = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=6,
        population_size=50,
        seed=42
    )

    print("Calculant normalitzacio estructural...")
    normalization_bounds = prepare_structural_normalization(
        objective=objective,
        line_length=ga.line_length,
    )
    print("normalization_method =", DEFAULT_NORMALIZATION_METHOD)
    print("service_min =", normalization_bounds.service_min)
    print("service_max =", normalization_bounds.service_max)
    print("cost_min =", normalization_bounds.cost_min)
    print("cost_max =", normalization_bounds.cost_max)
    print("cost_range =", normalization_bounds.cost_max - normalization_bounds.cost_min)
    print("service_range =", normalization_bounds.service_max - normalization_bounds.service_min)

    print("Executant GA...")
    best_individual, best_score = ga.run(
        lambda_=lambda_,
        generations=100,
        n_elite=2,
        tournament_size=3,
        mutation_prob=0.2
    )

    best_eval = objective.evaluate(best_individual, lambda_)
    line_edges = [normalize_edge(u, v) for (u, v) in best_eval.line_edges]
    stop_demand = aggregate_stop_demand(network)

    # Nodes de la lÃ­nia
    route_nodes = set()
    for u, v in line_edges:
        route_nodes.add(u)
        route_nodes.add(v)

    line_bus_stops = [n for n in route_nodes if n in network.bus_stops]

    pos = network.positions
    rng = random.Random(42)

    granate = "#7A0019"
    gris_carretera = "0.75"
    gris_nodes = "0.85"
    color_linia = "#7A0019"

    fig, ax = plt.subplots(figsize=(8, 8))

    # Guardem traÃ§ats consistents per cada aresta
    edge_paths = {}
    for u, v in network.graph.edges():
        e = normalize_edge(u, v)
        if e not in edge_paths:
            edge_paths[e] = jittered_polyline(pos[u], pos[v], rng, amp=0.06, kinks=2)

    # 1) Xarxa base
    for u, v in network.graph.edges():
        e = normalize_edge(u, v)
        pts = edge_paths[e]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=gris_carretera, linewidth=1.0, alpha=0.75, zorder=1)

    # 2) Nodes generals molt subtils
    x_all = [pos[n][0] for n in network.graph.nodes()]
    y_all = [pos[n][1] for n in network.graph.nodes()]
    ax.scatter(
        x_all, y_all,
        s=8,
        color=gris_nodes,
        edgecolors="none",
        zorder=2
    )

    # 3) Parades de bus amb demanda agregada
    xb = [pos[n][0] for n in network.bus_stops]
    yb = [pos[n][1] for n in network.bus_stops]
    demand_values = [stop_demand[n] for n in network.bus_stops]
    min_demand = min(demand_values) if demand_values else 0.0
    max_demand = max(demand_values) if demand_values else 0.0
    size_values = []

    for demand_value in demand_values:
        normalized_demand = 0.0
        if max_demand > min_demand:
            normalized_demand = (demand_value - min_demand) / (max_demand - min_demand)

        size_values.append(30.0 + 95.0 * normalized_demand)

    demand_scatter = ax.scatter(
        xb, yb,
        s=size_values,
        c=demand_values,
        cmap="YlOrRd",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.82,
        zorder=3
    )
    colorbar = fig.colorbar(demand_scatter, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("Demanda agregada per parada")

    # 4) LÃ­nia optimitzada
    for u, v in line_edges:
        pts = edge_paths[normalize_edge(u, v)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color_linia, linewidth=1.8, alpha=0.98, zorder=4.5)

    # 5) Ressaltar nodes de la lÃ­nia
    line_bus_stops = list(best_individual)

    xl = [pos[n][0] for n in line_bus_stops]
    yl = [pos[n][1] for n in line_bus_stops]
    ax.scatter(
        xl, yl,
        s=88,
        color=color_linia,
        edgecolors="white",
        linewidths=1.0,
        zorder=5.5
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.margins(0.03)

    out_pdf = os.path.join(results_dir, "line_ga_styled.pdf")

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nMillor lÃ­nia:", best_individual)
    print("Fitness:", best_score)
    print("Cost:", best_eval.cost)
    print("Passenger service:", best_eval.passenger_service)
    print("Cost norm:", best_eval.cost_norm)
    print("Passenger service norm:", best_eval.passenger_service_norm)
    print("Figura PDF guardada a:")
    print(" -", out_pdf)


if __name__ == "__main__":
    main()

