from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.core.network_model import Node, load_network_model

from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.config.od_scenarios import resolve_od_case_path


ODPair = Tuple[Node, Node, float]


def jittered_polyline(
    u: Tuple[float, float],
    v: Tuple[float, float],
    rng: random.Random,
    amp: float = 0.08,
    kinks: int = 2,
) -> List[Tuple[float, float]]:
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
    result = pts
    return result


def normalize_edge(u: Node, v: Node) -> Tuple[Node, Node]:
    result = (u, v) if u <= v else (v, u)
    return result


def collect_od_pairs(network) -> List[ODPair]:
    od_pairs: List[ODPair] = []
    bus_stops = network.bus_stops

    for i in range(len(bus_stops)):
        for j in range(i + 1, len(bus_stops)):
            stop_i = bus_stops[i]
            stop_j = bus_stops[j]
            demand = float(network.get_od_value(stop_i, stop_j))

            if demand > 0.0:
                od_pairs.append((stop_i, stop_j, demand))

    result = od_pairs
    return result


def compute_density_center(network) -> Tuple[float, float]:
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    for stop in network.bus_stops:
        stop_weight = 0.0
        for other_stop in network.bus_stops:
            if other_stop == stop:
                continue
            stop_weight += float(network.get_od_value(stop, other_stop))
            stop_weight += float(network.get_od_value(other_stop, stop))

        x, y = network.positions[stop]
        weighted_x += x * stop_weight
        weighted_y += y * stop_weight
        total_weight += stop_weight

    if total_weight > 0.0:
        center = (weighted_x / total_weight, weighted_y / total_weight)
    else:
        x_values = [network.positions[stop][0] for stop in network.bus_stops]
        y_values = [network.positions[stop][1] for stop in network.bus_stops]
        center = (float(np.mean(x_values)), float(np.mean(y_values)))

    return center


def demand_style(
    demand: float,
    low_threshold: float,
    high_threshold: float,
) -> Tuple[float, float]:
    alpha = 0.08
    linewidth = 0.7

    if demand >= high_threshold:
        alpha = 0.35
        linewidth = 3.0
    elif demand >= low_threshold:
        alpha = 0.18
        linewidth = 1.7

    result = alpha, linewidth
    return result


def draw_density_center(ax, center: Tuple[float, float], color: str) -> None:
    cx, cy = center

    ax.scatter(
        [cx], [cy],
        s=12000,
        color=color,
        alpha=0.05,
        edgecolors="none",
        zorder=2.2,
    )
    ax.scatter(
        [cx], [cy],
        s=4500,
        color=color,
        alpha=0.08,
        edgecolors="none",
        zorder=2.25,
    )
    ax.scatter(
        [cx], [cy],
        s=180,
        color=color,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        zorder=2.3,
    )


def draw_od_overlay(
    ax,
    network,
    od_pairs: List[ODPair],
    color: str,
) -> None:
    if not od_pairs:
        return

    demand_values = [demand for _, _, demand in od_pairs]
    low_threshold = float(np.quantile(demand_values, 0.33))
    high_threshold = float(np.quantile(demand_values, 0.66))

    sorted_pairs = sorted(od_pairs, key=lambda x: x[2])

    for stop_i, stop_j, demand in sorted_pairs:
        x1, y1 = network.positions[stop_i]
        x2, y2 = network.positions[stop_j]
        alpha, linewidth = demand_style(demand, low_threshold, high_threshold)

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=2.6,
        )

    legend_elements = [
        Line2D([0], [0], color=color, lw=0.9, alpha=0.10, label="Demanda baixa"),
        Line2D([0], [0], color=color, lw=1.8, alpha=0.20, label="Demanda mitjana"),
        Line2D([0], [0], color=color, lw=3.0, alpha=0.35, label="Demanda alta"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a visualitzar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viÃ ria")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), args.od_case, args.map_case)
    results_dir = os.path.join(case_results_dir, "line_visualization_od")
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
        od_matrix_path,
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
        seed=42,
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
        mutation_prob=0.2,
    )

    best_eval = objective.evaluate(best_individual, lambda_)
    line_edges = [normalize_edge(u, v) for (u, v) in best_eval.line_edges]
    od_pairs = collect_od_pairs(network)
    density_center = compute_density_center(network)

    pos = network.positions
    rng = random.Random(42)

    granate = "#7A0019"
    gris_carretera = "0.78"
    gris_nodes = "0.86"
    color_linia = "#7A0019"
    color_od = "#B22222"
    color_selected_stop = "#FFD23F"

    fig, ax = plt.subplots(figsize=(8, 8))

    edge_paths: Dict[Tuple[Node, Node], List[Tuple[float, float]]] = {}
    for u, v in network.graph.edges():
        edge = normalize_edge(u, v)
        if edge not in edge_paths:
            edge_paths[edge] = jittered_polyline(pos[u], pos[v], rng, amp=0.06, kinks=2)

    for u, v in network.graph.edges():
        edge = normalize_edge(u, v)
        pts = edge_paths[edge]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=gris_carretera, linewidth=1.0, alpha=0.70, zorder=1)

    x_all = [pos[n][0] for n in network.graph.nodes()]
    y_all = [pos[n][1] for n in network.graph.nodes()]
    ax.scatter(
        x_all,
        y_all,
        s=8,
        color=gris_nodes,
        edgecolors="none",
        zorder=1.5,
    )

    draw_density_center(ax, density_center, color_od)
    draw_od_overlay(ax, network, od_pairs, color_od)

    xb = [pos[n][0] for n in network.bus_stops]
    yb = [pos[n][1] for n in network.bus_stops]
    ax.scatter(
        xb,
        yb,
        s=38,
        color=granate,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.62,
        zorder=3.0,
    )

    for u, v in line_edges:
        pts = edge_paths[normalize_edge(u, v)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color_linia, linewidth=2.0, alpha=0.98, zorder=4.0)

    line_bus_stops = list(best_individual)
    xl = [pos[n][0] for n in line_bus_stops]
    yl = [pos[n][1] for n in line_bus_stops]
    ax.scatter(
        xl,
        yl,
        s=135,
        color=color_selected_stop,
        edgecolors=color_linia,
        linewidths=1.45,
        zorder=6.5,
    )

    for stop in line_bus_stops:
        x, y = pos[stop]
        ax.text(
            x + 0.10,
            y + 0.10,
            str(stop),
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
            zorder=7.0,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color_linia,
                "linewidth": 0.45,
                "alpha": 0.82,
            },
        )

    ax.set_title("LÃ­nia de bus optimitzada amb demanda OD i centre de densitat (GA)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.margins(0.03)

    out_pdf = os.path.join(results_dir, "line_ga_od_overlay.pdf")

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nMillor lÃ­nia:", best_individual)
    print("Fitness:", best_score)
    print("Cost:", best_eval.cost)
    print("Passenger service:", best_eval.passenger_service)
    print("Cost norm:", best_eval.cost_norm)
    print("Passenger service norm:", best_eval.passenger_service_norm)
    print("Centre de densitat:", density_center)
    print("Figura PDF guardada a:")
    print(" -", out_pdf)


if __name__ == "__main__":
    main()

