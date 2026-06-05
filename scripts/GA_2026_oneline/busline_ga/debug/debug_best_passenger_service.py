from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.visualization.ga_snapshot_plotter import (
    build_edge_paths,
    draw_base_network,
    draw_od_overlay,
    normalize_edge,
)
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.core.network_model import NetworkModel, Node, load_network_model
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.config.od_scenarios import resolve_od_case_path


@dataclass
class SearchResult:
    stop_set: List[Node]
    ordered_line: List[Node]
    passenger_service: float
    route_cost: float
    method: str


def compute_passenger_service_for_set(
    network: NetworkModel,
    stops: Sequence[Node],
) -> float:
    total_service = 0.0

    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            total_service += float(network.get_od_value(stops[i], stops[j]))

    return total_service


def incremental_gain(
    network: NetworkModel,
    current_stops: Sequence[Node],
    candidate_stop: Node,
) -> float:
    gain = 0.0

    for stop in current_stops:
        gain += float(network.get_od_value(stop, candidate_stop))

    return gain


def greedy_completion(
    network: NetworkModel,
    bus_stops: Sequence[Node],
    line_length: int,
    start_stops: Sequence[Node],
) -> List[Node]:
    selected = list(dict.fromkeys(start_stops))
    remaining = [stop for stop in bus_stops if stop not in selected]

    while len(selected) < line_length and remaining:
        best_stop = remaining[0]
        best_gain = incremental_gain(network, selected, best_stop)

        for candidate_stop in remaining[1:]:
            candidate_gain = incremental_gain(network, selected, candidate_stop)
            if candidate_gain > best_gain or (
                math.isclose(candidate_gain, best_gain) and candidate_stop < best_stop
            ):
                best_stop = candidate_stop
                best_gain = candidate_gain

        selected.append(best_stop)
        remaining.remove(best_stop)

    return selected


def improve_by_swaps(
    network: NetworkModel,
    bus_stops: Sequence[Node],
    initial_stops: Sequence[Node],
) -> List[Node]:
    current = list(initial_stops)
    improved = True

    while improved:
        improved = False
        current_service = compute_passenger_service_for_set(network, current)
        outside = [stop for stop in bus_stops if stop not in current]
        best_candidate = list(current)
        best_service = current_service

        for remove_index, removed_stop in enumerate(current):
            candidate_base = [stop for stop in current if stop != removed_stop or current.count(stop) > 1]
            if len(candidate_base) != len(current) - 1:
                candidate_base = current[:remove_index] + current[remove_index + 1 :]

            for added_stop in outside:
                candidate = list(candidate_base) + [added_stop]
                candidate_service = compute_passenger_service_for_set(network, candidate)

                if candidate_service > best_service:
                    best_candidate = candidate
                    best_service = candidate_service

        if best_service > current_service:
            current = best_candidate
            improved = True

    return current


def search_best_passenger_service_set(
    network: NetworkModel,
    line_length: int,
    search_seed: int = 2026,
    exact_threshold: int = 2_000_000,
    random_starts: int = 300,
) -> Tuple[List[Node], str]:
    bus_stops = list(network.bus_stops)
    combination_count = math.comb(len(bus_stops), line_length)
    best_set: List[Node] = []
    best_service = float("-inf")
    method = "random_greedy_local_search"

    if combination_count <= exact_threshold:
        method = "exact_combinations"
        for candidate_tuple in combinations(bus_stops, line_length):
            candidate = list(candidate_tuple)
            candidate_service = compute_passenger_service_for_set(network, candidate)
            if candidate_service > best_service:
                best_set = candidate
                best_service = candidate_service
    else:
        search_rng = random.Random(search_seed)
        top_demand_stops = sorted(
            bus_stops,
            key=lambda stop: sum(
                float(network.get_od_value(stop, other_stop))
                + float(network.get_od_value(other_stop, stop))
                for other_stop in bus_stops
                if other_stop != stop
            ),
            reverse=True,
        )
        seed_candidates: List[List[Node]] = []

        for stop in top_demand_stops[: min(12, len(top_demand_stops))]:
            seed_candidates.append(greedy_completion(network, bus_stops, line_length, [stop]))

        if len(top_demand_stops) >= 2:
            for stop_a, stop_b in combinations(top_demand_stops[: min(10, len(top_demand_stops))], 2):
                seed_candidates.append(
                    greedy_completion(network, bus_stops, line_length, [stop_a, stop_b])
                )

        for _ in range(random_starts):
            seed_size = min(2, line_length)
            random_seed_stops = search_rng.sample(bus_stops, k=seed_size)
            seed_candidates.append(
                greedy_completion(network, bus_stops, line_length, random_seed_stops)
            )

        for seed_candidate in seed_candidates:
            improved_candidate = improve_by_swaps(network, bus_stops, seed_candidate)
            candidate_service = compute_passenger_service_for_set(network, improved_candidate)
            if candidate_service > best_service:
                best_set = improved_candidate
                best_service = candidate_service

    return best_set, method


def visualize_service_only_line(
    network: NetworkModel,
    ordered_line: List[Node],
    passenger_service: float,
    route_cost: float,
    output_path: str,
    title_prefix: str,
) -> None:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    od_pairs = [
        (stop_i, stop_j, float(network.get_od_value(stop_i, stop_j)))
        for i, stop_i in enumerate(network.bus_stops)
        for stop_j in network.bus_stops[i + 1 :]
        if float(network.get_od_value(stop_i, stop_j)) > 0.0
    ]
    line_data = build_line_from_stops(network, ordered_line)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_base_network(ax, network, edge_paths)
    draw_od_overlay(ax, network, od_pairs, "#B22222")

    line_edges = [normalize_edge(u, v) for (u, v) in line_data["line_edges"]]
    for u, v in line_edges:
        points = edge_paths[normalize_edge(u, v)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="#7A0019", linewidth=2.0, alpha=0.98, zorder=4.0)

    xs_stops = [network.positions[stop][0] for stop in ordered_line]
    ys_stops = [network.positions[stop][1] for stop in ordered_line]
    ax.scatter(
        xs_stops,
        ys_stops,
        s=135,
        color="#FFD23F",
        edgecolors="#7A0019",
        linewidths=1.45,
        zorder=6.5,
    )

    for stop in ordered_line:
        x, y = network.positions[stop]
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
                "edgecolor": "#7A0019",
                "linewidth": 0.45,
                "alpha": 0.82,
            },
        )

    ax.set_title(
        f"{title_prefix}\n"
        f"passenger_service={passenger_service:.6f} | route_cost={route_cost:.6f}"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_case(
    od_case: str,
    map_case: str,
    line_length: int,
    search_seed: int,
) -> SearchResult:
    road_nodes_path, road_edges_path = resolve_map_case_paths(str(PROJECT_ROOT), map_case, verbose=False)
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)
    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )
    objective = ObjectiveFunction(network)
    objective.precompute_edge_costs()
    ga = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=line_length,
        population_size=10,
        seed=42,
    )

    best_set, method = search_best_passenger_service_set(
        network=network,
        line_length=line_length,
        search_seed=search_seed,
    )
    passenger_service = compute_passenger_service_for_set(network, best_set)
    ordered_line = ga.repair_order_min_cost(best_set)
    line_data = build_line_from_stops(network, ordered_line)
    route_cost = objective.compute_line_cost(line_data["line_edges"])
    case_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), od_case, map_case)
    output_path = os.path.join(case_results_dir, "debug_best_passenger_service.pdf")
    visualize_service_only_line(
        network=network,
        ordered_line=ordered_line,
        passenger_service=passenger_service,
        route_cost=route_cost,
        output_path=output_path,
        title_prefix=f"Service-only best line ({od_case}, {map_case})",
    )

    result = SearchResult(
        stop_set=list(best_set),
        ordered_line=ordered_line,
        passenger_service=passenger_service,
        route_cost=route_cost,
        method=method,
    )
    print(f"\n=== {map_case.upper()} ===")
    print("search_method =", method)
    print("best_stop_set =", result.stop_set)
    print("ordered_line =", result.ordered_line)
    print("passenger_service =", result.passenger_service)
    print("route_cost =", result.route_cost)
    print("pdf =", output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a analitzar")
    parser.add_argument("--map-case", default="dense_fill", help="Cas de mapa a analitzar")
    parser.add_argument("--line-length", type=int, default=6, help="Nombre de parades seleccionades")
    parser.add_argument("--search-seed", type=int, default=2026, help="Seed del diagnostic service-only")
    args = parser.parse_args()

    od_case = args.od_case
    line_length = args.line_length
    search_seed = args.search_seed
    map_case = args.map_case

    print("Debug service-only search")
    print("od_case =", od_case)
    print("map_case =", map_case)
    print("line_length =", line_length)
    print("search_seed =", search_seed)

    run_case(
        od_case=od_case,
        map_case=map_case,
        line_length=line_length,
        search_seed=search_seed,
    )


if __name__ == "__main__":
    main()

