from __future__ import annotations

import argparse
import os
from math import comb
from typing import Any, Dict, Optional

from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.visualization.ga_snapshot_plotter import save_final_line_with_od
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.core.network_model import load_network_model
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.config.od_scenarios import resolve_od_case_path


def run_ga_experiment(
    od_case: str = "base",
    map_case: str = "base",
    output_dir: Optional[str] = None,
    lambda_: float = 0.5,
) -> Dict[str, Any]:
    results_dir = output_dir or get_road_experiment_results_dir(str(SCRIPT_DIR), od_case, map_case)
    snapshot_dir = os.path.join(results_dir, "ga_improvements")
    os.makedirs(results_dir, exist_ok=True)

    road_nodes_path, road_edges_path = resolve_map_case_paths(str(PROJECT_ROOT), map_case, verbose=True)
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)

    print("Carregant model de xarxa...")
    print("Cas OD =", od_case)
    print("Cas mapa =", map_case)
    print("Fitxer nodes =", road_nodes_path)
    print("Fitxer edges =", road_edges_path)
    print("Fitxer OD =", od_matrix_path)
    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )

    print("Preparant funcio objectiu...")
    objective = ObjectiveFunction(network)
    objective.precompute_edge_costs()
    objective.precompute_edge_service()

    generations = 100
    population_size = 150
    line_length = 6
    seed = 42

    print("\nConfiguracio:")
    print("lambda =", lambda_)
    print("generations =", generations)
    print("population_size =", population_size)
    print("line_length =", line_length)
    print("seed =", seed)

    ga = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=line_length,
        population_size=population_size,
        seed=seed,
        init_ratio_service=0.30,
        init_ratio_demand=0.25,
        init_ratio_spatial=0.15,
        init_ratio_hybrid=0.15,
        init_ratio_random=0.15,
        inject_best_service_candidate=True,
        nearest_neighbors_k=8,
        init_top_k=8,
    )

    print("\nCalculant normalitzacio estructural...")
    normalization_bounds = prepare_structural_normalization(
        objective=objective,
        line_length=line_length,
    )
    print("normalization_method =", DEFAULT_NORMALIZATION_METHOD)
    print("service_min =", normalization_bounds.service_min)
    print("service_max =", normalization_bounds.service_max)
    print("service_max_type =", "structural_upper_bound_top_od_pairs")
    print("service_pair_count =", comb(line_length, 2))
    print("cost_min =", normalization_bounds.cost_min)
    print("cost_max =", normalization_bounds.cost_max)
    print("cost_min_type =", "structural_lower_bound_shortest_stop_transitions")
    print("cost_max_type =", "structural_upper_bound_shortest_stop_transitions")
    print("cost_transition_count =", line_length - 1)
    cost_range = normalization_bounds.cost_max - normalization_bounds.cost_min
    service_range = normalization_bounds.service_max - normalization_bounds.service_min
    print("cost_range =", cost_range)
    print("service_range =", service_range)

    if normalization_bounds.cost_max == normalization_bounds.cost_min:
        print("[WARNING] Structural cost normalization range is zero.")
    if normalization_bounds.service_max == normalization_bounds.service_min:
        print("[WARNING] Structural passenger-service normalization range is zero.")

    print("\nExecutant GA...")
    best_individual, best_score = ga.run(
        lambda_=lambda_,
        generations=generations,
        n_elite=2,
        tournament_size=3,
        mutation_prob=0.30,
        weight_adjacent=0.20,
        weight_reverse=0.15,
        weight_neighbor=0.25,
        weight_service_replace=0.40,
        snapshot_output_dir=snapshot_dir,
    )

    best_eval = objective.evaluate(best_individual, lambda_)
    repair_diagnostic = ga.compare_repaired_order(best_individual)
    best_service_candidate = list(ga.best_service_candidate_ordered)
    best_service_candidate_eval = objective.evaluate(best_service_candidate, lambda_)
    summary_path = os.path.join(results_dir, "ga_summary.txt")
    final_figure_path = os.path.join(results_dir, "final_line_with_od.pdf")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"od_case: {od_case}\n")
        file.write(f"map_case: {map_case}\n")
        file.write(f"od_matrix_path: {od_matrix_path}\n")
        file.write(f"road_nodes_path: {road_nodes_path}\n")
        file.write(f"road_edges_path: {road_edges_path}\n")
        file.write(f"lambda: {lambda_}\n")
        file.write(f"normalization_method: {DEFAULT_NORMALIZATION_METHOD}\n")
        file.write(f"normalization_cost_min: {normalization_bounds.cost_min}\n")
        file.write(f"normalization_cost_max: {normalization_bounds.cost_max}\n")
        file.write("normalization_cost_min_type: structural_lower_bound_shortest_stop_transitions\n")
        file.write("normalization_cost_max_type: structural_upper_bound_shortest_stop_transitions\n")
        file.write(f"normalization_cost_transition_count: {line_length - 1}\n")
        file.write(f"normalization_service_min: {normalization_bounds.service_min}\n")
        file.write(f"normalization_service_max: {normalization_bounds.service_max}\n")
        file.write("normalization_service_max_type: structural_upper_bound_top_od_pairs\n")
        file.write(f"normalization_service_pair_count: {comb(line_length, 2)}\n")
        file.write(f"init_ratio_service: {ga.init_ratio_service}\n")
        file.write(f"init_ratio_demand: {ga.init_ratio_demand}\n")
        file.write(f"init_ratio_spatial: {ga.init_ratio_spatial}\n")
        file.write(f"init_ratio_hybrid: {ga.init_ratio_hybrid}\n")
        file.write(f"init_ratio_random: {ga.init_ratio_random}\n")
        file.write(f"inject_best_service_candidate: {ga.inject_best_service_candidate}\n")
        file.write("mutation_prob: 0.3\n")
        file.write("weight_service_replace: 0.4\n")
        file.write(f"best_individual: {best_individual}\n")
        file.write(f"best_individual_repaired: {repair_diagnostic['repaired']}\n")
        file.write(f"best_individual_order_cost: {repair_diagnostic['original_order_cost']}\n")
        file.write(f"best_individual_repaired_order_cost: {repair_diagnostic['repaired_order_cost']}\n")
        file.write(f"best_individual_order_improvement: {repair_diagnostic['improvement']}\n")
        file.write(f"best_service_candidate: {ga.best_service_candidate}\n")
        file.write(f"best_service_candidate_ordered: {best_service_candidate}\n")
        file.write(
            "best_service_candidate_passenger_service: "
            f"{ga.best_service_candidate_passenger_service}\n"
        )
        file.write(f"best_service_candidate_cost: {best_service_candidate_eval.cost}\n")
        file.write(
            "best_service_candidate_fitness_at_lambda: "
            f"{best_service_candidate_eval.fitness}\n"
        )
        file.write(f"best_fitness: {best_score}\n")
        file.write(f"cost: {best_eval.cost}\n")
        file.write(f"cost_norm: {best_eval.cost_norm}\n")
        file.write(f"passenger_service: {best_eval.passenger_service}\n")
        file.write(f"passenger_service_norm: {best_eval.passenger_service_norm}\n")

    save_final_line_with_od(
        network=network,
        individual=best_individual,
        evaluation=best_eval,
        output_path=final_figure_path,
        title_prefix=f"Linia final amb OD ({od_case})",
    )

    print("\n=== RESULTAT FINAL ===")
    print("Millor individu:", best_individual)
    print("Diagnostic ordre original:", repair_diagnostic["original"])
    print("Cost ordre original:", repair_diagnostic["original_order_cost"])
    print("Diagnostic ordre reparat:", repair_diagnostic["repaired"])
    print("Cost ordre reparat:", repair_diagnostic["repaired_order_cost"])
    print("Millora ordre:", repair_diagnostic["improvement"])
    print("Best service candidate:", best_service_candidate)
    print("Best service candidate passenger service:", ga.best_service_candidate_passenger_service)
    print("Best service candidate cost:", best_service_candidate_eval.cost)
    print("Best service candidate fitness:", best_service_candidate_eval.fitness)
    print("Fitness:", best_score)
    print("Cost:", best_eval.cost)
    print("Cost norm:", best_eval.cost_norm)
    print("Passenger service:", best_eval.passenger_service)
    print("Passenger service norm:", best_eval.passenger_service_norm)
    print("Nombre d'arestes linia:", len(best_eval.line_edges))
    print("Arestes linia:", best_eval.line_edges)
    print("Resum guardat a:", summary_path)
    print("Figura final guardada a:", final_figure_path)

    return {
        "od_case": od_case,
        "map_case": map_case,
        "od_matrix_path": od_matrix_path,
        "road_nodes_path": road_nodes_path,
        "road_edges_path": road_edges_path,
        "best_individual": best_individual,
        "best_score": best_score,
        "best_evaluation": best_eval,
        "best_service_candidate": best_service_candidate,
        "best_service_candidate_evaluation": best_service_candidate_eval,
        "results_dir": results_dir,
        "snapshot_dir": snapshot_dir,
        "summary_path": summary_path,
        "final_figure_path": final_figure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a executar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viaria")
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.1,
        help="Pes de cost a la funcio objectiu",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de resultats per a aquest cas",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ga_experiment(
        od_case=args.od_case,
        map_case=args.map_case,
        output_dir=args.output_dir,
        lambda_=args.lambda_,
    )


if __name__ == "__main__":
    main()

