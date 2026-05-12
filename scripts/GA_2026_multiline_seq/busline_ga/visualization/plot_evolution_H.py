from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.network_model import Node, load_network_model
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import EvaluationResult, ObjectiveFunction


def get_project_paths() -> Tuple[str, str, str, str, str]:
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(base_dir))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    road_nodes_path = os.path.join(project_dir, "data", "road_network", "nodes.csv")
    road_edges_path = os.path.join(project_dir, "data", "road_network", "edges.csv")
    bus_stops_path = os.path.join(project_dir, "data", "bus_network", "nodes.csv")
    od_matrix_path = os.path.join(project_dir, "data", "bus_network", "od_matrix_fixed.csv")

    return road_nodes_path, road_edges_path, bus_stops_path, od_matrix_path, results_dir


def record_generation_best(
    ga: GeneticOptimizer,
    lambda_: float,
) -> Tuple[List[Node], EvaluationResult]:
    scored_population = ga.evaluate_population(lambda_)
    best_individual, best_evaluation = max(
        scored_population,
        key=lambda item: item[1].fitness,
    )
    result = list(best_individual), best_evaluation
    return result


def main() -> None:
    (
        road_nodes_path,
        road_edges_path,
        bus_stops_path,
        od_matrix_path,
        results_dir,
    ) = get_project_paths()

    print("Carregant dades...")
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

    lambda_ = 0.5
    generations = 50
    population_size = 50
    line_length = 6
    seed = 42

    ga = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=line_length,
        population_size=population_size,
        seed=seed,
    )

    print("Calculant normalitzacio estructural...")
    normalization_bounds = prepare_structural_normalization(
        objective=objective,
        line_length=line_length,
    )
    print("normalization_method =", DEFAULT_NORMALIZATION_METHOD)
    print("service_min =", normalization_bounds.service_min)
    print("service_max =", normalization_bounds.service_max)
    print("cost_min =", normalization_bounds.cost_min)
    print("cost_max =", normalization_bounds.cost_max)
    print("cost_range =", normalization_bounds.cost_max - normalization_bounds.cost_min)
    print("service_range =", normalization_bounds.service_max - normalization_bounds.service_min)

    ga.generate_initial_population()

    generation_indices: List[int] = []
    best_fitness_values: List[float] = []

    final_best_individual: List[Node] = []
    final_best_evaluation: Optional[EvaluationResult] = None

    for generation in range(generations + 1):
        best_individual, best_evaluation = record_generation_best(ga, lambda_)
        generation_indices.append(generation)
        best_fitness_values.append(best_evaluation.fitness)

        final_best_individual = best_individual
        final_best_evaluation = best_evaluation

        if generation < generations:
            ga.evolve_one_generation(
                lambda_=lambda_,
                n_elite=2,
                tournament_size=3,
                mutation_prob=0.2,
            )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        generation_indices,
        best_fitness_values,
        color="#2E6F40",
        linewidth=2.2,
        label=r"$H(k)$",
    )

    ax.set_title("Evolucio del valor objectiu H (lambda = 0.5)")
    ax.set_xlabel("Generacio")
    ax.set_ylabel("Valor de H")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=True)
    fig.tight_layout()

    out_pdf = os.path.join(results_dir, "evolution_H_lambda_05.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    if final_best_evaluation is not None:
        print("\n=== RESUM EVOLUCIO H ===")
        print("Generacions:", generations)
        print("Millor individu final:", final_best_individual)
        print("Millor fitness final:", final_best_evaluation.fitness)
        print("Figura PDF guardada a:")
        print(" -", out_pdf)


if __name__ == "__main__":
    main()

