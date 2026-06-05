from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, RESULTS_DIR
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.network_model import Node, load_network_model
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import EvaluationResult, ObjectiveFunction


def apply_tfg_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.8,
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
        }
    )


def configure_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Generació", fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=12.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.55")
    ax.spines["bottom"].set_color("0.55")
    ax.tick_params(axis="both", colors="0.25", labelsize=10.5)


def get_project_paths() -> Tuple[str, str, str, str, str]:
    results_dir = os.path.join(str(RESULTS_DIR), "thesis_evolution_plots")
    os.makedirs(results_dir, exist_ok=True)
    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        "base",
        verbose=False,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), "base")

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
    apply_tfg_plot_style()
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
        init_ratio_service=0.0,
        init_ratio_demand=0.0,
        init_ratio_spatial=0.0,
        init_ratio_hybrid=0.0,
        init_ratio_random=1.0,
        inject_best_service_candidate=False,
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
    print("initialization_mode =", "100% random")

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
        color="#7A0019",
        linewidth=1.6,
        label=r"$H$",
    )

    configure_axis(ax, r"Millor valor de $H$")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
    )
    fig.tight_layout()

    out_pdf = os.path.join(results_dir, "evolution_H_lambda_05.pdf")
    out_png = os.path.join(results_dir, "evolution_H_lambda_05.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)

    if final_best_evaluation is not None:
        print("\n=== RESUM EVOLUCIO H ===")
        print("Generacions:", generations)
        print("Millor individu final:", final_best_individual)
        print("Millor fitness final:", final_best_evaluation.fitness)
        print("Figures guardades a:")
        print(" -", out_pdf)
        print(" -", out_png)


if __name__ == "__main__":
    main()

