from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional

from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.core.network_model import load_network_model
from busline_ga.core.sequential_multi_line_optimizer import SequentialMultiLineOptimizer
from busline_ga.visualization.multiline_evolution_plotter import save_multiline_evolution_outputs


def run_multiline_ga_experiment(
    od_case: str = "base",
    map_case: str = "base",
    n_lines: int = 2,
    line_length: int = 6,
    population_size: int = 150,
    generations: int = 100,
    lambda_: float = 0.5,
    seed: int = 42,
    shared_stop_penalty: float = 0.0,
    output_dir: Optional[str] = None,
    save_improvement_snapshots: bool = False,
    snapshot_output_dir: Optional[str] = None,
    save_evolution_plots: bool = False,
) -> Dict[str, Any]:
    if n_lines < 1:
        raise ValueError("n_lines must be >= 1")
    if line_length < 2:
        raise ValueError("line_length must be >= 2")

    base_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), od_case, map_case)
    results_dir = output_dir or os.path.join(base_results_dir, f"multiline_{n_lines}_lines")
    improvement_snapshot_dir = snapshot_output_dir or os.path.join(results_dir, "multiline_improvements")
    evolution_output_dir = os.path.join(results_dir, "multiline_evolution")
    os.makedirs(results_dir, exist_ok=True)
    if save_improvement_snapshots:
        os.makedirs(improvement_snapshot_dir, exist_ok=True)
    if save_evolution_plots:
        os.makedirs(evolution_output_dir, exist_ok=True)

    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        map_case,
        verbose=True,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)

    print("\n=== Multi-line GA configuration ===")
    print(f"od_case: {od_case}")
    print(f"map_case: {map_case}")
    print(f"line_length: {line_length}")
    print(f"n_lines: {n_lines}")
    print(f"population_size: {population_size}")
    print(f"generations: {generations}")
    print(f"lambda: {lambda_}")
    print(f"seed: {seed}")
    print(f"shared_stop_penalty: {shared_stop_penalty}")
    print(f"save_improvement_snapshots: {save_improvement_snapshots}")
    print(f"save_evolution_plots: {save_evolution_plots}")
    if save_improvement_snapshots:
        print(f"snapshot_output_dir: {improvement_snapshot_dir}")
    if save_evolution_plots:
        print(f"evolution_output_dir: {evolution_output_dir}")
    print(f"output_dir: {results_dir}")
    print(f"road_nodes_path: {road_nodes_path}")
    print(f"road_edges_path: {road_edges_path}")
    print(f"od_matrix_path: {od_matrix_path}")

    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )
    optimizer = SequentialMultiLineOptimizer(
        network=network,
        line_length=line_length,
        population_size=population_size,
        n_lines=n_lines,
        lambda_=lambda_,
        generations=generations,
        seed=seed,
        shared_stop_penalty=shared_stop_penalty,
        save_improvement_snapshots=save_improvement_snapshots,
        snapshot_output_dir=improvement_snapshot_dir,
        save_evolution_history=save_evolution_plots,
        evolution_output_dir=evolution_output_dir,
        od_case=od_case,
        map_case=map_case,
    )
    run_result = optimizer.run()
    evolution_output_paths: Dict[str, str] = {}

    if save_evolution_plots:
        evolution_output_paths = save_multiline_evolution_outputs(
            evolution_history=run_result["evolution_history"],
            system_evolution_history=run_result["system_evolution_history"],
            output_dir=evolution_output_dir,
            title_context={
                "od_case": od_case,
                "map_case": map_case,
                "n_lines": n_lines,
                "lambda": lambda_,
                "shared_stop_penalty": shared_stop_penalty,
            },
        )

    summary_path = os.path.join(results_dir, "multiline_ga_summary.txt")
    config_path = os.path.join(results_dir, "multiline_config.txt")
    system_metrics_path = os.path.join(results_dir, "multiline_system_metrics.csv")
    lines_path = os.path.join(results_dir, "multiline_lines.csv")
    adjusted_line_metrics_path = os.path.join(results_dir, "multiline_adjusted_line_metrics.csv")
    pairwise_path = os.path.join(results_dir, "multiline_pairwise_overlap.csv")
    improvements_summary_path = os.path.join(improvement_snapshot_dir, "improvements_summary.csv")

    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(f"od_case={od_case}\n")
        config_file.write(f"map_case={map_case}\n")
        config_file.write(f"line_length={line_length}\n")
        config_file.write(f"n_lines={n_lines}\n")
        config_file.write(f"population_size={population_size}\n")
        config_file.write(f"generations={generations}\n")
        config_file.write(f"lambda={lambda_}\n")
        config_file.write(f"seed={seed}\n")
        config_file.write(f"shared_stop_penalty={shared_stop_penalty}\n")
        config_file.write(f"save_improvement_snapshots={save_improvement_snapshots}\n")
        config_file.write(f"save_evolution_plots={save_evolution_plots}\n")
        if save_improvement_snapshots:
            config_file.write(f"snapshot_output_dir={improvement_snapshot_dir}\n")
        if save_evolution_plots:
            config_file.write(f"evolution_output_dir={evolution_output_dir}\n")
        config_file.write(f"road_nodes_path={road_nodes_path}\n")
        config_file.write(f"road_edges_path={road_edges_path}\n")
        config_file.write(f"od_matrix_path={od_matrix_path}\n")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("=== CONFIGURATION ===\n")
        file.write(f"od_case: {od_case}\n")
        file.write(f"map_case: {map_case}\n")
        file.write(f"line_length: {line_length}\n")
        file.write(f"n_lines: {n_lines}\n")
        file.write(f"population_size: {population_size}\n")
        file.write(f"generations: {generations}\n")
        file.write(f"lambda: {lambda_}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"shared_stop_penalty: {shared_stop_penalty}\n")
        file.write(f"save_improvement_snapshots: {save_improvement_snapshots}\n")
        file.write(f"save_evolution_plots: {save_evolution_plots}\n")
        if save_improvement_snapshots:
            file.write(f"snapshot_output_dir: {improvement_snapshot_dir}\n")
        if save_evolution_plots:
            file.write(f"evolution_output_dir: {evolution_output_dir}\n")
        file.write(f"output_dir: {results_dir}\n")
        file.write(f"road_nodes_path: {road_nodes_path}\n")
        file.write(f"road_edges_path: {road_edges_path}\n")
        file.write(f"od_matrix_path: {od_matrix_path}\n")

        file.write("\n=== NORMALIZATION BOUNDS ===\n")
        file.write(f"normalization_cost_min: {run_result['normalization_bounds'].cost_min}\n")
        file.write(f"normalization_cost_max: {run_result['normalization_bounds'].cost_max}\n")
        file.write(f"normalization_service_min: {run_result['normalization_bounds'].service_min}\n")
        file.write(f"normalization_service_max: {run_result['normalization_bounds'].service_max}\n")

        file.write("\n=== PER-LINE RESULTS ===\n")
        for index, evaluation in enumerate(run_result["evaluations"], start=1):
            line = run_result["lines"][index - 1]
            file.write(f"line_{index}: {line}\n")
            file.write(f"line_{index}_cost: {evaluation.cost}\n")
            file.write(f"line_{index}_passenger_service: {evaluation.passenger_service}\n")
            file.write(f"line_{index}_cost_norm: {evaluation.cost_norm}\n")
            file.write(f"line_{index}_service_norm: {evaluation.service_norm}\n")
            file.write(f"line_{index}_fitness: {evaluation.fitness}\n")

        file.write("\n=== FINAL ADJUSTED PER-LINE METRICS ===\n")
        file.write(
            "The original per-line fitness is the value obtained when the line was optimized sequentially.\n"
        )
        file.write(
            "The adjusted final fitness is recomputed after all lines are selected, "
            "so shared links/edges affect all lines, including previously fixed ones.\n"
        )
        for entry in run_result["adjusted_line_metrics"]:
            line_index = entry["line_index"]
            file.write(f"line_{line_index}_original_fitness: {entry['original_fitness']}\n")
            file.write(f"line_{line_index}_adjusted_final_fitness: {entry['adjusted_fitness']}\n")
            file.write(
                f"line_{line_index}_original_passenger_service: "
                f"{entry['original_passenger_service']}\n"
            )
            file.write(
                f"line_{line_index}_adjusted_passenger_service: "
                f"{entry['adjusted_passenger_service']}\n"
            )
            file.write(
                f"line_{line_index}_shared_edge_ratio: "
                f"{entry['shared_edge_ratio_for_line']}\n"
            )

        file.write("\n=== GLOBAL SYSTEM METRICS ===\n")
        for key, value in run_result["system_metrics"].items():
            file.write(f"{key}: {value}\n")

        file.write("\n=== PAIRWISE OVERLAP ===\n")
        for entry in run_result["pairwise_overlap"]:
            file.write(
                f"line_{entry['line_i']}_line_{entry['line_j']}_shared_stops: {entry['shared_stops']}\n"
            )
            file.write(
                f"line_{entry['line_i']}_line_{entry['line_j']}_shared_edges: {entry['shared_edges']}\n"
            )

        if save_evolution_plots:
            file.write("\n=== EVOLUTION OUTPUTS ===\n")
            file.write(
                f"multiline_evolution_history: "
                f"{evolution_output_paths.get('multiline_evolution_history_csv', '')}\n"
            )
            file.write(
                f"multiline_system_evolution_history: "
                f"{evolution_output_paths.get('multiline_system_evolution_history_csv', '')}\n"
            )
            file.write(
                f"evolution_H_per_line_pdf: "
                f"{evolution_output_paths.get('evolution_H_per_line_pdf', '')}\n"
            )
            file.write(
                f"evolution_Pnorm_Cnorm_per_line_pdf: "
                f"{evolution_output_paths.get('evolution_Pnorm_Cnorm_per_line_pdf', '')}\n"
            )
            file.write(
                f"evolution_system_fitness_by_line_pdf: "
                f"{evolution_output_paths.get('evolution_system_fitness_by_line_pdf', '')}\n"
            )
            file.write(
                f"evolution_system_PC_by_line_pdf: "
                f"{evolution_output_paths.get('evolution_system_PC_by_line_pdf', '')}\n"
            )

        file.write("\n=== OUTPUT FILES ===\n")
        file.write(f"multiline_config: {config_path}\n")
        file.write(f"multiline_system_metrics: {system_metrics_path}\n")
        file.write(f"multiline_lines: {lines_path}\n")
        file.write(f"multiline_adjusted_line_metrics: {adjusted_line_metrics_path}\n")
        file.write(f"multiline_pairwise_overlap: {pairwise_path}\n")
        if save_improvement_snapshots:
            file.write(f"improvements_summary: {improvements_summary_path}\n")
        if save_evolution_plots:
            for path_name, path_value in sorted(evolution_output_paths.items()):
                file.write(f"{path_name}: {path_value}\n")

    with open(system_metrics_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "od_case",
                "map_case",
                "n_lines",
                "line_length",
                "population_size",
                "generations",
                "lambda",
                "seed",
                "shared_stop_penalty",
                *sorted(run_result["system_metrics"].keys()),
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "od_case": od_case,
                "map_case": map_case,
                "n_lines": n_lines,
                "line_length": line_length,
                "population_size": population_size,
                "generations": generations,
                "lambda": lambda_,
                "seed": seed,
                "shared_stop_penalty": shared_stop_penalty,
                **run_result["system_metrics"],
            }
        )

    with open(lines_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "line_index",
            "stops",
            "cost",
            "passenger_service",
            "cost_norm",
            "service_norm",
            "fitness",
        ])
        for index, evaluation in enumerate(run_result["evaluations"], start=1):
            line = run_result["lines"][index - 1]
            writer.writerow([
                index,
                ";".join(str(stop) for stop in line),
                evaluation.cost,
                evaluation.passenger_service,
                evaluation.cost_norm,
                evaluation.service_norm,
                evaluation.fitness,
            ])

    # Original metrics are the sequential optimization values. Adjusted metrics are
    # recomputed after all lines are known, so shared edges affect every line.
    with open(adjusted_line_metrics_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "line_index",
            "stops",
            "original_fitness",
            "original_passenger_service",
            "adjusted_passenger_service",
            "cost",
            "cost_norm",
            "adjusted_service_norm",
            "adjusted_fitness",
            "shared_edge_count_for_line",
            "shared_edge_ratio_for_line",
        ])
        for entry in run_result["adjusted_line_metrics"]:
            line = run_result["lines"][entry["line_index"] - 1]
            writer.writerow([
                entry["line_index"],
                ";".join(str(stop) for stop in line),
                entry["original_fitness"],
                entry["original_passenger_service"],
                entry["adjusted_passenger_service"],
                entry["cost"],
                entry["cost_norm"],
                entry["adjusted_service_norm"],
                entry["adjusted_fitness"],
                entry["shared_edge_count_for_line"],
                entry["shared_edge_ratio_for_line"],
            ])

    with open(pairwise_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["line_i", "line_j", "shared_stops", "shared_edges"])
        for entry in run_result["pairwise_overlap"]:
            writer.writerow(
                [
                    entry["line_i"],
                    entry["line_j"],
                    entry["shared_stops"],
                    entry["shared_edges"],
                ]
            )

    if save_improvement_snapshots:
        with open(improvements_summary_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "line_index",
                    "generation",
                    "fitness",
                    "cost",
                    "passenger_service",
                    "cost_norm",
                    "service_norm",
                    "pdf_path",
                ],
            )
            writer.writeheader()
            for snapshot in run_result["improvement_snapshots"]:
                writer.writerow(snapshot)

    print("\n=== RESULTAT MULTI-LINIA ===")
    for index, evaluation in enumerate(run_result["evaluations"], start=1):
        line = run_result["lines"][index - 1]
        print(f"Line {index}: {line}")
        print(
            f"  cost={evaluation.cost} "
            f"passenger_service={evaluation.passenger_service} "
            f"cost_norm={evaluation.cost_norm} "
            f"service_norm={evaluation.service_norm} "
            f"fitness={evaluation.fitness}"
        )

    print("\n=== GLOBAL SYSTEM RESULT ===")
    for name, value in run_result["system_metrics"].items():
        print(f"{name} = {value}")

    print("\nSummary saved to:", summary_path)
    print("Config saved to:", config_path)
    print("System metrics saved to:", system_metrics_path)
    print("Line metrics saved to:", lines_path)
    print("Adjusted line metrics saved to:", adjusted_line_metrics_path)
    print("Pairwise overlap saved to:", pairwise_path)
    if save_improvement_snapshots:
        print("Improvement snapshots saved to:", improvement_snapshot_dir)
        print("Improvement summary saved to:", improvements_summary_path)
    if save_evolution_plots:
        print("Evolution outputs saved to:", evolution_output_dir)

    result = {
        "results_dir": results_dir,
        "summary_path": summary_path,
        "config_path": config_path,
        "system_metrics_path": system_metrics_path,
        "lines_path": lines_path,
        "adjusted_line_metrics_path": adjusted_line_metrics_path,
        "pairwise_path": pairwise_path,
        "improvements_summary_path": improvements_summary_path if save_improvement_snapshots else None,
        "improvement_snapshot_dir": improvement_snapshot_dir if save_improvement_snapshots else None,
        "evolution_output_dir": evolution_output_dir if save_evolution_plots else None,
        "evolution_output_paths": evolution_output_paths,
        "network": network,
        **run_result,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a executar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viaria")
    parser.add_argument(
        "--n-lines",
        "--lines",
        dest="n_lines",
        type=int,
        default=2,
        help="Nombre de linies a optimitzar",
    )
    parser.add_argument("--line-length", type=int, default=6, help="Parades per linia")
    parser.add_argument("--population-size", type=int, default=150, help="Mida de poblacio")
    parser.add_argument("--generations", type=int, default=100, help="Nombre de generacions")
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.5,
        help="Pes de cost a la funcio objectiu",
    )
    parser.add_argument("--seed", type=int, default=42, help="Llavor del GA")
    parser.add_argument(
        "--shared-stop-penalty",
        type=float,
        default=0.0,
        help="Optional fitness penalty for reusing stops from previous fixed lines.",
    )
    parser.add_argument("--output-dir", default=None, help="Carpeta de resultats")
    parser.add_argument(
        "--save-improvement-snapshots",
        action="store_true",
        help="Save a multi-line PDF snapshot whenever a line optimization improves.",
    )
    parser.add_argument(
        "--save-evolution-plots",
        "--save-evolution-history",
        dest="save_evolution_plots",
        action="store_true",
        help="Save multi-line GA evolution CSV files and plots.",
    )
    args = parser.parse_args()

    if args.n_lines < 1:
        parser.error("--n-lines / --lines must be >= 1")
    if args.line_length < 2:
        parser.error("--line-length must be >= 2")

    return args


def main() -> None:
    args = parse_args()
    run_multiline_ga_experiment(
        od_case=args.od_case,
        map_case=args.map_case,
        n_lines=args.n_lines,
        line_length=args.line_length,
        population_size=args.population_size,
        generations=args.generations,
        lambda_=args.lambda_,
        seed=args.seed,
        shared_stop_penalty=args.shared_stop_penalty,
        output_dir=args.output_dir,
        save_improvement_snapshots=args.save_improvement_snapshots,
        save_evolution_plots=args.save_evolution_plots,
    )


if __name__ == "__main__":
    main()
