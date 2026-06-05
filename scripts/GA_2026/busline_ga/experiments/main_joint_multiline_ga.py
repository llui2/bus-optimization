from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from typing import Any, Dict, Optional

from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.core.joint_multi_line_optimizer import JointMultiLineOptimizer
from busline_ga.core.network_model import load_network_model
from busline_ga.core.normalization_utils import prepare_structural_normalization
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.visualization.joint_multiline_evolution_plotter import save_joint_multiline_evolution_outputs
from busline_ga.visualization.visualize_joint_multiline_ga import save_joint_multiline_map_from_run_result


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _format_line(line) -> str:
    result = ";".join(str(stop) for stop in line)
    return result


def format_float_for_path(value: float) -> str:
    result = f"{value:.2f}".replace("-", "m").replace(".", "p")
    return result


def _sanitize_run_name(run_name: Optional[str]) -> str:
    result = ""

    if run_name is not None:
        result = re.sub(r"\s+", "_", run_name.strip())
        result = re.sub(r"[^A-Za-z0-9_.-]+", "_", result)
        result = result.strip("._-")[:40]

    return result


def make_joint_run_folder_name(
    n_lines: int,
    line_length: int,
    lambda_: float,
    population_size: int,
    generations: int,
    seed: int,
    mutation_prob: float,
    crossover_prob: float,
    compactness_penalty: float,
    max_segment_cost_soft: Optional[float] = None,
    max_line_cost_soft: Optional[float] = None,
    max_edges_per_line_soft: Optional[float] = None,
    run_name: Optional[str] = None,
) -> str:
    folder_name = (
        f"joint_l{n_lines}_len{line_length}_lam{format_float_for_path(lambda_)}_"
        f"pop{population_size}_gen{generations}_seed{seed}_"
        f"mut{format_float_for_path(mutation_prob)}_cross{format_float_for_path(crossover_prob)}_"
        f"comp{format_float_for_path(compactness_penalty)}"
    )

    if max_segment_cost_soft is not None:
        folder_name = f"{folder_name}_seg{format_float_for_path(max_segment_cost_soft)}"
    if max_line_cost_soft is not None:
        folder_name = f"{folder_name}_line{format_float_for_path(max_line_cost_soft)}"
    if max_edges_per_line_soft is not None:
        folder_name = f"{folder_name}_edge{format_float_for_path(max_edges_per_line_soft)}"

    safe_run_name = _sanitize_run_name(run_name)

    if safe_run_name:
        folder_name = f"{folder_name}_{safe_run_name}"

    result = folder_name
    return result


def _prepare_normalization_bounds(network, line_length):
    objective = ObjectiveFunction(network=network)
    bounds = prepare_structural_normalization(objective, line_length)
    return bounds


def run_joint_multiline_ga_experiment(
    od_case: str = "base",
    map_case: str = "base",
    n_lines: int = 2,
    line_length: int = 6,
    population_size: int = 150,
    generations: int = 100,
    lambda_: float = 0.5,
    seed: int = 42,
    mutation_prob: float = 0.20,
    crossover_prob: float = 0.80,
    compactness_penalty: float = 0.0,
    max_segment_cost_soft: Optional[float] = None,
    max_line_cost_soft: Optional[float] = None,
    max_edges_per_line_soft: Optional[float] = None,
    n_elite: int = 2,
    tournament_size: int = 3,
    save_improvement_snapshots: bool = False,
    save_evolution_plots: bool = False,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    if n_lines < 1:
        raise ValueError("n_lines must be >= 1")
    if line_length < 2:
        raise ValueError("line_length must be >= 2")

    base_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), od_case, map_case)
    if output_dir is None:
        run_folder_name = make_joint_run_folder_name(
            n_lines=n_lines,
            line_length=line_length,
            lambda_=lambda_,
            population_size=population_size,
            generations=generations,
            seed=seed,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            compactness_penalty=compactness_penalty,
            max_segment_cost_soft=max_segment_cost_soft,
            max_line_cost_soft=max_line_cost_soft,
            max_edges_per_line_soft=max_edges_per_line_soft,
            run_name=run_name,
        )
        results_dir = os.path.join(base_results_dir, run_folder_name)
    else:
        results_dir = output_dir
    improvements_dir = os.path.join(results_dir, "joint_multiline_improvements")
    evolution_dir = os.path.join(results_dir, "joint_multiline_evolution")
    os.makedirs(_filesystem_path(results_dir), exist_ok=True)

    if save_improvement_snapshots:
        os.makedirs(_filesystem_path(improvements_dir), exist_ok=True)
    if save_evolution_plots:
        os.makedirs(_filesystem_path(evolution_dir), exist_ok=True)

    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        map_case,
        verbose=True,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)

    print("\n=== Joint multi-line GA configuration ===")
    print(f"od_case: {od_case}")
    print(f"map_case: {map_case}")
    print(f"n_lines: {n_lines}")
    print(f"line_length: {line_length}")
    print(f"population_size: {population_size}")
    print(f"generations: {generations}")
    print(f"lambda: {lambda_}")
    print(f"seed: {seed}")
    print(f"mutation_prob: {mutation_prob}")
    print(f"crossover_prob: {crossover_prob}")
    print(f"compactness_penalty: {compactness_penalty}")
    print(f"max_segment_cost_soft: {max_segment_cost_soft}")
    print(f"max_line_cost_soft: {max_line_cost_soft}")
    print(f"max_edges_per_line_soft: {max_edges_per_line_soft}")
    print(f"n_elite: {n_elite}")
    print(f"tournament_size: {tournament_size}")
    print(f"run_name: {run_name or ''}")
    print(f"save_improvement_snapshots: {save_improvement_snapshots}")
    print(f"save_evolution_plots: {save_evolution_plots}")
    print(f"results_dir: {results_dir}")

    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )
    normalization_bounds = _prepare_normalization_bounds(network, line_length)
    optimizer = JointMultiLineOptimizer(
        network=network,
        n_lines=n_lines,
        line_length=line_length,
        population_size=population_size,
        generations=generations,
        lambda_=lambda_,
        seed=seed,
        normalization_bounds=normalization_bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        compactness_penalty=compactness_penalty,
        max_segment_cost_soft=max_segment_cost_soft,
        max_line_cost_soft=max_line_cost_soft,
        max_edges_per_line_soft=max_edges_per_line_soft,
        n_elite=n_elite,
        tournament_size=tournament_size,
        save_improvement_snapshots=save_improvement_snapshots,
        snapshot_output_dir=improvements_dir,
        save_evolution_history=save_evolution_plots,
        evolution_output_dir=evolution_dir,
        od_case=od_case,
        map_case=map_case,
    )
    run_result = optimizer.run()
    evolution_output_paths: Dict[str, str] = {}

    if save_evolution_plots:
        evolution_output_paths = save_joint_multiline_evolution_outputs(
            evolution_history=run_result["evolution_history"],
            output_dir=evolution_dir,
            title_context={
                "od_case": od_case,
                "map_case": map_case,
                "n_lines": n_lines,
                "lambda": lambda_,
                "compactness_penalty": compactness_penalty,
            },
        )

    config_path = os.path.join(results_dir, "joint_multiline_config.txt")
    summary_path = os.path.join(results_dir, "joint_multiline_ga_summary.txt")
    legacy_summary_path = os.path.join(results_dir, "joint_multiline_summary.txt")
    system_metrics_path = os.path.join(results_dir, "joint_multiline_system_metrics.csv")
    lines_path = os.path.join(results_dir, "joint_multiline_lines.csv")
    adjusted_line_metrics_path = os.path.join(results_dir, "joint_multiline_adjusted_line_metrics.csv")
    pairwise_path = os.path.join(results_dir, "joint_multiline_pairwise_overlap.csv")
    improvements_summary_path = os.path.join(improvements_dir, "improvements_summary.csv")

    with open(_filesystem_path(config_path), "w", encoding="utf-8") as config_file:
        config_file.write(f"od_case={od_case}\n")
        config_file.write(f"map_case={map_case}\n")
        config_file.write(f"n_lines={n_lines}\n")
        config_file.write(f"line_length={line_length}\n")
        config_file.write(f"population_size={population_size}\n")
        config_file.write(f"generations={generations}\n")
        config_file.write(f"lambda={lambda_}\n")
        config_file.write(f"seed={seed}\n")
        config_file.write(f"mutation_prob={mutation_prob}\n")
        config_file.write(f"crossover_prob={crossover_prob}\n")
        config_file.write(f"compactness_penalty={compactness_penalty}\n")
        config_file.write(f"max_segment_cost_soft={max_segment_cost_soft}\n")
        config_file.write(f"max_line_cost_soft={max_line_cost_soft}\n")
        config_file.write(f"max_edges_per_line_soft={max_edges_per_line_soft}\n")
        config_file.write(f"n_elite={n_elite}\n")
        config_file.write(f"tournament_size={tournament_size}\n")
        config_file.write(f"run_name={run_name or ''}\n")
        config_file.write(f"results_dir={results_dir}\n")
        config_file.write(f"save_improvement_snapshots={save_improvement_snapshots}\n")
        config_file.write(f"save_evolution_plots={save_evolution_plots}\n")

    with open(_filesystem_path(summary_path), "w", encoding="utf-8") as file:
        file.write("=== CONFIGURATION ===\n")
        file.write(f"od_case: {od_case}\n")
        file.write(f"map_case: {map_case}\n")
        file.write(f"n_lines: {n_lines}\n")
        file.write(f"line_length: {line_length}\n")
        file.write(f"population_size: {population_size}\n")
        file.write(f"generations: {generations}\n")
        file.write(f"lambda: {lambda_}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"mutation_prob: {mutation_prob}\n")
        file.write(f"crossover_prob: {crossover_prob}\n")
        file.write(f"compactness_penalty: {compactness_penalty}\n")
        file.write(f"max_segment_cost_soft: {max_segment_cost_soft}\n")
        file.write(f"max_line_cost_soft: {max_line_cost_soft}\n")
        file.write(f"max_edges_per_line_soft: {max_edges_per_line_soft}\n")
        file.write(f"run_name: {run_name or ''}\n")
        file.write(f"results_dir: {results_dir}\n")
        file.write("\n=== NORMALIZATION BOUNDS ===\n")
        file.write(f"normalization_cost_min: {normalization_bounds.cost_min}\n")
        file.write(f"normalization_cost_max: {normalization_bounds.cost_max}\n")
        file.write(f"normalization_service_min: {normalization_bounds.service_min}\n")
        file.write(f"normalization_service_max: {normalization_bounds.service_max}\n")
        file.write("\n=== FINAL LINES ===\n")
        for index, line in enumerate(run_result["lines"], start=1):
            file.write(f"line_{index}: {line}\n")
        file.write("\n=== ADJUSTED PER-LINE METRICS ===\n")
        file.write(
            "Adjusted per-line metrics are recomputed after the full joint system "
            "is known and are the main per-line interpretation for the final joint solution.\n"
        )
        for entry in run_result["adjusted_line_metrics"]:
            file.write(
                f"line_{entry['line_index']}: adjusted_fitness={entry['adjusted_fitness']} "
                f"adjusted_service={entry['adjusted_passenger_service']} "
                f"adjusted_service_norm={entry['adjusted_service_norm']} "
                f"cost={entry['cost']} cost_norm={entry['cost_norm']} "
                f"shared_edge_ratio={entry['shared_edge_ratio_for_line']}\n"
            )
        file.write("\n=== RAW PER-LINE METRICS (REFERENCE ONLY) ===\n")
        file.write(
            "Raw per-line metrics come from evaluating each line individually. "
            "They are kept for reference/debugging and are not the main final "
            "per-line interpretation for the joint GA.\n"
        )
        for index, evaluation in enumerate(run_result["evaluations"], start=1):
            file.write(
                f"line_{index}: raw_fitness={evaluation.fitness} "
                f"raw_passenger_service={evaluation.passenger_service} "
                f"raw_service_norm={evaluation.service_norm} "
                f"cost={evaluation.cost} cost_norm={evaluation.cost_norm}\n"
            )
        file.write("\n=== GLOBAL SYSTEM METRICS ===\n")
        for key, value in run_result["system_metrics"].items():
            file.write(f"{key}: {value}\n")
        file.write("\n=== JOINT NORMALIZATION REFERENCES ===\n")
        normalization_reference_keys = [
            "single_line_service_min",
            "single_line_service_max",
            "system_service_min",
            "system_service_max",
            "single_line_cost_min",
            "single_line_cost_max",
            "system_cost_min",
            "system_cost_max",
            "adjusted_service",
            "raw_route_service_sum",
            "unique_edge_service_sum",
            "total_route_cost",
            "system_service_norm",
            "system_cost_norm",
            "system_fitness",
        ]
        for key in normalization_reference_keys:
            file.write(f"{key}: {run_result['system_metrics'].get(key, '')}\n")
        file.write("\n=== COMPACTNESS / LENGTH CONTROL ===\n")
        file.write(
            "The compactness penalty discourages visually unrealistic long or "
            "indirect bus lines. It does not replace the cost term; it adds an "
            "optional extra penalty to reduce excessive line length or very long "
            "segments between consecutive selected stops.\n"
        )
        compactness_keys = [
            "compactness_penalty",
            "max_line_cost_soft",
            "max_segment_cost_soft",
            "max_edges_per_line_soft",
            "compactness_penalty_value",
            "compactness_total_cost_component",
            "compactness_max_line_excess",
            "compactness_max_segment_excess",
            "compactness_max_edges_excess",
            "base_system_fitness",
            "system_fitness",
            "average_line_cost",
            "max_line_cost",
            "average_segment_cost",
            "max_segment_cost",
            "average_edges_per_line",
            "max_edges_per_line",
        ]
        for key in compactness_keys:
            file.write(f"{key}: {run_result['system_metrics'].get(key, '')}\n")
        file.write("\n=== SERVICE METRIC DEFINITIONS ===\n")
        file.write(
            "raw_line_passenger_service_sum: Sum of the individual line "
            "passenger_service values returned by the single-line evaluations.\n"
        )
        file.write(
            "naive_edge_passenger_service: Edge-based service counted once per "
            "line usage. Shared edges are counted multiple times.\n"
        )
        file.write(
            "unique_edge_passenger_service: Edge-based service counted once per "
            "unique road edge.\n"
        )
        file.write(
            "adjusted_passenger_service: Edge-based service adjusted using "
            "shared-edge counts. This is the service used for system_service_norm "
            "and system_fitness.\n"
        )
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
            for key, value in sorted(evolution_output_paths.items()):
                file.write(f"{key}: {value}\n")
        file.write("\n=== OUTPUT FILES ===\n")
        file.write(f"joint_multiline_config: {config_path}\n")
        file.write(f"joint_multiline_system_metrics: {system_metrics_path}\n")
        file.write(f"joint_multiline_lines: {lines_path}\n")
        file.write(f"joint_multiline_adjusted_line_metrics: {adjusted_line_metrics_path}\n")
        file.write(f"joint_multiline_pairwise_overlap: {pairwise_path}\n")

    shutil.copyfile(_filesystem_path(summary_path), _filesystem_path(legacy_summary_path))

    with open(_filesystem_path(system_metrics_path), "w", newline="", encoding="utf-8") as csvfile:
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
                **run_result["system_metrics"],
            }
        )

    with open(_filesystem_path(lines_path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "line_index",
            "stops",
            "cost",
            "cost_norm",
            "raw_passenger_service",
            "raw_service_norm",
            "raw_fitness",
            "adjusted_passenger_service",
            "adjusted_service_norm",
            "adjusted_fitness",
            "shared_edge_count_for_line",
            "shared_edge_ratio_for_line",
        ])
        adjusted_metrics_by_line = {
            entry["line_index"]: entry
            for entry in run_result["adjusted_line_metrics"]
        }
        for index, evaluation in enumerate(run_result["evaluations"], start=1):
            adjusted_metrics = adjusted_metrics_by_line[index]
            writer.writerow([
                index,
                _format_line(run_result["lines"][index - 1]),
                evaluation.cost,
                evaluation.cost_norm,
                evaluation.passenger_service,
                evaluation.service_norm,
                evaluation.fitness,
                adjusted_metrics["adjusted_passenger_service"],
                adjusted_metrics["adjusted_service_norm"],
                adjusted_metrics["adjusted_fitness"],
                adjusted_metrics["shared_edge_count_for_line"],
                adjusted_metrics["shared_edge_ratio_for_line"],
            ])

    with open(_filesystem_path(adjusted_line_metrics_path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for entry in run_result["adjusted_line_metrics"]:
            row = dict(entry)
            row["stops"] = _format_line(run_result["lines"][entry["line_index"] - 1])
            writer.writerow(row)

    with open(_filesystem_path(pairwise_path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["line_i", "line_j", "shared_stops", "shared_edges"])
        for entry in run_result["pairwise_overlap"]:
            writer.writerow([entry["line_i"], entry["line_j"], entry["shared_stops"], entry["shared_edges"]])

    if save_improvement_snapshots:
        with open(_filesystem_path(improvements_summary_path), "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "generation",
                    "fitness",
                    "base_system_fitness",
                    "compactness_penalty",
                    "compactness_penalty_value",
                    "compactness_total_cost_component",
                    "compactness_max_line_excess",
                    "compactness_max_segment_excess",
                    "compactness_max_edges_excess",
                    "adjusted_passenger_service",
                    "raw_line_passenger_service_sum",
                    "naive_edge_passenger_service",
                    "unique_edge_passenger_service",
                    "total_route_cost",
                    "average_line_cost",
                    "max_line_cost",
                    "average_segment_cost",
                    "max_segment_cost",
                    "average_edges_per_line",
                    "max_edges_per_line",
                    "system_cost_norm",
                    "system_service_norm",
                    "shared_stop_ratio",
                    "shared_edge_ratio",
                    "pdf_path",
                ],
            )
            writer.writeheader()
            for entry in run_result["improvement_history"]:
                writer.writerow(entry)

    print("\n=== JOINT MULTI-LINE RESULT ===")
    for index, line in enumerate(run_result["lines"], start=1):
        print(f"Line {index}: {line}")
    print("\n=== ADJUSTED PER-LINE RESULT ===")
    for entry in run_result["adjusted_line_metrics"]:
        print(
            f"Line {entry['line_index']}: "
            f"adjusted_fitness={entry['adjusted_fitness']} "
            f"adjusted_service={entry['adjusted_passenger_service']} "
            f"shared_edge_ratio={entry['shared_edge_ratio_for_line']}"
        )
    print("\n=== GLOBAL SYSTEM RESULT ===")
    for key, value in run_result["system_metrics"].items():
        print(f"{key} = {value}")
    print("\nSummary saved to:", summary_path)
    print("System metrics saved to:", system_metrics_path)
    print("Line metrics saved to:", lines_path)
    print("Adjusted line metrics saved to:", adjusted_line_metrics_path)
    print("Pairwise overlap saved to:", pairwise_path)
    if save_improvement_snapshots:
        print("Improvement snapshots saved to:", improvements_dir)
    if save_evolution_plots:
        print("Evolution outputs saved to:", evolution_dir)
    print("\nResults directory:")
    print(results_dir)

    result = {
        "results_dir": results_dir,
        "summary_path": summary_path,
        "legacy_summary_path": legacy_summary_path,
        "config_path": config_path,
        "system_metrics_path": system_metrics_path,
        "lines_path": lines_path,
        "adjusted_line_metrics_path": adjusted_line_metrics_path,
        "pairwise_path": pairwise_path,
        "improvements_summary_path": improvements_summary_path if save_improvement_snapshots else None,
        "improvement_snapshot_dir": improvements_dir if save_improvement_snapshots else None,
        "evolution_output_dir": evolution_dir if save_evolution_plots else None,
        "evolution_output_paths": evolution_output_paths,
        "network": network,
        **run_result,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base")
    parser.add_argument("--map-case", default="base")
    parser.add_argument("--n-lines", "--lines", dest="n_lines", type=int, default=2)
    parser.add_argument("--line-length", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-prob", type=float, default=0.20)
    parser.add_argument("--crossover-prob", type=float, default=0.80)
    parser.add_argument("--compactness-penalty", type=float, default=0.0)
    parser.add_argument("--max-segment-cost-soft", type=float, default=None)
    parser.add_argument("--max-line-cost-soft", type=float, default=None)
    parser.add_argument("--max-edges-per-line-soft", type=float, default=None)
    parser.add_argument("--n-elite", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-improvement-snapshots", action="store_true")
    parser.add_argument("--save-evolution-plots", action="store_true")
    parser.add_argument("--od-min-demand", type=float, default=3.0)
    args = parser.parse_args()

    if args.n_lines < 1:
        parser.error("--n-lines / --lines must be >= 1")
    if args.line_length < 2:
        parser.error("--line-length must be >= 2")

    return args


def main() -> None:
    args = parse_args()
    run_result = run_joint_multiline_ga_experiment(
        od_case=args.od_case,
        map_case=args.map_case,
        n_lines=args.n_lines,
        line_length=args.line_length,
        population_size=args.population_size,
        generations=args.generations,
        lambda_=args.lambda_,
        seed=args.seed,
        mutation_prob=args.mutation_prob,
        crossover_prob=args.crossover_prob,
        compactness_penalty=args.compactness_penalty,
        max_segment_cost_soft=args.max_segment_cost_soft,
        max_line_cost_soft=args.max_line_cost_soft,
        max_edges_per_line_soft=args.max_edges_per_line_soft,
        n_elite=args.n_elite,
        tournament_size=args.tournament_size,
        run_name=args.run_name,
        save_improvement_snapshots=args.save_improvement_snapshots,
        save_evolution_plots=args.save_evolution_plots,
    )
    pdf_path, png_path = save_joint_multiline_map_from_run_result(
        run_result,
        od_threshold=args.od_min_demand,
    )
    print("Joint multiline map PDF saved to:", pdf_path)
    print("Joint multiline map PNG saved to:", png_path)


if __name__ == "__main__":
    main()
