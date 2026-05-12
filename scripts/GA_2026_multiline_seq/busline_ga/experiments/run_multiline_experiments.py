from __future__ import annotations

import argparse
import csv
import os
from itertools import product
from typing import List, Optional

from busline_ga.config.map_cases import get_road_experiment_results_dir
from busline_ga.config.project_paths import PROJECT_ROOT, SCRIPT_DIR
from busline_ga.core.network_model import NetworkModel
from busline_ga.experiments.main_multiline_ga import run_multiline_ga_experiment


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a executar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viaria")
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
        "--line-counts",
        default="1,2,3,4",
        help="Comma-separated values for n_lines",
    )
    parser.add_argument(
        "--shared-stop-penalties",
        default="0.0,0.01,0.03,0.05",
        help="Comma-separated values for shared_stop_penalty",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional base output folder for the batch experiments",
    )
    args = parser.parse_args()

    if args.line_length < 2:
        parser.error("--line-length must be >= 2")

    args.line_counts = parse_int_list(args.line_counts)
    args.shared_stop_penalties = parse_float_list(args.shared_stop_penalties)

    if any(line_count < 1 for line_count in args.line_counts):
        parser.error("All values in --line-counts must be >= 1")

    return args


def main() -> None:
    args = parse_args()
    batch_root = args.output_dir or os.path.join(
        get_road_experiment_results_dir(str(SCRIPT_DIR), args.od_case, args.map_case),
        "multiline_experiments",
    )
    os.makedirs(batch_root, exist_ok=True)
    summary_path = os.path.join(batch_root, "multiline_experiments_summary.csv")

    fieldnames = [
        "od_case",
        "map_case",
        "n_lines",
        "shared_stop_penalty",
        "line_length",
        "population_size",
        "generations",
        "lambda",
        "seed",
        "system_total_route_cost",
        "system_unique_edge_cost",
        "system_adjusted_passenger_service",
        "system_service_per_total_cost",
        "system_service_per_unique_edge_cost",
        "system_fitness",
        "shared_stop_ratio",
        "shared_edge_ratio",
        "results_dir",
        "summary_path",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line_count, shared_stop_penalty in product(
            args.line_counts,
            args.shared_stop_penalties,
        ):
            print(
                f"\nRunning n_lines={line_count}, shared_stop_penalty={shared_stop_penalty}"
            )
            result = run_multiline_ga_experiment(
                od_case=args.od_case,
                map_case=args.map_case,
                n_lines=line_count,
                line_length=args.line_length,
                population_size=args.population_size,
                generations=args.generations,
                lambda_=args.lambda_,
                seed=args.seed,
                shared_stop_penalty=shared_stop_penalty,
                output_dir=os.path.join(
                    batch_root,
                    f"multiline_{line_count}_lines_penalty_{shared_stop_penalty}",
                ),
            )

            system_metrics = result["system_metrics"]
            writer.writerow(
                {
                    "od_case": args.od_case,
                    "map_case": args.map_case,
                    "n_lines": line_count,
                    "shared_stop_penalty": shared_stop_penalty,
                    "line_length": args.line_length,
                    "population_size": args.population_size,
                    "generations": args.generations,
                    "lambda": args.lambda_,
                    "seed": args.seed,
                    "system_total_route_cost": system_metrics["total_route_cost"],
                    "system_unique_edge_cost": system_metrics["unique_edge_cost"],
                    "system_adjusted_passenger_service": system_metrics["adjusted_passenger_service"],
                    "system_service_per_total_cost": system_metrics["service_per_total_cost"],
                    "system_service_per_unique_edge_cost": system_metrics["service_per_unique_edge_cost"],
                    "system_fitness": system_metrics["system_fitness"],
                    "shared_stop_ratio": system_metrics["shared_stop_ratio"],
                    "shared_edge_ratio": system_metrics["shared_edge_ratio"],
                    "results_dir": result["results_dir"],
                    "summary_path": result["summary_path"],
                }
            )

    print("\nBatch experiment summary saved to:", summary_path)


if __name__ == "__main__":
    main()
