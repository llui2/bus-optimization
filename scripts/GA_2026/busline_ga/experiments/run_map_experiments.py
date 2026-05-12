from __future__ import annotations

import argparse
import os
from typing import Dict, List

from busline_ga.config.project_paths import SCRIPT_RESULTS_DIR
from busline_ga.experiments.main_ga import run_ga_experiment
from busline_ga.config.map_cases import get_registered_map_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD fix per al batch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_results_dir = os.path.join(
        str(SCRIPT_RESULTS_DIR),
        "road_experiments",
        f"od_{args.od_case}",
    )
    os.makedirs(batch_results_dir, exist_ok=True)

    completed_runs: List[Dict[str, str]] = []
    failed_cases: List[str] = []

    for map_case in get_registered_map_cases():
        print(f"\n=== EXECUTANT MAP CASE: {map_case} | OD CASE: {args.od_case} ===")

        try:
            result = run_ga_experiment(
                od_case=args.od_case,
                map_case=map_case,
            )
            best_evaluation = result["best_evaluation"]
            completed_runs.append(
                {
                    "od_case": args.od_case,
                    "map_case": map_case,
                    "results_dir": result["results_dir"],
                    "summary_path": result["summary_path"],
                    "final_figure_path": result["final_figure_path"],
                    "best_individual": str(result["best_individual"]),
                    "best_score": str(result["best_score"]),
                    "cost": str(best_evaluation.cost),
                    "passenger_service": str(best_evaluation.passenger_service),
                    "cost_norm": str(best_evaluation.cost_norm),
                    "passenger_service_norm": str(best_evaluation.passenger_service_norm),
                }
            )
        except Exception as exc:
            failed_cases.append(f"{map_case}: {exc}")
            print(f"[WARNING] Ha fallat el map case {map_case}: {exc}")

    batch_summary_path = os.path.join(batch_results_dir, "map_batch_summary.txt")
    with open(batch_summary_path, "w", encoding="utf-8") as file:
        file.write("=== RESUM EXPERIMENTS MAP CASE ===\n")
        file.write(f"od_case: {args.od_case}\n")
        for item in completed_runs:
            file.write(f"\nmap_case: {item['map_case']}\n")
            file.write(f"results_dir: {item['results_dir']}\n")
            file.write(f"summary_path: {item['summary_path']}\n")
            file.write(f"final_figure_path: {item['final_figure_path']}\n")
            file.write(f"best_individual: {item['best_individual']}\n")
            file.write(f"best_score: {item['best_score']}\n")
            file.write(f"cost: {item['cost']}\n")
            file.write(f"passenger_service: {item['passenger_service']}\n")
            file.write(f"cost_norm: {item['cost_norm']}\n")
            file.write(f"passenger_service_norm: {item['passenger_service_norm']}\n")

        if failed_cases:
            file.write("\nCasos fallits:\n")
            for failed_case in failed_cases:
                file.write(f"- {failed_case}\n")

    print("\n=== RESUM FINAL MAP CASES ===")
    print("OD case:", args.od_case)
    print("Casos completats:", len(completed_runs))
    print("Casos fallits:", len(failed_cases))
    print("Resum batch guardat a:", batch_summary_path)

    for item in completed_runs:
        print(f" - {item['map_case']}: {item['results_dir']}")

    if failed_cases:
        print("Casos fallits:")
        for failed_case in failed_cases:
            print(" -", failed_case)


if __name__ == "__main__":
    main()

