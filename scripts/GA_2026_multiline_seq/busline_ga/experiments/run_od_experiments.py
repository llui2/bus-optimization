from __future__ import annotations

import argparse
import os
from typing import Dict, List

from busline_ga.config.project_paths import SCRIPT_RESULTS_DIR
from busline_ga.experiments.main_ga import run_ga_experiment
from busline_ga.config.od_scenarios import get_registered_od_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa tots els casos OD registrats per a un map_case fix."
    )
    parser.add_argument(
        "--map-case",
        default="dense_fill",
        help="Variant de xarxa viÃ ria a utilitzar per a tots els casos OD.",
    )
    result = parser.parse_args()
    return result


def main() -> None:
    args = parse_args()
    batch_results_dir = os.path.join(str(SCRIPT_RESULTS_DIR), "od_experiments")
    os.makedirs(batch_results_dir, exist_ok=True)

    completed_runs: List[Dict[str, str]] = []
    failed_cases: List[str] = []

    for od_case in get_registered_od_cases():
        print(f"\n=== EXECUTANT CAS OD: {od_case} ===")

        try:
            result = run_ga_experiment(od_case=od_case, map_case=args.map_case)
            best_evaluation = result["best_evaluation"]
            case_summary = {
                "od_case": od_case,
                "map_case": result["map_case"],
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
            completed_runs.append(case_summary)
        except Exception as exc:
            failed_cases.append(f"{od_case}: {exc}")
            print(f"[WARNING] Ha fallat el cas {od_case}: {exc}")

    batch_summary_path = os.path.join(batch_results_dir, "batch_summary.txt")
    with open(batch_summary_path, "w", encoding="utf-8") as file:
        file.write("=== RESUM EXPERIMENTS OD ===\n")
        file.write(f"map_case_batch: {args.map_case}\n")
        for item in completed_runs:
            file.write(f"\nCas: {item['od_case']}\n")
            file.write(f"map_case: {item['map_case']}\n")
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

    print("\n=== RESUM FINAL ===")
    print("Casos completats:", len(completed_runs))
    print("Casos fallits:", len(failed_cases))
    print("Resum batch guardat a:", batch_summary_path)

    for item in completed_runs:
        print(f" - {item['od_case']}: {item['results_dir']}")

    if failed_cases:
        print("Casos fallits:")
        for failed_case in failed_cases:
            print(" -", failed_case)


if __name__ == "__main__":
    main()

