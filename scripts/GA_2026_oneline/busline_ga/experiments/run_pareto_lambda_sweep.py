from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, OD_SCENARIOS_DIR, PROJECT_ROOT, RESULTS_DIR
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.network_model import NetworkModel, Node, load_network_model
from busline_ga.core.normalization_utils import DEFAULT_NORMALIZATION_METHOD, prepare_structural_normalization
from busline_ga.core.objective_function import EvaluationResult, ObjectiveFunction, NormalizationBounds
from busline_ga.visualization.ga_snapshot_plotter import save_final_line_with_od


@dataclass
class ParetoRunResult:
    run_id: int
    lambda_value: float
    seed: int
    fitness: float
    cost: float
    cost_norm: float
    passenger_service: float
    passenger_service_norm: float
    service: float
    service_norm: float
    is_pareto: bool
    pareto_label: str
    best_individual: str
    line_edges_count: int


def pareto_result_to_row(item: ParetoRunResult) -> Dict[str, object]:
    row = asdict(item)
    row["lambda"] = row.pop("lambda_value")
    ordered_keys = [
        "run_id",
        "lambda",
        "seed",
        "fitness",
        "cost",
        "cost_norm",
        "passenger_service",
        "passenger_service_norm",
        "service",
        "service_norm",
        "is_pareto",
        "pareto_label",
        "best_individual",
        "line_edges_count",
    ]
    result = {key: row[key] for key in ordered_keys}
    return result


def parse_seed_list(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    result = seeds
    return result


def build_lambda_values(step: float) -> List[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("El pas de lambda ha d'estar dins de l'interval (0, 1].")

    n_steps = int(round(1.0 / step))
    values = [round(index * step, 10) for index in range(n_steps + 1)]

    if values[-1] != 1.0:
        values.append(1.0)

    result = sorted(set(values))
    return result


def find_base_od_path() -> Path:
    candidates = [
        OD_SCENARIOS_DIR / "od_base.csv",
        BUS_NETWORK_DIR / "od_matrix_fixed.csv",
    ]

    selected = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            selected = candidate
            break

    if not selected.exists():
        raise FileNotFoundError(
            "No s'ha trobat cap OD base. He buscat od_scenarios/od_base.csv "
            "i bus_network/od_matrix_fixed.csv."
        )

    result = selected
    return result


def generate_dense_pareto_od(
    output_path: Path,
    n_strong_pairs: int,
    seed: int,
    high_min: float,
    high_max: float,
) -> Path:
    """
    Genera una matriu OD amb moltes parelles no nul·les i valors alts.

    Aquesta opció està pensada per a l'experiment del front de Pareto:
    tenir prou demanda repartida perquè diferents valors de lambda puguin
    donar solucions amb compromisos diferents entre servei i cost.
    """
    base_od_path = find_base_od_path()
    base_od = pd.read_csv(base_od_path, index_col=0)
    labels = [str(label) for label in base_od.index.tolist()]
    rng = random.Random(seed)

    all_pairs: List[Tuple[str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            all_pairs.append((labels[i], labels[j]))

    effective_pairs = min(n_strong_pairs, len(all_pairs))
    selected_pairs = rng.sample(all_pairs, effective_pairs)

    matrix = pd.DataFrame(0.1, index=labels, columns=labels)
    for stop_a, stop_b in selected_pairs:
        value = rng.uniform(high_min, high_max)
        matrix.loc[stop_a, stop_b] = value
        matrix.loc[stop_b, stop_a] = value

    np.fill_diagonal(matrix.values, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)

    result = output_path
    return result


def load_experiment_network(
    map_case: str,
    od_case: str,
    od_matrix_path: Optional[str],
    verbose: bool,
) -> Tuple[NetworkModel, str, str, str]:
    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        map_case,
        verbose=verbose,
    )
    bus_stops_path = str(BUS_NETWORK_DIR / "nodes.csv")

    selected_od_path = od_matrix_path
    if selected_od_path is None:
        selected_od_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)

    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=str(selected_od_path),
    )

    result = network, road_nodes_path, road_edges_path, str(selected_od_path)
    return result


def prepare_objective(network: NetworkModel, line_length: int) -> Tuple[ObjectiveFunction, NormalizationBounds]:
    objective = ObjectiveFunction(network)
    objective.precompute_edge_costs()
    objective.precompute_edge_service()
    normalization_bounds = prepare_structural_normalization(
        objective=objective,
        line_length=line_length,
    )

    result = objective, normalization_bounds
    return result


def create_optimizer(
    network: NetworkModel,
    objective: ObjectiveFunction,
    line_length: int,
    population_size: int,
    seed: int,
) -> GeneticOptimizer:
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

    result = ga
    return result


def run_single_ga(
    network: NetworkModel,
    objective: ObjectiveFunction,
    lambda_value: float,
    seed: int,
    generations: int,
    population_size: int,
    line_length: int,
    n_elite: int,
    tournament_size: int,
    mutation_prob: float,
) -> Tuple[List[Node], float, EvaluationResult]:
    ga = create_optimizer(
        network=network,
        objective=objective,
        line_length=line_length,
        population_size=population_size,
        seed=seed,
    )

    best_individual, best_score = ga.run(
        lambda_=lambda_value,
        generations=generations,
        n_elite=n_elite,
        tournament_size=tournament_size,
        mutation_prob=mutation_prob,
        weight_adjacent=0.20,
        weight_reverse=0.15,
        weight_neighbor=0.25,
        weight_service_replace=0.40,
        save_improvement_snapshots=False,
    )

    best_eval = objective.evaluate(best_individual, lambda_value)

    result = best_individual, best_score, best_eval
    return result


def compute_pareto_flags(results: Sequence[ParetoRunResult]) -> List[bool]:
    flags: List[bool] = []

    for i, candidate in enumerate(results):
        dominated = False

        for j, other in enumerate(results):
            if i == j:
                continue

            other_is_no_worse = (
                other.cost_norm <= candidate.cost_norm
                and other.passenger_service_norm >= candidate.passenger_service_norm
            )
            other_is_strictly_better = (
                other.cost_norm < candidate.cost_norm
                or other.passenger_service_norm > candidate.passenger_service_norm
            )

            if other_is_no_worse and other_is_strictly_better:
                dominated = True
                break

        flags.append(not dominated)

    result = flags
    return result


def write_results_csv(results: Sequence[ParetoRunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pareto_result_to_row(results[0]).keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(pareto_result_to_row(item))


def load_results_csv(input_path: Path) -> List[ParetoRunResult]:
    required_columns = {
        "run_id", "lambda", "seed", "fitness", "cost", "cost_norm",
        "passenger_service", "passenger_service_norm", "service", "service_norm",
        "is_pareto", "best_individual", "line_edges_count",
    }
    results: List[ParetoRunResult] = []

    if not input_path.is_file():
        raise FileNotFoundError(
            f"No s'ha trobat el fitxer de resultats existent: {input_path}. "
            "Executa primer l'experiment complet sense --plot-only."
        )

    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"El fitxer {input_path} no conté les columnes necessàries: "
                f"{', '.join(missing_columns)}."
            )

        for row in reader:
            results.append(
                ParetoRunResult(
                    run_id=int(row["run_id"]),
                    lambda_value=float(row["lambda"]),
                    seed=int(row["seed"]),
                    fitness=float(row["fitness"]),
                    cost=float(row["cost"]),
                    cost_norm=float(row["cost_norm"]),
                    passenger_service=float(row["passenger_service"]),
                    passenger_service_norm=float(row["passenger_service_norm"]),
                    service=float(row["service"]),
                    service_norm=float(row["service_norm"]),
                    is_pareto=str(row["is_pareto"]).strip().lower() in {"true", "1", "yes"},
                    pareto_label=row.get("pareto_label", ""),
                    best_individual=row["best_individual"],
                    line_edges_count=int(row["line_edges_count"]),
                )
            )

    if not results:
        raise ValueError(
            f"El fitxer de resultats està buit: {input_path}. "
            "Executa primer l'experiment complet sense --plot-only."
        )

    return results


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    od_matrix_path: str,
    road_nodes_path: str,
    road_edges_path: str,
    normalization_bounds: NormalizationBounds,
    results: Sequence[ParetoRunResult],
    all_csv_path: Path,
    pareto_csv_path: Path,
    pdf_path: Path,
    png_path: Path,
    representative_points: Sequence[ParetoRunResult],
) -> None:
    pareto_count = sum(1 for item in results if item.is_pareto)
    representative_rows = [
        pareto_result_to_row(item)
        for item in representative_points
    ]

    summary: Dict[str, object] = {
        "map_case": args.map_case,
        "od_case": args.od_case,
        "od_matrix_path": od_matrix_path,
        "road_nodes_path": road_nodes_path,
        "road_edges_path": road_edges_path,
        "lambda_step": args.lambda_step,
        "seeds": parse_seed_list(args.seeds),
        "generations": args.generations,
        "population_size": args.population_size,
        "line_length": args.line_length,
        "normalization_method": DEFAULT_NORMALIZATION_METHOD,
        "normalization_cost_min": normalization_bounds.cost_min,
        "normalization_cost_max": normalization_bounds.cost_max,
        "normalization_service_min": normalization_bounds.service_min,
        "normalization_service_max": normalization_bounds.service_max,
        "n_lambdas": len(build_lambda_values(args.lambda_step)),
        "n_seeds": len(parse_seed_list(args.seeds)),
        "total_runs": len(results),
        "pareto_points": pareto_count,
        "unique_pareto_points": len(unique_pareto_points(results)),
        "representative_points": [],
        "outputs": {
            "all_points_csv": str(all_csv_path),
            "pareto_points_csv": str(pareto_csv_path),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
        "generated_pareto_od": bool(args.generate_pareto_od),
        "pareto_od_pairs": args.pareto_od_pairs if args.generate_pareto_od else None,
        "pareto_od_seed": args.pareto_od_seed if args.generate_pareto_od else None,
        "pareto_od_high_min": args.pareto_od_high_min if args.generate_pareto_od else None,
        "pareto_od_high_max": args.pareto_od_high_max if args.generate_pareto_od else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def unique_pareto_points(results: Sequence[ParetoRunResult]) -> List[ParetoRunResult]:
    seen: Dict[Tuple[float, float], ParetoRunResult] = {}

    for item in results:
        if item.is_pareto:
            key = (round(item.cost_norm, 12), round(item.passenger_service_norm, 12))
            if key not in seen:
                seen[key] = item

    ordered = sorted(seen.values(), key=lambda item: (item.cost_norm, item.passenger_service_norm))
    result = ordered
    return result


def plot_pareto_results(results: Sequence[ParetoRunResult], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    granate = "#7A0019"
    cmap = LinearSegmentedColormap.from_list(
        "lambda_blue_lilac_burgundy",
        ["#D7ECFF", "#8E6BBE", "#7A0019"],
    )
    dominated = [item for item in results if not item.is_pareto]
    non_dominated = [item for item in results if item.is_pareto]
    pareto_points = unique_pareto_points(results)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    lambda_scatter = None
    if dominated:
        ax.scatter(
            [item.cost_norm for item in dominated],
            [item.passenger_service_norm for item in dominated],
            color="0.60",
            s=54,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
        )
    if non_dominated:
        lambda_scatter = ax.scatter(
            [item.cost_norm for item in non_dominated],
            [item.passenger_service_norm for item in non_dominated],
            c=[item.lambda_value for item in non_dominated],
            cmap=cmap,
            norm=norm,
            s=74,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    if pareto_points:
        pareto_x = [item.cost_norm for item in pareto_points]
        pareto_y = [item.passenger_service_norm for item in pareto_points]

        ax.plot(
            pareto_x,
            pareto_y,
            linestyle="--",
            linewidth=1.6,
            color=granate,
            zorder=3,
        )
        ax.scatter(
            pareto_x,
            pareto_y,
            s=150,
            facecolors="none",
            edgecolors=granate,
            linewidths=1.8,
            zorder=4,
        )

    if lambda_scatter is None:
        lambda_scatter = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    colorbar = fig.colorbar(lambda_scatter, ax=ax)
    colorbar.set_label(r"Valor de $\lambda$", fontsize=11)

    ax.set_xlabel(r"Cost normalitzat $C_{\mathrm{norm}}(E_\ell)$", fontsize=12)
    ax.set_ylabel(r"Servei normalitzat $P_{\mathrm{norm}}(E_\ell)$", fontsize=12)
    ax.grid(True, color="0.88", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    legend_handles = []
    if dominated:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6.5,
                markerfacecolor="0.60",
                markeredgecolor="white",
                label="Solucions dominades",
            )
        )
    if pareto_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                linewidth=1.6,
                color=granate,
                marker="o",
                markersize=7.5,
                markerfacecolor="none",
                markeredgecolor=granate,
                markeredgewidth=1.5,
                label="Front de Pareto",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        frameon=False,
        fontsize=8.4,
        handlelength=1.8,
        labelspacing=0.35,
        borderaxespad=0.2,
    )

    plt.tight_layout()

    pdf_path = output_dir / "pareto_lambda_sweep.pdf"
    png_path = output_dir / "pareto_lambda_sweep.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = pdf_path, png_path
    return result


def save_pareto_line_figures(
    network: NetworkModel,
    objective: ObjectiveFunction,
    results: Sequence[ParetoRunResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_points = [item for item in unique_pareto_points(results) if item.pareto_label]
    if not selected_points:
        selected_points = unique_pareto_points(results)

    for index, item in enumerate(selected_points, start=1):
        individual = json.loads(item.best_individual)
        evaluation = objective.evaluate(individual, item.lambda_value)
        label = item.pareto_label or f"pareto_{index:02d}"
        output_path = output_dir / f"{label}_lambda_{item.lambda_value:.2f}_seed_{item.seed}.pdf"
        save_final_line_with_od(
            network=network,
            individual=individual,
            evaluation=evaluation,
            output_path=str(output_path),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa el GA per diversos valors de lambda i seeds, i genera el front de Pareto."
    )
    parser.add_argument("--od-case", default="base", help="Cas OD registrat a od_scenarios.py")
    parser.add_argument("--od-matrix-path", default=None, help="Ruta directa a una matriu OD CSV")
    parser.add_argument("--map-case", default="dense_fill", help="Variant de xarxa: base, light_fill o dense_fill")
    parser.add_argument("--lambda-step", type=float, default=0.05, help="Pas entre valors de lambda")
    parser.add_argument("--seeds", default="42,43,44,45", help="Seeds separades per comes")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--line-length", type=int, default=6)
    parser.add_argument("--n-elite", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mutation-prob", type=float, default=0.30)
    parser.add_argument("--output-dir", default=None, help="Carpeta de sortida")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenera només la figura a partir del CSV existent, sense executar el GA.",
    )
    parser.add_argument("--quiet", action="store_true", help="Redueix la sortida per terminal")
    parser.add_argument("--save-pareto-lines", action="store_true", help="Guarda una figura de línia per cada punt de Pareto")

    parser.add_argument("--generate-pareto-od", action="store_true", help="Genera una OD densa amb parelles de demanda alta")
    parser.add_argument("--pareto-od-pairs", type=int, default=700, help="Nombre de parelles fortes de la OD generada")
    parser.add_argument("--pareto-od-seed", type=int, default=2026, help="Seed per generar la OD")
    parser.add_argument("--pareto-od-high-min", type=float, default=50.0, help="Valor mínim de demanda forta")
    parser.add_argument("--pareto-od-high-max", type=float, default=100.0, help="Valor màxim de demanda forta")

    args = parser.parse_args()
    result = args
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "pareto_lambda_sweep"

    if args.plot_only:
        all_csv_path = output_dir / "pareto_lambda_sweep_points.csv"
        try:
            results = load_results_csv(all_csv_path)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"ERROR: {error}") from error

        pdf_path, png_path = plot_pareto_results(results, output_dir)
        print("\n=== Pareto plot-only ===")
        print("CSV carregat:", all_csv_path)
        print("Total punts:", len(results))
        print("Punts Pareto:", sum(1 for item in results if item.is_pareto))
        print("Figura PDF:", pdf_path)
        print("Figura PNG:", png_path)
        return

    seeds = parse_seed_list(args.seeds)
    lambda_values = build_lambda_values(args.lambda_step)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_od_matrix_path = args.od_matrix_path
    if args.generate_pareto_od:
        generated_od_path = output_dir / f"od_pareto_dense_{args.pareto_od_pairs}_pairs.csv"
        selected_od_matrix_path = str(
            generate_dense_pareto_od(
                output_path=generated_od_path,
                n_strong_pairs=args.pareto_od_pairs,
                seed=args.pareto_od_seed,
                high_min=args.pareto_od_high_min,
                high_max=args.pareto_od_high_max,
            )
        )

    print("\n=== Pareto lambda sweep ===")
    print("map_case =", args.map_case)
    print("od_case =", args.od_case)
    print("od_matrix_path =", selected_od_matrix_path)
    print("lambdas =", lambda_values)
    print("seeds =", seeds)
    print("generations =", args.generations)
    print("population_size =", args.population_size)

    network, road_nodes_path, road_edges_path, od_matrix_path = load_experiment_network(
        map_case=args.map_case,
        od_case=args.od_case,
        od_matrix_path=selected_od_matrix_path,
        verbose=not args.quiet,
    )
    objective, normalization_bounds = prepare_objective(network, args.line_length)

    results: List[ParetoRunResult] = []
    run_id = 0

    for lambda_value in lambda_values:
        for seed in seeds:
            run_id += 1
            print(f"\n[RUN {run_id:03d}] lambda={lambda_value:.2f} seed={seed}")
            best_individual, best_score, best_eval = run_single_ga(
                network=network,
                objective=objective,
                lambda_value=lambda_value,
                seed=seed,
                generations=args.generations,
                population_size=args.population_size,
                line_length=args.line_length,
                n_elite=args.n_elite,
                tournament_size=args.tournament_size,
                mutation_prob=args.mutation_prob,
            )

            item = ParetoRunResult(
                run_id=run_id,
                lambda_value=lambda_value,
                seed=seed,
                fitness=best_score,
                cost=best_eval.cost,
                cost_norm=best_eval.cost_norm,
                passenger_service=best_eval.passenger_service,
                passenger_service_norm=best_eval.passenger_service_norm,
                service=best_eval.service,
                service_norm=best_eval.service_norm,
                is_pareto=False,
                pareto_label="",
                best_individual=json.dumps(best_individual),
                line_edges_count=len(best_eval.line_edges),
            )
            results.append(item)

            print(
                "  best:",
                "cost_norm=", f"{best_eval.cost_norm:.4f}",
                "service_norm=", f"{best_eval.passenger_service_norm:.4f}",
                "fitness=", f"{best_score:.4f}",
                "line=", best_individual,
            )

    pareto_flags = compute_pareto_flags(results)
    for item, is_pareto in zip(results, pareto_flags):
        item.is_pareto = is_pareto
    representative_points: List[ParetoRunResult] = []

    all_csv_path = output_dir / "pareto_lambda_sweep_points.csv"
    pareto_csv_path = output_dir / "pareto_front_points.csv"
    summary_path = output_dir / "pareto_summary.json"

    write_results_csv(results, all_csv_path)
    write_results_csv(unique_pareto_points(results), pareto_csv_path)
    pdf_path, png_path = plot_pareto_results(results, output_dir)
    write_summary(
        output_path=summary_path,
        args=args,
        od_matrix_path=od_matrix_path,
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        normalization_bounds=normalization_bounds,
        results=results,
        all_csv_path=all_csv_path,
        pareto_csv_path=pareto_csv_path,
        pdf_path=pdf_path,
        png_path=png_path,
        representative_points=representative_points,
    )

    if args.save_pareto_lines:
        save_pareto_line_figures(
            network=network,
            objective=objective,
            results=results,
            output_dir=output_dir / "pareto_line_figures",
        )

    print("\n=== Fitxers generats ===")
    print("Nombre de lambdas:", len(lambda_values))
    print("Nombre de seeds:", len(seeds))
    print("Total execucions:", len(results))
    print("Punts Pareto:", sum(1 for item in results if item.is_pareto))
    print("Directori de sortida:", output_dir)
    print("CSV tots els punts:", all_csv_path)
    print("CSV punts Pareto:", pareto_csv_path)
    print("Resum:", summary_path)
    print("Figura PDF:", pdf_path)
    print("Figura PNG:", png_path)


if __name__ == "__main__":
    main()
