#python -B -m busline_ga.experiments.run_initialization_comparison --od-case one_center --map-case base --lambda 0.5 --seeds 42,43,44,45 --mutation-prob 0.30s
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, RESULTS_DIR
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.network_model import Node, load_network_model
from busline_ga.core.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    prepare_structural_normalization,
)
from busline_ga.core.objective_function import ObjectiveFunction


# =============================================================================
# Estil gràfic del TFG
# =============================================================================

GRANATE = "#7A0019"
GUIDED_COLOR = GRANATE
RANDOM_COLOR = "#4E79A7"
GRAY_DARK = "0.25"
GRAY = "0.55"
GRID_COLOR = "0.88"

SHOW_SEED_CURVES = False
SHOW_MINMAX_BAND = True
SEED_LINE_ALPHA = 0.08
MINMAX_BAND_ALPHA = 0.08
SEED_LINE_WIDTH = 0.45
MEAN_LINE_WIDTH = 1.6

MODE_ORDER = ["random", "guided"]

MODE_LABELS = {
    "random": "Inicialització aleatòria",
    "guided": "Inicialització guiada",
}

MODE_COLORS = {
    "random": RANDOM_COLOR,
    "guided": GUIDED_COLOR,
}


# =============================================================================
# Configuracions d'inicialització
# =============================================================================

INIT_CONFIGS = {
    "random": {
        "init_ratio_service": 0.0,
        "init_ratio_demand": 0.0,
        "init_ratio_spatial": 0.0,
        "init_ratio_hybrid": 0.0,
        "init_ratio_random": 1.0,
        "inject_best_service_candidate": False,
    },
    "guided": {
        "init_ratio_service": 0.30,
        "init_ratio_demand": 0.25,
        "init_ratio_spatial": 0.15,
        "init_ratio_hybrid": 0.15,
        "init_ratio_random": 0.15,
        "inject_best_service_candidate": True,
    },
}


# =============================================================================
# Utilitats generals
# =============================================================================

def apply_tfg_plot_style() -> None:
    """Defineix un estil net i coherent amb les figures del TFG."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
        }
    )


def parse_seed_list(text: str) -> List[int]:
    """Converteix una cadena tipus '42,43,44,45' en una llista d'enters."""
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]

    if not seeds:
        raise ValueError("Cal indicar almenys una seed.")

    return seeds


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Guarda una llista de diccionaris en format CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Càrrega del model
# =============================================================================

def load_experiment_context(od_case: str, map_case: str):
    """Carrega la xarxa, la matriu OD i la funció objectiu."""
    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        map_case,
        verbose=True,
    )

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
    objective.precompute_edge_service()

    return network, objective, road_nodes_path, road_edges_path, od_matrix_path


def create_optimizer(
    mode: str,
    network,
    objective: ObjectiveFunction,
    line_length: int,
    population_size: int,
    seed: int,
) -> GeneticOptimizer:
    """Crea el GA amb la inicialització corresponent."""
    if mode not in INIT_CONFIGS:
        raise ValueError(f"Mode d'inicialització desconegut: {mode}")

    optimizer = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=line_length,
        population_size=population_size,
        seed=seed,
        nearest_neighbors_k=8,
        init_top_k=8,
        **INIT_CONFIGS[mode],
    )

    return optimizer


# =============================================================================
# Execució del GA per a un mode i una seed
# =============================================================================

def run_single_mode_seed(
    mode: str,
    seed: int,
    network,
    objective: ObjectiveFunction,
    lambda_: float,
    generations: int,
    population_size: int,
    line_length: int,
    n_elite: int,
    tournament_size: int,
    mutation_prob: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Executa el GA per a una inicialització i una seed concreta."""
    ga = create_optimizer(
        mode=mode,
        network=network,
        objective=objective,
        line_length=line_length,
        population_size=population_size,
        seed=seed,
    )

    ga.generate_initial_population()

    history_rows: List[Dict[str, object]] = []
    global_best_individual: List[Node] = []
    global_best_fitness = float("-inf")
    global_best_evaluation = None

    for generation in range(generations):
        scored_population = ga.evaluate_population(lambda_)
        generation_best = max(scored_population, key=lambda item: item[1].fitness)

        generation_fitness_values = [
            evaluation.fitness for _, evaluation in scored_population
        ]

        best_individual = list(generation_best[0])
        best_evaluation = generation_best[1]

        if best_evaluation.fitness > global_best_fitness:
            global_best_individual = list(best_individual)
            global_best_fitness = best_evaluation.fitness
            global_best_evaluation = best_evaluation

        history_rows.append(
            {
                "generation": generation,
                "initialization_mode": mode,
                "seed": seed,
                "best_fitness": best_evaluation.fitness,
                "best_cost": best_evaluation.cost,
                "best_cost_norm": best_evaluation.cost_norm,
                "best_passenger_service": best_evaluation.passenger_service,
                "best_passenger_service_norm": best_evaluation.passenger_service_norm,
                "mean_fitness": mean(generation_fitness_values),
                "best_individual": json.dumps(best_individual),
            }
        )

        if generation < generations - 1:
            ga.evolve_one_generation(
                lambda_=lambda_,
                n_elite=n_elite,
                tournament_size=tournament_size,
                mutation_prob=mutation_prob,
                weight_adjacent=0.20,
                weight_reverse=0.15,
                weight_neighbor=0.25,
                weight_service_replace=0.40,
            )

    if global_best_evaluation is None:
        raise RuntimeError("No s'ha pogut obtenir cap avaluació del GA.")

    summary_row = {
        "initialization_mode": mode,
        "seed": seed,
        "final_best_fitness": global_best_fitness,
        "final_cost": global_best_evaluation.cost,
        "final_cost_norm": global_best_evaluation.cost_norm,
        "final_passenger_service": global_best_evaluation.passenger_service,
        "final_passenger_service_norm": global_best_evaluation.passenger_service_norm,
        "final_best_individual": json.dumps(global_best_individual),
        "line_edges_count": len(global_best_evaluation.line_edges),
    }

    return history_rows, summary_row


# =============================================================================
# Resums
# =============================================================================

def build_summary_by_mode(
    summary_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Calcula el resum mitjà per tipus d'inicialització."""
    summary_df = pd.DataFrame(summary_rows)
    grouped_rows: List[Dict[str, object]] = []

    for mode in MODE_ORDER:
        mode_df = summary_df[summary_df["initialization_mode"] == mode]

        if not mode_df.empty:
            grouped_rows.append(
                {
                    "initialization_mode": mode,
                    "runs": int(len(mode_df)),
                    "mean_final_best_fitness": float(
                        mode_df["final_best_fitness"].mean()
                    ),
                    "std_final_best_fitness": float(
                        mode_df["final_best_fitness"].std(ddof=0)
                    ),
                    "mean_final_cost_norm": float(mode_df["final_cost_norm"].mean()),
                    "std_final_cost_norm": float(
                        mode_df["final_cost_norm"].std(ddof=0)
                    ),
                    "mean_final_passenger_service_norm": float(
                        mode_df["final_passenger_service_norm"].mean()
                    ),
                    "std_final_passenger_service_norm": float(
                        mode_df["final_passenger_service_norm"].std(ddof=0)
                    ),
                }
            )

    return grouped_rows


# =============================================================================
# Figures
# =============================================================================

def configure_axis(ax, ylabel: str) -> None:
    """Aplica l'estil comú als eixos de les figures del TFG."""
    ax.set_xlabel("Generació", fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=12.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

    # Sense quadrícula de fons per mantenir una figura més neta.
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)

    ax.tick_params(axis="both", colors=GRAY_DARK, labelsize=10.5)


def set_y_limits_with_margin(ax, values: List[float]) -> None:
    """Afegeix un marge vertical perquè les corbes no quedin enganxades als límits."""
    if values:
        data_min = min(values)
        data_max = max(values)
        span = data_max - data_min
        margin = 0.03 if span == 0 else span * 0.08
        ax.set_ylim(data_min - margin, data_max + margin)


def plot_metric(
    history_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_base: Path,
) -> Tuple[Path, Path]:
    """
    Genera una figura comparant l'evolució d'una mètrica.

    La línia principal representa la mitjana de les diferents seeds.
    La banda ombrejada representa el rang mínim-màxim entre execucions.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    plotted_values: List[float] = []

    for mode in MODE_ORDER:
        mode_df = history_df[history_df["initialization_mode"] == mode]

        if mode_df.empty:
            continue

        pivot = (
            mode_df
            .pivot(index="generation", columns="seed", values=metric)
            .sort_index()
        )

        generations = pivot.index.to_numpy()
        mean_values = pivot.mean(axis=1).to_numpy()
        min_values = pivot.min(axis=1).to_numpy()
        max_values = pivot.max(axis=1).to_numpy()

        color = MODE_COLORS[mode]
        label = MODE_LABELS[mode]

        plotted_values.extend(min_values.tolist())
        plotted_values.extend(max_values.tolist())

        # Corbes individuals de cada seed, desactivades per defecte.
        if SHOW_SEED_CURVES:
            for seed_column in pivot.columns:
                ax.plot(
                    generations,
                    pivot[seed_column].to_numpy(),
                    color=color,
                    alpha=SEED_LINE_ALPHA,
                    linewidth=SEED_LINE_WIDTH,
                    zorder=1,
                )

        # Banda min-max molt suau per mostrar la variabilitat entre seeds.
        if SHOW_MINMAX_BAND:
            ax.fill_between(
                generations,
                min_values,
                max_values,
                color=color,
                alpha=MINMAX_BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )

        # Corba mitjana principal.
        ax.plot(
            generations,
            mean_values,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            label=label,
            zorder=3,
        )

    configure_axis(ax, ylabel)
    set_y_limits_with_margin(ax, plotted_values)

    ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)

    plt.close(fig)

    return pdf_path, png_path


def save_figures(
    history_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, str]:
    """Genera les figures finals de fitness, servei i cost."""
    apply_tfg_plot_style()

    history_df = pd.DataFrame(history_rows)
    figure_paths: Dict[str, str] = {}

    fitness_pdf, fitness_png = plot_metric(
        history_df=history_df,
        metric="best_fitness",
        ylabel=r"Millor valor de $H$",
        output_base=output_dir / "initialization_comparison_fitness",
    )

    service_pdf, service_png = plot_metric(
        history_df=history_df,
        metric="best_passenger_service_norm",
        ylabel=r"Servei normalitzat $P_{\mathrm{norm}}(E_\ell)$",
        output_base=output_dir / "initialization_comparison_service",
    )

    cost_pdf, cost_png = plot_metric(
        history_df=history_df,
        metric="best_cost_norm",
        ylabel=r"Cost normalitzat $C_{\mathrm{norm}}(E_\ell)$",
        output_base=output_dir / "initialization_comparison_cost",
    )

    figure_paths["fitness_pdf"] = str(fitness_pdf)
    figure_paths["fitness_png"] = str(fitness_png)
    figure_paths["service_pdf"] = str(service_pdf)
    figure_paths["service_png"] = str(service_png)
    figure_paths["cost_pdf"] = str(cost_pdf)
    figure_paths["cost_png"] = str(cost_png)

    return figure_paths


# =============================================================================
# Arguments i execució principal
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara la inicialització aleatòria i la guiada del GA d'una línia."
        )
    )

    parser.add_argument("--od-case", default="one_center")
    parser.add_argument("--map-case", default="base")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--line-length", type=int, default=6)
    parser.add_argument("--mutation-prob", type=float, default=0.30)
    parser.add_argument("--n-elite", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--output-dir", default=None)

    return parser.parse_args()


def print_experiment_header(
    args: argparse.Namespace,
    seeds: List[int],
    output_dir: Path,
) -> None:
    print("\n=== Comparació d'inicialització ===")
    print("OD:", args.od_case)
    print("Mapa:", args.map_case)
    print("lambda:", args.lambda_)
    print("seeds:", seeds)
    print("generacions:", args.generations)
    print("mida població:", args.population_size)
    print("longitud línia:", args.line_length)
    print("probabilitat mutació:", args.mutation_prob)
    print("directori de sortida:", output_dir)
    print("Només canvia el mode d'inicialització: random vs guided")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else RESULTS_DIR / "initialization_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    network, objective, road_nodes_path, road_edges_path, od_matrix_path = (
        load_experiment_context(
            od_case=args.od_case,
            map_case=args.map_case,
        )
    )

    normalization_bounds = prepare_structural_normalization(
        objective=objective,
        line_length=args.line_length,
    )

    all_history_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    print_experiment_header(args, seeds, output_dir)

    for mode in MODE_ORDER:
        for seed in seeds:
            print(f"\n[RUN] mode={mode} seed={seed}")

            history_rows, summary_row = run_single_mode_seed(
                mode=mode,
                seed=seed,
                network=network,
                objective=objective,
                lambda_=args.lambda_,
                generations=args.generations,
                population_size=args.population_size,
                line_length=args.line_length,
                n_elite=args.n_elite,
                tournament_size=args.tournament_size,
                mutation_prob=args.mutation_prob,
            )

            all_history_rows.extend(history_rows)
            summary_rows.append(summary_row)

            print(
                "  final_best_fitness =",
                f"{summary_row['final_best_fitness']:.6f}",
                "| service_norm =",
                f"{summary_row['final_passenger_service_norm']:.6f}",
                "| cost_norm =",
                f"{summary_row['final_cost_norm']:.6f}",
            )

    history_csv_path = output_dir / "initialization_comparison_history.csv"
    summary_csv_path = output_dir / "initialization_comparison_summary.csv"
    summary_by_mode_csv_path = output_dir / "initialization_comparison_summary_by_mode.csv"
    summary_json_path = output_dir / "initialization_comparison_summary.json"

    write_csv(all_history_rows, history_csv_path)
    write_csv(summary_rows, summary_csv_path)
    write_csv(build_summary_by_mode(summary_rows), summary_by_mode_csv_path)

    figure_paths = save_figures(all_history_rows, output_dir)

    summary_json = {
        "od_case": args.od_case,
        "map_case": args.map_case,
        "od_matrix_path": od_matrix_path,
        "road_nodes_path": road_nodes_path,
        "road_edges_path": road_edges_path,
        "lambda": args.lambda_,
        "seeds": seeds,
        "generations": args.generations,
        "population_size": args.population_size,
        "line_length": args.line_length,
        "mutation_prob": args.mutation_prob,
        "n_elite": args.n_elite,
        "tournament_size": args.tournament_size,
        "normalization_method": DEFAULT_NORMALIZATION_METHOD,
        "normalization_cost_min": normalization_bounds.cost_min,
        "normalization_cost_max": normalization_bounds.cost_max,
        "normalization_service_min": normalization_bounds.service_min,
        "normalization_service_max": normalization_bounds.service_max,
        "initialization_modes": INIT_CONFIGS,
        "outputs": {
            "history_csv": str(history_csv_path),
            "summary_csv": str(summary_csv_path),
            "summary_by_mode_csv": str(summary_by_mode_csv_path),
            **figure_paths,
        },
    }

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2, ensure_ascii=False)

    print("\n=== Outputs ===")
    print("Historial CSV:", history_csv_path)
    print("Resum CSV:", summary_csv_path)
    print("Resum per mode CSV:", summary_by_mode_csv_path)
    print("Resum JSON:", summary_json_path)
    print("Fitness PDF:", figure_paths["fitness_pdf"])
    print("Fitness PNG:", figure_paths["fitness_png"])
    print("Directori de sortida:", output_dir)


if __name__ == "__main__":
    main()