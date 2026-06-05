from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


EVOLUTION_FIELDS = [
    "line_index",
    "generation",
    "best_individual",
    "fitness",
    "cost",
    "cost_norm",
    "passenger_service",
    "service_norm",
]

SYSTEM_EVOLUTION_FIELDS = [
    "line_index_added",
    "n_lines_so_far",
    "system_fitness",
    "adjusted_passenger_service",
    "raw_line_passenger_service_sum",
    "naive_edge_passenger_service",
    "unique_edge_passenger_service",
    "total_route_cost",
    "unique_edge_cost",
    "system_cost_norm",
    "system_service_norm",
    "shared_stop_ratio",
    "shared_edge_ratio",
]


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(_filesystem_path(path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _group_by_line(rows: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[int(row["line_index"])].append(row)

    for line_rows in grouped.values():
        line_rows.sort(key=lambda item: int(item["generation"]))

    return dict(grouped)


def _title(title_context: Dict[str, Any], label: str) -> str:
    title = (
        f"{label} | od={title_context.get('od_case', '')} | "
        f"map={title_context.get('map_case', '')} | "
        f"n_lines={title_context.get('n_lines', '')} | "
        f"lambda={title_context.get('lambda', '')} | "
        f"shared_stop_penalty={title_context.get('shared_stop_penalty', '')}"
    )
    return title


def _save_figure(fig, pdf_path: str, png_path: str) -> None:
    fig.savefig(_filesystem_path(pdf_path), bbox_inches="tight")
    fig.savefig(_filesystem_path(png_path), bbox_inches="tight", dpi=220)
    plt.close(fig)


def _plot_h_per_line(
    grouped: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for line_index, rows in sorted(grouped.items()):
        generations = [int(row["generation"]) for row in rows]
        fitness = [float(row["fitness"]) for row in rows]
        ax.plot(generations, fitness, linewidth=1.8, label=f"Line {line_index}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("H(k), best fitness")
    ax.set_title(_title(title_context, "Per-line GA fitness evolution"))
    ax.grid(True, alpha=0.25)
    ax.legend()

    pdf_path = os.path.join(output_dir, "evolution_H_per_line.pdf")
    png_path = os.path.join(output_dir, "evolution_H_per_line.png")
    _save_figure(fig, pdf_path, png_path)
    return {"evolution_H_per_line_pdf": pdf_path, "evolution_H_per_line_png": png_path}


def _plot_pnorm_cnorm_per_line(
    grouped: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for line_index, rows in sorted(grouped.items()):
        generations = [int(row["generation"]) for row in rows]
        service_norm = [float(row["service_norm"]) for row in rows]
        cost_norm = [float(row["cost_norm"]) for row in rows]
        axes[0].plot(generations, service_norm, linewidth=1.7, label=f"Line {line_index}")
        axes[1].plot(generations, cost_norm, linewidth=1.7, label=f"Line {line_index}")

    axes[0].set_ylabel("P_norm(k)")
    axes[1].set_ylabel("C_norm(k)")
    axes[1].set_xlabel("Generation")
    axes[0].set_title(_title(title_context, "Per-line normalized service/cost evolution"))

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()

    pdf_path = os.path.join(output_dir, "evolution_Pnorm_Cnorm_per_line.pdf")
    png_path = os.path.join(output_dir, "evolution_Pnorm_Cnorm_per_line.png")
    _save_figure(fig, pdf_path, png_path)

    output_paths = {
        "evolution_Pnorm_Cnorm_per_line_pdf": pdf_path,
        "evolution_Pnorm_Cnorm_per_line_png": png_path,
    }

    if len(grouped) > 4:
        for line_index, rows in sorted(grouped.items()):
            fig_line, ax_line = plt.subplots(figsize=(8, 4.8))
            generations = [int(row["generation"]) for row in rows]
            service_norm = [float(row["service_norm"]) for row in rows]
            cost_norm = [float(row["cost_norm"]) for row in rows]
            ax_line.plot(generations, service_norm, linewidth=1.8, label="P_norm(k)")
            ax_line.plot(generations, cost_norm, linewidth=1.8, label="C_norm(k)")
            ax_line.set_xlabel("Generation")
            ax_line.set_ylabel("Normalized value")
            ax_line.set_title(_title(title_context, f"Line {line_index} P_norm/C_norm"))
            ax_line.grid(True, alpha=0.25)
            ax_line.legend()
            line_pdf = os.path.join(output_dir, f"line_{line_index:02d}_Pnorm_Cnorm.pdf")
            line_png = os.path.join(output_dir, f"line_{line_index:02d}_Pnorm_Cnorm.png")
            _save_figure(fig_line, line_pdf, line_png)
            output_paths[f"line_{line_index:02d}_Pnorm_Cnorm_pdf"] = line_pdf
            output_paths[f"line_{line_index:02d}_Pnorm_Cnorm_png"] = line_png

    return output_paths


def _plot_system_fitness(
    rows: Sequence[Dict[str, Any]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    sorted_rows = sorted(rows, key=lambda item: int(item["n_lines_so_far"]))
    x_values = [int(row["n_lines_so_far"]) for row in sorted_rows]
    y_values = [float(row["system_fitness"]) for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x_values, y_values, marker="o", linewidth=2.0)
    ax.set_xlabel("Number of lines added")
    ax.set_ylabel("System fitness")
    ax.set_title(_title(title_context, "System fitness by added line"))
    ax.grid(True, alpha=0.25)
    ax.set_xticks(x_values)
    pdf_path = os.path.join(output_dir, "evolution_system_fitness_by_line.pdf")
    png_path = os.path.join(output_dir, "evolution_system_fitness_by_line.png")
    _save_figure(fig, pdf_path, png_path)
    return {
        "evolution_system_fitness_by_line_pdf": pdf_path,
        "evolution_system_fitness_by_line_png": png_path,
    }


def _plot_system_pc(
    rows: Sequence[Dict[str, Any]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    sorted_rows = sorted(rows, key=lambda item: int(item["n_lines_so_far"]))
    x_values = [int(row["n_lines_so_far"]) for row in sorted_rows]
    service_norm = [float(row["system_service_norm"]) for row in sorted_rows]
    cost_norm = [float(row["system_cost_norm"]) for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x_values, service_norm, marker="o", linewidth=2.0, label="system_service_norm")
    ax.plot(x_values, cost_norm, marker="o", linewidth=2.0, label="system_cost_norm")
    ax.set_xlabel("Number of lines added")
    ax.set_ylabel("Normalized value")
    ax.set_title(_title(title_context, "System P_norm/C_norm by added line"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xticks(x_values)
    pdf_path = os.path.join(output_dir, "evolution_system_PC_by_line.pdf")
    png_path = os.path.join(output_dir, "evolution_system_PC_by_line.png")
    _save_figure(fig, pdf_path, png_path)
    return {
        "evolution_system_PC_by_line_pdf": pdf_path,
        "evolution_system_PC_by_line_png": png_path,
    }


def save_multiline_evolution_outputs(
    evolution_history: Sequence[Dict[str, Any]],
    system_evolution_history: Sequence[Dict[str, Any]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    os.makedirs(_filesystem_path(output_dir), exist_ok=True)

    evolution_csv = os.path.join(output_dir, "multiline_evolution_history.csv")
    system_csv = os.path.join(output_dir, "multiline_system_evolution_history.csv")
    _write_csv(evolution_csv, evolution_history, EVOLUTION_FIELDS)
    _write_csv(system_csv, system_evolution_history, SYSTEM_EVOLUTION_FIELDS)

    grouped = _group_by_line(evolution_history)
    output_paths = {
        "multiline_evolution_history_csv": evolution_csv,
        "multiline_system_evolution_history_csv": system_csv,
    }

    if grouped:
        output_paths.update(_plot_h_per_line(grouped, output_dir, title_context))
        output_paths.update(_plot_pnorm_cnorm_per_line(grouped, output_dir, title_context))

    if system_evolution_history:
        output_paths.update(_plot_system_fitness(system_evolution_history, output_dir, title_context))
        output_paths.update(_plot_system_pc(system_evolution_history, output_dir, title_context))

    return output_paths
