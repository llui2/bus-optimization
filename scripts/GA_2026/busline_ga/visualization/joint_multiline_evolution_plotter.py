from __future__ import annotations

import csv
import os
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt


JOINT_EVOLUTION_FIELDS = [
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
    "unique_edge_cost",
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
    "best_system",
]


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _title(title_context: Dict[str, Any], label: str) -> str:
    title = (
        f"{label} | od={title_context.get('od_case', '')} | "
        f"map={title_context.get('map_case', '')} | "
        f"n_lines={title_context.get('n_lines', '')} | "
        f"lambda={title_context.get('lambda', '')} | "
        f"comp={title_context.get('compactness_penalty', '')}"
    )
    return title


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(_filesystem_path(path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=JOINT_EVOLUTION_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in JOINT_EVOLUTION_FIELDS})


def _save_figure(fig, pdf_path: str, png_path: str) -> None:
    fig.savefig(_filesystem_path(pdf_path), bbox_inches="tight")
    fig.savefig(_filesystem_path(png_path), bbox_inches="tight", dpi=220)
    plt.close(fig)


def _plot_system_h(rows: Sequence[Dict[str, Any]], output_dir: str, title_context: Dict[str, Any]) -> Dict[str, str]:
    generations = [int(row["generation"]) for row in rows]
    fitness = [float(row["fitness"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(generations, fitness, linewidth=1.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("System fitness")
    ax.set_title(_title(title_context, "Joint GA system H evolution"))
    ax.grid(True, alpha=0.25)
    pdf_path = os.path.join(output_dir, "evolution_system_H.pdf")
    png_path = os.path.join(output_dir, "evolution_system_H.png")
    _save_figure(fig, pdf_path, png_path)
    return {"evolution_system_H_pdf": pdf_path, "evolution_system_H_png": png_path}


def _plot_system_pc(rows: Sequence[Dict[str, Any]], output_dir: str, title_context: Dict[str, Any]) -> Dict[str, str]:
    generations = [int(row["generation"]) for row in rows]
    service_norm = [float(row["system_service_norm"]) for row in rows]
    cost_norm = [float(row["system_cost_norm"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(generations, service_norm, linewidth=1.9, label="system_service_norm")
    ax.plot(generations, cost_norm, linewidth=1.9, label="system_cost_norm")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized value")
    ax.set_title(_title(title_context, "Joint GA system P/C evolution"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    pdf_path = os.path.join(output_dir, "evolution_system_PC.pdf")
    png_path = os.path.join(output_dir, "evolution_system_PC.png")
    _save_figure(fig, pdf_path, png_path)
    return {"evolution_system_PC_pdf": pdf_path, "evolution_system_PC_png": png_path}


def _plot_shared_overlap(
    rows: Sequence[Dict[str, Any]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    generations = [int(row["generation"]) for row in rows]
    shared_stop_ratio = [float(row["shared_stop_ratio"]) for row in rows]
    shared_edge_ratio = [float(row["shared_edge_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(generations, shared_stop_ratio, linewidth=1.9, label="shared_stop_ratio")
    ax.plot(generations, shared_edge_ratio, linewidth=1.9, label="shared_edge_ratio")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Ratio")
    ax.set_title(_title(title_context, "Joint GA shared overlap evolution"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    pdf_path = os.path.join(output_dir, "evolution_shared_overlap.pdf")
    png_path = os.path.join(output_dir, "evolution_shared_overlap.png")
    _save_figure(fig, pdf_path, png_path)
    return {"evolution_shared_overlap_pdf": pdf_path, "evolution_shared_overlap_png": png_path}


def save_joint_multiline_evolution_outputs(
    evolution_history: Sequence[Dict[str, Any]],
    output_dir: str,
    title_context: Dict[str, Any],
) -> Dict[str, str]:
    os.makedirs(_filesystem_path(output_dir), exist_ok=True)
    evolution_csv = os.path.join(output_dir, "joint_evolution_history.csv")
    _write_csv(evolution_csv, evolution_history)
    output_paths = {"joint_evolution_history_csv": evolution_csv}

    if evolution_history:
        output_paths.update(_plot_system_h(evolution_history, output_dir, title_context))
        output_paths.update(_plot_system_pc(evolution_history, output_dir, title_context))
        output_paths.update(_plot_shared_overlap(evolution_history, output_dir, title_context))

    return output_paths
