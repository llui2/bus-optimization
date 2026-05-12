from __future__ import annotations

import os
from typing import Dict, List, Tuple

from busline_ga.config.project_paths import PROJECT_ROOT, SCRIPT_RESULTS_DIR
from busline_ga.generators.generate_road_variants import ensure_road_variants


MAP_CASES: Dict[str, str] = {
    "base": "base",
    "light_fill": "light_fill",
    "dense_fill": "dense_fill",
}


def get_registered_map_cases() -> List[str]:
    return list(MAP_CASES.keys())


def resolve_map_case_paths(
    project_dir: str,
    map_case: str,
    verbose: bool = True,
) -> Tuple[str, str]:
    if map_case not in MAP_CASES:
        valid_cases = ", ".join(get_registered_map_cases())
        raise ValueError(f"Map case desconegut: {map_case}. Casos vàlids: {valid_cases}")

    variant_info = ensure_road_variants(str(PROJECT_ROOT), verbose=verbose)
    selected = variant_info[map_case]
    return str(selected["nodes_path"]), str(selected["edges_path"])


def get_road_experiment_results_dir(
    base_dir: str,
    od_case: str,
    map_case: str,
) -> str:
    results_dir = os.path.join(
        str(SCRIPT_RESULTS_DIR),
        "road_experiments",
        f"od_{od_case}",
        f"map_{map_case}",
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
