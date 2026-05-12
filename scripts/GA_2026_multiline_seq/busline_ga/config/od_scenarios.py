from __future__ import annotations

import os
from typing import Dict, List

from busline_ga.config.project_paths import BUS_NETWORK_DIR, SCRIPT_RESULTS_DIR


OD_SCENARIO_FILENAMES: Dict[str, str] = {
    "base": "od_base.csv",
    "one_center": "od_one_center.csv",
    "feeder_core": "od_feeder_core.csv",
    "two_centers": "od_two_centers.csv",
    "center_periphery": "od_center_periphery.csv",
}


def get_registered_od_cases() -> List[str]:
    return list(OD_SCENARIO_FILENAMES.keys())


def resolve_od_case_path(project_dir: str, od_case: str) -> str:
    if od_case not in OD_SCENARIO_FILENAMES:
        valid_cases = ", ".join(get_registered_od_cases())
        raise ValueError(f"OD case desconegut: {od_case}. Casos vàlids: {valid_cases}")

    scenario_dir = os.path.join(str(BUS_NETWORK_DIR), "od_scenarios")
    expected_path = os.path.join(scenario_dir, OD_SCENARIO_FILENAMES[od_case])

    if os.path.exists(expected_path):
        return expected_path

    fallback_path = os.path.join(str(BUS_NETWORK_DIR), "od_matrix_fixed.csv")
    if od_case == "base" and os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        f"No s'ha trobat el fitxer OD per al cas '{od_case}'. "
        f"Ruta esperada: {expected_path}"
    )


def get_od_case_results_dir(base_dir: str, od_case: str) -> str:
    results_dir = os.path.join(str(SCRIPT_RESULTS_DIR), "od_experiments", od_case)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
