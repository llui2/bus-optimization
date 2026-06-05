from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent] + list(current.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("No s'ha pogut detectar PROJECT_ROOT: falta el directori 'data'.")


def _find_ga_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent] + list(current.parents):
        if (candidate / "busline_ga").is_dir():
            return candidate
    raise RuntimeError("No s'ha pogut detectar GA_ROOT: falta el directori 'busline_ga'.")


PROJECT_ROOT = _find_project_root()
GA_ROOT = _find_ga_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = GA_ROOT / "results"
BUS_NETWORK_DIR = DATA_DIR / "bus_network"
ROAD_NETWORK_DIR = DATA_DIR / "road_network"
OD_SCENARIOS_DIR = BUS_NETWORK_DIR / "od_scenarios"

# Aliases de compatibilitat interna
SCRIPT_DIR = GA_ROOT
SCRIPT_RESULTS_DIR = RESULTS_DIR
