from __future__ import annotations

from typing import Tuple

from busline_ga.core.objective_function import NormalizationBounds, ObjectiveFunction


DEFAULT_NORMALIZATION_METHOD = "structural_cost_and_service_bounds"

def prepare_structural_normalization(
    objective: ObjectiveFunction,
    line_length: int,
) -> NormalizationBounds:
    bounds = objective.estimate_structural_bounds_for_fixed_length(
        line_length=line_length,
    )
    objective.set_normalization_bounds(bounds)
    return bounds

