from __future__ import annotations

import itertools
import math
import random
import networkx as nx

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from busline_ga.visualization.ga_snapshot_plotter import (
    ImprovementSnapshotData,
    get_snapshot_pdf_path,
    save_improvement_snapshot,
)
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.objective_function import EvaluationResult, ObjectiveFunction

@dataclass
class GAHistory:
    best_scores: List[float] = field(default_factory=list)
    mean_scores: List[float] = field(default_factory=list)
    best_costs: List[float] = field(default_factory=list)
    best_services: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        result = {
            "best_scores": self.best_scores,
            "mean_scores": self.mean_scores,
            "best_costs": self.best_costs,
            "best_services": self.best_services,
        }
        return result

class GeneticOptimizer:
    def __init__(
        self,
        network: NetworkModel,
        objective_function: ObjectiveFunction,
        line_length: int,
        population_size: int,
        seed: int = 42,
        nearest_neighbors_k: int = 5,
        init_ratio_service: float = 0.30,
        init_ratio_demand: float = 0.25,
        init_ratio_spatial: float = 0.15,
        init_ratio_hybrid: float = 0.15,
        init_ratio_random: float = 0.15,
        seed_top_fraction: float = 0.20,
        init_top_k: int = 5,
        hybrid_alpha: float = 0.70,
        diversity_overlap_threshold: float = 0.80,
        diversity_max_similar: int = 2,
        initialization_max_attempts: int = 25,
        inject_best_service_candidate: bool = True,
    ) -> None:
        self.network = network
        self.objective_function = objective_function
        self.line_length = line_length
        self.population_size = population_size
        self.nearest_neighbors_k = nearest_neighbors_k
        self.init_ratio_service = init_ratio_service
        self.init_ratio_demand = init_ratio_demand
        self.init_ratio_spatial = init_ratio_spatial
        self.init_ratio_hybrid = init_ratio_hybrid
        self.init_ratio_random = init_ratio_random
        self.seed_top_fraction = seed_top_fraction
        self.init_top_k = init_top_k
        self.hybrid_alpha = hybrid_alpha
        self.diversity_overlap_threshold = diversity_overlap_threshold
        self.diversity_max_similar = diversity_max_similar
        self.initialization_max_attempts = initialization_max_attempts
        self.inject_best_service_candidate = inject_best_service_candidate

        self.rng = random.Random(seed)
        self.population: List[List[Node]] = []
        self.population_records: List[Dict[str, Any]] = []
        self.history = GAHistory()
        self.best_service_candidate: List[Node] = []
        self.best_service_candidate_ordered: List[Node] = []
        self.best_service_candidate_passenger_service: float = 0.0

        self.stop_costs: Dict[Tuple[Node, Node], float] = {}
        self.stop_demand_scores: Dict[Node, float] = {}
        self.nearest_neighbors: Dict[Node, List[Node]] = {}
        self.demand_weighted_center: Tuple[float, float] = (0.0, 0.0)
        self.stops_by_demand: List[Node] = []
        self.stops_by_centrality: List[Node] = []
        self.stops_by_hybrid_seed_score: List[Node] = []
        self.normalized_seed_demand: Dict[Node, float] = {}
        self.normalized_seed_proximity: Dict[Node, float] = {}
        self.hybrid_seed_scores: Dict[Node, float] = {}
        self.top_service_seed_pairs: List[Tuple[Node, Node, float]] = []

        self._precompute_stop_metadata()

    def _route_order_cost(self, individual: List[Node]) -> float:
        total_cost = 0.0

        for index in range(len(individual) - 1):
            stop_a = individual[index]
            stop_b = individual[index + 1]
            total_cost += self.stop_costs.get((stop_a, stop_b), math.inf)

        return total_cost

    def _is_valid_individual(self, individual: List[Node]) -> bool:
        valid_bus_stops = set(self.network.bus_stops)
        is_valid = (
            len(individual) == self.line_length
            and len(set(individual)) == len(individual)
            and all(stop in valid_bus_stops for stop in individual)
        )
        return is_valid

    def repair_order_min_cost(self, individual: List[Node]) -> List[Node]:
        best_order = list(individual)
        best_cost = self._route_order_cost(best_order)

        if len(individual) > 1 and self._is_valid_individual(individual):
            for permutation in itertools.permutations(individual):
                candidate = list(permutation)
                candidate_cost = self._route_order_cost(candidate)

                if candidate_cost < best_cost:
                    best_order = candidate
                    best_cost = candidate_cost

        result = best_order
        return result

    def compare_repaired_order(self, individual: List[Node]) -> Dict[str, object]:
        original = list(individual)
        repaired = self.repair_order_min_cost(original)
        original_cost = self._route_order_cost(original)
        repaired_cost = self._route_order_cost(repaired)
        result = {
            "original": original,
            "repaired": repaired,
            "original_order_cost": original_cost,
            "repaired_order_cost": repaired_cost,
            "improvement": original_cost - repaired_cost,
        }
        return result

    def _precompute_stop_metadata(self) -> None:
        bus_stops = list(self.network.bus_stops)

        for stop in bus_stops:
            total_demand = 0.0
            for other_stop in bus_stops:
                if other_stop == stop:
                    continue
                total_demand += float(self.network.get_od_value(stop, other_stop))
                total_demand += float(self.network.get_od_value(other_stop, stop))
            self.stop_demand_scores[stop] = total_demand

        for stop in bus_stops:
            lengths = nx.single_source_dijkstra_path_length(
                self.network.graph,
                stop,
                weight="cost",
            )
            ranked_neighbors: List[Tuple[Node, float]] = []

            for other_stop in bus_stops:
                if other_stop == stop:
                    continue

                cost = float(lengths.get(other_stop, math.inf))
                self.stop_costs[(stop, other_stop)] = cost

                if math.isfinite(cost):
                    ranked_neighbors.append((other_stop, cost))

            ranked_neighbors.sort(key=lambda x: x[1])
            self.nearest_neighbors[stop] = [
                neighbor for neighbor, _ in ranked_neighbors[: self.nearest_neighbors_k]
            ]

        self._precompute_seed_rankings(bus_stops)
        self.top_service_seed_pairs = self._top_od_pairs(limit=max(12, self.init_top_k * 3))

    def _precompute_seed_rankings(self, bus_stops: List[Node]) -> None:
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0

        for stop in bus_stops:
            weight = max(0.0, self.stop_demand_scores.get(stop, 0.0))
            x, y = self.network.positions[stop]
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        if total_weight > 0.0:
            self.demand_weighted_center = (
                weighted_x / total_weight,
                weighted_y / total_weight,
            )
        elif bus_stops:
            mean_x = sum(self.network.positions[stop][0] for stop in bus_stops) / len(bus_stops)
            mean_y = sum(self.network.positions[stop][1] for stop in bus_stops) / len(bus_stops)
            self.demand_weighted_center = (mean_x, mean_y)

        self.stops_by_demand = sorted(
            bus_stops,
            key=lambda stop: self.stop_demand_scores.get(stop, 0.0),
            reverse=True,
        )
        self.stops_by_centrality = sorted(
            bus_stops,
            key=lambda stop: self._distance_to_demand_center(stop),
        )
        (
            self.normalized_seed_demand,
            self.normalized_seed_proximity,
        ) = self._normalized_seed_components(bus_stops)
        self.hybrid_seed_scores = {
            stop: (
                0.5 * self.normalized_seed_demand.get(stop, 0.0)
                + 0.5 * self.normalized_seed_proximity.get(stop, 0.0)
            )
            for stop in bus_stops
        }
        self.stops_by_hybrid_seed_score = sorted(
            bus_stops,
            key=lambda stop: self.hybrid_seed_scores.get(stop, 0.0),
            reverse=True,
        )

    def _weighted_choice(self, items: List[Node], weights: List[float]) -> Node:
        positive_weights = [max(0.0, weight) for weight in weights]
        chosen = items[self.rng.randrange(len(items))]

        if sum(positive_weights) > 0.0:
            chosen = self.rng.choices(items, weights=positive_weights, k=1)[0]

        return chosen

    def _distance_to_demand_center(self, stop: Node) -> float:
        x, y = self.network.positions[stop]
        center_x, center_y = self.demand_weighted_center
        distance = math.dist((x, y), (center_x, center_y))
        return distance

    def _demand_center_proximity(self, stop: Node) -> float:
        proximity = 1.0 / (1.0 + self._distance_to_demand_center(stop))
        return proximity

    def _normalize_node_values(self, values: Dict[Node, float]) -> Dict[Node, float]:
        normalized_values: Dict[Node, float] = {}
        raw_values = list(values.values())
        minimum = min(raw_values) if raw_values else 0.0
        maximum = max(raw_values) if raw_values else 0.0

        for stop, value in values.items():
            normalized_value = 0.0

            if maximum > minimum:
                normalized_value = (value - minimum) / (maximum - minimum)
            elif maximum > 0.0:
                normalized_value = 1.0

            normalized_values[stop] = normalized_value

        return normalized_values

    def _normalized_seed_components(
        self,
        bus_stops: Sequence[Node],
    ) -> Tuple[Dict[Node, float], Dict[Node, float]]:
        demand_values = {
            stop: self.stop_demand_scores.get(stop, 0.0)
            for stop in bus_stops
        }
        proximity_values = {
            stop: self._demand_center_proximity(stop)
            for stop in bus_stops
        }
        normalized_demand = self._normalize_node_values(demand_values)
        normalized_proximity = self._normalize_node_values(proximity_values)
        return normalized_demand, normalized_proximity

    def _normalize_ratios(self, ratios: Sequence[float]) -> List[float]:
        clipped_ratios = [max(0.0, ratio) for ratio in ratios]
        total = sum(clipped_ratios)

        if total <= 0.0:
            normalized = [1.0 / len(clipped_ratios)] * len(clipped_ratios)
        else:
            normalized = [ratio / total for ratio in clipped_ratios]

        return normalized

    def _candidate_pool(self, current_stop: Node, remaining: Sequence[Node]) -> List[Node]:
        local_candidates = [
            stop for stop in self.nearest_neighbors.get(current_stop, []) if stop in remaining
        ]
        candidate_pool = local_candidates if local_candidates else list(remaining)
        return candidate_pool

    def _choose_seed_stop(self, family: str) -> Node:
        ranked_stops = self.stops_by_demand

        if family == "spatial":
            ranked_stops = self.stops_by_centrality
        elif family == "hybrid":
            ranked_stops = self.stops_by_hybrid_seed_score

        top_count = max(1, math.ceil(len(ranked_stops) * self.seed_top_fraction))
        shortlist = ranked_stops[:top_count]
        weights: List[float] = []

        for stop in shortlist:
            demand_component = self.stop_demand_scores.get(stop, 0.0)
            proximity_component = self._demand_center_proximity(stop)

            if family == "spatial":
                weight = proximity_component
            elif family == "hybrid":
                weight = (
                    0.5 * self.normalized_seed_demand.get(stop, 0.0)
                    + 0.5 * self.normalized_seed_proximity.get(stop, 0.0)
                )
            else:
                weight = demand_component

            weights.append(weight + 1e-6)

        seed_stop = self._weighted_choice(shortlist, weights)
        return seed_stop

    def compute_set_passenger_service(self, stops: Sequence[Node]) -> float:
        total_service = 0.0

        for index_a in range(len(stops)):
            stop_a = stops[index_a]
            for index_b in range(index_a + 1, len(stops)):
                stop_b = stops[index_b]
                total_service += float(self.network.get_od_value(stop_a, stop_b))

        return total_service

    def _incremental_service_gain(
        self,
        current_stops: Sequence[Node],
        candidate_stop: Node,
    ) -> float:
        incremental_gain = 0.0

        for existing_stop in current_stops:
            incremental_gain += float(self.network.get_od_value(existing_stop, candidate_stop))

        return incremental_gain

    def _choose_service_seed_stop(self) -> Node:
        ranked_stops = self.stops_by_demand
        top_count = max(1, math.ceil(len(ranked_stops) * self.seed_top_fraction))
        shortlist = ranked_stops[:top_count]
        weights = [
            self.stop_demand_scores.get(stop, 0.0) + 1e-6
            for stop in shortlist
        ]
        seed_stop = self._weighted_choice(shortlist, weights)
        return seed_stop

    def _top_od_pairs(self, limit: int) -> List[Tuple[Node, Node, float]]:
        od_pairs: List[Tuple[Node, Node, float]] = []
        bus_stops = list(self.network.bus_stops)

        for index_a in range(len(bus_stops)):
            stop_a = bus_stops[index_a]
            for index_b in range(index_a + 1, len(bus_stops)):
                stop_b = bus_stops[index_b]
                demand = float(self.network.get_od_value(stop_a, stop_b))

                if demand > 0.0:
                    od_pairs.append((stop_a, stop_b, demand))

        od_pairs.sort(key=lambda item: item[2], reverse=True)
        top_pairs = od_pairs[: max(0, limit)]
        return top_pairs

    def _demand_guided_score(self, current_stop: Node, candidate_stop: Node) -> float:
        demand_score = self.stop_demand_scores.get(candidate_stop, 0.0)
        transition_cost = self.stop_costs.get((current_stop, candidate_stop), math.inf)
        score = 0.0

        if math.isfinite(transition_cost):
            score = demand_score / (1.0 + transition_cost)

        return score

    def _spatial_guided_score(self, current_stop: Node, candidate_stop: Node) -> float:
        transition_cost = self.stop_costs.get((current_stop, candidate_stop), math.inf)
        proximity_score = 0.0

        if math.isfinite(transition_cost):
            proximity_score = 1.0 / (1.0 + transition_cost)

        score = proximity_score
        return score

    def _hybrid_guided_score(
        self,
        current_stop: Node,
        candidate_stop: Node,
        candidate_pool: Sequence[Node],
    ) -> float:
        demand_values = [self.stop_demand_scores.get(stop, 0.0) for stop in candidate_pool]
        proximity_values = [
            1.0 / (1.0 + self.stop_costs.get((current_stop, stop), math.inf))
            if math.isfinite(self.stop_costs.get((current_stop, stop), math.inf))
            else 0.0
            for stop in candidate_pool
        ]
        demand_min = min(demand_values) if demand_values else 0.0
        demand_max = max(demand_values) if demand_values else 0.0
        proximity_min = min(proximity_values) if proximity_values else 0.0
        proximity_max = max(proximity_values) if proximity_values else 0.0
        demand_value = self.stop_demand_scores.get(candidate_stop, 0.0)
        proximity_value = 0.0
        transition_cost = self.stop_costs.get((current_stop, candidate_stop), math.inf)

        if math.isfinite(transition_cost):
            proximity_value = 1.0 / (1.0 + transition_cost)

        demand_norm = 0.0
        proximity_norm = 0.0

        if demand_max > demand_min:
            demand_norm = (demand_value - demand_min) / (demand_max - demand_min)
        elif demand_max > 0.0:
            demand_norm = 1.0

        if proximity_max > proximity_min:
            proximity_norm = (proximity_value - proximity_min) / (proximity_max - proximity_min)
        elif proximity_max > 0.0:
            proximity_norm = 1.0

        score = self.hybrid_alpha * demand_norm + (1.0 - self.hybrid_alpha) * proximity_norm
        return score

    def _choose_next_stop(
        self,
        current_stop: Node,
        remaining: Sequence[Node],
        family: str,
    ) -> Node:
        candidate_pool = self._candidate_pool(current_stop, remaining)
        scored_candidates: List[Tuple[Node, float]] = []

        for candidate_stop in candidate_pool:
            score = 0.0

            if family == "spatial":
                score = self._spatial_guided_score(current_stop, candidate_stop)
            elif family == "hybrid":
                score = self._hybrid_guided_score(current_stop, candidate_stop, candidate_pool)
            else:
                score = self._demand_guided_score(current_stop, candidate_stop)

            scored_candidates.append((candidate_stop, score))

        ranked_candidates = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        top_k = max(1, min(self.init_top_k, len(ranked_candidates)))
        shortlist = ranked_candidates[:top_k]
        chosen_stop = self._weighted_choice(
            [stop for stop, _ in shortlist],
            [score + 1e-6 for _, score in shortlist],
        )
        return chosen_stop

    def _build_demand_guided_individual(self) -> List[Node]:
        individual: List[Node] = []
        remaining = set(self.network.bus_stops)
        first_stop = self._choose_seed_stop("demand")
        individual.append(first_stop)
        remaining.remove(first_stop)

        while len(individual) < self.line_length and remaining:
            current_stop = individual[-1]
            chosen_stop = self._choose_next_stop(current_stop, list(remaining), "demand")
            individual.append(chosen_stop)
            remaining.remove(chosen_stop)

        return individual

    def _build_spatial_guided_individual(self) -> List[Node]:
        individual: List[Node] = []
        remaining = set(self.network.bus_stops)
        first_stop = self._choose_seed_stop("spatial")
        individual.append(first_stop)
        remaining.remove(first_stop)

        while len(individual) < self.line_length and remaining:
            current_stop = individual[-1]
            chosen_stop = self._choose_next_stop(current_stop, list(remaining), "spatial")
            individual.append(chosen_stop)
            remaining.remove(chosen_stop)

        return individual

    def _build_hybrid_guided_individual(self) -> List[Node]:
        individual: List[Node] = []
        remaining = set(self.network.bus_stops)
        first_stop = self._choose_seed_stop("hybrid")
        individual.append(first_stop)
        remaining.remove(first_stop)

        while len(individual) < self.line_length and remaining:
            current_stop = individual[-1]
            chosen_stop = self._choose_next_stop(current_stop, list(remaining), "hybrid")
            individual.append(chosen_stop)
            remaining.remove(chosen_stop)

        return individual

    def _build_random_individual(self) -> List[Node]:
        individual = self.rng.sample(list(self.network.bus_stops), k=self.line_length)
        return individual

    def _build_service_guided_individual(self) -> List[Node]:
        individual: List[Node] = []
        remaining = set(self.network.bus_stops)
        use_pair_seed = bool(self.top_service_seed_pairs) and self.rng.random() < 0.70

        if use_pair_seed:
            pair_top_k = max(1, min(self.init_top_k, len(self.top_service_seed_pairs)))
            shortlist = self.top_service_seed_pairs[:pair_top_k]
            chosen_pair = self._weighted_choice(
                [(stop_a, stop_b) for stop_a, stop_b, _ in shortlist],
                [demand + 1e-6 for _, _, demand in shortlist],
            )
            individual = self._build_service_guided_individual_from_seed_pair(chosen_pair)
        else:
            first_stop = self._choose_service_seed_stop()
            individual.append(first_stop)
            remaining.remove(first_stop)

            while len(individual) < self.line_length and remaining:
                scored_candidates: List[Tuple[Node, float]] = []

                for candidate_stop in remaining:
                    gain = self._incremental_service_gain(individual, candidate_stop)
                    scored_candidates.append((candidate_stop, gain))

                scored_candidates.sort(key=lambda item: item[1], reverse=True)
                top_k = max(1, min(self.init_top_k, len(scored_candidates)))
                shortlist = scored_candidates[:top_k]
                chosen_stop = self._weighted_choice(
                    [stop for stop, _ in shortlist],
                    [gain + 1e-6 for _, gain in shortlist],
                )
                individual.append(chosen_stop)
                remaining.remove(chosen_stop)

        return individual

    def _build_service_guided_individual_from_seed_pair(
        self,
        seed_pair: Tuple[Node, Node],
        stochastic: bool = True,
    ) -> List[Node]:
        selected_stops = list(dict.fromkeys(seed_pair))
        remaining_stops = [
            stop for stop in self.network.bus_stops if stop not in selected_stops
        ]

        while len(selected_stops) < self.line_length and remaining_stops:
            scored_candidates: List[Tuple[Node, float]] = []

            for candidate_stop in remaining_stops:
                gain = self._incremental_service_gain(selected_stops, candidate_stop)
                scored_candidates.append((candidate_stop, gain))

            scored_candidates.sort(key=lambda item: item[1], reverse=True)
            chosen_stop = scored_candidates[0][0]

            if stochastic:
                top_k = max(1, min(self.init_top_k, len(scored_candidates)))
                shortlist = scored_candidates[:top_k]
                chosen_stop = self._weighted_choice(
                    [stop for stop, _ in shortlist],
                    [gain + 1e-6 for _, gain in shortlist],
                )
            selected_stops.append(chosen_stop)
            remaining_stops.remove(chosen_stop)

        result = selected_stops
        return result

    def _improve_service_candidate_by_swaps(
        self,
        candidate: List[Node],
    ) -> List[Node]:
        selected_stops = list(candidate)
        improved = True

        while improved:
            improved = False
            best_candidate = list(selected_stops)
            best_service = self.compute_set_passenger_service(selected_stops)
            remaining_stops = [
                stop for stop in self.network.bus_stops if stop not in selected_stops
            ]

            for index, current_stop in enumerate(selected_stops):
                for replacement_stop in remaining_stops:
                    trial_candidate = list(selected_stops)
                    trial_candidate[index] = replacement_stop

                    if len(set(trial_candidate)) != len(trial_candidate):
                        continue

                    trial_service = self.compute_set_passenger_service(trial_candidate)
                    if trial_service > best_service:
                        best_candidate = trial_candidate
                        best_service = trial_service

            if best_candidate != selected_stops:
                selected_stops = best_candidate
                improved = True

        result = self.repair_order_min_cost(selected_stops)
        return result

    def _build_best_service_candidate(self) -> List[Node]:
        best_candidate: List[Node] = []
        best_service = float("-inf")
        best_cost = math.inf
        top_pairs = self._top_od_pairs(limit=max(12, self.init_top_k * 3))
        seed_candidates: List[List[Node]] = []

        for stop_a, stop_b, _ in top_pairs:
            seed_candidates.append(
                self._build_service_guided_individual_from_seed_pair(
                    (stop_a, stop_b),
                    stochastic=False,
                )
            )

        if not seed_candidates:
            seed_candidates.append(self._build_service_guided_individual())

        for candidate in seed_candidates:
            candidate = self._improve_service_candidate_by_swaps(candidate)
            candidate_service = self.compute_set_passenger_service(candidate)
            candidate_cost = self._route_order_cost(candidate)

            if (
                candidate_service > best_service
                or (
                    math.isclose(candidate_service, best_service)
                    and candidate_cost < best_cost
                )
            ):
                best_candidate = candidate
                best_service = candidate_service
                best_cost = candidate_cost

        result = best_candidate
        return result

    def _individual_overlap_ratio(self, ind1: List[Node], ind2: List[Node]) -> float:
        shared_stops = len(set(ind1) & set(ind2))
        denominator = max(1, self.line_length)
        overlap_ratio = shared_stops / denominator
        return overlap_ratio

    def _is_diverse_enough(
        self,
        candidate: List[Node],
        population: List[List[Node]],
        overlap_threshold: float = 0.8,
        max_similar: int = 2,
    ) -> bool:
        similar_count = 0

        for existing in population:
            if self._individual_overlap_ratio(candidate, existing) >= overlap_threshold:
                similar_count += 1

            if similar_count >= max_similar:
                break

        is_diverse = similar_count < max_similar
        return is_diverse

    def _count_similar_individuals(
        self,
        candidate: List[Node],
        population: List[List[Node]],
    ) -> int:
        similar_count = 0

        for existing in population:
            overlap_ratio = self._individual_overlap_ratio(candidate, existing)
            if overlap_ratio >= self.diversity_overlap_threshold:
                similar_count += 1

        return similar_count

    def _candidate_signature(self, individual: List[Node]) -> Tuple[Node, ...]:
        signature = tuple(individual)
        return signature

    def _best_unique_candidate(
        self,
        family: str,
        population: List[List[Node]],
        seen_signatures: set[Tuple[Node, ...]],
        attempts: int,
    ) -> Optional[List[Node]]:
        best_candidate: Optional[List[Node]] = None
        best_similarity_count = math.inf

        for _attempt in range(max(1, attempts)):
            candidate = self._generate_candidate_by_family(family)
            signature = self._candidate_signature(candidate)

            if signature in seen_signatures:
                continue

            similarity_count = self._count_similar_individuals(candidate, population)
            if similarity_count < best_similarity_count:
                best_candidate = candidate
                best_similarity_count = similarity_count

            if self._is_diverse_enough(
                candidate,
                population,
                overlap_threshold=self.diversity_overlap_threshold,
                max_similar=self.diversity_max_similar,
            ):
                best_candidate = candidate
                break

        return best_candidate

    def _finalize_unique_candidate(
        self,
        family: str,
        candidate: Optional[List[Node]],
        population: List[List[Node]],
        seen_signatures: set[Tuple[Node, ...]],
    ) -> Optional[List[Node]]:
        final_candidate = candidate
        retry_best: Optional[List[Node]] = None
        retry_best_similarity = math.inf

        for _attempt in range(max(1, self.initialization_max_attempts)):
            if final_candidate is not None:
                signature = self._candidate_signature(final_candidate)
                if signature not in seen_signatures:
                    break

            retry_candidate = self._generate_candidate_by_family(family)
            retry_signature = self._candidate_signature(retry_candidate)

            if retry_signature in seen_signatures:
                continue

            similarity_count = self._count_similar_individuals(retry_candidate, population)
            if similarity_count < retry_best_similarity:
                retry_best = retry_candidate
                retry_best_similarity = similarity_count

            if self._is_diverse_enough(
                retry_candidate,
                population,
                overlap_threshold=self.diversity_overlap_threshold,
                max_similar=self.diversity_max_similar,
            ):
                final_candidate = retry_candidate
                break

            final_candidate = retry_candidate

        if final_candidate is not None:
            signature = self._candidate_signature(final_candidate)
            if signature in seen_signatures:
                final_candidate = retry_best
        elif retry_best is not None:
            final_candidate = retry_best

        return final_candidate

    def _generate_candidate_by_family(self, family: str) -> List[Node]:
        candidate = self._build_demand_guided_individual()

        if family == "service":
            candidate = self._build_service_guided_individual()
        elif family == "spatial":
            candidate = self._build_spatial_guided_individual()
        elif family == "hybrid":
            candidate = self._build_hybrid_guided_individual()
        elif family == "random":
            candidate = self._build_random_individual()

        return candidate

    def _build_population_record(
        self,
        family: str,
        individual: List[Node],
    ) -> Dict[str, Any]:
        seed_stop = individual[0] if individual else None
        record = {
            "family": family,
            "individual": list(individual),
            "seed_stop": seed_stop,
        }
        return record

    def _add_family_individuals(
        self,
        family: str,
        target_count: int,
        population: List[List[Node]],
        seen_signatures: set[Tuple[Node, ...]],
        population_records: List[Dict[str, Any]],
    ) -> None:
        for _ in range(target_count):
            best_candidate = self._best_unique_candidate(
                family,
                population,
                seen_signatures,
                self.initialization_max_attempts,
            )

            if best_candidate is None:
                best_candidate = self._best_unique_candidate(
                    family,
                    population,
                    seen_signatures,
                    max(1, self.initialization_max_attempts // 2),
                )

            best_candidate = self._finalize_unique_candidate(
                family,
                best_candidate,
                population,
                seen_signatures,
            )

            if best_candidate is not None:
                best_candidate = self.repair_order_min_cost(best_candidate)
                fallback_signature = self._candidate_signature(best_candidate)
                if fallback_signature not in seen_signatures:
                    population.append(best_candidate)
                    seen_signatures.add(fallback_signature)
                    population_records.append(
                        self._build_population_record(family, best_candidate)
                    )

    def _family_counts_from_ratios(
        self,
        total_count: int,
        ratios: Sequence[float],
    ) -> List[int]:
        normalized_ratios = self._normalize_ratios(ratios)
        raw_counts = [ratio * total_count for ratio in normalized_ratios]
        family_counts = [int(count) for count in raw_counts]
        assigned = sum(family_counts)
        remainders = [
            (index, raw_counts[index] - family_counts[index])
            for index in range(len(family_counts))
        ]
        remainders.sort(key=lambda item: item[1], reverse=True)

        for index, _ in remainders[: total_count - assigned]:
            family_counts[index] += 1

        return family_counts

    def _initialization_family_counts(self, total_count: int) -> List[int]:
        family_counts = self._family_counts_from_ratios(
            total_count,
            [
                self.init_ratio_service,
                self.init_ratio_demand,
                self.init_ratio_spatial,
                self.init_ratio_hybrid,
                self.init_ratio_random,
            ],
        )
        return family_counts

    def sample_guided_candidates(self, n_samples: int) -> List[List[Node]]:
        rng_state = self.rng.getstate()
        candidates: List[List[Node]] = []
        sample_count = max(0, n_samples)
        family_counts = self._family_counts_from_ratios(
            sample_count,
            [
                self.init_ratio_service,
                self.init_ratio_demand,
                self.init_ratio_spatial,
                self.init_ratio_hybrid,
            ],
        )
        family_plan = [
            ("service", family_counts[0]),
            ("demand", family_counts[1]),
            ("spatial", family_counts[2]),
            ("hybrid", family_counts[3]),
        ]

        try:
            for family, count in family_plan:
                for _ in range(count):
                    candidate = self._generate_candidate_by_family(family)
                    candidates.append(candidate)
        finally:
            self.rng.setstate(rng_state)

        return candidates

    def sample_mixed_candidates(self, n_samples: int) -> List[List[Node]]:
        rng_state = self.rng.getstate()
        candidates: List[List[Node]] = []
        sample_count = max(0, n_samples)
        family_counts = self._initialization_family_counts(sample_count)
        family_plan = [
            ("service", family_counts[0]),
            ("demand", family_counts[1]),
            ("spatial", family_counts[2]),
            ("hybrid", family_counts[3]),
            ("random", family_counts[4]),
        ]

        try:
            for family, count in family_plan:
                for _ in range(count):
                    candidate = self._generate_candidate_by_family(family)
                    candidates.append(candidate)
        finally:
            self.rng.setstate(rng_state)

        return candidates

    def sample_pseudorandom_reference_lines(
        self,
        n_samples: int,
        reference_seed: int = 2026,
        repair_order: bool = True,
        unique: bool = True,
    ) -> List[List[Node]]:
        sample_count = max(0, n_samples)
        bus_stops = list(self.network.bus_stops)
        reference_rng = random.Random(reference_seed)
        reference_lines: List[List[Node]] = []
        seen_signatures: set[Tuple[Node, ...]] = set()
        attempts = 0
        max_attempts = max(sample_count * 20, 100)

        if self.line_length > len(bus_stops):
            raise ValueError("line_length cannot exceed the number of available bus stops")

        while len(reference_lines) < sample_count and attempts < max_attempts:
            attempts += 1
            candidate = reference_rng.sample(bus_stops, k=self.line_length)

            if repair_order:
                candidate = self.repair_order_min_cost(candidate)

            signature = self._candidate_signature(candidate)

            if not unique or signature not in seen_signatures:
                reference_lines.append(candidate)
                seen_signatures.add(signature)

        result = reference_lines
        return result

    def generate_initial_population(self) -> None:
        population: List[List[Node]] = []
        population_records: List[Dict[str, Any]] = []
        seen_signatures: set[Tuple[Node, ...]] = set()
        self.best_service_candidate = []
        self.best_service_candidate_ordered = []
        self.best_service_candidate_passenger_service = 0.0

        if self.inject_best_service_candidate and self.population_size > 0:
            injected_candidate = self._build_best_service_candidate()
            injected_signature = self._candidate_signature(injected_candidate)

            if injected_signature not in seen_signatures:
                population.append(injected_candidate)
                seen_signatures.add(injected_signature)
                population_records.append(
                    self._build_population_record("service_best", injected_candidate)
                )
                self.best_service_candidate = list(injected_candidate)
                self.best_service_candidate_ordered = list(injected_candidate)
                self.best_service_candidate_passenger_service = (
                    self.compute_set_passenger_service(injected_candidate)
                )

        remaining_count = max(0, self.population_size - len(population))
        family_counts = self._initialization_family_counts(remaining_count)
        family_plan = [
            ("service", family_counts[0]),
            ("demand", family_counts[1]),
            ("spatial", family_counts[2]),
            ("hybrid", family_counts[3]),
            ("random", family_counts[4]),
        ]

        for family, count in family_plan:
            self._add_family_individuals(
                family,
                count,
                population,
                seen_signatures,
                population_records,
            )

        self.population = population
        self.population_records = population_records

        if not self.best_service_candidate:
            fallback_candidate = self._build_best_service_candidate()
            self.best_service_candidate = list(fallback_candidate)
            self.best_service_candidate_ordered = list(fallback_candidate)
            self.best_service_candidate_passenger_service = (
                self.compute_set_passenger_service(fallback_candidate)
            )

    def fitness(self, individual: List[Node], lambda_: float) -> float:
        evaluation = self.objective_function.evaluate(individual, lambda_)
        score = evaluation.fitness
        return score

    def evaluate_individual(
        self,
        individual: List[Node],
        lambda_: float,
    ) -> EvaluationResult:
        result = self.objective_function.evaluate(individual, lambda_)
        return result

    def evaluate_population(
        self,
        lambda_: float,
    ) -> List[Tuple[List[Node], EvaluationResult]]:
        scored_population: List[Tuple[List[Node], EvaluationResult]] = []

        for individual in self.population:
            evaluation = self.evaluate_individual(individual, lambda_)
            scored_population.append((individual, evaluation))

        return scored_population

    def select_elite(
        self,
        scored_population: List[Tuple[List[Node], EvaluationResult]],
        n_elite: int,
    ) -> List[List[Node]]:
        elite: List[List[Node]] = []
        sorted_population = sorted(
            scored_population,
            key=lambda x: x[1].fitness,
            reverse=True,
        )

        for individual, _ in sorted_population[:n_elite]:
            elite.append(list(individual))

        return elite

    def tournament_selection(
        self,
        scored_population: List[Tuple[List[Node], EvaluationResult]],
        tournament_size: int,
    ) -> List[Node]:
        candidates = self.rng.sample(scored_population, k=tournament_size)
        winner = max(candidates, key=lambda x: x[1].fitness)[0]
        result = list(winner)
        return result

    def crossover_ox(
        self,
        parent1: List[Node],
        parent2: List[Node],
    ) -> List[Node]:
        size = len(parent1)
        start, end = sorted(self.rng.sample(range(size), 2))
        child: List[Optional[Node]] = [None] * size
        child[start : end + 1] = parent1[start : end + 1]
        parent2_idx = 0

        for i in range(size):
            if child[i] is not None:
                continue

            while parent2[parent2_idx] in child:
                parent2_idx += 1

            child[i] = parent2[parent2_idx]

        result = [node for node in child if node is not None]
        return result

    def _local_transition_cost(
        self,
        prev_stop: Optional[Node],
        current_stop: Node,
        next_stop: Optional[Node],
    ) -> float:
        total_cost = 0.0

        if prev_stop is not None:
            total_cost += self.stop_costs.get((prev_stop, current_stop), math.inf)

        if next_stop is not None:
            total_cost += self.stop_costs.get((current_stop, next_stop), math.inf)

        return total_cost

    def mutate_adjacent_swap(self, individual: List[Node]) -> List[Node]:
        child = list(individual)
        i = self.rng.randrange(len(child) - 1)
        child[i], child[i + 1] = child[i + 1], child[i]
        result = child
        return result

    def mutate_reverse_segment(
        self,
        individual: List[Node],
        max_segment_length: int = 4,
    ) -> List[Node]:
        child = list(individual)
        start, end = sorted(self.rng.sample(range(len(child)), 2))

        if end - start + 1 > max_segment_length:
            end = start + max_segment_length - 1

        child[start : end + 1] = reversed(child[start : end + 1])
        result = child
        return result

    def mutate_replace_by_neighbor(
        self,
        individual: List[Node],
        acceptance_tolerance: float = 0.00,
    ) -> List[Node]:
        child = list(individual)
        position = self.rng.randrange(len(child))
        current_stop = child[position]
        present_stops = set(child)
        candidate_stops = [
            stop
            for stop in self.nearest_neighbors.get(current_stop, [])
            if stop not in present_stops
        ]

        if not candidate_stops:
            child = self.mutate_adjacent_swap(individual)
            result = child
            return result

        prev_stop = child[position - 1] if position > 0 else None
        next_stop = child[position + 1] if position < len(child) - 1 else None
        base_cost = self._local_transition_cost(prev_stop, current_stop, next_stop)
        candidate_stops.sort(
            key=lambda stop: self._local_transition_cost(prev_stop, stop, next_stop)
        )

        top_k = max(1, min(3, len(candidate_stops)))
        shortlist = candidate_stops[:top_k]
        chosen_stop = self.rng.choice(shortlist)
        chosen_cost = self._local_transition_cost(prev_stop, chosen_stop, next_stop)

        if chosen_cost <= base_cost * (1.0 + acceptance_tolerance):
            child[position] = chosen_stop

        result = child
        return result

    def mutate_replace_by_service_gain(self, individual: List[Node]) -> List[Node]:
        child = list(individual)
        position = self.rng.randrange(len(child))
        remaining_stops = [
            stop for stop in self.network.bus_stops
            if stop not in child or stop == child[position]
        ]
        scored_candidates: List[Tuple[Node, float]] = []

        for candidate_stop in remaining_stops:
            if candidate_stop == child[position]:
                continue

            trial_candidate = list(child)
            trial_candidate[position] = candidate_stop

            if len(set(trial_candidate)) != len(trial_candidate):
                continue

            trial_service = self.compute_set_passenger_service(trial_candidate)
            scored_candidates.append((candidate_stop, trial_service))

        if scored_candidates:
            scored_candidates.sort(key=lambda item: item[1], reverse=True)
            top_k = max(1, min(self.init_top_k, len(scored_candidates)))
            shortlist = scored_candidates[:top_k]
            chosen_stop = self._weighted_choice(
                [stop for stop, _ in shortlist],
                [score + 1e-6 for _, score in shortlist],
            )
            child[position] = chosen_stop

        result = child
        return result

    def mutate(
        self,
        individual: List[Node],
        mutation_prob: float,
        weight_adjacent: float = 0.25,
        weight_reverse: float = 0.20,
        weight_neighbor: float = 0.25,
        weight_service_replace: float = 0.30,
        max_reverse_length: int = 4,
        acceptance_tolerance: float = 0.00,
    ) -> List[Node]:
        child = list(individual)

        if self.rng.random() >= mutation_prob:
            result = child
            return result

        strategies = ["adjacent", "reverse", "neighbor", "service_replace"]
        weights = [
            weight_adjacent,
            weight_reverse,
            weight_neighbor,
            weight_service_replace,
        ]
        strategy = self.rng.choices(strategies, weights=weights, k=1)[0]

        if strategy == "adjacent":
            child = self.mutate_adjacent_swap(child)
        elif strategy == "reverse":
            child = self.mutate_reverse_segment(
                child,
                max_segment_length=max_reverse_length,
            )
        elif strategy == "neighbor":
            child = self.mutate_replace_by_neighbor(
                child,
                acceptance_tolerance=acceptance_tolerance,
            )
        else:
            child = self.mutate_replace_by_service_gain(child)

        result = child
        return result

    def evolve_one_generation(
        self,
        lambda_: float,
        n_elite: int = 1,
        tournament_size: int = 3,
        mutation_prob: float = 0.20,
        weight_adjacent: float = 0.25,
        weight_reverse: float = 0.20,
        weight_neighbor: float = 0.25,
        weight_service_replace: float = 0.30,
        max_reverse_length: int = 4,
        acceptance_tolerance: float = 0.00,
    ) -> None:
        scored_population = self.evaluate_population(lambda_)
        elite = self.select_elite(scored_population, n_elite)
        new_population: List[List[Node]] = list(elite)

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(scored_population, tournament_size)
            parent2 = self.tournament_selection(scored_population, tournament_size)
            child = self.crossover_ox(parent1, parent2)
            child = self.mutate(
                child,
                mutation_prob=mutation_prob,
                weight_adjacent=weight_adjacent,
                weight_reverse=weight_reverse,
                weight_neighbor=weight_neighbor,
                weight_service_replace=weight_service_replace,
                max_reverse_length=max_reverse_length,
                acceptance_tolerance=acceptance_tolerance,
            )
            child = self.repair_order_min_cost(child)
            new_population.append(child)

        self.population = new_population

    def _update_history(
        self,
        scored_population: List[Tuple[List[Node], EvaluationResult]],
    ) -> None:
        best_individual, best_evaluation = max(
            scored_population,
            key=lambda x: x[1].fitness,
        )
        mean_score = 0.0

        if scored_population:
            mean_score = sum(item[1].fitness for item in scored_population) / len(scored_population)

        self.history.best_scores.append(best_evaluation.fitness)
        self.history.mean_scores.append(mean_score)
        self.history.best_costs.append(best_evaluation.cost)
        self.history.best_services.append(best_evaluation.service)

    def _save_global_best_snapshot(
        self,
        generation: int,
        best_individual: List[Node],
        best_evaluation: EvaluationResult,
        output_dir: Optional[str] = None,
        od_min_to_plot: float = 10.0,
    ) -> Optional[str]:
        snapshot = ImprovementSnapshotData(
            generation=generation,
            individual=list(best_individual),
            fitness=best_evaluation.fitness,
            cost=best_evaluation.cost,
            passenger_service=best_evaluation.passenger_service,
            cost_norm=best_evaluation.cost_norm,
            passenger_service_norm=best_evaluation.passenger_service_norm,
            evaluation=best_evaluation,
        )
        attempted_pdf_path = get_snapshot_pdf_path(generation, output_dir=output_dir)
        saved_pdf_path: Optional[str] = None

        try:
            saved_pdf_path = save_improvement_snapshot(
                self.network,
                snapshot,
                output_dir=output_dir,
                od_min_to_plot=od_min_to_plot,
            )
            print(
                f"[SNAPSHOT SAVED] gen={generation:03d} "
                f"fitness={best_evaluation.fitness:.12f} "
                f"cost={best_evaluation.cost:.12f} "
                f"passenger_service={best_evaluation.passenger_service:.12f} "
                f"cost_norm={best_evaluation.cost_norm:.12f} "
                f"passenger_service_norm={best_evaluation.passenger_service_norm:.12f} "
                f"pdf={saved_pdf_path}"
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[SNAPSHOT WARNING] gen={generation:03d} "
                f"pdf={attempted_pdf_path} "
                f"error={exc}"
            )

        return saved_pdf_path

    def run(
        self,
        lambda_: float,
        generations: int,
        n_elite: int = 1,
        tournament_size: int = 3,
        mutation_prob: float = 0.20,
        weight_adjacent: float = 0.25,
        weight_reverse: float = 0.20,
        weight_neighbor: float = 0.25,
        weight_service_replace: float = 0.30,
        max_reverse_length: int = 4,
        acceptance_tolerance: float = 0.00,
        stagnation_generations: Optional[int] = None,
        return_history: bool = False,
        reinitialize_population: bool = True,
        save_improvement_snapshots: bool = True,
        snapshot_output_dir: Optional[str] = None,
        od_min_to_plot: float = 10.0,
    ):
        best_individual: List[Node] = []
        best_score = float("-inf")
        no_improvement_counter = 0

        if reinitialize_population or not self.population:
            self.generate_initial_population()

        self.history = GAHistory()

        for generation in range(generations):
            scored_population = self.evaluate_population(lambda_)
            generation_best = max(scored_population, key=lambda x: x[1].fitness)
            self._update_history(scored_population)

            if generation_best[1].fitness > best_score:
                best_individual = list(generation_best[0])
                best_score = generation_best[1].fitness
                no_improvement_counter = 0

                if save_improvement_snapshots:
                    self._save_global_best_snapshot(
                        generation=generation,
                        best_individual=best_individual,
                        best_evaluation=generation_best[1],
                        output_dir=snapshot_output_dir,
                        od_min_to_plot=od_min_to_plot,
                    )
            else:
                no_improvement_counter += 1

            should_stop = False
            is_last_generation = generation == generations - 1

            if stagnation_generations is not None and no_improvement_counter >= stagnation_generations:
                should_stop = True

            if not should_stop and not is_last_generation:
                self.evolve_one_generation(
                    lambda_=lambda_,
                    n_elite=n_elite,
                    tournament_size=tournament_size,
                    mutation_prob=mutation_prob,
                    weight_adjacent=weight_adjacent,
                    weight_reverse=weight_reverse,
                    weight_neighbor=weight_neighbor,
                    weight_service_replace=weight_service_replace,
                    max_reverse_length=max_reverse_length,
                    acceptance_tolerance=acceptance_tolerance,
                )
            elif should_stop:
                break

        result = (best_individual, best_score)

        if return_history:
            result = (best_individual, best_score, self.history.as_dict())

        return result

