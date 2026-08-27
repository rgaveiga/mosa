"""Internal types and helpers used by the MOSA algorithm."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeAlias, TypedDict

import numpy as np

from numba import njit

from ._error import MOSAError

Number: TypeAlias = int | float
"""@private"""

ObjectiveValues: TypeAlias = list[Number]
"""@private"""

ObjectiveWeightValues: TypeAlias = list[Number]
"""@private"""

Solution: TypeAlias = dict[str, Any]
"""@private"""

PopulationGroup: TypeAlias = list[Any] | tuple[Number, ...]
"""@private"""

Population: TypeAlias = dict[str, PopulationGroup]
"""@private"""

ObjectiveFunction: TypeAlias = Callable[..., Sequence[Number]]
"""@private"""


_ADAPTATIVE_SELECTION_MIN_PROBABILITY = 0.01


def _selection_probability_floor(number_of_groups: int) -> float:
    """Return a feasible minimum probability for every selectable group."""

    return min(_ADAPTATIVE_SELECTION_MIN_PROBABILITY, 1.0 / number_of_groups)


def _apply_selection_probability_floor(
    probabilities: np.ndarray, minimum_probability: float
) -> np.ndarray:
    """Apply the exploration floor only when a probability violates it."""

    if np.all(probabilities >= minimum_probability):
        return probabilities

    return (
        minimum_probability
        + (1.0 - minimum_probability * probabilities.size) * probabilities
    )


def _normalize_selection_weights(
    weights: Sequence[Number], minimum_probability: float
) -> np.ndarray:
    """Normalize selection weights while preserving an exploration floor."""

    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise MOSAError(
            "Adaptative group selection weights must be finite, non-negative, "
            "and have a positive sum!"
        )

    probabilities = values / values.sum()
    return _apply_selection_probability_floor(probabilities, minimum_probability)


def _adaptative_selection_probabilities(
    qualities: Sequence[Number], temperature: float, minimum_probability: float
) -> np.ndarray:
    """Convert adaptative qualities into stable Boltzmann probabilities."""

    scaled_qualities = np.asarray(qualities, dtype=float) / temperature
    scaled_qualities -= np.max(scaled_qualities)
    probabilities = np.exp(scaled_qualities)
    probabilities /= probabilities.sum()
    return _apply_selection_probability_floor(probabilities, minimum_probability)


def _adaptative_selection_reward(
    previous: Sequence[Number],
    candidate: Sequence[Number],
    maximum_deltas: np.ndarray,
) -> float:
    """Return the mean normalized objective variation and update its maxima."""

    previous_values = np.asarray(previous, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    return _adaptative_selection_reward_kernel(
        previous_values, candidate_values, maximum_deltas
    )


@njit(cache=True)
def _adaptative_selection_reward_kernel(
    previous: np.ndarray,
    candidate: np.ndarray,
    maximum_deltas: np.ndarray,
) -> float:
    """Compute adaptive reward without per-call temporary arrays."""

    total = 0.0
    for index in range(previous.size):
        previous_value = previous[index]
        candidate_value = candidate[index]

        if np.isfinite(previous_value) and np.isfinite(candidate_value):
            delta = abs(candidate_value - previous_value)
            if delta > maximum_deltas[index]:
                maximum_deltas[index] = delta
            if maximum_deltas[index] > 0.0:
                total += delta / maximum_deltas[index]
        elif previous_value != candidate_value:
            total += 1.0

    return total / previous.size


def corana_step_length(
    step_length: float, accepted_moves: int, attempted_moves: int
) -> float:
    """Adjust a continuous step length using Corana's acceptance rule.

    A coordinate that was not attempted keeps its current step length.
    """

    if attempted_moves == 0:
        return step_length

    acceptance_rate = accepted_moves / attempted_moves

    if acceptance_rate > 0.6:
        return step_length * (1.0 + 2.0 * (acceptance_rate - 0.6) / 0.4)
    if acceptance_rate < 0.4:
        return step_length / (1.0 + 2.0 * (0.4 - acceptance_rate) / 0.4)
    return step_length


class Archive(TypedDict):
    """@private"""

    x: list[Solution]
    f: list[ObjectiveValues]


def _semantic_key(value: Any) -> Any | None:
    """Return a hashable equality key for common categorical values."""

    if isinstance(value, list):
        items = tuple(_semantic_key(item) for item in value)
        return None if any(item is None for item in items) else (list, items)

    if isinstance(value, tuple):
        items = tuple(_semantic_key(item) for item in value)
        return None if any(item is None for item in items) else (tuple, items)

    if isinstance(value, dict):
        items = [
            (_semantic_key(key), _semantic_key(item)) for key, item in value.items()
        ]
        if any(key is None or item is None for key, item in items):
            return None
        return (dict, frozenset(items))

    try:
        hash(value)
    except (TypeError, ValueError):
        return None

    return (object, value)


def _values_equal(left: Any, right: Any) -> bool:
    """Compare arbitrary category values while requiring scalar truth."""

    try:
        result = left == right
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except (TypeError, ValueError):
        return left is right


def _categorical_equivalence(values: Sequence[Any]) -> np.ndarray:
    """Assign equality classes without merging population position codes."""

    equivalence = np.empty(len(values), dtype=np.int64)
    keyed_classes: dict[Any, int] = {}
    fallback_representatives: list[tuple[Any, int]] = []
    next_class = 0

    for index, value in enumerate(values):
        key = _semantic_key(value)
        if key is not None:
            equivalent = keyed_classes.get(key)
            if equivalent is None:
                equivalent = next_class
                keyed_classes[key] = equivalent
                next_class += 1
        else:
            equivalent = None
            for representative, class_id in fallback_representatives:
                if _values_equal(value, representative):
                    equivalent = class_id
                    break
            if equivalent is None:
                equivalent = next_class
                fallback_representatives.append((value, equivalent))
                next_class += 1
        equivalence[index] = equivalent

    return equivalence


def _is_numeric_population(values: Sequence[Any]) -> bool:
    """Return whether a discrete population can use a native numeric dtype."""

    return len(values) > 0 and all(
        isinstance(value, (int, float, np.integer, np.floating)) for value in values
    )


@dataclass
class _GroupState:
    """Numeric representation of one solution/population group."""

    continuous: bool
    categorical: bool
    scalar_output: bool
    population: np.ndarray
    solution: np.ndarray
    categories: tuple[Any, ...] = ()
    equivalence: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        population: PopulationGroup,
        number_of_elements: int,
        solution: Any | None = None,
    ) -> "_GroupState":
        scalar_output = number_of_elements == 1

        if isinstance(population, tuple):
            if solution is None:
                solution_values: list[Any] = []
            elif scalar_output:
                solution_values = [solution]
            else:
                solution_values = list(solution)
            return cls(
                continuous=True,
                categorical=False,
                scalar_output=scalar_output,
                population=np.asarray(population, dtype=np.float64),
                solution=np.asarray(solution_values, dtype=np.float64),
            )

        population_values = list(population)
        if solution is None:
            solution_values = []
        elif scalar_output:
            solution_values = [solution]
        else:
            solution_values = list(solution)
        numeric_values = population_values + solution_values

        if _is_numeric_population(numeric_values):
            dtype = (
                np.float64
                if any(
                    isinstance(value, (float, np.floating)) for value in numeric_values
                )
                else np.int64
            )
            return cls(
                continuous=False,
                categorical=False,
                scalar_output=scalar_output,
                population=np.asarray(population_values, dtype=dtype),
                solution=np.asarray(solution_values, dtype=dtype),
            )

        categories = tuple(population_values + solution_values)
        return cls(
            continuous=False,
            categorical=True,
            scalar_output=scalar_output,
            population=np.arange(len(population_values), dtype=np.int64),
            solution=np.arange(len(population_values), len(categories), dtype=np.int64),
            categories=categories,
            equivalence=_categorical_equivalence(categories),
        )

    def equal(self, left: Any, right: Any) -> bool:
        if self.categorical:
            assert self.equivalence is not None
            return bool(self.equivalence[int(left)] == self.equivalence[int(right)])
        return bool(left == right)

    def decode_value(self, value: Any) -> Any:
        if self.categorical:
            return self.categories[int(value)]
        return value.item() if isinstance(value, np.generic) else value

    def decode(self, values: np.ndarray) -> Any:
        if self.scalar_output:
            return self.decode_value(values[0])
        if self.categorical:
            return [self.categories[int(value)] for value in values]
        return values.tolist()

    def decode_solution(self) -> Any:
        return self.decode(self.solution)

    def decode_population(self) -> PopulationGroup:
        if self.continuous:
            return tuple(float(value) for value in self.population)
        if self.categorical:
            return [self.categories[int(value)] for value in self.population]
        return self.population.tolist()


@njit(cache=True)
def _dominance_masks_kernel(
    archive_arr: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compare an archive with one candidate in a single compiled pass."""

    row_count, objective_count = archive_arr.shape
    archive_dominates = np.ones(row_count, dtype=np.bool_)
    candidate_dominates = np.ones(row_count, dtype=np.bool_)

    for row in range(row_count):
        archive_row_dominates = True
        candidate_dominates_row = True

        for objective in range(objective_count):
            archive_value = archive_arr[row, objective]
            candidate_value = candidate[objective]

            if not archive_value <= candidate_value:
                archive_row_dominates = False
            if not archive_value >= candidate_value:
                candidate_dominates_row = False

            if not archive_row_dominates and not candidate_dominates_row:
                break

        archive_dominates[row] = archive_row_dominates
        candidate_dominates[row] = candidate_dominates_row

    return archive_dominates, candidate_dominates


@njit(cache=True)
def _non_dominated_mask_kernel(f_arr: np.ndarray) -> np.ndarray:
    """Return the non-dominated rows using compiled early-exit comparisons."""

    row_count, objective_count = f_arr.shape
    keep = np.ones(row_count, dtype=np.bool_)

    for candidate in range(row_count):
        for other in range(row_count):
            if candidate == other:
                continue

            dominates = True

            for objective in range(objective_count):
                if not f_arr[other, objective] <= f_arr[candidate, objective]:
                    dominates = False
                    break

            if dominates:
                keep[candidate] = False
                break

    return keep
