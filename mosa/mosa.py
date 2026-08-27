"""This module defines the `Anneal` class, which implements the MOSA algorithm."""

import json
import os
import warnings
from copy import deepcopy
from math import ceil, exp, floor, inf, isclose, isfinite, isnan, log10
from numbers import Real
from typing import Any, Sequence

import numpy as np
from numpy.random import choice, triangular, uniform
from . import __version__
from ._error import MOSAError
from ._support import (
    Archive,
    Number,
    ObjectiveFunction,
    ObjectiveValues,
    ObjectiveWeightValues,
    Population,
    PopulationGroup,
    Solution,
    _GroupState,
    _adaptative_selection_probabilities,
    _adaptative_selection_reward,
    _categorical_equivalence,  # noqa: F401 - re-exported for compatibility
    corana_step_length,
    _dominance_masks_kernel,
    _is_numeric_population,  # noqa: F401 - re-exported for compatibility
    _non_dominated_mask_kernel,
    _normalize_selection_weights,
    _selection_probability_floor,
    _semantic_key,  # noqa: F401 - re-exported for compatibility
    _values_equal,  # noqa: F401 - re-exported for compatibility
)


_GROUP_PARAMS = frozenset(
    {
        "number_of_elements",
        "maximum_number_of_elements",
        "distinct_elements",
        "mc_step_size",
        "mc_step_increment",
        "change_value_move",
        "insert_or_delete_move",
        "swap_move",
        "sort_elements",
        "group_selection_weights",
    }
)


class Anneal:
    """This class implements the MOSA algorithm."""

    def __init__(self) -> None:
        """@private
        Initializes object attributes.
        """

        print("--------------------------------------------------")
        print(f" MULTI-OBJECTIVE SIMULATED ANNEALING (MOSA) {__version__}  ")
        print("--------------------------------------------------")

        self._initemp: float = 1.0
        self._initempset: bool = False
        self._autohightemp: bool = False
        self._hightempaccthresh: float = 0.8
        self._decrease: float = 0.9
        self._ntemp: int = 10
        self._population: Population = {}
        self._groupstates: dict[str, _GroupState] = {}
        self._changemove: dict[str, Number] = {}
        self._swapmove: dict[str, Number] = {}
        self._insordelmove: dict[str, Number] = {}
        self._xnel: dict[str, int] = {}
        self._maxnel: dict[str, int] = {}
        self._xdistinct: dict[str, bool] = {}
        self._xincrement: dict[str, Number] = {}
        self._xstep: dict[str, Number] = {}
        self._adaptxstep: bool = False
        self._adaptsel: bool = False
        self._xsort: dict[str, bool] = {}
        self._xselalpha: float = 0.2
        self._xselweight: dict[str, Number] = {}
        self._archivex: list[Solution] = []
        self._archivefarr: np.ndarray = np.empty((0, 0), dtype=float)
        self._archivefcap: int = 0
        self._xcache: bool = False
        self._xcachesize: int = 10000
        self._cache: Archive = {"x": [], "f": []}
        self._archive_lookup: dict[Any, int] = {}
        self._cache_lookup: dict[Any, tuple[Solution, ObjectiveValues]] = {}
        self._temp: list[float] = []
        self._weight: ObjectiveWeightValues = []
        self._niter: int = 1000
        self._archivefile: str = "archive.json"
        self._archivesaveint: int = 10
        self._archivesize: int = 1000
        self._maxarchivereject: int = 1000
        self._alpha: float = 0.0
        self._restart: bool = True
        self._trackoptprogress: bool = False
        self._f: list[Number | ObjectiveValues] = []
        self._verbose: bool = False

    def set_population(self, **groups: PopulationGroup) -> None:
        """
        Sets the population.

        ### Parameters

        `**groups`: series of key-value pairs where each key corresponds to a
        group in the solution and contains the data that can be used to achieve
        an optimized solution to the problem.
        """

        if len(groups) > 0:
            for key, value in groups.items():
                self._population[key] = value
        else:
            raise MOSAError("No keyword was provided!")

    def set_group_params(self, group: str, **params: Any) -> None:
        """
        Sets the optimization parameters for the specified group in the solution.

        ### Parameters

        `group`: group in the solution.

        `**params`: names of the optimization parameters with respective values.

        They can be any of the alternatives below:

        - `number_of_elements`

        - `maximum_number_of_elements`

        - `distinct_elements`

        - `mc_step_size`

        - `mc_step_increment`

        - `change_value_move`

        - `insert_or_delete_move`

        - `swap_move`

        - `sort_elements`

        - `group_selection_weights`
        """

        if len(params) == 0:
            raise MOSAError("No keyword was provided!")

        for param, value in params.items():
            if param not in _GROUP_PARAMS:
                raise MOSAError("Optimization parameter does not exist!")

            setattr(self, param, {group: value})

    def set_opt_param(self, param: str, **groups: Any) -> None:
        """
        Sets the values of the optimization parameter for the specified solution
        groups.

        ### Parameters

        `param`: name of the optimization parameter.

        It must be one of the alternatives below:

        - `number_of_elements`

        - `maximum_number_of_elements`

        - `distinct_elements`

        - `mc_step_size`

        - `mc_step_increment`

        - `change_value_move`

        - `insert_or_delete_move`

        - `swap_move`

        - `sort_elements`

        - `group_selection_weights`

        `**groups`: series of key-value pairs where each key corresponds to a
        group in the solution to the problem.
        """

        if len(groups) == 0:
            raise MOSAError("No keyword was provided!")

        if param not in _GROUP_PARAMS:
            raise MOSAError("Optimization parameter does not exist!")

        setattr(self, param, groups)

    def evolve(self, func: ObjectiveFunction) -> None:
        """
        Performs the optimization of the objective function.

        ### Parameters

        `func`: objective function.
        """

        print("--- BEGIN: Evolving a solution ---\n")

        if not callable(func):
            raise MOSAError("A Python function must be provided!")

        from_archive: bool = False
        from_saved_state: bool = False
        updated: int = 0
        nupdated: int = 0
        naccept: int = 0
        narchivereject: int = 0
        fcurr: ObjectiveValues = []
        ftmp: ObjectiveValues = []
        weight: ObjectiveWeightValues = []
        lstep: dict[str, int] = {}
        population: Population = {}
        xcurr: Solution = {}
        xtmp: Solution = {}
        xstep: dict[str, Number] = {}
        xstep_bounds: dict[str, tuple[float, float]] = {}
        xincrement: dict[str, float] = {}
        xincrement_count: dict[str, int] = {}
        xincrement_warned: set[str] = set()
        corana_attempts: dict[str, int] = {}
        corana_accepts: dict[str, int] = {}
        xsampling: dict[str, int] = {}
        xbounds: dict[str, list[Number]] = {}
        changemove: dict[str, float] = {}
        swapmove: dict[str, float] = {}
        insordelmove: dict[str, float] = {}
        xdistinct: dict[str, bool] = {}
        xnel: dict[str, int] = {}
        maxnel: dict[str, Number] = {}
        xsort: dict[str, bool] = {}
        totlength: float = 0.0
        sellength: dict[str, float] = {}
        selection_weights: dict[str, float] = {}
        displayed_selection_probabilities: tuple[float, ...] = ()
        adaptative_quality: dict[str, float] = {}
        adaptative_maximum_deltas = np.empty(0, dtype=float)
        minimum_selection_probability: float = 0.0
        groups: list[str] = []
        MAX_FAILED: int = 10
        MIN_STEP_LENGTH: int = 10

        def update_increment_count(group: str) -> None:
            interval_count = 2.0 * float(xstep[group]) / xincrement[group]
            rounded_count = round(interval_count)
            includes_upper = isclose(
                interval_count, rounded_count, rel_tol=1e-12, abs_tol=1e-12
            )
            if includes_upper:
                interval_count = rounded_count
            else:
                interval_count = floor(interval_count)
                if group not in xincrement_warned:
                    warnings.warn(
                        f"The interval from -{xstep[group]} to {xstep[group]} is not "
                        f"divisible by step increment {xincrement[group]} for group "
                        f"'{group}'; the upper limit {xstep[group]} will not be included.",
                        UserWarning,
                        stacklevel=2,
                    )
                    xincrement_warned.add(group)

            xincrement_count[group] = int(interval_count) + 1

        if self._restart:
            if len(self._archivex) == 0:
                print(f"Trying to load the archive from file {self._archivefile}...")

                if not self.__load_archive_file(self._archivefile):
                    print(
                        f"File {self._archivefile} not found or invalid! "
                        "Initializing an empty archive..."
                    )
                    self.__set_archive_data([], [])
                else:
                    print(f"File {self._archivefile} found!")

            if len(self._archivex) > 0 and self._population:
                print("Restarting from a previous run...")

                xcurr = deepcopy(self._archivex[-1])
                fcurr = (
                    self._archivefarr[len(self._archivex) - 1].astype(float).tolist()
                )
                population = {
                    group: tuple(values) if isinstance(values, tuple) else list(values)
                    for group, values in self._population.items()
                }
                from_archive = True
        else:
            print("Initializing an empty archive...")

            self.__set_archive_data([], [])

        print("Done!")

        if population and xcurr and len(fcurr) > 0:
            if set(population.keys()) == set(xcurr.keys()):
                from_saved_state = True
            else:
                raise MOSAError("Solution and population must have the same groups!")
        else:
            if self._restart and len(self._archivex) > 0 and not self._population:
                raise MOSAError(
                    "A population must be configured to restart from the archive!"
                )

            if self._population:
                xcurr = {}
                fcurr = []
                population = deepcopy(self._population)
            else:
                raise MOSAError("A population must be provided!")

        groups = list(population.keys())

        if self._adaptsel:
            minimum_selection_probability = _selection_probability_floor(len(groups))
            initial_probabilities = _normalize_selection_weights(
                [self._xselweight.get(group, 1.0) for group in groups],
                minimum_selection_probability,
            )
            selection_weights = dict(zip(groups, initial_probabilities.tolist()))
            self._xselweight.update(selection_weights)
        else:
            selection_weights = {
                group: float(self._xselweight.get(group, 1.0)) for group in groups
            }

        selection_weight_total = sum(selection_weights.values())
        displayed_selection_probabilities = tuple(
            selection_weights[group] / selection_weight_total for group in groups
        )

        print("------\n")
        print("Groups in the solution:\n======================\n")

        for group in groups:
            print(f"    {group}:")

            if group in self._xnel.keys() and self._xnel[group] > 0:
                xnel[group] = self._xnel[group]
            else:
                xnel[group] = 1

            print(f"        Number of elements: {xnel[group]}")

            if isinstance(population[group], tuple):
                print("        Sample space: continuous")

                if len(population[group]) <= 1:
                    raise MOSAError(f"Two numbers are expected in group {group}!")

                xsampling[group] = 1
                xbounds[group] = list(population[group])

                if xbounds[group][1] < xbounds[group][0]:
                    xbounds[group][0], xbounds[group][1] = (
                        xbounds[group][1],
                        xbounds[group][0],
                    )
                elif xbounds[group][1] == xbounds[group][0]:
                    raise MOSAError(
                        f"Second element in group {group} must be larger than the first one!"
                    )

                print(f"        Boundaries: ({xbounds[group][0]},{xbounds[group][1]})")
            elif isinstance(population[group], list):
                print("        Sample space: discrete")
                print(f"        Size of population group: {len(population[group])}")

                if len(population[group]) <= 1 and not from_saved_state:
                    raise MOSAError(
                        "Number of elements in the population group must be greater than one!"
                    )

                xsampling[group] = 0

                if group in self._xdistinct.keys():
                    xdistinct[group] = bool(self._xdistinct[group])
                else:
                    xdistinct[group] = False

                print(f"        Distinct elements: {xdistinct[group]}")
            else:
                raise MOSAError(f"Wrong format of group {group}!")

            totlength += selection_weights[group]
            selection_probability = selection_weights[group] / selection_weight_total
            print(f"        Selection probability: {selection_probability:.6f}")

            sellength[group] = totlength

            if group in self._changemove.keys() and self._changemove[group] >= 0.0:
                changemove[group] = float(self._changemove[group])
            else:
                changemove[group] = 1.0

            if changemove[group] > 0.0:
                print(
                    f"        Weight of 'change value' trial move: {changemove[group]}"
                )

            if group in self._swapmove.keys() and self._swapmove[group] > 0.0:
                swapmove[group] = float(self._swapmove[group])

                print(f"        Weight of 'swap' trial move: {swapmove[group]}")
            else:
                swapmove[group] = 0.0

            if group in self._insordelmove.keys() and self._insordelmove[group] > 0.0:
                insordelmove[group] = float(self._insordelmove[group])

                print(
                    f"        Weight of 'insert or delete' trial move: {insordelmove[group]}"
                )

                if group in self._maxnel.keys() and self._maxnel[group] >= xnel[group]:
                    maxnel[group] = int(self._maxnel[group])

                    if maxnel[group] <= 1:
                        maxnel[group] = 2
                else:
                    maxnel[group] = inf

                print(f"        Maximum number of elements: {maxnel[group]}")
            else:
                insordelmove[group] = 0.0

            if swapmove[group] == 0.0 and group in self._xsort.keys():
                xsort[group] = bool(self._xsort[group])
            else:
                xsort[group] = False

            print(f"        Sort values: {xsort[group]}")

            if xsampling[group] == 1:
                boundary_range = float(xbounds[group][1] - xbounds[group][0])
                minimum_xstep = boundary_range / 100.0
                maximum_xstep = boundary_range / 2.0
                xstep_bounds[group] = (minimum_xstep, maximum_xstep)

                if group in self._xstep:
                    xstep[group] = float(self._xstep[group])

                    if xstep[group] < minimum_xstep:
                        print(
                            f"WARNING: Monte Carlo step size for continuous group "
                            f"'{group}' is below the minimum {minimum_xstep}. "
                            f"Using {minimum_xstep}."
                        )
                        xstep[group] = minimum_xstep
                    elif xstep[group] > maximum_xstep:
                        print(
                            f"WARNING: Monte Carlo step size for continuous group "
                            f"'{group}' is above the maximum {maximum_xstep}. "
                            f"Using {maximum_xstep}."
                        )
                        xstep[group] = maximum_xstep
                else:
                    xstep[group] = boundary_range / 10.0

                self._xstep[group] = xstep[group]
            elif group in self._xstep:
                xstep[group] = int(self._xstep[group])
            elif changemove[group] > 0.0:
                xstep[group] = int(len(population[group]) / 2)
            else:
                xstep[group] = 0

            if xsampling[group] == 1:
                print(f"        Maximum step size: {xstep[group]}")

                if group in self._xincrement:
                    try:
                        configured_increment = float(self._xincrement[group])
                    except (TypeError, ValueError, OverflowError) as error:
                        raise MOSAError(
                            f"Step increment for group '{group}' must be a positive number!"
                        ) from error

                    if (
                        not isfinite(configured_increment)
                        or configured_increment <= 0.0
                    ):
                        raise MOSAError(
                            f"Step increment for group '{group}' must be a positive number!"
                        )

                    xincrement[group] = configured_increment
                    update_increment_count(group)
                    print(f"        Step increment: {xincrement[group]}")

                if self._xcache and group not in self._xincrement:
                    print(
                        "WARNING: The solution cache is enabled, but no Monte Carlo "
                        f"step increment was set for continuous group '{group}'. "
                        "This may prevent effective use of the cache."
                    )
            elif (
                xsampling[group] == 0
                and (changemove[group] + insordelmove[group]) > 0.0
            ):
                if xstep[group] > len(population[group]) / 2 or xstep[group] <= 0:
                    xstep[group] = int(len(population[group]) / 2)

                if xstep[group] >= MIN_STEP_LENGTH:
                    print(f"        Maximum step size: {xstep[group]}")
                else:
                    print("        Elements selected at random from the population")

            if (
                xsampling[group] == 0
                and (changemove[group] + insordelmove[group]) > 0.0
            ):
                if len(population[group]) == 1:
                    lstep[group] = 0
                else:
                    lstep[group] = choice(len(population[group]))

            if xnel[group] == 1 and insordelmove[group] == 0.0:
                changemove[group] = 1.0
                swapmove[group] = 0.0

            if len(population[group]) == 0 and insordelmove[group] == 0:
                changemove[group] = 0.0
                swapmove[group] = 1.0

        self._groupstates = {}

        for group in groups:
            try:
                self._groupstates[group] = _GroupState.create(
                    population[group],
                    xnel[group],
                    xcurr[group] if from_saved_state else None,
                )
            except (TypeError, ValueError, OverflowError) as error:
                if from_saved_state:
                    raise MOSAError(
                        f"Saved solution group '{group}' has an incompatible format!"
                    ) from error

                raise

        if from_archive:
            for group in groups:
                self.__restore_archive_group_state(
                    group,
                    self._groupstates[group],
                    xdistinct.get(group, False),
                )

        if from_saved_state:
            xcurr = {
                group: self._groupstates[group].decode_solution() for group in groups
            }

        print("------")

        if from_archive:
            print("Initial solution loaded from the archive...")
        else:
            print("Initializing with a random solution from scratch...")

            for group in groups:
                state = self._groupstates[group]

                if xnel[group] == 1:
                    if xsampling[group] == 0:
                        m = choice(len(state.population))
                        state.solution = np.asarray(
                            [state.population[m]], dtype=state.population.dtype
                        )

                        if xdistinct[group]:
                            state.population = np.delete(state.population, m)
                    else:
                        state.solution = np.asarray(
                            [uniform(xbounds[group][0], xbounds[group][1])],
                            dtype=np.float64,
                        )
                else:
                    values: list[Any] = []

                    for j in range(xnel[group]):
                        if xsampling[group] == 0:
                            m = choice(len(state.population))
                            values.append(state.population[m])

                            if xdistinct[group]:
                                state.population = np.delete(state.population, m)
                        else:
                            values.append(uniform(xbounds[group][0], xbounds[group][1]))

                    state.solution = np.asarray(
                        values,
                        dtype=(
                            np.float64 if state.continuous else state.population.dtype
                        ),
                    )

                    if xsort[group]:
                        state.solution.sort()

                xcurr[group] = state.decode_solution()

            fcurr = self.__evaluate_solution(func, xcurr)

            updated = self.__updatearchive(xcurr, fcurr)

            if self._trackoptprogress:
                if len(fcurr) == 1:
                    self._f.append(fcurr[0])
                else:
                    self._f.append(fcurr)

        print("Done!")
        print("------")

        if len(fcurr) == len(self._weight):
            weight = self._weight.copy()
        else:
            weight = [1.0 for k in range(len(fcurr))]
        self.__validate_calibration_weights(weight)

        automatic_high_temperature = self._autohightemp and not self._initempset
        if automatic_high_temperature:
            initial_temperature = self.__estimate_initial_temperature(fcurr)
            initial_scale = sum(abs(float(value)) for value in fcurr) / len(fcurr)
            self._temp = [
                initial_temperature * self._decrease**i for i in range(self._ntemp)
            ]
            if self._verbose:
                print("Automatic high-temperature calibration enabled.")
                print(f"Initial objective scale: {initial_scale:.6e}")
                print("Initial calibration temperature: " f"{initial_temperature:.6e}")
        else:
            self._temp = [self._initemp * self._decrease**i for i in range(self._ntemp)]
            if self._verbose and self._autohightemp and self._initempset:
                print(
                    "Explicit initial temperature provided; automatic "
                    "high-temperature calibration is disabled for this run."
                )

        if self._adaptsel:
            adaptative_quality = {group: 0.0 for group in groups}
            adaptative_maximum_deltas = np.zeros(len(fcurr), dtype=float)

        if not self._verbose:
            print(f"Starting at temperature: {self._temp[0]:.6f}")
            print("Evolving solutions to the problem, please wait...")

        archive_dirty = updated == 1

        for temperature_index, temp in enumerate(self._temp, start=1):
            collect_calibration = automatic_high_temperature and temperature_index == 1
            reduced_delta_samples: list[list[float]] = []
            evaluated_trials = 0
            accepted_evaluated_trials = 0
            if self._verbose:
                print(f"TEMPERATURE: {temp:.6f}")
                current_selection_probabilities = tuple(
                    selection_weights[selected_group] / totlength
                    for selected_group in groups
                )
                if current_selection_probabilities != displayed_selection_probabilities:
                    print("    Group selection probabilities:")
                    for selected_group, selection_probability in zip(
                        groups, current_selection_probabilities
                    ):
                        print(
                            f"        {selected_group}: " f"{selection_probability:.6f}"
                        )
                    displayed_selection_probabilities = current_selection_probabilities

            nupdated = 0
            naccept = 0
            if self._adaptxstep:
                corana_attempts = {
                    group: 0
                    for group in groups
                    if xsampling[group] == 1 and group not in xincrement
                }
                corana_accepts = corana_attempts.copy()

            for j in range(self._niter):
                selstep = chosen = old = new = None
                population_update: tuple[str, int | None, Any] | None = None

                r = uniform(0.0, totlength)

                for group in groups:
                    if r < sellength[group]:
                        break

                r = uniform(
                    0.0, (changemove[group] + swapmove[group] + insordelmove[group])
                )

                state = self._groupstates[group]
                candidate = (
                    state.solution[0] if state.scalar_output else state.solution.copy()
                )
                encoded_population = state.population

                if r < changemove[group] or r >= (changemove[group] + swapmove[group]):
                    if xnel[group] > 1:
                        old = choice(len(candidate))

                    if xsampling[group] == 0 and len(encoded_population) > 0:
                        for _ in range(MAX_FAILED):
                            if len(encoded_population) == 1:
                                new = 0
                            elif xstep[group] >= MIN_STEP_LENGTH:
                                selstep = int(
                                    round(triangular(-xstep[group], 0, xstep[group]), 0)
                                )
                                new = lstep[group] + selstep

                                if new >= len(encoded_population):
                                    new -= len(encoded_population)
                                elif new < 0:
                                    new += len(encoded_population)
                            else:
                                new = choice(len(encoded_population))

                            if r >= changemove[group] or xdistinct[group]:
                                break
                            else:
                                if xnel[group] == 1:
                                    if not state.equal(
                                        candidate, encoded_population[new]
                                    ):
                                        break
                                else:
                                    if not state.equal(
                                        candidate[old], encoded_population[new]
                                    ):
                                        break
                        else:
                            new = None

                if xsampling[group] == 0 and r < changemove[group] and new is None:
                    if insordelmove[group] > 0.0:
                        r = changemove[group] + swapmove[group]
                    elif swapmove[group] > 0.0 and xnel[group] > 1:
                        r = changemove[group]
                    else:
                        if self._verbose:
                            print(
                                f"WARNING!!!!!! It was not possible to find an element in group '{group}' in the population to update the solution at iteration {j}!"
                            )

                        continue

                if r < changemove[group]:
                    if xsampling[group] == 0:
                        if xdistinct[group]:
                            if xnel[group] == 1:
                                population_update = (
                                    "replace",
                                    new,
                                    candidate,
                                )
                                candidate = encoded_population[new]
                            else:
                                population_update = (
                                    "replace",
                                    new,
                                    candidate[old],
                                )
                                candidate[old] = encoded_population[new]
                        else:
                            if xnel[group] == 1:
                                candidate = encoded_population[new]
                            else:
                                candidate[old] = encoded_population[new]
                    else:
                        if group in xincrement_count:
                            mc_step_increment = -float(xstep[group]) + (
                                choice(xincrement_count[group]) * xincrement[group]
                            )
                        else:
                            mc_step_increment = uniform(-xstep[group], xstep[group])

                        if xnel[group] == 1:
                            candidate += mc_step_increment
                            if (
                                candidate > xbounds[group][1]
                                or candidate < xbounds[group][0]
                            ):
                                candidate = xbounds[group][0] + (
                                    (candidate - xbounds[group][0])
                                    % (xbounds[group][1] - xbounds[group][0])
                                )
                        else:
                            candidate[old] += mc_step_increment
                            if (
                                candidate[old] > xbounds[group][1]
                                or candidate[old] < xbounds[group][0]
                            ):
                                candidate[old] = xbounds[group][0] + (
                                    (candidate[old] - xbounds[group][0])
                                    % (xbounds[group][1] - xbounds[group][0])
                                )

                    if xsort[group] and xnel[group] > 1:
                        candidate.sort()
                elif r < (changemove[group] + swapmove[group]):
                    for _ in range(int(len(candidate) / 2)):
                        chosen = choice(len(candidate), 2, False)

                        if not state.equal(candidate[chosen[0]], candidate[chosen[1]]):
                            candidate[chosen[0]], candidate[chosen[1]] = (
                                candidate[chosen[1]],
                                candidate[chosen[0]],
                            )

                            break
                    else:
                        if self._verbose:
                            print(
                                f"WARNING!!!!!! Failed {int(len(candidate)/2)} times to find different elements in group '{group}' for swapping at iteration {j}!"
                            )

                        continue
                else:
                    if len(candidate) == 1:
                        r = 0.0
                    elif (
                        xsampling[group] == 0 and len(encoded_population) == 0
                    ) or len(candidate) >= maxnel[group]:
                        r = 1.0
                    else:
                        r = uniform(0.0, 1.0)

                    if r < 0.5:
                        if xsampling[group] == 0:
                            candidate = np.append(candidate, encoded_population[new])

                            if xdistinct[group]:
                                population_update = ("remove", new, None)
                        else:
                            candidate = np.append(
                                candidate, uniform(xbounds[group][0], xbounds[group][1])
                            )

                        if xsort[group]:
                            candidate.sort()
                    else:
                        if xsampling[group] == 0 and xdistinct[group]:
                            population_update = ("append", None, candidate[old])

                        candidate = np.delete(candidate, old)

                xtmp = xcurr.copy()
                xtmp[group] = (
                    state.decode_value(candidate)
                    if state.scalar_output
                    else state.decode(candidate)
                )
                continuous_change = (
                    self._adaptxstep
                    and xsampling[group] == 1
                    and group not in xincrement
                    and r < changemove[group]
                )
                if continuous_change:
                    corana_attempts[group] += 1

                ftmp = self.__evaluate_solution(func, xtmp)

                reduced_delta, gamma = self.__trial_acceptance(
                    fcurr,
                    ftmp,
                    weight,
                    temp,
                    collect_reduced_delta=collect_calibration,
                )
                if collect_calibration:
                    assert reduced_delta is not None
                    reduced_delta_samples.append(reduced_delta)
                    evaluated_trials += 1

                if gamma == 1.0 or uniform(0.0, 1.0) < gamma:
                    if collect_calibration:
                        accepted_evaluated_trials += 1
                    if xsampling[group] == 0 and new is not None:
                        lstep[group] = new

                    if self._adaptsel:
                        reward = _adaptative_selection_reward(
                            fcurr, ftmp, adaptative_maximum_deltas
                        )
                        previous_quality = adaptative_quality[group]
                        adaptative_quality[group] = previous_quality + (
                            self._xselalpha * (reward - previous_quality)
                        )

                    fcurr = ftmp
                    xcurr = xtmp
                    if state.scalar_output:
                        state.solution[0] = candidate
                    else:
                        state.solution = candidate

                    if population_update is not None:
                        action, index, value = population_update

                        if action == "replace":
                            assert index is not None
                            state.population[index] = value
                        elif action == "remove":
                            assert index is not None
                            state.population = np.delete(state.population, index)
                        else:
                            state.population = np.append(state.population, value)

                    naccept += 1
                    if continuous_change:
                        corana_accepts[group] += 1
                    updated = self.__updatearchive(xcurr, fcurr)
                    nupdated += updated
                    archive_dirty = archive_dirty or updated == 1

                    if updated == 1:
                        narchivereject = 0
                    else:
                        narchivereject += 1
                else:
                    narchivereject += 1

                if self._trackoptprogress:
                    if len(fcurr) == 1:
                        self._f.append(fcurr[0])
                    else:
                        self._f.append(fcurr)

                if narchivereject >= self._maxarchivereject:
                    if self._verbose:
                        print(
                            f"    Insertion in the archive consecutively rejected {self._maxarchivereject} times!"
                        )
                        print(f"    Stoping at iteration {j}...")
                    else:
                        print(
                            "Too many attempts to insert a solution in the archive failed!"
                        )
                        print(f"Stopping at temperature: {temp:.6f}")

                    print("------")
                    print("\n--- THE END ---")

                    if archive_dirty:
                        self.savex()

                    self.__remove_json_backup(self._archivefile)
                    return

            if collect_calibration:
                if reduced_delta_samples:
                    expected_acceptance = self.__expected_acceptance(
                        reduced_delta_samples, temp
                    )
                    observed_acceptance = accepted_evaluated_trials / evaluated_trials
                    if self._verbose:
                        print(
                            "    Expected acceptance at initial temperature: "
                            f"{expected_acceptance:.6f}"
                        )
                        print(
                            "    Observed acceptance at initial temperature: "
                            f"{observed_acceptance:.6f}"
                        )
                        print(
                            "    Target high-temperature acceptance: "
                            f"{self._hightempaccthresh:.6f}"
                        )

                    if (
                        expected_acceptance < self._hightempaccthresh
                        and self._ntemp >= 2
                    ):
                        high_temperature = self.__estimate_high_temperature(
                            temp,
                            reduced_delta_samples,
                            self._hightempaccthresh,
                        )
                        self._temp[1:] = [
                            high_temperature * self._decrease**i
                            for i in range(self._ntemp - 1)
                        ]
                        if self._verbose:
                            print(
                                "    Estimated high temperature: "
                                f"{high_temperature:.6e}"
                            )
                            print(
                                "    Temperature scale factor: "
                                f"{high_temperature / temp:.6f}"
                            )
                            print(
                                "    Starting geometric quench after the "
                                "calibrated high-temperature stage."
                            )
                    elif self._verbose:
                        if expected_acceptance >= self._hightempaccthresh:
                            print(
                                "    Initial calibration temperature satisfies "
                                "the target acceptance."
                            )
                            print("    Starting geometric quench.")
                        else:
                            print(
                                "    A higher calibrated stage cannot be executed "
                                "because only one temperature is configured."
                            )
                elif self._verbose:
                    print(
                        "    Automatic high-temperature calibration could not "
                        "estimate acceptance because no trial move was evaluated "
                        "at the initial temperature."
                    )

            if self._adaptxstep:
                for continuous_group, attempted_moves in corana_attempts.items():
                    adjusted_xstep = corana_step_length(
                        float(xstep[continuous_group]),
                        corana_accepts[continuous_group],
                        attempted_moves,
                    )
                    minimum_xstep, maximum_xstep = xstep_bounds[continuous_group]
                    xstep[continuous_group] = min(
                        max(adjusted_xstep, minimum_xstep), maximum_xstep
                    )

            if self._adaptsel:
                selection_temperature = (
                    self._temp[temperature_index]
                    if temperature_index < len(self._temp)
                    else temp
                )
                probabilities = _adaptative_selection_probabilities(
                    [adaptative_quality[group] for group in groups],
                    selection_temperature,
                    minimum_selection_probability,
                )
                selection_weights = dict(zip(groups, probabilities.tolist()))
                self._xselweight.update(selection_weights)
                totlength = 0.0
                for selected_group in groups:
                    totlength += selection_weights[selected_group]
                    sellength[selected_group] = totlength

            final_temperature = temperature_index == len(self._temp)
            archive_save_due = final_temperature or (
                self._archivesaveint > 0
                and (
                    temperature_index == 1
                    or temperature_index % self._archivesaveint == 0
                )
            )

            if self._verbose:
                if naccept > 0:
                    print(f"    Number of accepted moves: {naccept}.")
                    print(f"    Fraction of accepted moves: {naccept/self._niter:.6f}.")

                    if nupdated > 0:
                        print(f"    Number of archive updates: {nupdated}.")
                        print(
                            f"    Fraction of archive updates in accepted moves: {nupdated/naccept:.6f}."
                        )
                    else:
                        print("    No archive update.")
                else:
                    print("    No move accepted.")

                print("------")

            if archive_dirty and archive_save_due:
                self.savex()
                archive_dirty = False

        if not self._verbose:
            print("Maximum number of temperatures reached!")
            print(f"Stopping at temperature:  {temp:.6f}.")
            print("------")

        print("\n--- THE END ---")
        self.__remove_json_backup(self._archivefile)

    def prune_dominated(self, xset: Archive | None = None) -> Archive:
        """
        Returns a subset of the full or reduced solution archive containing only
        non-dominated solutions.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        ### Returns

        Solution archive with non-dominated solutions.
        """

        xset = self.__checkarchive(xset)

        if len(xset["x"]) <= 1:
            return xset

        tmpdict: dict[str, list] = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]
        f_arr = np.asarray(f, dtype=float)
        keep_mask = self.__non_dominated_mask(f_arr)

        tmpdict["x"] = [v for i, v in enumerate(x) if keep_mask[i]]
        tmpdict["f"] = [v for i, v in enumerate(f) if keep_mask[i]]

        return tmpdict

    def savex(self, xset: Archive | None = None, archive_file: str = "") -> None:
        """
        Saves the solution archive into a text file in JSON format.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `archive_file`: name of the archive file.

        The default is an empty string, which means the main archive file.
        """

        xset = self.__checkarchive(xset)

        if isinstance(archive_file, str):
            archive_file = archive_file.strip()

            if len(archive_file) == 0:
                archive_file = self._archivefile
        else:
            raise MOSAError("The name of the archive file must be a string!")

        self.__write_json_atomic(xset, archive_file)

    def loadx(self, archive_file: str = "") -> None:
        """
        Loads solutions from a JSON file into the solution archive.

        ### Parameters

        `archive_file`: name of the archive file.

        The default is an empty string, which means the main archive file will
        be used.
        """

        if isinstance(archive_file, str):
            archive_file = archive_file.strip()

            if len(archive_file) == 0:
                archive_file = self._archivefile
        else:
            raise MOSAError("Name of the archive file must be a string!")

        if not self.__load_archive_file(archive_file):
            print(f"File {archive_file} not found or invalid!")

    def trimx(
        self,
        xset: Archive | None = None,
        thresholds: Sequence[Number | None] | np.ndarray | None = None,
    ) -> Archive:
        """
        Extracts solutions where the objective values are less than or equal to
        the thresholds.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `thresholds`: maximum values of the objective functions.

        The default is an empty list.

        ### Returns

        Solution archive with only the selected solutions.
        """

        xset = self.__checkarchive(xset)

        tmpdict: Archive = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]
        f_arr = np.array(f)

        threshold_values = [] if thresholds is None else list(thresholds)

        for i, value in enumerate(threshold_values):
            if value is None:
                threshold_values[i] = np.inf

        threshold_array = np.asarray(threshold_values)
        included = np.flatnonzero(np.all(f_arr <= threshold_array, axis=-1))

        if len(included) > 0:
            included_indices = included.tolist()
            tmpdict["x"] = [x[i] for i in included_indices]
            tmpdict["f"] = [f[i] for i in included_indices]
        else:
            raise RuntimeError("No solution remained in the reduced archive!")

        return tmpdict

    def reducex(
        self, xset: Archive | None = None, index: int = 0, nel: int = 5
    ) -> Archive:
        """
        Reduces and sorts in ascending order the archive according to the selected
        objective function.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `index`: index of the objective function.

        The default is 0.

        `nel`: number of solutions stored in the reduced solution archive.

        The default is 5.

        ### Returns

        Reduced solution archive.
        """

        xset = self.__checkarchive(xset)

        tmpdict: Archive = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]

        if nel > len(f):
            nel = len(f)

        indexlist = sorted(range(len(f)), key=lambda i: f[i][index])[:nel]

        tmpdict["x"] = [x[i] for i in indexlist]
        tmpdict["f"] = [f[i] for i in indexlist]

        return tmpdict

    def bestx(
        self,
        xset: Archive | None = None,
        weights: Sequence[Number] | np.ndarray | None = None,
    ) -> Archive:
        """
        Selects the best solution in the archive by applying the TOPSIS method
        to the objective values.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `weights`: weights of the objective functions.

        The default is an empty list, which means the same weight (1.0) for all
        objective functions.

        ### Returns

        Solution archive containing only the best solution.
        """

        xset = self.__checkarchive(xset)

        if len(xset["x"]) == 1:
            return xset

        tmpdict: Archive = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]
        f_arr = np.asarray(f, dtype=float)

        if f_arr.ndim != 2:
            raise MOSAError(
                "The objective values in the solution archive must define a 2D array!"
            )

        if weights is None or len(weights) == 0:
            weights = np.ones(f_arr.shape[1], dtype=float)
        else:
            if len(weights) != f_arr.shape[1]:
                raise MOSAError(
                    "The number of weights must be equal to the number of objective functions!"
                )

            weights = np.asarray(weights, dtype=float)

            if np.any(weights < 0.0):
                raise MOSAError("The weights must be non-negative!")

            if weights.sum() == 0.0:
                raise MOSAError("The sum of the weights must be greater than zero!")

        weights = weights / weights.sum()

        col_norms = np.linalg.norm(f_arr, axis=0)
        col_norms[col_norms == 0.0] = 1.0

        weighted = (f_arr / col_norms) * weights

        ideal_positive = weighted.min(axis=0)
        ideal_negative = weighted.max(axis=0)

        dist_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
        dist_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))

        denominator = dist_positive + dist_negative

        with np.errstate(invalid="ignore", divide="ignore"):
            closeness = np.where(denominator == 0.0, 0.0, dist_negative / denominator)

        ibest = int(np.argmax(closeness))

        tmpdict["x"].append(x[ibest])
        tmpdict["f"].append(f[ibest])

        return tmpdict

    def mergex(self, xset_list: list[Archive] | tuple[Archive, ...]) -> Archive:
        """
        Merges two or more solution archives into a single solution archive.

        ### Parameters

        `xset_list`: solution archives to be merged.

        ### Returns

        Merged solution archives.
        """

        tmpdict: Archive = {"x": [], "f": []}

        if len(xset_list) < 2:
            raise MOSAError("Nothing to be done!")

        for xset in xset_list:
            xset = self.__checkarchive(xset)

            tmpdict["x"] += xset["x"]
            tmpdict["f"] += xset["f"]

        return tmpdict

    def copyx(self, xset: Archive | None = None) -> Archive:
        """
        Returns a copy of the solution archive.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        ### Returns

        Copy of the solution archive.
        """

        xset = self.__checkarchive(xset)

        return deepcopy(xset)

    def printx(self, xset: Archive | None = None) -> None:
        """
        Prints the solutions in the solution archive in human readable format.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.
        """

        xset = self.__checkarchive(xset)

        for i in range(len(xset["x"])):
            s = str(xset["x"][i]).translate(str.maketrans("", "", "{}'\""))

            print(f"{i}) {s} ===> {xset['f'][i]}")

    def sizex(self, xset: Archive | None = None) -> int:
        """
        Returns the number of solutions stored in the archive.

        ### Parameters

        `xset`: full or reduced solution archive.

        ### Returns

        Number of solutions stored in the archive.
        """

        xset = self.__checkarchive(xset)

        return len(xset["x"])

    def plot_front(
        self,
        xset: Archive | None = None,
        index1: int = 0,
        index2: int = 1,
        index3: int | None = None,
        file: str | None = None,
        label: Sequence[str] = (),
    ) -> None:
        """
        Plots 2D or 3D scatter plots of selected objective values.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `index1`: index of the objective function displayed along x-axis.

        The default is 0.

        `index2`: index of the objective function displayed along y-axis.

        The default is 1.

        `index3`: index of the objective function displayed along z-axis.

        The default is `None`, which means a 2D plot will be created.

        `file`: name of the image file where the plot will be saved.

        The default is `None`, which means that no figure will be created.

        `label`: axis labels in objective order, provided as a list or tuple.

        The default is an empty tuple, which uses `f0`, `f1`, ... as labels.
        """

        try:
            import matplotlib.pyplot as plt
        except:
            raise MOSAError("Matplotlib is not available in your system!")

        xset = self.__checkarchive(xset)

        nobj = len(xset["f"][0])
        if not isinstance(label, (list, tuple)):
            raise MOSAError("Axis labels must be provided in a list or tuple!")

        if len(label) not in (0, nobj):
            raise MOSAError(
                "The number of axis labels must be zero or equal to the number "
                "of objective functions!"
            )

        axis_labels = list(label) if len(label) > 0 else [f"f{i}" for i in range(nobj)]
        indices = [index1, index2]

        if index3 is not None:
            indices.append(index3)

        if any(index < 0 or index >= nobj for index in indices):
            raise MOSAError("Index out of range!")

        if len(set(indices)) != len(indices):
            raise MOSAError("Objective function indices must be different!")

        f: list[list[Number]] = [[] for _ in indices]

        for objective_values in xset["f"]:
            for axis, index in enumerate(indices):
                f[axis].append(objective_values[index])

        fig = plt.figure()

        if index3 is None:
            ax = fig.add_subplot()
            ax.set_xlabel(axis_labels[index1])
            ax.set_ylabel(axis_labels[index2])
            ax.grid()
            ax.scatter(f[0], f[1])
        else:
            ax = fig.add_subplot(projection="3d")
            ax.set_xlabel(axis_labels[index1])
            ax.set_ylabel(axis_labels[index2])
            ax.set_zlabel(axis_labels[index3])
            ax.grid()
            ax.scatter(f[0], f[1], f[2])

        if file is not None and len(file) > 0:
            fig.savefig(file)

        plt.show()

    def get_stats(self, xset: Archive | None = None) -> dict[str, list[float]]:
        """
        Retrieves the minimum, maximum, average and standard deviation values of
        the objectives.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        ### Returns

        Minimum, maximum, average and standard deviation values of the objectives.
        """

        xset = self.__checkarchive(xset)

        f_arr = np.array(xset["f"])
        fmin = f_arr.min(axis=0)
        fmax = f_arr.max(axis=0)
        favg = f_arr.mean(axis=0)
        fstd = f_arr.std(axis=0)

        return {
            "Min": fmin.astype(float).tolist(),
            "Max": fmax.astype(float).tolist(),
            "Avg": favg.astype(float).tolist(),
            "Std": fstd.astype(float).tolist(),
        }

    def __estimate_initial_temperature(
        self, objective_values: ObjectiveValues
    ) -> float:
        if not objective_values:
            raise MOSAError(
                "Initial objective values must be a non-empty sequence of finite numbers!"
            )

        try:
            values = [float(value) for value in objective_values]
        except (TypeError, ValueError, OverflowError) as error:
            raise MOSAError(
                "Initial objective values must be a non-empty sequence of finite numbers!"
            ) from error

        if not all(isfinite(value) for value in values):
            raise MOSAError(
                "Initial objective values must be a non-empty sequence of finite numbers!"
            )

        objective_scale = sum(abs(value) for value in values) / len(values)
        if not isfinite(objective_scale):
            raise MOSAError("Initial objective scale must be finite!")

        if objective_scale == 0.0:
            temperature = 1.0
        else:
            try:
                temperature = 10.0 ** ceil(log10(objective_scale))
            except (OverflowError, ValueError) as error:
                raise MOSAError(
                    "Automatic initial temperature must be finite and greater than zero!"
                ) from error

        if not isfinite(temperature) or temperature <= 0.0:
            raise MOSAError(
                "Automatic initial temperature must be finite and greater than zero!"
            )
        return float(temperature)

    @staticmethod
    def __validate_calibration_weights(weights: ObjectiveWeightValues) -> None:
        try:
            valid = all(
                not isinstance(weight, bool)
                and isfinite(float(weight))
                and float(weight) > 0.0
                for weight in weights
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not weights or not valid:
            raise MOSAError(
                "Objective weights used for automatic high-temperature calibration "
                "must be finite numbers greater than zero!"
            )

    def __reduced_objective_deltas(
        self,
        current_values: ObjectiveValues,
        trial_values: ObjectiveValues,
        weights: ObjectiveWeightValues,
    ) -> list[float]:
        if not (
            len(current_values) == len(trial_values) == len(weights)
            and len(trial_values) > 0
        ):
            raise MOSAError(
                "Current objectives, trial objectives, and weights must have "
                "the same non-zero length!"
            )
        self.__validate_calibration_weights(weights)

        try:
            current = [float(value) for value in current_values]
            trial = [float(value) for value in trial_values]
        except (TypeError, ValueError, OverflowError) as error:
            raise MOSAError("Objective values must be numbers!") from error

        reduced_delta: list[float] = []
        for current_value, trial_value, weight in zip(current, trial, weights):
            if isnan(trial_value) or trial_value == inf:
                # NaN and +inf are invalid or maximally bad trial objectives for a
                # minimization problem. Mapping either to +inf makes the complete
                # proposal's acceptance probability zero.
                delta = inf
            elif trial_value == -inf:
                # -inf is the best possible value in a minimization problem, so its
                # individual Metropolis probability is exp(0) == 1.
                delta = 0.0
            elif isnan(current_value) or current_value == inf:
                # A finite trial improves upon an invalid or +inf current value.
                delta = 0.0
            elif current_value == -inf:
                # Moving from -inf to a finite value is a maximal worsening.
                delta = inf
            else:
                delta = max((trial_value - current_value) / float(weight), 0.0)
            reduced_delta.append(delta)

        if any(isnan(delta) or delta < 0.0 for delta in reduced_delta):
            raise MOSAError(
                "Reduced objective deltas must be non-negative numbers other than NaN!"
            )
        return reduced_delta

    def __trial_acceptance(
        self,
        current_values: ObjectiveValues,
        trial_values: ObjectiveValues,
        weights: ObjectiveWeightValues,
        temperature: float,
        *,
        collect_reduced_delta: bool,
    ) -> tuple[list[float] | None, float]:
        """Compute trial deltas and acceptance in one pass inside evolve."""

        if not (
            len(current_values) == len(trial_values) == len(weights)
            and len(trial_values) > 0
        ):
            raise MOSAError(
                "Current objectives, trial objectives, and weights must have "
                "the same non-zero length!"
            )

        reduced_delta: list[float] | None = [] if collect_reduced_delta else None
        gamma_product = 1.0
        maximum_probability = 0.0
        invalid_delta = False

        try:
            for current_value, trial_value, weight in zip(
                current_values, trial_values, weights
            ):
                current = float(current_value)
                trial = float(trial_value)

                if isnan(trial) or trial == inf:
                    delta = inf
                elif trial == -inf:
                    delta = 0.0
                elif isnan(current) or current == inf:
                    delta = 0.0
                elif current == -inf:
                    delta = inf
                else:
                    delta = max((trial - current) / float(weight), 0.0)

                if reduced_delta is not None:
                    reduced_delta.append(delta)

                if not isfinite(delta):
                    invalid_delta = True
                elif not invalid_delta:
                    probability = exp(-delta / temperature)
                    gamma_product *= probability
                    if probability > maximum_probability:
                        maximum_probability = probability
        except (TypeError, ValueError, OverflowError) as error:
            raise MOSAError("Objective values must be numbers!") from error

        if invalid_delta:
            return reduced_delta, 0.0

        gamma = (1.0 - self._alpha) * gamma_product + (
            self._alpha * maximum_probability
        )
        return reduced_delta, min(max(float(gamma), 0.0), 1.0)

    def __acceptance_probability_from_reduced_delta(
        self, reduced_delta: Sequence[float], temperature: float
    ) -> float:
        if not isinstance(temperature, Real) or isinstance(temperature, bool):
            raise MOSAError("Temperature must be a finite number greater than zero!")
        temperature = float(temperature)
        if not isfinite(temperature) or temperature <= 0.0:
            raise MOSAError("Temperature must be a finite number greater than zero!")
        if not reduced_delta:
            raise MOSAError("At least one reduced objective delta is required!")

        try:
            deltas = [float(delta) for delta in reduced_delta]
        except (TypeError, ValueError, OverflowError) as error:
            raise MOSAError("Reduced objective deltas must be numbers!") from error
        if any(not isfinite(delta) for delta in deltas):
            return 0.0
        if not all(delta >= 0.0 for delta in deltas):
            raise MOSAError("Finite reduced objective deltas must be non-negative!")

        probabilities = [exp(-delta / temperature) for delta in deltas]
        gamma_product = 1.0
        for probability in probabilities:
            gamma_product *= probability
        gamma = (1.0 - self._alpha) * gamma_product + self._alpha * max(probabilities)
        if not isfinite(gamma):
            raise MOSAError("MOSA acceptance probability must be finite!")
        return min(max(float(gamma), 0.0), 1.0)

    def __expected_acceptance(
        self, samples: Sequence[Sequence[float]], temperature: float
    ) -> float:
        if not samples:
            raise MOSAError(
                "At least one calibration sample is required to estimate acceptance!"
            )
        return sum(
            self.__acceptance_probability_from_reduced_delta(sample, temperature)
            for sample in samples
        ) / len(samples)

    @staticmethod
    def __calibration_sample_statistics(
        samples: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute temperature-independent calibration sample statistics."""

        sums = np.empty(len(samples), dtype=float)
        minima = np.empty(len(samples), dtype=float)
        valid = np.ones(len(samples), dtype=bool)

        for index, sample in enumerate(samples):
            if not sample:
                raise MOSAError("At least one reduced objective delta is required!")
            try:
                deltas = np.asarray(sample, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise MOSAError("Reduced objective deltas must be numbers!") from error

            if np.any(deltas < 0.0):
                raise MOSAError("Finite reduced objective deltas must be non-negative!")
            if not np.all(np.isfinite(deltas)):
                valid[index] = False
                sums[index] = 0.0
                minima[index] = 0.0
            else:
                sums[index] = float(deltas.sum())
                minima[index] = float(deltas.min())

        return sums, minima, valid

    def __expected_acceptance_from_statistics(
        self,
        sums: np.ndarray,
        minima: np.ndarray,
        valid: np.ndarray,
        temperature: float,
    ) -> float:
        """Evaluate all calibration samples with vectorized NumPy operations."""

        if not isinstance(temperature, Real) or isinstance(temperature, bool):
            raise MOSAError("Temperature must be a finite number greater than zero!")
        temperature = float(temperature)
        if not isfinite(temperature) or temperature <= 0.0:
            raise MOSAError("Temperature must be a finite number greater than zero!")

        probabilities = np.zeros(sums.size, dtype=float)
        probabilities[valid] = (1.0 - self._alpha) * np.exp(
            -sums[valid] / temperature
        ) + self._alpha * np.exp(-minima[valid] / temperature)
        return float(probabilities.mean())

    def __estimate_high_temperature(
        self,
        initial_temperature: float,
        samples: Sequence[Sequence[float]],
        target_acceptance: float,
    ) -> float:
        if not samples:
            raise MOSAError(
                "At least one calibration sample is required to estimate acceptance!"
            )
        statistics = self.__calibration_sample_statistics(samples)

        def expected(temperature: float) -> float:
            return self.__expected_acceptance_from_statistics(*statistics, temperature)

        if expected(initial_temperature) >= target_acceptance:
            return float(initial_temperature)

        low = float(initial_temperature)
        high = 2.0 * low
        for _ in range(60):
            if not isfinite(high):
                break
            if expected(high) >= target_acceptance:
                break
            low = high
            high *= 2.0
        else:
            raise MOSAError(
                "Unable to bracket a temperature satisfying the target acceptance!"
            )

        if not isfinite(high) or expected(high) < target_acceptance:
            raise MOSAError(
                "Unable to bracket a temperature satisfying the target acceptance!"
            )

        for _ in range(100):
            if (high - low) / high <= 1e-6:
                break
            middle = 0.5 * (low + high)
            if expected(middle) >= target_acceptance:
                high = middle
            else:
                low = middle
        return high

    def __updatearchive(self, x: Solution, f: ObjectiveValues) -> int:
        """
        Appends a solution to the archive if it is not dominated by other existing
        solutions.

        ### Parameters

        `x`: solution.

        `f`: objective values.

        ### Returns

        1, if the archive is updated, or 0, if not.
        """

        archive_len = len(self._archivex)
        f_arr = np.asarray(f, dtype=float)

        if archive_len == 0:
            updated = True
        else:
            archive_arr = self._archivefarr[:archive_len]
            archive_dominates, candidate_dominates = self.__dominance_masks(
                archive_arr, f_arr
            )
            dominated_by_archive = np.any(archive_dominates)

            if dominated_by_archive:
                updated = False
            else:
                dominated_rows = np.flatnonzero(candidate_dominates)

                if archive_len < self._archivesize or dominated_rows.size > 0:
                    updated = True
                else:
                    updated = False

                if updated and dominated_rows.size > 0:
                    removed_solutions = [
                        (
                            self._archivex[int(row)],
                            archive_arr[int(row)].astype(float).tolist(),
                        )
                        for row in dominated_rows
                    ]

                    keep_mask = np.ones(archive_len, dtype=bool)
                    keep_mask[dominated_rows] = False
                    self._archivex = [
                        value for i, value in enumerate(self._archivex) if keep_mask[i]
                    ]
                    kept_count = int(np.count_nonzero(keep_mask))

                    if kept_count > 0:
                        self._archivefarr[:kept_count] = archive_arr[keep_mask]

                    archive_len = kept_count
                    self.__rebuild_archive_lookup()

                    for removed_x, removed_f in removed_solutions:
                        self.__cache_solution(removed_x, removed_f)

        if updated:
            self.__ensure_archive_capacity(archive_len + 1, len(f_arr))
            self._archivex.append(x)
            self._archivefarr[archive_len] = f_arr
            solution_key = _semantic_key(x)
            if solution_key is not None:
                self._archive_lookup[solution_key] = archive_len
            self.__remove_cached_solution(x)

        return int(updated)

    def __evaluate_solution(
        self, func: ObjectiveFunction, x: Solution
    ) -> ObjectiveValues:
        """Return previously computed objectives or evaluate and cache the solution."""

        if self._xcache:
            solution_key = _semantic_key(x)
            archive_index = (
                self._archive_lookup.get(solution_key)
                if solution_key is not None
                else self.__solution_index(self._archivex, x)
            )

            if archive_index is not None:
                return self._archivefarr[archive_index].astype(float).tolist()

            if solution_key is not None:
                cached = self._cache_lookup.get(solution_key)
                if cached is not None:
                    return list(cached[1])
            else:
                cache_index = self.__solution_index(self._cache["x"], x)
                if cache_index is not None:
                    return list(self._cache["f"][cache_index])

        objective_values = list(func(**x))
        self.__cache_solution(x, objective_values)
        return objective_values

    @staticmethod
    def __solution_index(solutions: list[Solution], x: Solution) -> int | None:
        """Find a semantically equal solution in a list."""

        for index, solution in enumerate(solutions):
            if Anneal.__solution_values_equal(solution, x):
                return index

        return None

    @staticmethod
    def __solution_values_equal(left: Any, right: Any) -> bool:
        """Compare nested solution values, including unhashable NumPy categories."""

        left_key = _semantic_key(left)
        right_key = _semantic_key(right)

        if left_key is not None and right_key is not None:
            return left_key == right_key

        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                Anneal.__solution_values_equal(left[key], right[key]) for key in left
            )

        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                Anneal.__solution_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )

        return _values_equal(left, right)

    def __cache_solution(self, x: Solution, f: ObjectiveValues) -> None:
        """Store one unique non-archive solution, evicting the oldest if needed."""

        if not self._xcache:
            return

        solution_key = _semantic_key(x)
        if solution_key is not None:
            if solution_key in self._archive_lookup:
                return
            if solution_key in self._cache_lookup:
                return
        else:
            if self.__solution_index(self._archivex, x) is not None:
                return

            if self.__solution_index(self._cache["x"], x) is not None:
                return

        if len(self._cache["x"]) >= self._xcachesize:
            evicted = self._cache["x"].pop(0)
            self._cache["f"].pop(0)
            evicted_key = _semantic_key(evicted)
            if evicted_key is not None:
                self._cache_lookup.pop(evicted_key, None)

        cached_x = deepcopy(x)
        cached_f = list(f)
        self._cache["x"].append(cached_x)
        self._cache["f"].append(cached_f)
        if solution_key is not None:
            self._cache_lookup[solution_key] = (cached_x, cached_f)

    def __remove_cached_solution(self, x: Solution) -> None:
        """Remove a solution from the cache after it enters the archive."""

        solution_key = _semantic_key(x)
        cache_index: int | None = None
        if solution_key is not None:
            cached = self._cache_lookup.pop(solution_key, None)
            if cached is not None:
                cached_x = cached[0]
                cache_index = next(
                    (
                        index
                        for index, solution in enumerate(self._cache["x"])
                        if solution is cached_x
                    ),
                    None,
                )
        else:
            cache_index = self.__solution_index(self._cache["x"], x)

        if cache_index is not None:
            self._cache["x"].pop(cache_index)
            self._cache["f"].pop(cache_index)

    def __rebuild_archive_lookup(self) -> None:
        """Rebuild the hash index for semantically hashable archive solutions."""

        self._archive_lookup.clear()
        for index, solution in enumerate(self._archivex):
            solution_key = _semantic_key(solution)
            if solution_key is not None:
                self._archive_lookup[solution_key] = index

    def __rebuild_cache_lookup(self) -> None:
        """Rebuild the hash index for semantically hashable cached solutions."""

        self._cache_lookup.clear()
        for solution, objective_values in zip(self._cache["x"], self._cache["f"]):
            solution_key = _semantic_key(solution)
            if solution_key is not None:
                self._cache_lookup[solution_key] = (solution, objective_values)

    @staticmethod
    def __dominance_masks(
        archive_arr: np.ndarray, f_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compare archive objectives with one candidate in compiled code."""

        return _dominance_masks_kernel(archive_arr, f_arr)

    @staticmethod
    def __non_dominated_mask(f_arr: np.ndarray) -> np.ndarray:
        """Return a mask for Pareto-optimal rows in the compiled kernel."""

        return _non_dominated_mask_kernel(f_arr)

    @staticmethod
    def __write_json_atomic(data: Any, destination: str) -> None:
        """Write compact JSON while retaining one recoverable generation."""

        target = os.path.abspath(destination)
        temporary = f"{target}.tmp"
        backup = f"{target}.bak"

        try:
            encoded = json.dumps(data, separators=(",", ":"))

            with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

            if os.path.exists(target):
                os.replace(target, backup)

            os.replace(temporary, target)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)

            raise

    @staticmethod
    def __remove_json_backup(destination: str) -> None:
        """Remove the recoverable generation after successful optimization."""

        backup = f"{os.path.abspath(destination)}.bak"

        try:
            os.remove(backup)
        except FileNotFoundError:
            pass

    def __load_archive_file(self, archive_file: str) -> bool:
        """Load the primary archive or its last complete backup."""

        for candidate in (archive_file, f"{archive_file}.bak"):
            try:
                with open(candidate, "r", encoding="utf-8") as stream:
                    archive = json.load(stream)

                if not isinstance(archive, dict) or not {
                    "x",
                    "f",
                }.issubset(archive):
                    raise MOSAError("Archive does not contain 'x' and 'f'!")

                archive = self.__checkarchive(archive)
                self.__set_archive_data(archive["x"], archive["f"])
            except (FileNotFoundError, OSError, TypeError, ValueError, MOSAError):
                continue

            if candidate != archive_file:
                print(f"Recovered archive from backup file {candidate}.")

            return True

        return False

    @staticmethod
    def __restore_archive_group_state(
        group: str, state: _GroupState, distinct: bool
    ) -> None:
        """Validate an archived solution and rebuild its available population."""

        if state.solution.size == 0:
            raise MOSAError(f"Archived solution group '{group}' is empty!")

        if state.solution.ndim != 1:
            raise MOSAError(
                f"Archived solution group '{group}' has an incompatible format!"
            )

        if state.continuous:
            lower = min(float(state.population[0]), float(state.population[1]))
            upper = max(float(state.population[0]), float(state.population[1]))

            if np.any(state.solution < lower) or np.any(state.solution > upper):
                raise MOSAError(
                    f"Archived solution group '{group}' is outside its boundaries!"
                )

            return

        available = state.population.copy()

        for solution_index, selected in enumerate(state.solution):
            match_index = None

            for population_index, candidate in enumerate(available):
                if state.equal(selected, candidate):
                    match_index = population_index
                    break

            if match_index is None:
                raise MOSAError(
                    f"Archived solution group '{group}' is incompatible with its "
                    "configured population!"
                )

            state.solution[solution_index] = available[match_index]

            if distinct:
                available = np.delete(available, match_index)

        if distinct:
            state.population = available

    def __checkarchive(self, xset: Archive | None = None) -> Archive:
        """
        Performs checks on the archive.

        ### Parameters

        `xset`: full or reduced solution archive.

        ### Returns

        Solution archive.
        """

        if not xset:
            return self.__archive_dict()

        if not ("x" in xset and "f" in xset):
            raise MOSAError("'x' and 'f' must be present in the archive!")

        if not (isinstance(xset["x"], list) and isinstance(xset["f"], list)):
            raise MOSAError("'x' and 'f' must be Python lists!")

        if len(xset["x"]) == 0:
            raise MOSAError("Archive is empty!")

        if len(xset["x"]) != len(xset["f"]):
            raise MOSAError("'x' and 'f' must have the same number of elements!")

        return xset

    def __archive_dict(self) -> Archive:
        """@private"""

        return {
            "x": self._archivex.copy(),
            "f": self._archivefarr[: len(self._archivex)].astype(float).tolist(),
        }

    def __set_archive_data(
        self, x_values: list[Solution], f_values: list[ObjectiveValues]
    ) -> None:
        """@private"""

        self._archivex = list(x_values)
        self.__rebuild_archive_lookup()

        if len(f_values) == 0:
            self._archivefarr = np.empty((0, 0), dtype=float)
            self._archivefcap = 0
        else:
            archive_f_arr = np.asarray(f_values, dtype=float)

            if archive_f_arr.ndim == 1:
                archive_f_arr = archive_f_arr.reshape(1, -1)

            self._archivefarr = archive_f_arr.copy()
            self._archivefcap = self._archivefarr.shape[0]

        for solution in self._archivex:
            self.__remove_cached_solution(solution)

    def __ensure_archive_capacity(self, rows: int, nf: int) -> None:
        """@private"""

        if self._archivefcap >= rows and self._archivefarr.shape[1] == nf:
            return

        if self._archivefcap == 0 or self._archivefarr.shape[1] != nf:
            new_capacity = max(rows, 1)
            new_archive_f_arr = np.empty((new_capacity, nf), dtype=float)
        else:
            new_capacity = max(rows, self._archivefcap * 2)
            new_archive_f_arr = np.empty((new_capacity, nf), dtype=float)
            active_rows = len(self._archivex)

            if active_rows > 0:
                new_archive_f_arr[:active_rows] = self._archivefarr[:active_rows]

        self._archivefarr = new_archive_f_arr
        self._archivefcap = new_capacity

    @property
    def population(self) -> Population:
        """
        Population where each group represents the data that can be used to achieve
        an optimized solution to the problem.
        """

        return self._population

    @population.setter
    def population(self, val: Population) -> None:
        if isinstance(val, dict) and val:
            self._population = val
        else:
            raise MOSAError("Population must be a non-empty dictionary!")

    @property
    def archive(self) -> Archive:
        """
        Solution archive.

        > [!WARNING]
        > The archive should not be changed manually.
        """

        return self.__archive_dict()

    @archive.setter
    def archive(self, val: Archive) -> None:
        if isinstance(val, dict) and val:
            if not ("x" in val.keys() and "f" in val.keys()):
                raise MOSAError("'x' and 'f' must be present in the archive!")
            else:
                if not (isinstance(val["x"], list) and isinstance(val["f"], list)):
                    raise MOSAError("'x' and 'f' must be Python lists!")
        else:
            raise MOSAError("The archive must be a non-empty dictionary!")

        self.__set_archive_data(val["x"], val["f"])

    @property
    def restart(self) -> bool:
        """
        Restarts from the last retained solution when an archive is available.

        The configured population rebuilds the available search space.

        The default is `True`.
        """

        return self._restart

    @restart.setter
    def restart(self, val: bool) -> None:
        if isinstance(val, bool):
            self._restart = val
        else:
            raise MOSAError("Restart must be a boolean!")

    @property
    def objective_weights(self) -> ObjectiveWeightValues:
        """
        Weights for the objectives.

        The default is [], which means the same weight (1.0) for all objectives.
        """

        return self._weight

    @objective_weights.setter
    def objective_weights(self, val: ObjectiveWeightValues) -> None:
        if isinstance(val, list):
            self._weight = val
        else:
            raise MOSAError("The weights must be provided in a list!")

    @property
    def initial_temperature(self) -> float:
        """
        Initial temperature.

        The numerical default is 1.0. Explicitly assigning this property takes
        precedence over automatic high-temperature calibration, even when
        `auto_high_temperature` is `True`.
        """

        return self._initemp

    @initial_temperature.setter
    def initial_temperature(self, val: Number) -> None:
        if isinstance(val, (int, float)) and val > 0.0:
            self._initemp = float(val)
            self._initempset = True
        else:
            raise MOSAError("Initial temperature must be a number greater than zero!")

    @property
    def auto_high_temperature(self) -> bool:
        """
        Enables automatic calibration of the high-temperature stage.

        The default is `False`. When enabled and `initial_temperature` has not
        been explicitly assigned, the first temperature is estimated from the
        initial objective scale. If its expected mean MOSA move-acceptance
        probability is below the configured target, one higher calibration stage
        is used before geometric quenching begins.
        """

        return self._autohightemp

    @auto_high_temperature.setter
    def auto_high_temperature(self, val: bool) -> None:
        if isinstance(val, bool):
            self._autohightemp = val
        else:
            raise MOSAError("Automatic high-temperature calibration must be a boolean!")

    @property
    def high_temperature_acceptance_threshold(self) -> float:
        """
        Target expected mean MOSA move-acceptance probability during automatic
        high-temperature calibration.

        The default is 0.8 and the valid range is strictly between zero and one.
        """

        return self._hightempaccthresh

    @high_temperature_acceptance_threshold.setter
    def high_temperature_acceptance_threshold(self, val: Number) -> None:
        valid = isinstance(val, Real) and not isinstance(val, bool)
        if valid:
            value = float(val)
            valid = isfinite(value) and 0.0 < value < 1.0
        if not valid:
            raise MOSAError(
                "High-temperature acceptance threshold must be a finite number "
                "greater than zero and less than one!"
            )
        self._hightempaccthresh = value

    @property
    def temperature_decrease_factor(self) -> float:
        """
        Decrease factor of the temperature.

        The default is 0.9.
        """

        return self._decrease

    @temperature_decrease_factor.setter
    def temperature_decrease_factor(self, val: float) -> None:
        if isinstance(val, float) and val > 0.0 and val < 1.0:
            self._decrease = val
        else:
            raise MOSAError(
                "Decrease factor must be a number greater than zero and less than one!"
            )

    @property
    def number_of_temperatures(self) -> int:
        """
        Number of temperatures.

        The default is 10.
        """

        return self._ntemp

    @number_of_temperatures.setter
    def number_of_temperatures(self, val: int) -> None:
        if isinstance(val, int) and val > 0:
            self._ntemp = val
        else:
            raise MOSAError(
                "Number of annealing temperatures must be an integer greater than zero!"
            )

    @property
    def number_of_iterations(self) -> int:
        """
        Number of Monte Carlo iterations per temperature.

        The default is 1,000.
        """

        return self._niter

    @number_of_iterations.setter
    def number_of_iterations(self, val: int) -> None:
        if isinstance(val, int) and val > 0:
            self._niter = val
        else:
            raise MOSAError(
                "Number of iterations must be an integer greater than zero!"
            )

    @property
    def archive_size(self) -> int:
        """
        Maximum number of solutions in the archive.

        The default is 1,000.
        """

        return self._archivesize

    @archive_size.setter
    def archive_size(self, val: int) -> None:
        if isinstance(val, int) and val > 0:
            self._archivesize = val
        else:
            raise MOSAError("The archive size must be an integer greater than zero!")

    @property
    def solution_cache(self) -> bool:
        """
        Enable the solution cache.

        The cache should ideally be enabled only for objective functions that are
        very computationally expensive. The default is `False`.
        """

        return self._xcache

    @solution_cache.setter
    def solution_cache(self, val: bool) -> None:
        if isinstance(val, bool):
            self._xcache = val
        else:
            raise MOSAError("Solution cache must be a boolean!")

    @property
    def solution_cache_size(self) -> int:
        """
        Maximum number of solutions in the solution cache.

        The default is 10,000.
        """

        return self._xcachesize

    @solution_cache_size.setter
    def solution_cache_size(self, val: int) -> None:
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            self._xcachesize = val
            overflow = len(self._cache["x"]) - val

            if overflow > 0:
                del self._cache["x"][:overflow]
                del self._cache["f"][:overflow]
                self.__rebuild_cache_lookup()
        else:
            raise MOSAError(
                "The solution cache size must be an integer greater than zero!"
            )

    @property
    def archive_file(self) -> str:
        """
        Name of the archive file.

        The default is 'archive.json'.
        """

        return self._archivefile

    @archive_file.setter
    def archive_file(self, val: str) -> None:
        if isinstance(val, str) and len(val.strip()) > 0:
            self._archivefile = val.strip()
        else:
            raise MOSAError("A file name must be provided!")

    @property
    def archive_save_interval(self) -> int:
        """
        Number of completed temperatures between automatic archive writes.

        The default is 10. Set it to 0 to write only when evolution finishes.
        Positive intervals also persist after the first completed temperature.
        The archive is written only when it has changed since the previous save.
        """

        return self._archivesaveint

    @archive_save_interval.setter
    def archive_save_interval(self, val: int) -> None:
        if isinstance(val, int) and val >= 0:
            self._archivesaveint = val
        else:
            raise MOSAError("Archive save interval must be a non-negative integer!")

    @property
    def maximum_archive_rejections(self) -> int:
        """
        Maximum number of consecutive times a solution insertion into the archive
        can be rejected.

        The default is 1,000.
        """

        return self._maxarchivereject

    @maximum_archive_rejections.setter
    def maximum_archive_rejections(self, val: int) -> None:
        if isinstance(val, int) and val > 0:
            self._maxarchivereject = val
        else:
            raise MOSAError(
                "Maximum archive rejections must be an integer greater than zero!"
            )

    @property
    def alpha(self) -> float:
        """
        Alpha parameter.

        The default is 0.0.
        """

        return self._alpha

    @alpha.setter
    def alpha(self, val: float) -> None:
        if isinstance(val, float) and val >= 0.0 and val <= 1.0:
            self._alpha = val
        else:
            raise MOSAError("Alpha must be a number between zero and one!")

    @property
    def number_of_elements(self) -> dict[str, int]:
        """
        Number of elements for each group in the solution.

        The default is {}, which means one element for all groups in the solutions.
        """

        return self._xnel

    @number_of_elements.setter
    def number_of_elements(self, val: dict[str, int]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, int) and value > 0:
                    self._xnel[key] = value
                else:
                    raise MOSAError(
                        f"Group '{key}' must be an integer greater than zero!"
                    )
        else:
            raise MOSAError("Number of elements must be provided as a dictionary!")

    @property
    def maximum_number_of_elements(self) -> dict[str, int]:
        """
        Maximum number of elements for each group in the solution, if the number of elements
        is variable.

        The default is {}, which means an unlimited number of elements.
        """

        return self._maxnel

    @maximum_number_of_elements.setter
    def maximum_number_of_elements(self, val: dict[str, int]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, int) and value >= 2:
                    self._maxnel[key] = value
                else:
                    raise MOSAError(
                        f"Group '{key}' must be an integer greater than or equal to 2!"
                    )
        else:
            raise MOSAError(
                "Maximum number of elements must be provided as a dictionary!"
            )

    @property
    def distinct_elements(self) -> dict[str, bool]:
        """
        Determines that an element cannot be repeated in a group in the solution.

        The default is {}, which means that repetitions are allowed.
        """

        return self._xdistinct

    @distinct_elements.setter
    def distinct_elements(self, val: dict[str, bool]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, bool):
                    self._xdistinct[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a boolean!")
        else:
            raise MOSAError(
                "Whether or not to repeat elements in the group in the solution must be provided as a dictionary!"
            )

    @property
    def adaptative_mc_step(self) -> bool:
        """
        Whether Corana's adaptative maximum step-length algorithm is enabled.

        The default is `False`. Only continuous groups without a configured step
        increment are affected.
        """

        return self._adaptxstep

    @adaptative_mc_step.setter
    def adaptative_mc_step(self, val: bool) -> None:
        if isinstance(val, bool):
            self._adaptxstep = val
        else:
            raise MOSAError("Corana usage must be a boolean!")

    @property
    def adaptative_selection(self) -> bool:
        """
        Enable adaptative selection of solution groups.

        When enabled, accepted moves assign each selected group a reward equal
        to the mean normalized variation across all objectives. At the end of
        each temperature, an exponential moving average of these rewards is
        converted into selection probabilities using a Boltzmann distribution.
        Every group retains a minimum selection probability of 1% whenever that
        floor is feasible.

        The default is `False`.
        """

        return self._adaptsel

    @adaptative_selection.setter
    def adaptative_selection(self, val: bool) -> None:
        if isinstance(val, bool):
            self._adaptsel = val
        else:
            raise MOSAError("Adaptative group selection usage must be a boolean!")

    @property
    def mc_step_size(self) -> dict[str, Number]:
        """
        Monte Carlo maximum step size for each group in the solution.

        The default is {}, which means one tenth of the boundary range for a
        continuous search space and half the number of elements in a population
        group for a discrete search space. Continuous step sizes are constrained
        between one hundredth and one half of the boundary range.
        """

        return self._xstep

    @mc_step_size.setter
    def mc_step_size(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (int, float)):
                    self._xstep[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a number!")
        else:
            raise MOSAError("Monte Carlo step sizes must be provided as a dictionary!")

    @property
    def mc_step_increment(self) -> dict[str, Number]:
        """
        Increment used to discretize continuous solution changes.

        The default is {}, which samples each continuous change uniformly between
        the negative and positive Monte Carlo maximum step size.
        """

        return self._xincrement

    @mc_step_increment.setter
    def mc_step_increment(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isfinite(value)
                    and value > 0.0
                ):
                    self._xincrement[key] = value
                else:
                    raise MOSAError(
                        f"Step increment for group '{key}' must be a positive number!"
                    )
        else:
            raise MOSAError("Step increments must be provided as a dictionary!")

    @property
    def change_value_move(self) -> dict[str, Number]:
        """
        Weight (non-normalized probability) to select a trial move where the value
        of a randomly selected element in a group in the solution will be modified
        as follows:

        - Discrete search space: values between the solution and the population
        are exchanged.

        - Continuous search space: the value of the solution element is randomly
        incremented/decremented.

        The default is {}, which means the weight to select this trial move is
        equal to 1.0.
        """

        return self._changemove

    @change_value_move.setter
    def change_value_move(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._changemove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def insert_or_delete_move(self) -> dict[str, Number]:
        """
        Weight (non-normalized probability) to select a trial move where an element
        will be inserted into or deleted from a group in the solution.

        The default is {}, which means this trial move is not allowed, i.e., the
        weight is equal to zero.
        """

        return self._insordelmove

    @insert_or_delete_move.setter
    def insert_or_delete_move(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._insordelmove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def swap_move(self) -> dict[str, Number]:
        """
        Weight (non-normalized probability) to select a trial move where elements
        will be swaped in the solution.

        The default is {}, which means this trial move is not allowed, i.e., the
        weight is equal to zero.
        """

        return self._swapmove

    @swap_move.setter
    def swap_move(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._swapmove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def sort_elements(self) -> dict[str, bool]:
        """
        Elements in a group in the solution will be sorted in ascending order.

        The default is {}, which means no sorting at all.
        """

        return self._xsort

    @sort_elements.setter
    def sort_elements(self, val: dict[str, bool]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, bool):
                    self._xsort[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a boolean!")
        else:
            raise MOSAError("Sort group elements must be provided as a dictionary!")

    @property
    def group_selection_weights(self) -> dict[str, Number]:
        """
        Selection weight for each group in the solution in a Monte Carlo iteration.

        The default value is {}, which means that all groups have the same selection
        weight, i.e., the same probability of being selected. When
        `adaptative_selection` is enabled, configured weights are used as initial
        probabilities and this dictionary is updated after every temperature.
        """

        return self._xselweight

    @group_selection_weights.setter
    def group_selection_weights(self, val: dict[str, Number]) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (int, float)):
                    self._xselweight[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a number!")
        else:
            raise MOSAError("Group selection weights must be provided as a dictionary!")

    @property
    def group_selection_alpha(self) -> float:
        """
        EMA smoothing factor used by adaptative group selection.

        The value must be between zero and one. Higher values give more
        importance to recent accepted moves. The default is 0.2.
        """

        return self._xselalpha

    @group_selection_alpha.setter
    def group_selection_alpha(self, val: Number) -> None:
        if (
            isinstance(val, (int, float))
            and not isinstance(val, bool)
            and isfinite(val)
            and 0.0 <= val <= 1.0
        ):
            self._xselalpha = float(val)
        else:
            raise MOSAError(
                "Group selection alpha must be a number between zero and one!"
            )

    @property
    def track_optimization_progress(self) -> bool:
        """
        Tracks the optimization progress by saving the accepted objetive values
        into a Python list.

        The default is `False`.
        """

        return self._trackoptprogress

    @track_optimization_progress.setter
    def track_optimization_progress(self, val: bool) -> None:
        if isinstance(val, bool):
            self._trackoptprogress = val
        else:
            raise MOSAError("Tracking or not optimization progress must be a boolean!")

    @property
    def accepted_objective_values(self) -> list[Number | ObjectiveValues]:
        """Accepted objective values over Monte Carlo iterations."""

        return self._f

    @property
    def verbose(self) -> bool:
        """
        Displays verbose output.

        The default is `False`.
        """

        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        if isinstance(val, bool):
            self._verbose = val
        else:
            raise MOSAError("Displaying or not verbose output must be a boolean!")
