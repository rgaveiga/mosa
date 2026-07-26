"""This module defines the `Anneal` class, which implements the MOSA algorithm."""

from __future__ import print_function
from __future__ import division
import json
from copy import deepcopy
from typing import Any, Callable, Sequence, TypeAlias, TypedDict
import numpy as np
from numpy.random import choice, triangular, uniform
from math import exp, inf
from . import __version__

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


class Archive(TypedDict):
    """@private"""

    x: list[Solution]
    f: list[ObjectiveValues]


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
        self._decrease: float = 0.9
        self._ntemp: int = 10
        self._population: Population = {}
        self._changemove: dict[str, Number] = {}
        self._swapmove: dict[str, Number] = {}
        self._insordelmove: dict[str, Number] = {}
        self._xnel: dict[str, int] = {}
        self._maxnel: dict[str, int] = {}
        self._xdistinct: dict[str, bool] = {}
        self._xstep: dict[str, Number] = {}
        self._xsort: dict[str, bool] = {}
        self._xselweight: dict[str, Number] = {}
        self._archive_x: list[Solution] = []
        self._archive_f_arr: np.ndarray = np.empty((0, 0), dtype=float)
        self._archive_f_capacity: int = 0
        self._temp: list[float] = []
        self._weight: ObjectiveWeightValues = []
        self._niter: int = 1000
        self._archive_file: str = "archive.json"
        self._checkpoint_interval: int = 1
        self._archive_save_interval: int = 1
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

        - `change_value_move`

        - `insert_or_delete_move`

        - `swap_move`

        - `sort_elements`

        - `group_selection_weights`
        """

        allowed: dict[str, str]

        if len(params) > 0:
            allowed = {
                "number_of_elements": "self._xnel",
                "maximum_number_of_elements": "self._maxnel",
                "distinct_elements": "self._xdistinct",
                "mc_step_size": "self._xstep",
                "change_value_move": "self._changemove",
                "insert_or_delete_move": "self._insordelmove",
                "swap_move": "self._swapmove",
                "sort_elements": "self._xsort",
                "group_selection_weights": "self._xselweight",
            }

            for param, value in params.items():
                if param in allowed:
                    exec(f"{allowed[param]}[group]=value")
        else:
            raise MOSAError("No keyword was provided!")

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

        - `change_value_move`

        - `insert_or_delete_move`

        - `swap_move`

        - `sort_elements`

        - `group_selection_weights`

        `**groups`: series of key-value pairs where each key corresponds to a
        group in the solution to the problem.
        """

        params: dict[str, str]
        execstr: str

        if len(groups) > 0:
            params = {
                "number_of_elements": "self._xnel",
                "maximum_number_of_elements": "self._maxnel",
                "distinct_elements": "self._xdistinct",
                "mc_step_size": "self._xstep",
                "change_value_move": "self._changemove",
                "insert_or_delete_move": "self._insordelmove",
                "swap_move": "self._swapmove",
                "sort_elements": "self._xsort",
                "group_selection_weights": "self._xselweight",
            }

            if param in params:
                execstr = "for key,value in groups.items():\n"
                execstr += f"    {params[param]}[key]=value"
                exec(execstr)
            else:
                raise MOSAError("Optimization parameter does not exist!")
        else:
            raise MOSAError("No keyword was provided!")

    def evolve(self, func: ObjectiveFunction) -> None:
        """
        Performs the optimization of the objective function.

        ### Parameters

        `func`: objective function.
        """

        print("--- BEGIN: Evolving a solution ---\n")

        from_checkpoint: bool = False
        pmax: float = 0.0
        gamma: float = 1.0
        updated: int = 0
        nupdated: int = 0
        naccept: int = 0
        narchivereject: int = 0
        fcurr: ObjectiveValues = []
        ftmp: ObjectiveValues = []
        weight: ObjectiveWeightValues = []
        lstep: dict[str, int] = {}
        population: Population = {}
        poptmp: Population = {}
        xcurr: Solution = {}
        xtmp: Solution = {}
        xstep: dict[str, Number] = {}
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
        groups: list[str] = []
        MAX_FAILED: int = 10
        MIN_STEP_LENGTH: int = 10

        self._temp = [self._initemp * self._decrease**i for i in range(self._ntemp)]

        if self._restart:
            xcurr, fcurr, population = self.__getcheckpoint()

            if len(self._archive_x) == 0:
                try:
                    print(
                        f"Trying to load the archive from file {self._archive_file}..."
                    )

                    tmp_archive = json.load(open(self._archive_file, "r"))
                    tmp_archive = self.__checkarchive(tmp_archive)
                    self.__set_archive_data(tmp_archive["x"], tmp_archive["f"])
                except FileNotFoundError:
                    print(
                        f"File {self._archive_file} not found! Initializing an empty archive..."
                    )

                    self.__set_archive_data([], [])

                print("Done!")
        else:
            print("Initializing an empty archive...")

            self.__set_archive_data([], [])

            print("Done!")

        if population and xcurr and len(fcurr) > 0:
            if set(population.keys()) == set(xcurr.keys()):
                from_checkpoint = True
            else:
                raise MOSAError("Solution and population must have the same groups!")
        else:
            if self._population:
                xcurr = {}
                fcurr = []
                population = deepcopy(self._population)
            else:
                raise MOSAError("A population must be provided!")

        groups = list(population.keys())

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

                if len(population[group]) <= 1 and not from_checkpoint:
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

            if group in self._xselweight.keys():
                totlength += self._xselweight[group]

                print(f"        Selection weight: {self._xselweight[group]}")
            else:
                totlength += 1.0

                print("        Selection weight: 1.0")

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

            if group in self._xstep.keys():
                if xsampling[group] == 1:
                    xstep[group] = float(self._xstep[group])

                    if xstep[group] <= 0.0:
                        xstep[group] = 0.1
                else:
                    xstep[group] = int(self._xstep[group])
            else:
                if xsampling[group] == 1:
                    xstep[group] = 0.1
                else:
                    if changemove[group] > 0.0:
                        xstep[group] = int(len(population[group]) / 2)
                    else:
                        xstep[group] = 0

            if xsampling[group] == 1:
                print(f"        Maximum step size: {xstep[group]}")
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

        print("------")

        if from_checkpoint:
            print("Initial solution loaded from the checkpoint file...")
        else:
            print("Initializing with a random solution from scratch...")

            for group in groups:
                if xnel[group] == 1:
                    if xsampling[group] == 0:
                        m = choice(len(population[group]))
                        xcurr[group] = population[group][m]

                        if xdistinct[group]:
                            population[group].pop(m)
                    else:
                        xcurr[group] = uniform(xbounds[group][0], xbounds[group][1])
                else:
                    xcurr[group] = []

                    for j in range(xnel[group]):
                        if xsampling[group] == 0:
                            m = choice(len(population[group]))
                            xcurr[group].append(population[group][m])

                            if xdistinct[group]:
                                population[group].pop(m)
                        else:
                            xcurr[group].append(
                                uniform(xbounds[group][0], xbounds[group][1])
                            )

                    if xsort[group]:
                        xcurr[group].sort()

            if callable(func):
                fcurr = list(func(**xcurr))

                updated = self.__updatearchive(xcurr, fcurr)

                if self._trackoptprogress:
                    if len(fcurr) == 1:
                        self._f.append(fcurr[0])
                    else:
                        self._f.append(fcurr)
            else:
                raise MOSAError("A Python function must be provided!")

        print("Done!")
        print("------")

        if len(fcurr) == len(self._weight):
            weight = self._weight.copy()
        else:
            weight = [1.0 for k in range(len(fcurr))]

        if not self._verbose:
            print(f"Starting at temperature: {self._temp[0]:.6f}")
            print("Evolving solutions to the problem, please wait...")

        archive_dirty = updated == 1

        for temperature_index, temp in enumerate(self._temp, start=1):
            if self._verbose:
                print(f"TEMPERATURE: {temp:.6f}")

            nupdated = 0
            naccept = 0

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

                xtmp = xcurr.copy()

                if isinstance(xcurr[group], list):
                    xtmp[group] = xcurr[group].copy()

                if r < changemove[group] or r >= (changemove[group] + swapmove[group]):
                    if xnel[group] > 1:
                        old = choice(len(xtmp[group]))

                    if xsampling[group] == 0 and len(population[group]) > 0:
                        for _ in range(MAX_FAILED):
                            if len(population[group]) == 1:
                                new = 0
                            elif xstep[group] >= MIN_STEP_LENGTH:
                                selstep = int(
                                    round(triangular(-xstep[group], 0, xstep[group]), 0)
                                )
                                new = lstep[group] + selstep

                                if new >= len(population[group]):
                                    new -= len(population[group])
                                elif new < 0:
                                    new += len(population[group])
                            else:
                                new = choice(len(population[group]))

                            if r >= changemove[group] or xdistinct[group]:
                                break
                            else:
                                if xnel[group] == 1:
                                    if not xtmp[group] == population[group][new]:
                                        break
                                else:
                                    if not xtmp[group][old] == population[group][new]:
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
                                    xtmp[group],
                                )
                                xtmp[group] = population[group][new]
                            else:
                                population_update = (
                                    "replace",
                                    new,
                                    xtmp[group][old],
                                )
                                xtmp[group][old] = population[group][new]
                        else:
                            if xnel[group] == 1:
                                xtmp[group] = population[group][new]
                            else:
                                xtmp[group][old] = population[group][new]
                    else:
                        if xnel[group] == 1:
                            xtmp[group] += uniform(-xstep[group], xstep[group])

                            if xtmp[group] > xbounds[group][1]:
                                xtmp[group] -= xbounds[group][1] - xbounds[group][0]
                            elif xtmp[group] < xbounds[group][0]:
                                xtmp[group] += xbounds[group][1] - xbounds[group][0]
                        else:
                            xtmp[group][old] += uniform(-xstep[group], xstep[group])

                            if xtmp[group][old] > xbounds[group][1]:
                                xtmp[group][old] -= (
                                    xbounds[group][1] - xbounds[group][0]
                                )
                            elif xtmp[group][old] < xbounds[group][0]:
                                xtmp[group][old] += (
                                    xbounds[group][1] - xbounds[group][0]
                                )

                    if xsort[group] and xnel[group] > 1:
                        xtmp[group].sort()
                elif r < (changemove[group] + swapmove[group]):
                    for _ in range(int(len(xtmp[group]) / 2)):
                        chosen = choice(len(xtmp[group]), 2, False)

                        if not xtmp[group][chosen[0]] == xtmp[group][chosen[1]]:
                            xtmp[group][chosen[0]], xtmp[group][chosen[1]] = (
                                xtmp[group][chosen[1]],
                                xtmp[group][chosen[0]],
                            )

                            break
                    else:
                        if self._verbose:
                            print(
                                f"WARNING!!!!!! Failed {int(len(xtmp[group])/2)} times to find different elements in group '{group}' for swapping at iteration {j}!"
                            )

                        continue
                else:
                    if len(xtmp[group]) == 1:
                        r = 0.0
                    elif (xsampling[group] == 0 and len(population[group]) == 0) or len(
                        xtmp[group]
                    ) >= maxnel[group]:
                        r = 1.0
                    else:
                        r = uniform(0.0, 1.0)

                    if r < 0.5:
                        if xsampling[group] == 0:
                            xtmp[group].append(population[group][new])

                            if xdistinct[group]:
                                population_update = ("remove", new, None)
                        else:
                            xtmp[group].append(
                                uniform(xbounds[group][0], xbounds[group][1])
                            )

                        if xsort[group]:
                            xtmp[group].sort()
                    else:
                        if xsampling[group] == 0 and xdistinct[group]:
                            population_update = ("append", None, xtmp[group][old])

                        xtmp[group].pop(old)

                gamma = 1.0

                ftmp = list(func(**xtmp))

                for k in range(len(ftmp)):
                    if ftmp[k] < fcurr[k]:
                        pmax = p = 1.0
                    else:
                        p = exp(-(ftmp[k] - fcurr[k]) / (temp * weight[k]))

                        if pmax < p:
                            pmax = p

                    gamma *= p

                gamma = (1.0 - self._alpha) * gamma + self._alpha * pmax

                if gamma == 1.0 or uniform(0.0, 1.0) < gamma:
                    if xsampling[group] == 0 and new is not None:
                        lstep[group] = new

                    fcurr = ftmp
                    xcurr = xtmp

                    if population_update is not None:
                        action, index, value = population_update

                        if action == "replace":
                            assert index is not None
                            population[group][index] = value
                        elif action == "remove":
                            assert index is not None
                            population[group].pop(index)
                        else:
                            population[group].append(value)

                    naccept += 1
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

                    self.__savecheckpoint(xcurr, fcurr, population)

                    if archive_dirty:
                        self.savex()

                    return

            final_temperature = temperature_index == len(self._temp)
            checkpoint_due = final_temperature or (
                self._checkpoint_interval > 0
                and temperature_index % self._checkpoint_interval == 0
            )
            archive_save_due = final_temperature or (
                self._archive_save_interval > 0
                and temperature_index % self._archive_save_interval == 0
            )

            if checkpoint_due:
                self.__savecheckpoint(xcurr, fcurr, population)

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
                archive_file = self._archive_file
        else:
            raise MOSAError("The name of the archive file must be a string!")

        json.dump(xset, open(archive_file, "w"), indent=4)

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
                archive_file = self._archive_file
        else:
            raise MOSAError("Name of the archive file must be a string!")

        try:
            tmpdict = json.load(open(archive_file, "r"))
        except FileNotFoundError:
            print(f"File {archive_file} not found!")

            return

        try:
            tmpdict = self.__checkarchive(tmpdict)
        except:
            print(f"Something wrong with file {archive_file}!")

            return

        self.__set_archive_data(tmpdict["x"], tmpdict["f"])

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
            tmpdict["x"] = [v for i, v in enumerate(x) if i in included]
            tmpdict["f"] = [v for i, v in enumerate(f) if i in included]
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
        """

        try:
            import matplotlib.pyplot as plt
        except:
            raise MOSAError("Matplotlib is not available in your system!")

        xset = self.__checkarchive(xset)

        nobj = len(xset["f"][0])
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
            ax.set_xlabel(f"f{index1}")
            ax.set_ylabel(f"f{index2}")
            ax.grid()
            ax.scatter(f[0], f[1])
        else:
            ax = fig.add_subplot(projection="3d")
            ax.set_xlabel(f"f{index1}")
            ax.set_ylabel(f"f{index2}")
            ax.set_zlabel(f"f{index3}")
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
        nf = f_arr.shape[1]
        fmin: np.ndarray = np.zeros(nf)
        fmax: np.ndarray = np.zeros(nf)
        favg: np.ndarray = np.zeros(nf)
        fstd: np.ndarray = np.zeros(nf)

        for i in range(nf):
            fmin[i] = f_arr[:, i].min()
            fmax[i] = f_arr[:, i].max()
            favg[i] = f_arr[:, i].mean()
            fstd[i] = f_arr[:, i].std()

        return {
            "Min": fmin.astype(float).tolist(),
            "Max": fmax.astype(float).tolist(),
            "Avg": favg.astype(float).tolist(),
            "Std": fstd.astype(float).tolist(),
        }

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

        archive_len = len(self._archive_x)
        f_arr = np.asarray(f, dtype=float)

        if archive_len == 0:
            updated = True
        else:
            archive_arr = self._archive_f_arr[:archive_len]
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
                    keep_mask = np.ones(archive_len, dtype=bool)
                    keep_mask[dominated_rows] = False
                    self._archive_x = [
                        value for i, value in enumerate(self._archive_x) if keep_mask[i]
                    ]
                    kept_count = int(np.count_nonzero(keep_mask))

                    if kept_count > 0:
                        self._archive_f_arr[:kept_count] = archive_arr[keep_mask]

                    archive_len = kept_count

        if updated:
            self.__ensure_archive_capacity(archive_len + 1, len(f_arr))
            self._archive_x.append(x)
            self._archive_f_arr[archive_len] = f_arr

        return int(updated)

    @staticmethod
    def __dominance_masks(
        archive_arr: np.ndarray, f_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compare objectives by column to avoid 2-D reduction temporaries."""

        archive_dominates = archive_arr[:, 0] <= f_arr[0]
        candidate_dominates = archive_arr[:, 0] >= f_arr[0]

        for objective in range(1, len(f_arr)):
            archive_dominates &= archive_arr[:, objective] <= f_arr[objective]
            candidate_dominates &= archive_arr[:, objective] >= f_arr[objective]

        return archive_dominates, candidate_dominates

    @staticmethod
    def __non_dominated_mask(f_arr: np.ndarray, block_size: int = 256) -> np.ndarray:
        """Return a mask for Pareto-optimal rows without per-row allocations."""

        row_count = len(f_arr)
        dominated = np.zeros(row_count, dtype=bool)
        objective_count = f_arr.shape[1]
        memory_limited_block_size = max(1, 8_000_000 // max(row_count, 1))
        block_size = min(block_size, memory_limited_block_size)

        for start in range(0, row_count, block_size):
            end = min(start + block_size, row_count)
            candidates = f_arr[start:end]
            comparisons = f_arr[np.newaxis, :, 0] <= candidates[:, np.newaxis, 0]

            for objective in range(1, objective_count):
                comparisons &= (
                    f_arr[np.newaxis, :, objective]
                    <= candidates[:, np.newaxis, objective]
                )

            local_rows = np.arange(end - start)
            comparisons[local_rows, start + local_rows] = False
            dominated[start:end] = np.any(comparisons, axis=1)

        return ~dominated

    def __getcheckpoint(self) -> tuple[Solution, ObjectiveValues, Population]:
        """
        Initializes with a solution from a previous run.

        ### Returns

        Solution, objective values, and population compatible with the solution.
        """

        tmpdict: dict[str, Any] = {}
        x: Solution = {}
        f: ObjectiveValues = []
        population: Population = {}

        print("Looking for a solution in the checkpoint file...")

        try:
            tmpdict = json.load(open("checkpoint.json", "r"))

            if "x" in tmpdict and "f" in tmpdict and "Population" in tmpdict:
                x = tmpdict["x"]
                f = tmpdict["f"]
                population = tmpdict["Population"]

                if "SampleSpace" in tmpdict:
                    ss = tmpdict["SampleSpace"]

                    for key in ss.keys():
                        if ss[key] == 1:
                            population[key] = tuple(population[key])
        except:
            print("No checkpoint file!")

        print("Done!")

        return x, f, population

    def __savecheckpoint(
        self, x: Solution, f: ObjectiveValues, population: Population
    ) -> None:
        """
        Saves the solution passed as argument as JSON into a text file.

        ### Parameters

        `x`: solution.

        `f`: objective values.

        `population`: population compatible with the solution.
        """

        tmpdict: dict[str, Any] = {
            "x": x,
            "f": f,
            "Population": population,
            "SampleSpace": {},
        }

        for key in population.keys():
            if isinstance(population[key], list):
                tmpdict["SampleSpace"][key] = 0
            elif isinstance(population[key], tuple):
                tmpdict["SampleSpace"][key] = 1

        json.dump(tmpdict, open("checkpoint.json", "w"), indent=4)

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
            "x": self._archive_x.copy(),
            "f": self._archive_f_arr[: len(self._archive_x)].astype(float).tolist(),
        }

    def __set_archive_data(
        self, x_values: list[Solution], f_values: list[ObjectiveValues]
    ) -> None:
        """@private"""

        self._archive_x = list(x_values)

        if len(f_values) == 0:
            self._archive_f_arr = np.empty((0, 0), dtype=float)
            self._archive_f_capacity = 0
        else:
            archive_f_arr = np.asarray(f_values, dtype=float)

            if archive_f_arr.ndim == 1:
                archive_f_arr = archive_f_arr.reshape(1, -1)

            self._archive_f_arr = archive_f_arr.copy()
            self._archive_f_capacity = self._archive_f_arr.shape[0]

    def __ensure_archive_capacity(self, rows: int, nf: int) -> None:
        """@private"""

        if self._archive_f_capacity >= rows and self._archive_f_arr.shape[1] == nf:
            return

        if self._archive_f_capacity == 0 or self._archive_f_arr.shape[1] != nf:
            new_capacity = max(rows, 1)
            new_archive_f_arr = np.empty((new_capacity, nf), dtype=float)
        else:
            new_capacity = max(rows, self._archive_f_capacity * 2)
            new_archive_f_arr = np.empty((new_capacity, nf), dtype=float)
            active_rows = len(self._archive_x)

            if active_rows > 0:
                new_archive_f_arr[:active_rows] = self._archive_f_arr[:active_rows]

        self._archive_f_arr = new_archive_f_arr
        self._archive_f_capacity = new_capacity

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
        Restarts from a previous run if a checkpoint file is available.

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

        The default is 1.0.
        """

        return self._initemp

    @initial_temperature.setter
    def initial_temperature(self, val: Number) -> None:
        if isinstance(val, (int, float)) and val > 0.0:
            self._initemp = val
        else:
            raise MOSAError("Initial temperature must be a number greater than zero!")

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
    def archive_file(self) -> str:
        """
        Name of the archive file.

        The default is 'archive.json'.
        """

        return self._archive_file

    @archive_file.setter
    def archive_file(self, val: str) -> None:
        if isinstance(val, str) and len(val.strip()) > 0:
            self._archive_file = val.strip()
        else:
            raise MOSAError("A file name must be provided!")

    @property
    def checkpoint_interval(self) -> int:
        """
        Number of completed temperatures between checkpoint writes.

        The default is 1. Set it to 0 to write only when evolution finishes.
        A final checkpoint is always written, including on early termination.
        """

        return self._checkpoint_interval

    @checkpoint_interval.setter
    def checkpoint_interval(self, val: int) -> None:
        if isinstance(val, int) and val >= 0:
            self._checkpoint_interval = val
        else:
            raise MOSAError("Checkpoint interval must be a non-negative integer!")

    @property
    def archive_save_interval(self) -> int:
        """
        Number of completed temperatures between automatic archive writes.

        The default is 1. Set it to 0 to write only when evolution finishes.
        The archive is written only when it has changed since the previous save.
        """

        return self._archive_save_interval

    @archive_save_interval.setter
    def archive_save_interval(self, val: int) -> None:
        if isinstance(val, int) and val >= 0:
            self._archive_save_interval = val
        else:
            raise MOSAError("Archive save interval must be a non-negative integer!")

    @property
    def maximum_archive_rejections(self) -> int:
        """
        Maximum number of consecutive rejections of insertion of a solution
        in the archive.

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
    def mc_step_size(self) -> dict[str, Number]:
        """
        Monte Carlo step size for each group in the solution.

        The default is {}, which means 0.1 for continuous search space and half
        the number of elements in a population group for discrete search space.
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
        weight, i.e., the same probability of being selected.
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


class MOSAError(Exception):
    """@private
    This class defines exceptions raised by the MOSA algorithm.
    """

    def __init__(self, message: str = "") -> None:
        """Class constructor."""

        self._message = message

    def __str__(self) -> str:
        """Returns the error message."""

        return self._message
