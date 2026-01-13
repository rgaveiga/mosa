"""This module defines the `Anneal` class, which implements the MOSA algorithm."""

from __future__ import print_function
from __future__ import division
import json
from copy import deepcopy
import numpy as np
from numpy.random import choice, triangular, uniform
from math import exp, inf
from . import __version__


class Anneal:
    """This class implements the MOSA algorithm."""

    def __init__(self):
        """@private
        Initializes object attributes.
        """

        print("--------------------------------------------------")
        print(f" MULTI-OBJECTIVE SIMULATED ANNEALING (MOSA) {__version__}  ")
        print("--------------------------------------------------")

        self._initemp: float = 1.0
        self._decrease: float = 0.9
        self._ntemp: int = 10
        self._population: dict = {}
        self._changemove: dict = {}
        self._swapmove: dict = {}
        self._insordelmove: dict = {}
        self._xnel: dict = {}
        self._maxnel: dict = {}
        self._xdistinct: dict = {}
        self._xstep: dict = {}
        self._xsort: dict = {}
        self._xselweight: dict = {}
        self._archive: dict = {}
        self._temp: list = []
        self._weight: list = []
        self._niter: int = 1000
        self._archive_file: str = "archive.json"
        self._archivesize: int = 1000
        self._maxarchivereject: int = 1000
        self._alpha: float = 0.0
        self._restart: bool = True
        self._trackoptprogress: bool = False
        self._f: list = []
        self._verbose: bool = False

    def set_population(self, **groups):
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

    def set_group_params(self, group: str, **params):
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

        allowed: dict

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

    def set_opt_param(self, param: str, **groups):
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

        params: tuple
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

    def evolve(self, func: object):
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
        fcurr: list = []
        ftmp: list = []
        weight: list = []
        lstep: dict = {}
        population: dict = {}
        poptmp: dict = {}
        xcurr: dict = {}
        xtmp: dict = {}
        xstep: dict = {}
        xsampling: dict = {}
        xbounds: dict = {}
        changemove: dict = {}
        swapmove: dict = {}
        insordelmove: dict = {}
        xdistinct: dict = {}
        xnel: dict = {}
        maxnel: dict = {}
        xsort: dict = {}
        totlength: float = 0.0
        sellength: dict = {}
        groups: list = []
        args: str = ""
        MAX_FAILED: int = 10
        MIN_STEP_LENGTH: int = 10

        self._temp = [self._initemp * self._decrease**i for i in range(self._ntemp)]

        if self._restart:
            xcurr, fcurr, population = self.__getcheckpoint()

            if not self._archive:
                try:
                    print(
                        f"Trying to load the archive from file {self._archive_file}..."
                    )

                    self._archive = json.load(open(self._archive_file, "r"))
                except FileNotFoundError:
                    print(
                        f"File {self._archive_file} not found! Initializing an empty archive..."
                    )

                    self._archive = {"x": [], "f": []}

                print("Done!")
        else:
            print("Initializing an empty archive...")

            self._archive = {"x": [], "f": []}

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
                for group in groups:
                    args += f"{group} = {xcurr[group]}, "

                fcurr = eval(f"list(func({args}))")

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

        for temp in self._temp:
            if self._verbose:
                print(f"TEMPERATURE: {temp:.6f}")

            nupdated = 0
            naccept = 0

            for j in range(self._niter):
                selstep = chosen = old = new = None
                xtmp = deepcopy(xcurr)
                poptmp = deepcopy(population)

                r = uniform(0.0, totlength)

                for group in groups:
                    if r < sellength[group]:
                        break

                r = uniform(
                    0.0, (changemove[group] + swapmove[group] + insordelmove[group])
                )

                if r < changemove[group] or r >= (changemove[group] + swapmove[group]):
                    if xnel[group] > 1:
                        old = choice(len(xtmp[group]))

                    if xsampling[group] == 0 and len(poptmp[group]) > 0:
                        for _ in range(MAX_FAILED):
                            if len(poptmp[group]) == 1:
                                new = 0
                            elif xstep[group] >= MIN_STEP_LENGTH:
                                selstep = int(
                                    round(triangular(-xstep[group], 0, xstep[group]), 0)
                                )
                                new = lstep[group] + selstep

                                if new >= len(poptmp[group]):
                                    new -= len(poptmp[group])
                                elif new < 0:
                                    new += len(poptmp[group])
                            else:
                                new = choice(len(poptmp[group]))

                            if r >= changemove[group] or xdistinct[group]:
                                break
                            else:
                                if xnel[group] == 1:
                                    if not xtmp[group] == poptmp[group][new]:
                                        break
                                else:
                                    if not xtmp[group][old] == poptmp[group][new]:
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
                                xtmp[group], poptmp[group][new] = (
                                    poptmp[group][new],
                                    xtmp[group],
                                )
                            else:
                                xtmp[group][old], poptmp[group][new] = (
                                    poptmp[group][new],
                                    xtmp[group][old],
                                )
                        else:
                            if xnel[group] == 1:
                                xtmp[group] = poptmp[group][new]
                            else:
                                xtmp[group][old] = poptmp[group][new]
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
                    elif (xsampling[group] == 0 and len(poptmp[group]) == 0) or len(
                        xtmp[group]
                    ) >= maxnel[group]:
                        r = 1.0
                    else:
                        r = uniform(0.0, 1.0)

                    if r < 0.5:
                        if xsampling[group] == 0:
                            xtmp[group].append(poptmp[group][new])

                            if xdistinct[group]:
                                poptmp[group].pop(new)
                        else:
                            xtmp[group].append(
                                uniform(xbounds[group][0], xbounds[group][1])
                            )

                        if xsort[group]:
                            xtmp[group].sort()
                    else:
                        if xsampling[group] == 0 and xdistinct[group]:
                            poptmp[group].append(xtmp[group][old])

                        xtmp[group].pop(old)

                gamma = 1.0

                args = ""

                for group in groups:
                    args += f"{group} = {xtmp[group]}, "

                ftmp = eval(f"list(func({args}))")

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

                    fcurr = ftmp.copy()
                    xcurr = deepcopy(xtmp)
                    population = deepcopy(poptmp)
                    naccept += 1
                    updated = self.__updatearchive(xcurr, fcurr)
                    nupdated += updated

                    if updated == 1:
                        narchivereject = 0
                    else:
                        narchivereject += 1

                    self.__savecheckpoint(xcurr, fcurr, population)
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

                    self.savex()

                    return

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

            if nupdated > 0:
                self.savex()

        if not self._verbose:
            print("Maximum number of temperatures reached!")
            print(f"Stopping at temperature:  {temp:.6f}.")
            print("------")

        print("\n--- THE END ---")

    def prune_dominated(self, xset: dict = {}) -> dict:
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

        if len(xset["x"]) == 1:
            return xset

        tmpdict: dict[str, list] = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]
        f_arr = np.array(f)

        dominated = []

        for i in range(len(f_arr) - 1):
            fj = f_arr[i + 1 :]
            d_arr = np.all(f_arr[i] >= fj, axis=-1)

            if d_arr.sum() > 0:
                dominated.append(i)

        fj = f_arr[:-1]
        d_arr = np.all(f_arr[-1] >= fj, axis=-1)

        if d_arr.sum() > 0:
            dominated.append(len(f_arr) - 1)

        tmpdict["x"] = [v for i, v in enumerate(x) if i not in dominated]
        tmpdict["f"] = [v for i, v in enumerate(f) if i not in dominated]

        return tmpdict

    def savex(self, xset: dict = {}, archive_file: str = ""):
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

    def loadx(self, archive_file: str = ""):
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

        self._archive = tmpdict

    def trimx(
        self, xset: dict = {}, thresholds: tuple | np.ndarray | list = []
    ) -> dict:
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

        tmpdict: dict[str, list] = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]
        f_arr = np.array(f)

        for i in range(len(thresholds)):
            if thresholds[i] is None:
                thresholds[i] = np.inf

        thresholds = np.asarray(thresholds)
        included = np.flatnonzero(np.all(f_arr <= thresholds, axis=-1))

        if len(included) > 0:
            tmpdict["x"] = [v for i, v in enumerate(x) if i in included]
            tmpdict["f"] = [v for i, v in enumerate(f) if i in included]
        else:
            raise RuntimeError("No solution remained in the reduced archive!")

        return tmpdict

    def reducex(self, xset: dict = {}, index: int = 0, nel: int = 5) -> dict:
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

        tmpdict: dict[str, list] = {"x": [], "f": []}

        x = xset["x"]
        f = xset["f"]

        if nel > len(f):
            nel = len(f)

        indexlist = list(range(len(f)))

        for i in range(nel):
            k = 0

            for j in indexlist:
                if k == 0:
                    toadd = j
                    bestval = f[j][index]
                    k += 1
                else:
                    if f[j][index] < bestval:
                        toadd = j
                        bestval = f[j][index]

            tmpdict["x"].append(x[toadd])
            tmpdict["f"].append(f[toadd])
            indexlist.remove(toadd)

        return tmpdict

    def mergex(self, xset_list: list | tuple) -> dict:
        """
        Merges two or more solution archives into a single solution archive.

        ### Parameters

        `xset_list`: solution archives to be merged.

        ### Returns

        Merged solution archives.
        """

        tmpdict: dict[str, list] = {"x": [], "f": []}

        if len(xset_list) < 2:
            raise MOSAError("Nothing to be done!")

        for xset in xset_list:
            xset = self.__checkarchive(xset)

            tmpdict["x"] += xset["x"]
            tmpdict["f"] += xset["f"]

        return tmpdict

    def copyx(self, xset: dict = {}) -> dict:
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

    def printx(self, xset: dict = {}):
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

    def sizex(self, xset: dict = {}) -> int:
        """
        Returns the number of solutions stored in the archive.

        ### Parameters

        `xset`: full or reduced solution archive.

        ### Returns

        Number of solutions stored in the archive.
        """

        xset = self.__checkarchive(xset)

        return len(xset["x"])

    # TODO: Show up to three objective functions in a single plot
    def plot_front(
        self, xset: dict = {}, index1: int = 0, index2: int = 1, file: str | None = None
    ):
        """
        Plots 2D scatter plots of selected pairs of objective values.

        ### Parameters

        `xset`: full or reduced solution archive.

        The default is {}, meaning the full solution archive.

        `index1`: index of the objective function displayed along x-axis.

        The default is 0.

        `index2`: index of the objective function displayed along y-axis.

        The default is 1.

        `file`: name of the image file where the plot will be saved.

        The default is `None`, which means that no figure will be created.
        """

        try:
            import matplotlib.pyplot as plt
        except:
            raise MOSAError("Matplotlib is not available in your system!")

        xset = self.__checkarchive(xset)

        f: list = [[], []]

        if (
            index1 >= 0
            and index1 < len(xset["f"][0])
            and index2 >= 0
            and index2 < len(xset["f"][0])
        ):
            for i in range(len(xset["f"])):
                f[0].append(xset["f"][i][index1])
                f[1].append(xset["f"][i][index2])

            plt.xlabel(f"f{index1}")
            plt.ylabel(f"f{index2}")
            plt.grid()
            plt.scatter(f[0], f[1])

            if file is not None and len(file) > 0:
                plt.savefig(file)

            plt.show()
        else:
            raise MOSAError("Index out of range!")

    def get_stats(self, xset: dict = {}) -> dict:
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
            "Min": list(fmin),
            "Max": list(fmax),
            "Avg": list(favg),
            "Std": list(fstd),
        }

    def __updatearchive(self, x: dict, f: list) -> int:
        """
        Appends a solution to the archive if it is not dominated by other existing
        solutions.

        ### Parameters

        `x`: solution.

        `f`: objective values.

        ### Returns

        1, if the archive is updated, or 0, if not.
        """

        updated: int = 0

        if len(self._archive["x"]) == 0:
            updated = 1
        else:
            archive_arr = np.array(self._archive["f"])
            f_arr = np.array(f)
            c_arr = np.all(archive_arr <= f_arr, axis=-1)

            if c_arr.sum() > 0:
                updated = 0
            else:
                if len(self._archive["x"]) < self._archivesize:
                    updated = 1
                else:
                    d_arr = np.flatnonzero(np.all(archive_arr >= f_arr, axis=-1))

                    if d_arr.shape[0] > 0:
                        self._archive["x"] = [
                            v
                            for i, v in enumerate(self._archive["x"])
                            if i not in d_arr
                        ]
                        self._archive["f"] = [
                            v
                            for i, v in enumerate(self._archive["f"])
                            if i not in d_arr
                        ]
                    else:
                        updated = 0

        if updated == 1:
            self._archive["x"].append(x)
            self._archive["f"].append(f)

        return updated

    def __getcheckpoint(self) -> tuple:
        """
        Initializes with a solution from a previous run.

        ### Returns

        Solution, objective values, and population compatible with the solution.
        """

        tmpdict: dict = {}
        x: dict = {}
        f: list = []
        population: dict = {}

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

    def __savecheckpoint(self, x: dict, f: list, population: dict):
        """
        Saves the solution passed as argument as JSON into a text file.

        ### Parameters

        `x`: solution.

        `f`: objective values.

        `population`: population compatible with the solution.
        """

        tmpdict: dict = {
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

    def __checkarchive(self, xset) -> dict:
        """
        Performs checks on the archive.

        ### Parameters

        `xset`: full or reduced solution archive.

        ### Returns

        Solution archive.
        """

        if not xset:
            return self._archive

        if not ("x" in xset and "f" in xset):
            raise MOSAError("'x' and 'f' must be present in the archive!")

        if not (isinstance(xset["x"], list) and isinstance(xset["f"], list)):
            raise MOSAError("'x' and 'f' must be Python lists!")

        if len(xset["x"]) == 0:
            raise MOSAError("Archive is empty!")

        if len(xset["x"]) != len(xset["f"]):
            raise MOSAError("'x' and 'f' must have the same number of elements!")

        return xset

    @property
    def population(self) -> dict:
        """
        Population where each group represents the data that can be used to achieve
        an optimized solution to the problem.
        """

        return self._population

    @population.setter
    def population(self, val: dict):
        if isinstance(val, dict) and val:
            self._population = val
        else:
            raise MOSAError("Population must be a non-empty dictionary!")

    @property
    def archive(self) -> dict:
        """
        Solution archive.

        > [!WARNING]
        > The archive should not be changed manually.
        """

        return self._archive

    @archive.setter
    def archive(self, val: dict):
        if isinstance(val, dict) and val:
            if not ("x" in val.keys() and "f" in val.keys()):
                raise MOSAError("'x' and 'f' must be present in the archive!")
            else:
                if not (isinstance(val["x"], list) and isinstance(val["f"], list)):
                    raise MOSAError("'x' and 'f' must be Python lists!")
        else:
            raise MOSAError("The archive must be a non-empty dictionary!")

        self._archive = val

    @property
    def restart(self) -> bool:
        """
        Restarts from a previous run if a checkpoint file is available.

        The default is `True`.
        """

        return self._restart

    @restart.setter
    def restart(self, val: bool):
        if isinstance(val, bool):
            self._restart = val
        else:
            raise MOSAError("Restart must be a boolean!")

    @property
    def objective_weights(self) -> list:
        """
        Weights for the objectives.

        The default is [], which means the same weight (1.0) for all objectives.
        """

        return self._weight

    @objective_weights.setter
    def objective_weights(self, val: list):
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
    def initial_temperature(self, val: int | float):
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
    def temperature_decrease_factor(self, val: float):
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
    def number_of_temperatures(self, val: int):
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
    def number_of_iterations(self, val: int):
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
    def archive_size(self, val: int):
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
    def archive_file(self, val: str):
        if isinstance(val, str) and len(val.strip()) > 0:
            self._archive_file = val.strip()
        else:
            raise MOSAError("A file name must be provided!")

    @property
    def maximum_archive_rejections(self) -> int:
        """
        Maximum number of consecutive rejections of insertion of a solution
        in the archive.

        The default is 1,000.
        """

        return self._maxarchivereject

    @maximum_archive_rejections.setter
    def maximum_archive_rejections(self, val: int):
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
    def alpha(self, val: float):
        if isinstance(val, float) and val >= 0.0 and val <= 1.0:
            self._alpha = val
        else:
            raise MOSAError("Alpha must be a number between zero and one!")

    @property
    def number_of_elements(self) -> dict:
        """
        Number of elements for each group in the solution.

        The default is {}, which means one element for all groups in the solutions.
        """

        return self._xnel

    @number_of_elements.setter
    def number_of_elements(self, val: dict):
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
    def maximum_number_of_elements(self) -> dict:
        """
        Maximum number of elements for each group in the solution, if the number of elements
        is variable.

        The default is {}, which means an unlimited number of elements.
        """

        return self._maxnel

    @maximum_number_of_elements.setter
    def maximum_number_of_elements(self, val: dict):
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
    def distinct_elements(self) -> dict:
        """
        Determines that an element cannot be repeated in a group in the solution.

        The default is {}, which means that repetitions are allowed.
        """

        return self._xdistinct

    @distinct_elements.setter
    def distinct_elements(self, val: dict):
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
    def mc_step_size(self) -> dict:
        """
        Monte Carlo step size for each group in the solution.

        The default is {}, which means 0.1 for continuous search space and half
        the number of elements in a population group for discrete search space.
        """

        return self._xstep

    @mc_step_size.setter
    def mc_step_size(self, val: dict):
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (int, float)):
                    self._xstep[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a number!")
        else:
            raise MOSAError("Monte Carlo step sizes must be provided as a dictionary!")

    @property
    def change_value_move(self) -> dict:
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
    def change_value_move(self, val: dict):
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._changemove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def insert_or_delete_move(self) -> dict:
        """
        Weight (non-normalized probability) to select a trial move where an element
        will be inserted into or deleted from a group in the solution.

        The default is {}, which means this trial move is not allowed, i.e., the
        weight is equal to zero.
        """

        return self._insordelmove

    @insert_or_delete_move.setter
    def insert_or_delete_move(self, val: dict):
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._insordelmove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def swap_move(self) -> dict:
        """
        Weight (non-normalized probability) to select a trial move where elements
        will be swaped in the solution.

        The default is {}, which means this trial move is not allowed, i.e., the
        weight is equal to zero.
        """

        return self._swapmove

    @swap_move.setter
    def swap_move(self, val: dict):
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._swapmove[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a positive number!")
        else:
            raise MOSAError("Weights of trial moves must be provided as a dictionary!")

    @property
    def sort_elements(self) -> dict:
        """
        Elements in a group in the solution will be sorted in ascending order.

        The default is {}, which means no sorting at all.
        """

        return self._xsort

    @sort_elements.setter
    def sort_elements(self, val: dict):
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, bool):
                    self._xsort[key] = value
                else:
                    raise MOSAError(f"Group '{key}' must be a boolean!")
        else:
            raise MOSAError("Sort group elements must be provided as a dictionary!")

    @property
    def group_selection_weights(self) -> dict:
        """
        Selection weight for each group in the solution in a Monte Carlo iteration.

        The default value is {}, which means that all groups have the same selection
        weight, i.e., the same probability of being selected.
        """

        return self._xselweight

    @group_selection_weights.setter
    def group_selection_weights(self, val: dict):
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
    def track_optimization_progress(self, val: bool):
        if isinstance(val, bool):
            self._trackoptprogress = val
        else:
            raise MOSAError("Tracking or not optimization progress must be a boolean!")

    @property
    def accepted_objective_values(self) -> list:
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
    def verbose(self, val: bool):
        if isinstance(val, bool):
            self._verbose = val
        else:
            raise MOSAError("Displaying or not verbose output must be a boolean!")


class MOSAError(Exception):
    """@private
    This class defines exceptions raised by the MOSA algorithm.
    """

    def __init__(self, message: str = ""):
        """Class constructor."""

        self._message = message

    def __str__(self) -> str:
        """Returns the error message."""

        return self._message
