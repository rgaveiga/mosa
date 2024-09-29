from __future__ import print_function
from __future__ import division
import json
from copy import deepcopy
from numpy.random import choice, triangular, uniform, shuffle
from math import exp, inf


class Anneal:
    def __init__(self) -> None:
        """
        Class constructor, initializes object attributes.

        Returns
        -------
        None.
        """

        print("--------------------------------------------------")
        print("    MULTI-OBJECTIVE SIMULATED ANNEALING (MOSA)    ")
        print("--------------------------------------------------")
        print("         Developed by Prof. Roberto Gomes         ")
        print("   Universidade Federal do ABC (UFABC), Brazil    ")
        print("\n")

        self._initemp: float = 1.0
        self._decrease: float = 0.9
        self._ntemp: int = 10
        self._population: dict = {"X": (-1.0, 1.0)}
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
        self._archivefile: str = "archive.json"
        self._archivesize: int = 1000
        self._maxarchivereject: int = 1000
        self._alpha: float = 0.0
        self._restart: bool = True
        self._trackoptprogress: bool = False
        self._f: list = []
        self._verbose: bool = False

    def evolve(self, func: object) -> None:
        """
        Performs the optimization.

        Parameters
        ----------
        func : Python object
            A Python function that returns the value of the objective(s).

        Returns
        -------
        None.
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
        keys: list = []
        MAX_FAILED: int = 10
        MIN_STEP_LENGTH: int = 10

        self._temp = [self._initemp * self._decrease**i for i in range(self._ntemp)]

        if self._restart:
            xcurr, fcurr, population = self.__getcheckpoint()

            if not bool(self._archive):
                try:
                    print(
                        "Trying to load the archive from file %s..."
                        % self._archivefile
                    )

                    self._archive = json.load(open(self._archivefile, "r"))
                except FileNotFoundError:
                    print(
                        "File %s not found! Initializing an empty archive..."
                        % self._archivefile
                    )

                    self._archive = {"Solution": [], "Values": []}

                print("Done!")
        else:
            print("Initializing an empty archive...")

            self._archive = {"Solution": [], "Values": []}

            print("Done!")

        if bool(population) and bool(xcurr) and len(fcurr) > 0:
            if set(population.keys()) == set(xcurr.keys()):
                from_checkpoint = True
            else:
                raise MOSAError(
                    "Solution and population dictionaries must have the same keys!"
                )
        else:
            if bool(self._population):
                xcurr = {}
                fcurr = []
                population = deepcopy(self._population)
            else:
                raise MOSAError("A population must be provided!")

        keys = list(population.keys())

        print("------")
        print("Keys in the population/solution dictionaries:")

        for key in keys:
            print("    ['%s']:" % key)

            if key in self._xnel.keys() and self._xnel[key] > 0:
                xnel[key] = self._xnel[key]
            else:
                xnel[key] = 1

            print("        Number of elements in the solution: %d" % xnel[key])

            if isinstance(population[key], tuple):
                print("        Continuous sample space")

                if len(population[key]) <= 1:
                    raise MOSAError("Two numbers are expected in key %s!" % key)

                xsampling[key] = 1
                xbounds[key] = list(population[key])

                if xbounds[key][1] < xbounds[key][0]:
                    xbounds[key][0], xbounds[key][1] = xbounds[key][1], xbounds[key][0]
                elif xbounds[key][1] == xbounds[key][0]:
                    raise MOSAError(
                        "Second element in key %s must be larger than the first one!"
                        % key
                    )

                print(
                    "        Boundaries: (%f,%f)" % (xbounds[key][0], xbounds[key][1])
                )
            elif isinstance(population[key], list):
                print("        Discrete sample space")
                print(
                    "        Number of elements in the population: %d"
                    % (len(population[key]))
                )

                if len(population[key]) <= 1 and not from_checkpoint:
                    raise MOSAError(
                        "Number of elements in the population must be greater than one!"
                    )

                xsampling[key] = 0

                if key in self._xdistinct.keys():
                    xdistinct[key] = bool(self._xdistinct[key])
                else:
                    xdistinct[key] = False

                print(
                    "        Elements cannot be repeated in the solution: %s"
                    % xdistinct[key]
                )
            else:
                raise MOSAError("Wrong format of key %s!" % key)

            if key in self._xselweight.keys():
                totlength += self._xselweight[key]

                print(
                    "        Selection weight of this key: %f" % self._xselweight[key]
                )
            else:
                totlength += 1.0

                print("        Selection weight of this key: %f" % 1.0)

            sellength[key] = totlength

            if key in self._changemove.keys() and self._changemove[key] >= 0.0:
                changemove[key] = float(self._changemove[key])
            else:
                changemove[key] = 1.0

            if changemove[key] > 0.0:
                print(
                    "        Weight of 'change value' trial move: %f" % changemove[key]
                )

            if key in self._swapmove.keys() and self._swapmove[key] > 0.0:
                swapmove[key] = float(self._swapmove[key])

                print("        Weight of 'swap' trial move: %f" % swapmove[key])
            else:
                swapmove[key] = 0.0

            if key in self._insordelmove.keys() and self._insordelmove[key] > 0.0:
                insordelmove[key] = float(self._insordelmove[key])

                print(
                    "        Weight of 'insert or delete' trial move: %f"
                    % insordelmove[key]
                )

                if key in self._maxnel.keys() and self._maxnel[key] >= xnel[key]:
                    maxnel[key] = int(self._maxnel[key])

                    if maxnel[key] <= 1:
                        maxnel[key] = 2
                else:
                    maxnel[key] = inf

                print("        Maximum number of solution elements: %d" % maxnel[key])
            else:
                insordelmove[key] = 0.0

            if swapmove[key] == 0.0 and key in self._xsort.keys():
                xsort[key] = bool(self._xsort[key])
            else:
                xsort[key] = False

            print("        Solution sorted after trial move: %s" % xsort[key])

            if key in self._xstep.keys():
                if xsampling[key] == 1:
                    xstep[key] = float(self._xstep[key])

                    if xstep[key] <= 0.0:
                        xstep[key] = 0.1
                else:
                    xstep[key] = int(self._xstep[key])
            else:
                if xsampling[key] == 1:
                    xstep[key] = 0.1
                else:
                    if changemove[key] > 0.0:
                        xstep[key] = int(len(population[key]) / 2)
                    else:
                        xstep[key] = 0

            if xsampling[key] == 1:
                print(
                    "        Maximum step size to choose a new value in the solution: %f"
                    % xstep[key]
                )
            elif xsampling[key] == 0 and (changemove[key] + insordelmove[key]) > 0.0:
                if xstep[key] > len(population[key]) / 2 or xstep[key] <= 0:
                    xstep[key] = int(len(population[key]) / 2)

                if xstep[key] >= MIN_STEP_LENGTH:
                    print(
                        "        Maximum step size to select an element in the population, using a triangular distribution: %d"
                        % xstep[key]
                    )
                else:
                    print("        Elements selected at random from the population")

            if xsampling[key] == 0 and (changemove[key] + insordelmove[key]) > 0.0:
                if len(population[key]) == 1:
                    lstep[key] = 0
                else:
                    lstep[key] = choice(len(population[key]))

            if xnel[key] == 1 and insordelmove[key] == 0.0:
                changemove[key] = 1.0
                swapmove[key] = 0.0

            if len(population[key]) == 0 and insordelmove[key] == 0:
                changemove[key] = 0.0
                swapmove[key] = 1.0

        print("------")

        if from_checkpoint:
            print("Initial solution loaded from the checkpoint file...")
        else:
            print("Initializing with a random solution from scratch...")

            for key in keys:
                if xnel[key] == 1:
                    if xsampling[key] == 0:
                        m = choice(len(population[key]))
                        xcurr[key] = population[key][m]

                        if xdistinct[key]:
                            population[key].pop(m)
                    else:
                        xcurr[key] = uniform(xbounds[key][0], xbounds[key][1])
                else:
                    xcurr[key] = []

                    for j in range(xnel[key]):
                        if xsampling[key] == 0:
                            m = choice(len(population[key]))
                            xcurr[key].append(population[key][m])

                            if xdistinct[key]:
                                population[key].pop(m)
                        else:
                            xcurr[key].append(uniform(xbounds[key][0], xbounds[key][1]))

                    if xsort[key]:
                        xcurr[key].sort()

            if callable(func):
                fcurr = list(func(xcurr))
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
            print("Starting at temperature: %.6f" % self._temp[0])
            print("Evolving solutions to the problem, please wait...")

        for temp in self._temp:
            if self._verbose:
                print("TEMPERATURE: %.6f" % temp)

            nupdated = 0
            naccept = 0

            for j in range(self._niter):
                selstep = chosen = old = new = None
                xtmp = deepcopy(xcurr)
                poptmp = deepcopy(population)

                r = uniform(0.0, totlength)

                for key in keys:
                    if r < sellength[key]:
                        break

                r = uniform(0.0, (changemove[key] + swapmove[key] + insordelmove[key]))

                if r < changemove[key] or r >= (changemove[key] + swapmove[key]):
                    if xnel[key] > 1:
                        old = choice(len(xtmp[key]))

                    if xsampling[key] == 0 and len(poptmp[key]) > 0:
                        for _ in range(MAX_FAILED):
                            if len(poptmp[key]) == 1:
                                new = 0
                            elif xstep[key] >= MIN_STEP_LENGTH:
                                selstep = int(
                                    round(triangular(-xstep[key], 0, xstep[key]), 0)
                                )
                                new = lstep[key] + selstep

                                if new >= len(poptmp[key]):
                                    new -= len(poptmp[key])
                                elif new < 0:
                                    new += len(poptmp[key])
                            else:
                                new = choice(len(poptmp[key]))

                            if r >= changemove[key] or xdistinct[key]:
                                break
                            else:
                                if xnel[key] == 1:
                                    if not xtmp[key] == poptmp[key][new]:
                                        break
                                else:
                                    if not xtmp[key][old] == poptmp[key][new]:
                                        break
                        else:
                            new = None

                if xsampling[key] == 0 and r < changemove[key] and new is None:
                    if insordelmove[key] > 0.0:
                        r = changemove[key] + swapmove[key]
                    elif swapmove[key] > 0.0 and xnel[key] > 1:
                        r = changemove[key]
                    else:
                        if self._verbose:
                            print(
                                "WARNING!!!!!! It was not possible to find an element in the key '%s' in the population to update the solution at iteration %d!"
                                % (key, j)
                            )

                        continue

                if r < changemove[key]:
                    if xsampling[key] == 0:
                        if xdistinct[key]:
                            if xnel[key] == 1:
                                xtmp[key], poptmp[key][new] = (
                                    poptmp[key][new],
                                    xtmp[key],
                                )
                            else:
                                xtmp[key][old], poptmp[key][new] = (
                                    poptmp[key][new],
                                    xtmp[key][old],
                                )
                        else:
                            if xnel[key] == 1:
                                xtmp[key] = poptmp[key][new]
                            else:
                                xtmp[key][old] = poptmp[key][new]
                    else:
                        if xnel[key] == 1:
                            xtmp[key] += uniform(-xstep[key], xstep[key])

                            if xtmp[key] > xbounds[key][1]:
                                xtmp[key] -= xbounds[key][1] - xbounds[key][0]
                            elif xtmp[key] < xbounds[key][0]:
                                xtmp[key] += xbounds[key][1] - xbounds[key][0]
                        else:
                            xtmp[key][old] += uniform(-xstep[key], xstep[key])

                            if xtmp[key][old] > xbounds[key][1]:
                                xtmp[key][old] -= xbounds[key][1] - xbounds[key][0]
                            elif xtmp[key][old] < xbounds[key][0]:
                                xtmp[key][old] += xbounds[key][1] - xbounds[key][0]

                    if xsort[key] and xnel[key] > 1:
                        xtmp[key].sort()
                elif r < (changemove[key] + swapmove[key]):
                    for _ in range(int(len(xtmp[key]) / 2)):
                        chosen = choice(len(xtmp[key]), 2, False)

                        if not xtmp[key][chosen[0]] == xtmp[key][chosen[1]]:
                            xtmp[key][chosen[0]], xtmp[key][chosen[1]] = (
                                xtmp[key][chosen[1]],
                                xtmp[key][chosen[0]],
                            )

                            break
                    else:
                        if self._verbose:
                            print(
                                "WARNING!!!!!! Failed %d times to find different elements in key '%s' for swapping at iteration %d!"
                                % (int(len(xtmp[key]) / 2), key, j)
                            )

                        continue
                else:
                    if len(xtmp[key]) == 1:
                        r = 0.0
                    elif (xsampling[key] == 0 and len(poptmp[key]) == 0) or len(
                        xtmp[key]
                    ) >= maxnel[key]:
                        r = 1.0
                    else:
                        r = uniform(0.0, 1.0)

                    if r < 0.5:
                        if xsampling[key] == 0:
                            xtmp[key].append(poptmp[key][new])

                            if xdistinct[key]:
                                poptmp[key].pop(new)
                        else:
                            xtmp[key].append(uniform(xbounds[key][0], xbounds[key][1]))

                        if xsort[key]:
                            xtmp[key].sort()
                    else:
                        if xsampling[key] == 0 and xdistinct[key]:
                            poptmp[key].append(xtmp[key][old])

                        xtmp[key].pop(old)

                gamma = 1.0
                ftmp = list(func(xtmp))

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
                    if xsampling[key] == 0 and new is not None:
                        lstep[key] = new

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
                            "    Insertion in the archive consecutively rejected %d times!"
                            % self._maxarchivereject
                        )
                        print("    Quiting at iteration %d..." % j)
                    else:
                        print(
                            "Too many attempts to insert a solution in the archive failed!"
                        )
                        print("Stopping at temperature: %.6f" % temp)

                    print("------")
                    print("\n--- THE END ---")

                    self.savex()

                    return

            if self._verbose:
                if naccept > 0:
                    print("    Number of accepted moves: %d." % naccept)
                    print(
                        "    Fraction of accepted moves: %.6f."
                        % (naccept / self._niter)
                    )

                    if nupdated > 0:
                        print("    Number of archive updates: %d." % nupdated)
                        print(
                            "    Fraction of archive updates in accepted moves: %.6f."
                            % (nupdated / naccept)
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
            print("Stopping at temperature:  %.6f" % temp)
            print("------")

        print("\n--- THE END ---")

    def prunedominated(self, xset: dict = {}, delduplicated: bool = False) -> dict:
        """
        Returns a subset of the full or reduced archive that contains only
        non-dominated solutions.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.
        delduplicated : logical, optional
            Whether to delete or not a solution if the objective values are
            strictly equal to the values of a previous solution. The default
            is False.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are non-dominated.
        """

        tmpdict: dict = {}
        tmpdict["Solution"]: list = []
        tmpdict["Values"]: list = []
        nl: int
        ng: int
        ne: int

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        included: list = [True for i in range(len(xset["Values"]))]

        for i in range(len(xset["Values"]) - 1):
            if not included[i]:
                continue

            for j in range(i + 1, len(xset["Values"])):
                if not included[j]:
                    continue

                nl = ng = ne = 0

                for k in range(len(xset["Values"][i])):
                    if xset["Values"][i][k] < xset["Values"][j][k]:
                        nl += 1
                    elif xset["Values"][i][k] > xset["Values"][j][k]:
                        ng += 1
                    else:
                        ne += 1

                if delduplicated and ne == len(xset["Values"][i]):
                    included[j] = False
                elif nl > 0 and ng == 0:
                    included[j] = False
                elif ng > 0 and nl == 0:
                    included[i] = False

                    break

        for i in range(len(xset["Values"])):
            if included[i]:
                tmpdict["Solution"].append(xset["Solution"][i])
                tmpdict["Values"].append(xset["Values"][i])

        return tmpdict

    def savex(self, xset: dict = {}, archivefile: str = "") -> None:
        """
        Saves the archive into a text file in JSON format.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. Default is {}, meaning the full solution
            archive.
        archivefile : string, optional
            Name of the archive file. Default is '', which means the main
            archive file.

        Returns
        -------
        None.
        """

        if isinstance(xset, dict):
            if not bool(xset):
                xset = self._archive
        else:
            raise MOSAError("The solution archive must be provided as a dictionary!")

        if isinstance(archivefile, str):
            archivefile = archivefile.strip()

            if len(archivefile) == 0:
                archivefile = self._archivefile
        else:
            raise MOSAError("The name of the archive file must be a string!")

        json.dump(xset, open(archivefile, "w"), indent=4)

    def loadx(self, archivefile: str = "") -> None:
        """
        Loads solutions from a JSON file into the archive.

        Parameters
        ----------
        archivefile : string, optional
            Name of the archive file. Default is '', which means the main
            archive file will be used.

        Returns
        -------
        None.
        """

        if isinstance(archivefile, str):
            archivefile = archivefile.strip()

            if len(archivefile) == 0:
                archivefile = self._archivefile
        else:
            raise MOSAError("Name of the archive file must be a string!")

        try:
            tmpdict = json.load(open(archivefile, "r"))
        except FileNotFoundError:
            print("WARNING: File %s not found!" % archivefile)

            return
        except:
            print("WARNING: Something wrong with file %s!" % archivefile)

            return

        self._archive = tmpdict

    def trimx(self, xset: dict = {}, thresholds: list = []) -> dict:
        """
        Extracts from the archive the solutions where the objective values are
        less than the given threshold values.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.
        thresholds : list, optional
            Maximum values of the objective funcions required for a solution
            to be selected. The default is an empty list.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are in agreement with the thresholds.
        """

        tmpdict: dict = {}
        tmpdict["Solution"]: list = []
        tmpdict["Values"]: list = []
        included: bool

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        indexlist = list(range(len(xset["Values"])))

        if len(thresholds) == len(xset["Values"][0]):
            for i in indexlist:
                for j in range(len(xset["Values"][i])):
                    if thresholds[j] is None or xset["Values"][i][j] <= thresholds[j]:
                        included = True
                    else:
                        included = False

                        break

                if included:
                    tmpdict["Solution"].append(xset["Solution"][i])
                    tmpdict["Values"].append(xset["Values"][i])
        else:
            raise MOSAError("The threshold list cannot be empty!")

        return tmpdict

    def reducex(self, xset: dict = {}, index: int = 0, nel: int = 5) -> dict:
        """
        Reduces and sorts in ascending order the archive according to the selected
        objective function.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.
        index : integer, optional
            Index of the objective function that will be used when comparing
            solutions that will be sorted and introduced in the reduced
            solution archive. The default is 0.
        nel : integer, optional
            Number of solutions stored in the reduced archive. The default is 5.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the reduced solution archive.
        """

        tmpdict: dict = {}
        tmpdict["Solution"]: list = []
        tmpdict["Values"]: list = []

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        if nel > len(xset["Values"]):
            nel = len(xset["Values"])

        indexlist = list(range(len(xset["Values"])))

        for i in range(nel):
            k = 0

            for j in indexlist:
                if k == 0:
                    toadd = j
                    bestval = xset["Values"][j][index]
                    k += 1
                else:
                    if xset["Values"][j][index] < bestval:
                        toadd = j
                        bestval = xset["Values"][j][index]

            tmpdict["Solution"].append(xset["Solution"][toadd])
            tmpdict["Values"].append(xset["Values"][toadd])
            indexlist.remove(toadd)

        return tmpdict

    def mergex(self, xsetlist: list) -> dict:
        """
        Merges a list of solution archives into a single solution archive.

        Parameters
        ----------
        xsetlist : list
            A Python list containing the solution archives to be merged.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary containing the merged solution archives.
        """

        tmpdict: dict = {}

        if len(xsetlist) <= 1:
            raise MOSAError("Nothing to be done!")

        if not bool(xsetlist[0]):
            raise MOSAError("First solution archive is empty!")
        else:
            if not (
                "Solution" in xsetlist[0].keys() and "Values" in xsetlist[0].keys()
            ):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xsetlist[0]["Solution"], list)
                    and isinstance(xsetlist[0]["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

                tmpdict = deepcopy(xsetlist[0])

        for i in range(1, len(xsetlist)):
            if (
                bool(xsetlist[i])
                and "Solution" in xsetlist[i].keys()
                and "Values" in xsetlist[i].keys()
                and isinstance(xsetlist[i]["Solution"], list)
                and isinstance(xsetlist[i]["Values"], list)
            ):
                for j in range(len(xsetlist[i]["Values"])):
                    tmpdict["Solution"].append(xsetlist[i]["Solution"][j])
                    tmpdict["Values"].append(xsetlist[i]["Values"][j])
            else:
                raise MOSAError("Format of solution archive %d is wrong! % i")

        return tmpdict

    def copyx(self, xset: dict = {}) -> None:
        """
        Returns a copy of the solution archive.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.

        Returns
        -------
        None.
        """

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        return deepcopy(xset)

    def printx(self, xset: dict = {}) -> None:
        """
        Prints the solutions in the archive (full or reduced) in a more human
        readable format.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.

        Returns
        -------
        None.
        """

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        print("===")
        print("Solutions:")

        for i in range(len(xset["Solution"])):
            print("%d) %s" % (i + 1, xset["Solution"][i]))

        print("Values:")

        for i in range(len(xset["Values"])):
            print("%d) %s" % (i + 1, xset["Values"][i]))

    def plotfront(
        self, xset: dict = {}, index1: int = 0, index2: int = 1, file: str | None = None
    ) -> None:
        """
        Plots 2D scatter plots of selected pairs of objective values.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.
        index1 : integer, optional
            Index of the objective function the value of which will be
            displayed along x-axis. The default is 0.
        index2 : integer, optional
            Index of the objective function the value of which will be
            displayed along y-axis. The default is 1.
        file : string, optional
            Name of the image file where the plot will be saved. The default is
            None, which means that no figure will be created.

        Returns
        -------
        None.
        """

        try:
            import matplotlib.pyplot as plt
        except:
            raise MOSAError("Matplotlib is not available in your system!")

        f: list = [[], []]

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        if (
            index1 >= 0
            and index1 < len(xset["Values"][0])
            and index2 >= 0
            and index2 < len(xset["Values"][0])
        ):
            for i in range(len(xset["Values"])):
                f[0].append(xset["Values"][i][index1])
                f[1].append(xset["Values"][i][index2])

            plt.xlabel("f%d" % index1)
            plt.ylabel("f%d" % index2)
            plt.grid()
            plt.scatter(f[0], f[1])

            if file is not None and len(file) > 0:
                plt.savefig(file)

            plt.show()
        else:
            raise MOSAError("Index out of range!")

    def printstats(self, xset: dict = {}) -> None:
        """
        Prints the minimum, maximum and average values of the objectives.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a
            reduced solution archive. The default is {}, meaning the full
            solution archive.

        Returns
        -------
        None.
        """

        if not bool(xset):
            xset = self._archive
        else:
            if not ("Solution" in xset.keys() and "Values" in xset.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(xset["Solution"], list)
                    and isinstance(xset["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")

        fmin: list = []
        fmax: list = []
        favg: list = []

        for i in range(len(xset["Values"])):
            for j in range(len(xset["Values"][i])):
                if i == 0:
                    fmin.append(xset["Values"][i][j])
                    fmax.append(xset["Values"][i][j])
                    favg.append(xset["Values"][i][j])
                else:
                    if xset["Values"][i][j] < fmin[j]:
                        fmin[j] = xset["Values"][i][j]
                    elif xset["Values"][i][j] > fmax[j]:
                        fmax[j] = xset["Values"][i][j]

                    favg[j] += xset["Values"][i][j]

                    if i == len(xset["Values"]) - 1:
                        print("===")
                        print("Objective function %d: " % j)
                        print("    Minimum: %f" % fmin[j])
                        print("    Maximum: %f" % fmax[j])
                        print("    Average: %f" % (favg[j] / (i + 1)))

    def setpopulation(self, **keys) -> None:
        """
        Sets the population.

        Parameters
        ----------
        **keys : Keyword arguments
            A series of key-value pairs where each key contains the data that can
            be used to achieve an optimized solution to the problem.
            
        Returns
        -------
        None.
        """

        if len(keys) > 0:
            for key, value in keys.items():
                self._population[key] = value
        else:
            raise MOSAError("No keyword was provided!")
            
    def setkeyparams(self, key: str, **params) -> None:
        """
        Sets the optimization parameters for the specified key in the solution 
        to the problem.

        Parameters
        ----------
        key : str
            A key in the solution to the problem.
        **params : Keyword arguments
            Names of the optimization parameters with respective values. They can 
            any of the alternatives below:
                - number_of_solution_elements
                - maximum_number_of_solution_elements
                - no_repeated_elements
                - mc_step_size
                - change_value_move
                - insert_or_delete_move
                - swap_move
                - sort_solution_elements
                - solution_key_selection_weights
                
        Returns
        -------
        None.
        """
        
        allowed: dict
        
        if len(params)>0:
            allowed = {
                "number_of_solution_elements":"self._xnel",
                "maximum_number_of_solution_elements":"self._maxnel",
                "no_repeated_elements":"self._xdistinct",
                "mc_step_size":"self._xstep",
                "change_value_move":"self._changemove",
                "insert_or_delete_move":"self._insordelmove",
                "swap_move":"self._swapmove",
                "sort_solution_elements":"self._xsort",
                "solution_key_selection_weights":"self._xselweight",
            }
            
            for param,value in params.items():
                if param in allowed:
                    exec(f"{allowed[param]}[key]=value")
        else:
            raise MOSAError("No keyword was provided!")

    def setoptparam(self, param: str, **keys) -> None:
        """
        Sets the values of the specified optimization parameter.

        Parameters
        ----------
        param : str
            Name of the optimization parameter. It must be one of the alternatives below:
                - number_of_solution_elements
                - maximum_number_of_solution_elements
                - no_repeated_elements
                - mc_step_size
                - change_value_move
                - insert_or_delete_move
                - swap_move
                - sort_solution_elements
                - solution_key_selection_weights
        **keys : Keyword arguments
            A series of key-value pairs where each key corresponds to a key in the
            solution to the problem.
            
        Returns
        -------
        None.
        """

        params: tuple
        execstr: str

        if len(keys) > 0:
            params = {
                "number_of_solution_elements":"self._xnel",
                "maximum_number_of_solution_elements":"self._maxnel",
                "no_repeated_elements":"self._xdistinct",
                "mc_step_size":"self._xstep",
                "change_value_move":"self._changemove",
                "insert_or_delete_move":"self._insordelmove",
                "swap_move":"self._swapmove",
                "sort_solution_elements":"self._xsort",
                "solution_key_selection_weights":"self._xselweight",
            }

            if param in params:
                execstr = "for key,value in keys.items():\n"
                execstr += f"    {params[param]}[key]=value"
                exec(execstr)
            else:
                raise MOSAError("Optimization parameter does not exist!")
        else:
            raise MOSAError("No keyword was provided!")

    def __updatearchive(self, x: dict, f: list) -> int:
        """
        Checks if the solution given as argument is better than solutions randomly
        (and sequentially) chosen from the archive. If so, the archive is updated,
        this solution is appended and a worse solution is removed.

        Parameters
        ----------
        x : dictionary
            A Python dictionary containing the solution.
        f : list
            A Python list containing the values of the objectives associated
            with the solution.

        Returns
        -------
        updated : integer
            1, if the archive is updated, or 0, if not.
        """

        updated: int = 0
        indexlist = list(range(len(self._archive["Values"])))

        for i in indexlist:
            if f == self._archive["Values"][i]:
                return updated

        if len(self._archive["Solution"]) == 0:
            updated = 1
        else:
            shuffle(indexlist)

            for i in indexlist:
                nl = ng = 0

                for j in range(len(self._archive["Values"][i])):
                    if f[j] < self._archive["Values"][i][j]:
                        nl += 1
                    elif f[j] > self._archive["Values"][i][j]:
                        ng += 1

                if len(self._archive["Solution"]) < self._archivesize:
                    if nl > 0:
                        updated = 1

                        if ng == 0:
                            self._archive["Solution"].pop(i)
                            self._archive["Values"].pop(i)

                            break
                    else:
                        updated = 0

                        break
                else:
                    if nl > 0 and ng == 0:
                        self._archive["Solution"].pop(i)
                        self._archive["Values"].pop(i)

                        updated = 1

                        break

        if updated == 1:
            self._archive["Solution"].append(x)
            self._archive["Values"].append(f)

        return updated

    def __getcheckpoint(self) -> tuple:
        """
        Initializes with a solution from a previous run.

        Returns
        -------
        x : dictionary
            A Python dictionary containing a solution.
        f : list
            A Python list containing the values of the objective functions
            associated with the solution.
        population : dictionary
            A Python dictionary containing the population compatible with the
            solution.
        """

        tmpdict: dict = {}
        x: dict = {}
        f: list = []
        population: dict = {}

        print("Looking for a solution in the checkpoint file...")

        try:
            tmpdict = json.load(open("checkpoint.json", "r"))

            if (
                "Solution" in tmpdict.keys()
                and "Values" in tmpdict.keys()
                and "Population" in tmpdict.keys()
            ):
                x = tmpdict["Solution"]
                f = tmpdict["Values"]
                population = tmpdict["Population"]

                if "SampleSpace" in tmpdict.keys():
                    ss = tmpdict["SampleSpace"]

                    for key in ss.keys():
                        if ss[key] == 1:
                            population[key] = tuple(population[key])
        except:
            print("No checkpoint file!")

        print("Done!")

        return x, f, population

    def __savecheckpoint(self, x: dict, f: list, population: dict) -> None:
        """
        Saves the solution passed as argument as JSON into a text file.

        Parameters
        ----------
        x : dictionary
            A Python dictionary containing the solution.
        f : list
            A Python list containing the values of the objectives associated
            with the solution.
        population : dictionary
            A Python dictionary containing the population compatilbe with the
            solution.

        Returns
        -------
        None.
        """

        tmpdict: dict = {
            "Solution": x,
            "Values": f,
            "Population": population,
            "SampleSpace": {},
        }

        for key in population.keys():
            if isinstance(population[key], list):
                tmpdict["SampleSpace"][key] = 0
            elif isinstance(population[key], tuple):
                tmpdict["SampleSpace"][key] = 1

        json.dump(tmpdict, open("checkpoint.json", "w"), indent=4)

    """
    Class properties
    ----------------
    population : dictionary, optional
        A Python dictionary, each key of which contains the data that can be 
        used to achieve an optimized solution to the problem. Default is
        {"X":(-1.0,1.0)}.
    archive : dictionary
        A Python dictionary with two keys: "Solution", which contains a list of 
        the best solutions to the problem, and "Values", which contains a list 
        of the corresponding objective values. It should not be changed manually.
    restart : logical, optional
        Whether the optimization process must restart from a previous run (if 
        a checkpoint file is available) or not. Default is True.
    objective_weights : list, optional
        A Python list containing weights for the objectives, one per objective.
        Default is [], which means the same weight (1.0) for all objectives.
    initial_temperature : double, optional
        Initial temperature for the Simulated Annealing algorithm. Default 
        value is 1.0.
    temperature_decrease_factor : double, optional
        Decrease factor of the temperature during Simulated Annealing. It
        determines how fast the quench will occur. Default value is 0.9.
    number_of_temperatures : integer, optional
        Number of temperatures to be considered in Simulated Annealing.
        Default is 10.
    number_of_iterations : integer, optional
        Number of Monte Carlo iterations per temperature. Default is 1000.
    archive_size : integer, optional
        Maximum number of solutions in the archive. Default value is 1000.
    archive_file : string, optional
        Text file where the archive should be saved to. Default value is
        'archive.json'.
    maximum_archive_rejections : integer, optional
        Maximum number of consecutive rejections of insertion of a solution 
        in the archive. Once reached, the optimization process finishes.
        Default value is 1000.
    alpha : float, optional
        Value of the alpha parameter. Default value is 0.0.
    number_of_solution_elements : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the number of elements for that key in the solution 
        set. Default value is {}, which means one element for all keys in the 
        solution.
    maximum_number_of_solution_elements : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the maximum number of elements for that key in the 
        solution set, if the number of elements is variable. Default value is 
        {}, which means an unlimited number of elements can be present in the 
        solution key.
    no_repeated_elements : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies whether an element cannot be repeated in the solution. 
        Default value is {}, which means that repetitions are allowed.
    mc_step_size : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        and specifies the maximum number of steps, to the left or to the right, 
        that the Monte Carlo algorithm can take when randomly selecting an 
        element in the corresponding key in the population to insert in the 
        solution. Default is {}, which means 0.1 for continuous search 
        spaces and half the number of elements in the population for discrete 
        search spaces.
    change_value_move : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the weight (non-normalized probability) used to select
        a trial move in which the value of a randomly selected element in the 
        solution set will be modified. How this modification is done depends on 
        the sample space of solutions to the problem: (1) if discrete, the 
        exchange of values between the solution and the population; or (2) if 
        continuous, the random increment/decrement of the value of an element 
        in the solution set. Default value is {}, which means the weight to 
        select this trial move is equal to 1.0.
    insert_or_delete_move : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the weight (non-normalized probability) used to 
        select a trial move in which an element will be inserted into or deleted 
        from the solution set. Default value is {}, which means this trial move 
        is not allowed, i.e., weight equal to zero.
    swap_move : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the weight (non-normalized probability) used to 
        select a trial move in which the algorithm swaps two randomly chosen 
        elements in the solution set. Default value is {}, which means this 
        trial move is not allowed, i.e., weight equal to zero.
    sort_solution_elements : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies if the list in that key must be sorted in ascending 
        order. Default is {}, which means no sorting at all.
    solution_key_selection_weights : dictionary, optional
        A Python dictionary where each key corresponds to a key in the solution 
        set and specifies the selection weight of this key in a Monte Carlo 
        iteration. Default value is {}, which means that all keys have the same 
        selection weight, i.e., the same probability of being selected.
    track_optimization_progress : boolean, optional
        Whether to track or not optimization progress by saving the accepted 
        objetive values into a Python list. Default is False.
    accepted_objective_values : list, readonly
        A Python list of accepted objective values over Monte Carlo algorithm 
        iterations, useful for diagnostic purposes or for tracking optimization 
        progress.
    verbose : boolean, optional
        Whether to display verbose output or not. Default is False.
    """

    @property
    def population(self) -> dict:
        return self._population

    @population.setter
    def population(self, val: dict) -> None:
        if isinstance(val, dict) and bool(val):
            self._population = val
        else:
            raise MOSAError("Population must be a non-empty dictionary!")

    @property
    def archive(self) -> dict:
        return self._archive

    @archive.setter
    def archive(self, val: dict) -> None:
        if isinstance(val, dict) and bool(val):
            if not ("Solution" in val.keys() and "Values" in val.keys()):
                raise MOSAError(
                    "'Solution' and 'Values' must be present in the dictionary!"
                )
            else:
                if not (
                    isinstance(val["Solution"], list)
                    and isinstance(val["Values"], list)
                ):
                    raise MOSAError("'Solution' and 'Values' must be Python lists!")
        else:
            raise MOSAError("The archive must be a non-empty dictionary!")

        self._archive = val

    @property
    def restart(self) -> bool:
        return self._restart

    @restart.setter
    def restart(self, val: bool) -> None:
        if isinstance(val, bool):
            self._restart = val
        else:
            raise MOSAError("Restart must be a boolean!")

    @property
    def objective_weights(self) -> list:
        return self._weight

    @objective_weights.setter
    def objective_weights(self, val: list) -> None:
        if isinstance(val, list):
            self._weight = val
        else:
            raise MOSAError("The weights must be provided in a list!")

    @property
    def initial_temperature(self) -> float:
        return self._initemp

    @initial_temperature.setter
    def initial_temperature(self, val: int | float) -> None:
        if isinstance(val, (int, float)) and val > 0.0:
            self._initemp = val
        else:
            raise MOSAError("Initial temperature must be a number greater than zero!")

    @property
    def temperature_decrease_factor(self) -> float:
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
        return self._archivesize

    @archive_size.setter
    def archive_size(self, val: int) -> None:
        if isinstance(val, int) and val > 0:
            self._archivesize = val
        else:
            raise MOSAError("The archive size must be an integer greater than zero!")

    @property
    def archive_file(self) -> str:
        return self._archivefile

    @archive_file.setter
    def archive_file(self, val: str) -> None:
        if isinstance(val, str) and len(val.strip()) > 0:
            self._archivefile = val.strip()
        else:
            raise MOSAError("A file name must be provided!")

    @property
    def maximum_archive_rejections(self) -> int:
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
        return self._alpha

    @alpha.setter
    def alpha(self, val: float) -> None:
        if isinstance(val, float) and val >= 0.0 and val <= 1.0:
            self._alpha = val
        else:
            raise MOSAError("Alpha must be a number between zero and one!")

    @property
    def number_of_solution_elements(self) -> dict:
        return self._xnel

    @number_of_solution_elements.setter
    def number_of_solution_elements(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, int) and value > 0:
                    self._xnel[key] = value
                else:
                    raise MOSAError(
                        "Key '%s' must contain an integer greater than zero!" % key
                    )
        else:
            raise MOSAError(
                "Number of solution elements must be provided in a dictionary!"
            )

    @property
    def maximum_number_of_solution_elements(self) -> dict:
        return self._maxnel

    @maximum_number_of_solution_elements.setter
    def maximum_number_of_solution_elements(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, int) and value >= 2:
                    self._maxnel[key] = value
                else:
                    raise MOSAError(
                        "Key '%s' must contain an integer greater than or equal to 2!"
                        % key
                    )
        else:
            raise MOSAError(
                "Maximum number of solution elements must be provided in a dictionary!"
            )

    @property
    def no_repeated_elements(self) -> dict:
        return self._xdistinct

    @no_repeated_elements.setter
    def no_repeated_elements(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, bool):
                    self._xdistinct[key] = value
                else:
                    raise MOSAError("Key '%s' must contain a boolean!" % key)
        else:
            raise MOSAError(
                "Whether or not to repeat elements in the solution must be provided as a dictionary!"
            )

    @property
    def mc_step_size(self) -> dict:
        return self._xstep

    @mc_step_size.setter
    def mc_step_size(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (int, float)):
                    self._xstep[key] = value
                else:
                    raise MOSAError("Key '%s' must contain a number!" % key)
        else:
            raise MOSAError("Monte Carlo step size must be provided in a dictionary!")

    @property
    def change_value_move(self) -> dict:
        return self._changemove

    @change_value_move.setter
    def change_value_move(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._changemove[key] = value
                else:
                    raise MOSAError("Key '%s' must contain a positive number!" % key)
        else:
            raise MOSAError("Weight of trial move must be provided in a dictionary!")

    @property
    def insert_or_delete_move(self) -> dict:
        return self._insordelmove

    @insert_or_delete_move.setter
    def insert_or_delete_move(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._insordelmove[key] = value
                else:
                    raise MOSAError("Key '%s' must be a positive number!" % key)
        else:
            raise MOSAError("Weight of trial move must be provided in a dictionary!")

    @property
    def swap_move(self) -> dict:
        return self._swapmove

    @swap_move.setter
    def swap_move(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (float, int)) and value >= 0.0:
                    self._swapmove[key] = value
                else:
                    raise MOSAError("Key '%s' must be a positive number!" % key)
        else:
            raise MOSAError("Weight of trial move must be provided in a dictionary!")

    @property
    def sort_solution_elements(self) -> dict:
        return self._xsort

    @sort_solution_elements.setter
    def sort_solution_elements(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, bool):
                    self._xsort[key] = value
                else:
                    raise MOSAError("Key '%s' must contain a boolean!" % key)
        else:
            raise MOSAError("Sort solution elements must be provided as a dictionary!")

    @property
    def solution_key_selection_weights(self) -> dict:
        return self._xselweight

    @solution_key_selection_weights.setter
    def solution_key_selection_weights(self, val: dict) -> None:
        if isinstance(val, dict):
            for key, value in val.items():
                if isinstance(value, (int, float)):
                    self._xselweight[key] = value
                else:
                    raise MOSAError("Key '%s' must contain a number!" % key)
        else:
            raise MOSAError(
                "Solution key selection weights must be provided as a dictionary!"
            )

    @property
    def track_optimization_progress(self) -> bool:
        return self._trackoptprogress

    @track_optimization_progress.setter
    def track_optimization_progress(self, val: bool) -> None:
        if isinstance(val, bool):
            self._trackoptprogress = val
        else:
            raise MOSAError("Tracking or not optimization progress must be a boolean!")

    @property
    def accepted_objective_values(self) -> list:
        return self._f

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        if isinstance(val, bool):
            self._verbose = val
        else:
            raise MOSAError("Displaying or not verbose output must be a boolean!")


class MOSAError(Exception):
    def __init__(self, message: str = "") -> None:
        """
        __init__ -> class constructor.

        Parameters
        ----------
        message : string, optional
            Error message to be displayed. The default is "".

        Returns
        -------
        None.
        """
        self.message = message

    def __str__(self) -> str:
        """
        __str__ -> returns the error message.

        Returns
        -------
        message
            Error message to be displayed.
        """
        return self.message
