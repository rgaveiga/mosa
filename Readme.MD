This package implements in a single Python module, *mosa*, the Multi-Objective Simulated Annealing (MOSA) 
heuristic to approximate the Pareto front in multi-objective optimization problems. MOSA is a kind 
of Monte Carlo algorithm, therefore, it employs random numbers in the process of searching for optimal 
solutions to a problem. A broad discussion about Multi-Objective Simulated Annealing and its algorithm variants can be found in \[*Multiobjective Simulated Annealing: Principles and Algorithm Variants*, https://doi.org/10.1155/2019/8134674\]. In the current version of the *mosa* module, the acceptance probability of any randomly generated solution is given by Eq. 4 in this reference.

Jupyter notebooks in the test directory provide usage examples. In the following sections, the basic workflow of a MOSA run is briefly described.

## Problem definition

Initially, it is necessary to provide a Python dictionary, the **population**, from which the solutions will be sampled. For example, the population of the *alloy_optimization* problem in the test set is defined as follows:

```python
population={"Component":df.Component.values.tolist(),"Concentration":[]}
```

In the *population* dictionary above, the key "Component" is filled with data obtained from a *pandas* dataframe. If a key in the population contains a finite list of elements, the corresponding key in the solutions can only contain elements taken from this list. Therefore, the search space represented by the key is **discrete**. On the other hand, the "Concentration" key is defined as an empty list. This means that the corresponding key in the solutions consists of a list with one or more floats randomly chosen within a lower and upper bound, which enclose a **continuous** search space.

Subsequently, the user must implement a Python function that takes as its single argument a solution to the problem and returns the resulting objective values:

```python
def fobj(solution):
    '''
    The problem is implemented here. User-defined operations are carried out using 
    the solution as input.
    Values are assigned to the objectives, f1 and f2 (in the case of two objectives).
    '''
    return [f1,f2]
```

See the problems in the test set for examples on how to implement such a function. Remember that the solution also is a dictionary with the same keys as the population. 

## Setting up a MOSA run

First, the user has to import the class *Anneal* from the *mosa* module, instantiate it and assign to the *Anneal* object a population.

```python
from mosa import Anneal
opt=Anneal()
opt.setpopulation(population)
```

Next, it is necessary to configure the archive and the solutions. These tasks are accomplished by two methods of the *Anneal* class:

```
setarchiveconfig(archivesize,archivefile,maxarchivereject) -> sets configurations of the archive.
        
        Parameters
        ----------
        archivesize : integer, optional
            Maximum number of solutions in the archive. Default value is 100.
        archivefile : string, optional
            Text file where the archive should be saved to. Default value is
            'archive.json'.
        maxarchivereject : integer, optional
            Maximum number of consecutive rejections of insertion of a solution 
            in the archive. Once reached, the optimization process finishes.
            Default value is 1000.
        
        Returns
        -------
        None.
```

```
setxconfig(xnel,xdistinct,xstep,exchangeprob,xbounds,xsort,selweight) -> sets configurations of the solution.
        
        Parameters
        ----------
        xnel : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the number of elements for that key in
            the solution set. Default value is an empty dictionary.
        xdistinct : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies if the insertion of an element in that 
            key in the solution set requires the corresponding element to be 
            removed from the population. Default value is an empty dictionary.
        xstep : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution and specifies the maximum number of steps, to the left or
            to the right, that the algorithm can take when randomly selecting 
            an element in the corresponding key in the population to insert in 
            the solution. Default is an empty dictionary.
        exchangeprob : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the probability that elements will be
            exchanged between that key in the solution set and the population.
            If less than 1, it implies that there is a probability that elements
            will be added/removed from that key in the solution set. Default 
            value is an empty dictionary.
        xbounds : dictionary
            A Python dictionary where each key corresponds to a key in the 
            solution set and specifies the lower and upper values of solution
            elements. Default value is an empty dictionary.
        xsort : dictionary
            A Python dictionary where each key corresponds to a key in the
            solution set and specifies if the list in that key must be sorted
            in ascending order.
        selweight : dictionary
            A Python dictionary where each element corresponds to a key in the
            solution set and specifies the selection weight of this key in an
            MC iteration. Default value is an empty dictionary.
            
        Returns
        -------
        None.
```

## Running MOSA

The MOSA algorithm itself is contained in the *evolve* method, which must be called taking as its single argument the function where the user implemented the problem. For example:

```python
opt.evolve(fobj)
```

That is all.

## Purging dominated solutions

The best solutions are stored into a Python dictionary, the **archive**. However, at the end of a MOSA run, dominated solutions may still remain in the archive. The *purgedominated* method can be used to get rid of them, returning a version of the archive with only non-dominated solutions:

```
purgedominated(xset) -> returns a subset of the full or reduced archive that
        contains only non-dominated solutions.

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are non-dominated.
```

## Output

The *mosa* module provides methods to display the solutions saved in the archive, as well as to plot the Pareto front (using *matplotlib*).

```
printx(xset) -> prints the solutions in the archive (complete or reduced) in 
        a more human readable format.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary, 
            meaning the full solution archive.
        
        Returns
        -------
        None.
```

```
plotfront(xset,index1,index2) -> plots 2D scatter plots of selected pairs of objective 
        functions.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary, 
            meaning the full solution archive.
        index1 : integer, optional
            Index of the objective function the value of which will be 
            displayed along x-axis. The default is 0.
        index2 : integer, optional
            Index of the objective function the value of which will be 
            displayed along y-axis. The default is 1.

        Returns
        -------
        None.
```

## Decision making

The module also provides two methods that allow the user to reduce the amount of optimal solutions according to his/her needs, in order to decide which one will be selected:

```pseudocode
trimx(xset,thresholds) -> extracts from the archive the solutions whose values of the
        objective functions are less than the given threshold values.
        
        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.
        thresholds : list, optional
            Maximum values of the objective funcions required for a solution
            to be selected. The default is an empty list.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the solution archive with only
            the solutions that are in agreement with the thresholds.
```

```pseudocode
reducex(xset,index,nel) -> reduces and sorts the archive according to the value of
        an objective function.        

        Parameters
        ----------
        xset : dictionary, optional
            A Python dictionary containing the full solution archive or a 
            reduced solution archive. The default is an empty dictionary,
            meaning the full solution archive.
        index : integer, optional
            Index of the objective function value that will be used when 
            comparing solutions that will be sorted and introduced in the
            reduced solution archive. The default is 0.
        nel : integer, optional
            Number of solutions stored in the reduced archive. The default is 5.

        Returns
        -------
        tmpdict : dictionary
            A Python dictionary representing the reduced solution archive.
```
