# Multi-Objective Simulated Annealing (MOSA)

Simulated Annealing (SA) has been initially proposed in [*Optimization by Simulated Annealing*](https://doi.org/10.1126/science.220.4598.671) as an optimization heuristic. Multi-objective Simulated Annealing (MOSA) extends the original, single-objective SA to approximate the Pareto front in multi-objective optimization problems. A thorough discussion about MOSA and its algorithm variants can be found in [*Multi-objective Simulated Annealing: Principles and Algorithm Variants*](https://doi.org/10.1155/2019/8134674).

In the following sections, the basic workflow of a MOSA run with this package is briefly described. Jupyter notebooks in the *test*  directory provide usage examples.

## Installing MOSA

The easiest way to install MOSA is using **pip**:

> pip install mosa

## Setting up a MOSA run

First, the user has to import the class *Anneal* from the *mosa* module and instantiate it:

```python
from mosa import Anneal
opt=Anneal()
```

Then the user assigns values to *Anneal*'s properties that will control the optimization process. These properties are described below:

- population : dictionary, optional
    - A Python dictionary, each key of which contains the data that can be used to achieve an optimized solution to the problem. Default is {"X":(-1.0,1.0)}.

```
archive : dictionary
    A Python dictionary with two keys: "Solution", which contains a list of     the best solutions to the problem,
    and "Values", which contains a list     of the corresponding objective values. It should not be changed 
    manually.
restart : logical, optional
    Whether the optimization process must restart from a previous run (if a checkpoint file is available) or not.
    Default is True.
objective_weights : list, optional
    A Python list containing weights for the objectives, one per objective. Default is [], which means the same weight
    (1.0) for all objectives.
initial_temperature : double, optional
    Initial temperature for the Simulated Annealing algorithm. Default value is 1.0.
temperature_decrease_factor : double, optional
    Decrease factor of the temperature during Simulated Annealing. It determines how fast the quench will occur. Default
    value is 0.9.
number_of_temperatures : integer, optional
    Number of temperatures to be considered in Simulated Annealing. Default is 10.
number_of_iterations : integer, optional
    Number of Monte Carlo iterations per temperature. Default is 1000.
archive_size : integer, optional
    Maximum number of solutions in the archive. Default value is 1000.
archive_file : string, optional
    Text file where the archive should be saved to. Default value is 'archive.json'.
maximum_archive_rejections : integer, optional
    Maximum number of consecutive rejections of insertion of a solution in the archive. Once reached, the optimization 
    process finishes. Default value is 1000.
alpha : float, optional
    Value of the alpha parameter. Default value is 0.0.
number_of_solution_elements : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the number of elements for
    that key in the solution set. Default value is {}, which means one element for all keys in the solution.
maximum_number_of_solution_elements : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the maximum number of elements
    for that key in the solution set, if the number of elements is variable. Default value is {}, which means an unlimited 
    number of elements can be present in the solution keys.
no_repeated_elements : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies whether an element cannot be 
    repeated in the solution. Default value is {}, which means that repetitions are allowed. 
mc_step_size : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution and specifies the maximum number of steps, to the
    left or to the right, that the Monte Carlo algorithm can take when randomly selecting an element in the corresponding key
    in the population to insert in the solution. Default is {}, which means 0.1 for continuous search spaces and half the number
    of elements in the population for discrete search spaces.
change_value_move : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the weight (non-normalized 
    probability) used to select a trial move in which the value of a randomly selected element in the solution set will be modified.
    How this modification is done depends on the sample space of solutions to the problem: (1) if discrete, the exchange of values
    between the solution and the population; or (2) if continuous, the random increment/decrement of the value of an element in the 
    solution set. Default value is {}, which means the weight to select this trial move is equal to 1.0.
insert_or_delete_move : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the weight (non-normalized probability) 
    used to select a trial move in which an element will be inserted into or deleted from the solution set. Default value is {}, which 
    means this trial move is not allowed, i.e., weight equal to zero.
swap_move : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the weight (non-normalized probability) 
    used to select a trial move in which the algorithm swaps two randomly chosen elements in the solution set. Default value is {}, 
    which means this trial move is not allowed, i.e., weight equal to zero.
sort_solution_elements : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies if the list in that key must be sorted in 
    ascending order. Default is {}, which means no sorting at all.
solution_key_selection_weights : dictionary, optional
    A Python dictionary where each key corresponds to a key in the solution set and specifies the selection weight of this key in a 
    Monte Carlo iteration. Default value is {}, which means that all keys have the same selection weight, i.e., the same probability of 
    being selected.
track_optimization_progress : boolean, optional
    Whether to track or not optimization progress by saving the accepted objetive values into a Python list. Default is False.
accepted_objective_values : list, readonly
    A Python list of accepted objective values over Monte Carlo algorithm iterations, useful for diagnostic purposes or for tracking 
    optimization progress.
verbose : boolean, optional
    Whether to display verbose output or not. Default is False.
```

## Problem definition

The very first step is to provide a Python dictionary, the *population*, from which the solutions will be sampled. For example, the population of the *alloy_optimization* problem in the test directory is defined as follows:

```python
opt.population={"Component":Component.tolist(),"Concentration":(0.0,0.1)}
```

In the *population* dictionary above, the "Component" key is filled with data obtained from a *numpy* array converted to a Python list. If a key in the population contains a Python list, the corresponding key in the solutions can only contain elements taken from this list. Therefore, the sample space represented by the key is **discrete**. 

On the other hand, the value in the "Concentration" key is a Python tuple with two numbers, the lower and upper bounds of a **continuous** sample space. This indicates that the corresponding key in the solutions consists of a list that contains one or more float numbers randomly chosen within the lower and upper bounds.

Subsequently, the user must implement a Python function that takes as its single argument a trial solution to the problem and returns the resulting objective values:

```python
def fobj(solution):
    '''
    The problem is implemented here. User-defined operations are carried out using 
    the solution as input. f1 and f2, the return values, are the values of the 
    objectives.
    '''
    return f1,f2
```

See the problems in the test directory for examples on how to implement such a function. Remember that the solution must also be a dictionary with the same keys as the population.

## Running MOSA

The MOSA algorithm itself is contained in the *evolve* method, which must be called taking as its single argument the function where the user implemented the problem. For example:

```python
opt.evolve(fobj)
```

That is all.

## Pruning dominated solutions

The best solutions are stored into a Python dictionary, the **archive**. However, at the end of a MOSA run, dominated solutions may still remain in the archive. The *prunedominated* method can be used to get rid of them, returning a version of the archive with only non-dominated solutions:

```
    prunedominated(xset,delduplicated) -> returns a subset of the full or 
    reduced archive that contains only non-dominated solutions.

    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full solution 
        archive.
    delduplicated : logical, optional
        Whether to delete or not a solution if the objective values are 
        strictly equal to the values of a previous solution. The default 
        is False.

    Returns
    -------
    tmpdict : dictionary
        A Python dictionary representing the solution archive with only
        the solutions that are non-dominated.
```

## Output

The *mosa* module provides methods to display the solutions saved in the archive (*printx*), as well as to plot the Pareto front (*plotfront*). Moreover, basic statistics (minimum, maximum and average objective values) can also be shown on screen by using the *printstats* method.

```
    printx(xset) -> prints the solutions in the archive (complete or 
    reduced) in a more human readable format.
        
    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full solution 
        archive.
        
    Returns
    -------
    None.
```

```
    plotfront(xset,index1,index2) -> plots 2D scatter plots of selected 
    pairs of objective values.
        
    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full solution 
        archive.
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

```
    printstats(xset) -> prints the minimum, maximum and average values of 
    the objectives.
        
    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full 
        solution archive.
            
    Returns
    -------
    None.
```

## Decision-making

The module also provides two methods that allow the user to reduce the amount of optimal solutions according to his/her needs, in order to decide which one will be selected:

```
    trimx(xset,thresholds) -> extracts from the archive the solutions the 
    objective values are less than the given threshold values.
        
    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full solution 
        archive.
    thresholds : list, optional
        Maximum values of the objective funcions required for a solution
        to be selected. The default is an empty list.

    Returns
    -------
    tmpdict : dictionary
        A Python dictionary representing the solution archive with only
        the solutions that are in agreement with the thresholds.
```

```
    reducex(xset,index,nel) -> reduces and sorts in ascending order the 
    archive according to the selected objective function.        
    
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
```

## Saving, loading, copying and merging solution archives

The full solution archive is saved from times to times during the optimization process into a JSON file (default is *archive.json*, see the *archive_file* property above). There is a method, *savex*, that allows the user to save a solution archive, e.g., a reduced archive, also in the JSON format. Another method, *loadx*, can be used to load the full solution archive from a JSON file. The *copyx* method can be used to copy a solution archive. Finally, two or more solution archives from different MOSA runs can be merged into a single solution archive using the *mergex* method, which provides a simple way to run MOSA in parallel.

```
    savex(xset,archivefile) -> saves the archive into a text file in JSON 
    format.
        
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
```

```
    loadx(archivefile) -> loads solutions from a JSON file into the archive.
        
    Parameters
    ----------
    archivefile : string, optional
        Name of the archive file. Default is '', which means the main 
        archive file will be used.
                
    Returns
    -------
    None.
```

```
    copyx(xset) -> returns a copy of archive.
        
    Parameters
    ----------
    xset : dictionary, optional
        A Python dictionary containing the full solution archive or a 
        reduced solution archive. The default is {}, meaning the full 
        solution archive.
        
    Returns
    -------
    None.
```

```
    mergex(xsetlist) -> merges a list of solution archives into a single 
    solution archive.        

    Parameters
    ----------
    xsetlist : list
        A Python list containing the solution archives to be merged.
            
    Returns
    -------
    tmpdict : dictionary
        A Python dictionary containing the merged solution archives.
```

## Need to say that...

... the *mosa* module is a pure Python implementation of the MOSA algorithm. As such, one should not expect it to rival, for example, C/C++ implementations in terms of performance. On the other hand, by dealing with Python dictionaries with meaningful names rather than cryptic arrays, *mosa* provides a much more descriptive data model, which also facilitates the data analysis and decision-making process. The current version uses Python lists to store data. Perhaps replacing them with *numpy* arrays and using *numpy* functions in a vectorized fashion can add a few milliseconds to the speed of execution. But that's for future versions.

Finally, the code is provided as is. The author makes no guarantee that its results are accurate and is not responsible for any losses caused by the use of the code. If you have any questions, comments or suggestions about the code, just [drop a message](mailto:roberto.veiga@ufabc.edu.br).
