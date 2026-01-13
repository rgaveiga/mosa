"""
Simulated Annealing (SA) has been initially proposed in [S. Kirkpatrick et al., 
Optimization by Simulated Annealing, Science 220, 671-680 (1983)](https://doi.org/10.1126/science.220.4598.671) 
as a global optimization metaheuristic inspired by the metallurgical heat treatment 
process called *annealing*.

We can read in the abstract of this seminal work:

> *There is a deep and useful connection between statistical mechanics (the 
> behavior of systems with many degrees of freedom in thermal equilibrium 
> at a finite temperature) and multivariate or combinatorial optimization 
> (finding the minimum of a given function depending on many parameters). 
> A detailed analogy with annealing in solids provides a framework for 
> optimization of the properties of very large and complex systems. This 
> connection to statistical mechanics exposes new information and provides 
> an unfamiliar perspective on traditional optimization problems and methods.*

SA is typically used in the optimization of combinatorial problems (e.g., traveling 
salesman or knapsack problems), but can also be used to optimize problems where 
the search space is continuous.

Multi-Objective Simulation Annealing (MOSA) extends the original, single-objective 
SA algorithm to approximate the Pareto front in multi-objective optimization problems. 
A comprehensive discussion on MOSA and its algorithm variants can be found in 
[Multi-objective Simulated Annealing: Principles and Algorithm Variants, Advances 
in Operations Research, Volume 2019, Article ID 8134674](https://doi.org/10.1155/2019/8134674).

This library implements MOSA in Python as a probabilistic 
[Metropolis Monte Carlo algorithm](https://doi.org/10.1063/1.1699114).

## Installation

The easiest way to install **MOSA** is using **pip**:

```
pip install mosa
```

## Getting started

The *optimization problem* must be implemented as a Python function that takes 
the values ​​of a tentative solution as arguments and returns the *objective values* 
as a tuple of floating-point numbers.

For example, the function below implements the 
[Binh and Korn problem](https://github.com/rgaveiga/mosa/tree/main/examples/binh_and_korn):

```python
def fobj(X1: float, X2: float) -> tuple:
    f1 = 4.0 * (pow(X1, 2) + pow(X2, 2))
    f2 = pow((X1 - 5), 2) + pow((X2 - 5), 2)

    c1 = pow((X1 - 5), 2) + pow((X2), 2)
    c2 = pow((X1 - 8), 2) + pow((X2 + 3), 2)

    if c1 > 25.0 or c2 < 7.7:
        f1 = inf
        f2 = inf

    return f1, f2
``` 

> [!NOTE]
> MOSA treats the optimization problem as a *black box*: only the inputs and outputs
> matter. The algorithm imposes no restrictions on what the function does with the arguments.

Then, the user needs to import and create an instance of the `mosa.mosa.Anneal` class:

```python
from mosa import Anneal

opt = Anneal()
```

The population from which solutions to the problem will be sampled can be defined 
as follows:
    
```python
opt.set_population(X1=(0.0, 5.0), X2=(0.0, 3.0))
```

It is important to make clear that *population* here does not have the same meaning 
as in techniques such as Genetic Algorithm or Particle Swarm. In the context of 
this package, a population is the set of all elements that can be part of a solution 
to the problem.

Furthermore, similar elements of the population (e.g., elements with the same 
meaning, type, or boundaries) form *groups*. A group can be a discrete list 
of elements or a continuous range of numbers between a minimum and a maximum 
value. The same groups that make up the population will be present in the 
solutions sampled from it.

A number of MOSA hyperparameters can be set to control the optimization process. 
For example, `mosa.mosa.Anneal.initial_temperature` sets the initial fictitious 
temperature in the Monte Carlo acceptance rule, while `mosa.mosa.Anneal.number_of_temperatures` 
allows the user to determine the maximum number of temperatures in the annealing. 
See the `mosa.mosa.Anneal` properties documentation for more details.

The hyperparameters related to the solutions themselves are assigned using the 
`mosa.mosa.Anneal.set_group_params` or `mosa.mosa.Anneal.set_opt_param` methods, 
as seen in the following code snippet, also with the Binh and Korn problem in 
mind:
    
```python
opt.set_opt_param("number_of_elements", X1=1, X2=1)
opt.set_opt_param("mc_step_size", X1=0.5, X2=0.3)
```

The optimization process starts by calling the `mosa.mosa.Anneal.evolve` method:
    
```python
opt.evolve(fobj)
```

The solutions to the problem are stored in the *archive*. After the optimization 
process is finished, various operations can be performed on the archive, for 
example, pruning dominated solutions that still remain using the 
`mosa.mosa.Anneal.prune_dominated` method. See the `mosa.mosa.Anneal` methods 
documentation for additional information.

## Usage examples

The Binh and Korn problem above and other examples of optimization problems that 
can be solved with MOSA can be found in the Jupyter notebooks in the 
[examples](https://github.com/rgaveiga/mosa/tree/main/examples) directory.
"""

__version__ = "0.8.5"

from .mosa import Anneal
