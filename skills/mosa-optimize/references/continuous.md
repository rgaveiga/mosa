# Continuous objectives

Sources: `mosa/mosa.py` docstrings for `set_population`, `set_group_params`, `evolve`, `initial_temperature`, `adaptative_mc_step`, and `mc_step_increment`; notebooks under `examples/binh_and_korn`, `chankong_and_haimes`, `fonseca_and_fleming`, `rastrigin`, and `rosenbrock`.

## Separate scalar groups and constraints

Adapted from `examples/binh_and_korn/binh_and_korn.ipynb`. Run where the output filename is unused. The reduced schedule checks integration, not convergence.

```python
from math import inf
from numpy.random import seed
from mosa import Anneal

def fobj(X1, X2):
    if (X1 - 5)**2 + X2**2 > 25 or (X1 - 8)**2 + (X2 + 3)**2 < 7.7:
        return inf, inf
    return 4.0 * (X1**2 + X2**2), (X1 - 5)**2 + (X2 - 5)**2

seed(0)
opt = Anneal()
opt.set_population(X1=(0.0, 5.0), X2=(0.0, 3.0))
opt.restart = False
opt.archive_file = "binh-korn.json"
opt.archive_save_interval = 0
opt.initial_temperature = 1000.0
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.adaptative_mc_step = True
opt.evolve(fobj)
result = opt.copyx()
```

The constrained notebooks return positive infinity for infeasible candidates. Set a positive explicit `initial_temperature` with this pattern: automatic calibration rejects infinite objective values. Check retained solutions are finite and feasible. There is no separate constraint callback in `evolve`.

## One vector group and one objective

Adapted from `examples/rastrigin/rastrigin.ipynb`.

```python
from math import cos, pi
from numpy.random import seed
from mosa import Anneal

def fobj(X):
    return (20.0 + sum(x*x - 10.0*cos(2*pi*x) for x in X),)

seed(0)
opt = Anneal()
opt.set_population(X=(-5.12, 5.12))
opt.set_group_params("X", number_of_elements=2, mc_step_size=1.0,
                     mc_step_increment=0.0001)
opt.restart = False
opt.archive_file = "rastrigin.json"
opt.archive_save_interval = 0
opt.initial_temperature = 10.0
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.evolve(fobj)
result = opt.copyx()
```

`number_of_elements` defaults to one: set it explicitly for vectors. All elements of a continuous group share bounds. Use separate groups for different bounds.

Chankong-Haimes uses a two-element vector with constraints; Fonseca-Fleming uses three elements, two objectives and `mc_step_increment`; Rosenbrock uses three elements and `adaptative_mc_step`. Preserve the notebook's actual objective formula when reproducing its results.
