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
opt.setx({"X1": 1.0, "X2": 1.0})
opt.archive_save_interval = 0
opt.initial_temperature = 1000.0
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.adaptative_mc_step = True
opt.evolve(fobj)
result = opt.copyx()
```

The current notebook uses `setx(x)` without `f` to start from a known solution while leaving objective evaluation to `evolve()`. The supplied point must match the population keys and group shapes.

## One vector group with constraints: Chankong-Haimes

Adapted from `examples/chankong_and_haimes/chankong_and_haimes.ipynb` with a reduced schedule.

```python
from math import inf
from numpy import ndarray, where
from numpy.random import seed
from mosa import Anneal

def fobj(X: ndarray):
    x1, x2 = X
    f1 = 2.0 + (x1 - 2.0)**2 + (x2 - 1.0)**2
    f2 = 9.0 * x1 - (x2 - 1.0)**2
    invalid = (x1**2 + x2**2 > 225.0) | (x1 - 3.0 * x2 + 10.0 > 0.0)
    return where(invalid, inf, f1), where(invalid, inf, f2)

seed(0)
opt = Anneal()
opt.set_population(X=(-20.0, 20.0))
opt.set_group_params("X", number_of_elements=2, mc_step_size=2.0)
opt.restart = False
opt.archive_file = "chankong-haimes.json"
opt.archive_save_interval = 0
opt.initial_temperature = 1000.0
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.evolve(fobj)
result = opt.copyx()
```

The constrained notebooks return positive infinity for infeasible candidates. Set a positive explicit `initial_temperature` with this pattern: automatic initial-temperature sampling skips infinite results, but the first annealing calibration stage still raises if a trial returns infinity. Check retained solutions are finite and feasible. There is no separate constraint callback in `evolve`.

## Three-element vector and two objectives: Fonseca-Fleming

Adapted from `examples/fonseca_and_fleming/fonseca_and_fleming.ipynb` with a reduced schedule.

```python
from math import exp, sqrt
from numpy import ndarray
from numpy.random import seed
from mosa import Anneal

def fobj(X: ndarray):
    y = 1.0 / sqrt(3.0)
    return (
        1.0 - exp(-((X - y) ** 2).sum()),
        1.0 - exp(-((X + y) ** 2).sum()),
    )

seed(0)
opt = Anneal()
opt.set_population(X=(-4.0, 4.0))
opt.set_group_params("X", number_of_elements=3, mc_step_size=2.0,
                     mc_step_increment=0.0002)
opt.restart = False
opt.archive_file = "fonseca-fleming.json"
opt.archive_save_interval = 0
opt.auto_high_temperature = True
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.evolve(fobj)
result = opt.copyx()
```

`mc_step_increment` discretizes proposal changes for the group and prevents Corana step adaptation for that group. Automatic high-temperature calibration is appropriate here because the objectives remain finite over the configured bounds.

## Two-element vector and one objective: Rastrigin

Adapted from `examples/rastrigin/rastrigin.ipynb`.

```python
from math import pi
from numpy import array, cos, ndarray
from numpy.random import seed
from mosa import Anneal

def fobj(X: ndarray):
    f = 20.0 + (X**2).sum() - 10.0 * cos(2.0 * pi * X).sum()
    return f,

seed(0)
opt = Anneal()
opt.set_population(X=(-5.12, 5.12))
opt.setx({"X": array([1.0, 1.0])})
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

The current notebook initializes this run with `setx(x)` and lets `evolve()` calculate the objective. Because `X` is a multi-element continuous group, the initial value is a NumPy array with the configured length.

## Adjacent vector elements and one objective: Rosenbrock

Adapted from `examples/rosenbrock/rosenbrock.ipynb` with a reduced schedule. This preserves the notebook's formula, including the factor of `100` applied to the full summed expression.

```python
from numpy import ndarray
from numpy.random import seed
from mosa import Anneal

def fobj(X: ndarray):
    f = 100.0 * ((X[1:] - X[:-1] ** 2) ** 2 + (1.0 - X[:-1]) ** 2).sum()
    return (f,)

seed(0)
opt = Anneal()
opt.set_population(X=(-100.0, 100.0))
opt.set_group_params("X", number_of_elements=3, mc_step_size=1.0)
opt.restart = False
opt.archive_file = "rosenbrock.json"
opt.archive_save_interval = 0
opt.initial_temperature = 100.0
opt.number_of_temperatures = 5
opt.number_of_iterations = 100
opt.adaptative_mc_step = True
opt.evolve(fobj)
result = opt.copyx()
```

`number_of_elements` defaults to one: set it explicitly for vectors. A multi-element continuous group reaches the objective as a NumPy array, so vectorized NumPy expressions can evaluate it directly. All elements of a continuous group share bounds. Use separate groups for different bounds. These examples stop at `copyx()`; use the `mosa-analyze` skill for plotting or any other archive analysis.
