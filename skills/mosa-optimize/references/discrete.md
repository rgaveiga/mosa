# Discrete and mixed objectives

Sources: `mosa/mosa.py` group-parameter docstrings; notebooks under `examples/travelling_salesman`, `thief_in_the_treasure_room`, and `alloy_optimization`.

Discrete inputs must be Python lists (convert arrays with `.tolist()`). A group configured with `number_of_elements=1` passes a single candidate to the objective; larger configured groups pass lists. With `distinct_elements=True`, keep the requested size within the available candidates. For variable-size subsets, start with at least two elements, as in the thief example, to use the list representation.

## Permutations

Adapted from `examples/travelling_salesman/travelling_salesman.ipynb`, with smaller data and schedule. Run where the output filename is unused.

```python
from math import dist
from numpy.random import seed
from mosa import Anneal

cities = {"Airport": (0, 0), "A": (1, 0), "B": (1, 1), "C": (0, 1)}
stops = ["A", "B", "C"]

def fobj(Stops):
    route = ["Airport", *Stops, "Airport"]
    return (sum(dist(cities[a], cities[b]) for a, b in zip(route, route[1:])),)

seed(0)
opt = Anneal()
opt.set_population(Stops=stops)
opt.set_group_params("Stops", number_of_elements=len(stops),
                     distinct_elements=True, change_value_move=0.0, swap_move=1.0)
opt.restart = False
opt.archive_file = "route.json"
opt.archive_save_interval = 0
opt.initial_temperature = 10.0
opt.number_of_temperatures = 3
opt.number_of_iterations = 50
opt.evolve(fobj)
result = opt.copyx()
```

Full permutations need swaps because all candidates are already selected. Leave insertion/deletion disabled and preserve order; sorting would destroy route semantics.

## Variable-size subsets

The thief notebook uses integer item IDs, objective `(-total_value, total_weight)`, and these settings on `Items`:

| Parameter | Notebook value | Meaning |
| --- | --- | --- |
| `number_of_elements` | `5` | Initial subset size |
| `maximum_number_of_elements` | `20` | Cap when inserting |
| `distinct_elements` | `True` | No repeated item |
| `sort_elements` | `True` | Canonical order for an unordered subset |
| `change_value_move` | `0.7` | Replace an item |
| `insert_or_delete_move` | `0.3` | Change subset size |
| `mc_step_size` | `50` | Discrete proposal step |

Move values are relative weights, not required to sum to one. The default replacement weight is `1.0`; swap and insertion/deletion default to zero. Enable at least one usable move. The notebook filters weight afterward with `trimx(thresholds=[None, 20])`; encode a capacity constraint in the objective if it must apply during the search.

## Mixed alloy composition

The alloy notebook sets `Component=component.tolist()` and `Concentration=(0.0, 0.1)`. Configure two distinct components and one concentration. The callback `fobj(Component, Concentration)` unpacks the component list and uses the scalar concentration to compute `(-property, cost)`.

It enables component replacement and swaps, sets the concentration step to `0.05`, and initializes `group_selection_weights` to `Component=1.0, Concentration=4.0`. With `adaptative_selection=True`, these weights change after each temperature. Swapping components matters because their concentration coefficients differ.
