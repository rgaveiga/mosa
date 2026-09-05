---
name: mosa-optimize
description: Build and tune Python MOSA (mosa.Anneal) optimizations with continuous, discrete, or mixed variables, constraints, and single or multiple objectives. Use when implementing or debugging a MOSA optimization.
license: GPL-3.0
---

# MOSA optimization

Use `from mosa import Anneal`. Requires the MOSA Python package. Match the installed API; these instructions derive from `mosa/mosa.py` and the repository notebooks.

- `set_population(**groups)`: tuples `(low, high)` define continuous bounds; lists define discrete candidates. Names must match objective keyword arguments exactly.
- Groups configured with `number_of_elements=1` reach the objective as single values; groups configured with more elements reach it as lists. This applies to continuous and discrete groups.
- Return a fixed-length tuple of objectives, including `(value,)` for one objective. All objectives are minimized; negate quantities to maximize.
- Set group options with `set_group_params("X", number_of_elements=3)` or `set_opt_param("number_of_elements", X=3)`. Set global options as properties.
- `restart` defaults to `True`. For a fresh experiment set `restart=False` and choose an unused `archive_file`; `evolve(func)` writes JSON even with `archive_save_interval=0`.
- `evolve` returns `None`; retrieve results with `copyx()`. Never edit `archive` directly.

Read only the relevant reference:

- [Continuous problems](references/continuous.md): scalar/vector objectives, constraints, Binh-Korn and Rastrigin examples.
- [Discrete and mixed problems](references/discrete.md): permutations, variable-size subsets, alloy composition.
- [Tuning](references/tuning.md): temperatures, objective scales, adaptive moves, caching and restart.

Use small iteration budgets to check argument shapes and feasibility before a full run. Seed `numpy.random` when reproducibility is needed. Notebook schedules are examples, not universal defaults. Report the budget and observed results without claiming a guaranteed optimum.
