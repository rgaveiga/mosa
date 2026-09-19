---
name: mosa-optimize
description: Build and tune Python MOSA (mosa.Anneal) optimizations with continuous, discrete, or mixed variables, constraints, and single or multiple objectives. Use when implementing or debugging an optimization; use mosa-analyze for post-run archive analysis and plots.
license: GPL-3.0
---

# MOSA optimization

Use `from mosa import Anneal`. Requires the MOSA Python package. Match the installed API; these instructions derive from `mosa/mosa.py` and the repository notebooks.

- `set_population(**groups)`: tuples `(low, high)` define continuous bounds; lists define discrete candidates. Names must match objective keyword arguments exactly.
- `setx(x, f=None)`: configure a user-provided initial solution after setting the population. `x` must be a dictionary whose keys exactly match the population groups. Without `f`, the solution is kept in memory and used when no archive is available. With `f` (a float, list, or tuple), `setx()` seeds and persists a new archive, raising `MOSAError` if an in-memory archive or the configured archive file already exists.
- Groups configured with `number_of_elements=1` reach the objective as single values. Multi-element discrete groups arrive as Python lists; multi-element continuous groups arrive as NumPy arrays. Archived solutions remain JSON-compatible and store arrays as lists.
- Return a fixed-length tuple of objectives, including `(value,)` for one objective. All objectives are minimized; negate quantities to maximize.
- Set group options with `set_group_params("X", number_of_elements=3)` or `set_opt_param("number_of_elements", X=3)`. Set global options as properties.
- `restart` defaults to `True`. For a fresh experiment set `restart=False` and choose an unused `archive_file`; `evolve(func)` writes JSON even with `archive_save_interval=0`.
- `evolve` returns `None`; retrieve results with `copyx()`. Never edit `archive` directly.

Read only the relevant reference:

- [Continuous problems](references/continuous.md): scalar/vector objectives, constraints, and Binh-Korn, Chankong-Haimes, Fonseca-Fleming, Rastrigin, and Rosenbrock examples.
- [Discrete and mixed problems](references/discrete.md): permutations, variable-size subsets, alloy composition.
- [Tuning](references/tuning.md): random initial-temperature calibration, objective scales, adaptive moves, caching and restart.

Use small iteration budgets to check argument shapes and feasibility before a full run. Seed `numpy.random` when reproducibility is needed. Notebook schedules are examples, not universal defaults. Report the budget and observed results without claiming a guaranteed optimum.

## Analysis handoff

After optimization and basic run verification, use the `mosa-analyze` skill for all result analysis, including Pareto-front pruning or plotting, merging runs, threshold filtering, statistics, TOPSIS ranking, and selecting solutions for reporting. Pass it the archive returned by `copyx()` or the saved `archive_file`; do not duplicate those analysis workflows in this skill.

When a request covers both optimization and analysis, use this skill for problem formulation, search configuration, and execution, then use `mosa-analyze` for the post-run phase.
