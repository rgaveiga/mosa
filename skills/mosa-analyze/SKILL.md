---
name: mosa-analyze
description: Analyze Python MOSA solution archives by pruning Pareto fronts, merging runs, filtering thresholds, ranking with TOPSIS, and plotting objectives. Use with MOSA archive JSON or results from mosa.Anneal.
license: GPL-3.0
---

# MOSA archive analysis

Use `from mosa import Anneal`. An archive is `{"x": [solution_dict, ...], "f": [[objective, ...], ...]}` with aligned rows. Objectives are minimized; maximized quantities are usually stored negated.

Load JSON with `opt.loadx(archive_file="run.json")`, then check `opt.sizex()` before analysis. Missing/invalid files can print a message without raising. Use a new `Anneal` when loading an independent file to avoid confusing old state with loaded results.

- `copyx()` makes a deep copy. Do not modify `opt.archive` directly.
- `prune_dominated(xset)` returns a nondominated subset.
- `mergex([a, b])` returns a pruned merge; inputs must have the same objective definitions, order, units and compatible solution groups.
- `trimx(xset, thresholds=[None, limit])` keeps values <= thresholds; `None` skips an objective. Supply one entry per objective. No survivors raises `RuntimeError`.
- `reducex(xset, index=0, nel=5)` selects the smallest values of one objective.
- `bestx(xset, weights=[1.0, 0.25])` returns a one-solution archive using TOPSIS. Use finite nonnegative preferences, one per objective, with positive sum; defaults are equal.

Transforms return archives. Pass the returned `xset` explicitly to subsequent analysis and `savex(xset=..., archive_file=...)`; omitting it uses the main archive.

Read [analysis example](references/analysis.md) for filtering, saving, statistics and plots adapted from the alloy and thief notebooks. Report ranking preferences and restore negated quantities when explaining results. A TOPSIS choice expresses preferences, not a unique universal optimum.
