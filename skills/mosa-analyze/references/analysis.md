# Archive workflow

Sources: archive-method docstrings in `mosa/mosa.py`, `examples/alloy_optimization/alloy_optimization.ipynb` and `examples/thief_in_the_treasure_room/thief_in_the_treasure_room.ipynb`.

This small fixture uses the thief notebook's `(-value, weight)` convention. It demonstrates analysis without running optimization. Run where `selected.json` is unused.

```python
from mosa import Anneal

opt = Anneal()
a = {"x": [{"Items": [0]}, {"Items": [1]}],
     "f": [[-100.0, 5.0], [-200.0, 10.0]]}
b = {"x": [{"Items": [2]}, {"Items": [3]}],
     "f": [[-150.0, 12.0], [-300.0, 25.0]]}
merged = opt.mergex([a, b])
trimmed = opt.trimx(xset=merged, thresholds=[None, 20.0])
top = opt.reducex(xset=trimmed, index=0, nel=2)
best = opt.bestx(xset=top, weights=[1.0, 0.25])
stats = opt.get_stats(xset=trimmed)
opt.printx(xset=best)
opt.savex(xset=best, archive_file="selected.json")
loaded = Anneal()
loaded.loadx(archive_file="selected.json")
result = loaded.copyx()
```

`get_stats` returns `Min`, `Max`, `Avg`, and `Std`, each a list in objective order. Filtering `-value <= -200` means original value >= 200. Retain stored signs while using MOSA's dominance and ranking operations.

The alloy notebook runs several seeds, takes `copyx()` snapshots, merges, filters property with `thresholds=[-27.0, None]`, and ranks cost/property. Save a merged or filtered result explicitly: `opt.savex(xset=merged, archive_file="merged.json")`. The notebook's save call omits this argument and therefore saves the main archive instead of `merged`.

For two objectives, plot with `opt.plot_front(xset=trimmed, file="front.png", label=["Negative value", "Weight"])`. `file` controls saving; the method creates and shows a figure even without it. For three objectives supply `index3=2`. Indices must be distinct and in range; labels must cover all objectives in original order, even when plotting a subset. For headless execution select Matplotlib's `Agg` backend before plotting. Single-objective runs cannot use a two-axis objective front.

TOPSIS uses normalized objective columns and separate preference weights; it does not reuse the optimizer's `objective_weights`. Check finite values and matching dimensions before ranking or plotting imported archives. Use `copyx(xset)` before editing a returned subset whose solution dictionaries may share references with its input.
