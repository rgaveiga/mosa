# Tuning and persistence

Source: the corresponding property docstrings in `mosa/mosa.py`; alloy, Fonseca-Fleming, Binh-Korn, Rosenbrock and thief notebooks illustrate these options. Preserve the API spelling `adaptative_*`.

| Property | Default | Decision |
| --- | --- | --- |
| `initial_temperature` | `None` | Automatic calibration; explicit positive float overrides it |
| `auto_high_temperature` | `True` | Cannot override an explicitly assigned initial temperature |
| `high_temperature_acceptance_threshold` | `0.8` | Calibration target, strictly between 0 and 1 |
| `temperature_decrease_factor` | `0.9` | Float strictly between 0 and 1 |
| `number_of_temperatures` | `10` | Positive integer schedule length |
| `number_of_iterations` | `1000` | Positive integer iterations per temperature |
| `maximum_archive_rejections` | `1000` | Consecutive archive rejections can end the run early |
| `archive_size` | `1000` | Maximum retained solutions |
| `objective_weights` | `[]` | Equal objective scales unless configured |
| `adaptative_mc_step` | `False` | Corana adaptation only for continuous groups without increments |
| `adaptative_selection` | `False` | Adapt which group receives a move |
| `group_selection_alpha` | `0.2` | EMA factor in [0, 1] for adaptive group selection |
| `solution_cache` | `False` | Consider for expensive, repeatable objective evaluations |
| `solution_cache_size` | `10000` | Positive integer cache limit |
| `track_optimization_progress` | `False` | Record accepted values in `accepted_objective_values` |

`objective_weights` scales objective deltas in acceptance: worsening is divided by the corresponding weight. Use one positive finite scale per objective, e.g. the property/cost ranges in the alloy example. Larger values reduce that objective's influence on rejection. These are separate from TOPSIS preference weights passed to `bestx`.

Continuous `mc_step_size` defaults to range/10 and is clamped to [range/1000, range/2]. `mc_step_increment` must be positive and finite; it discretizes changes, not the absolute coordinates of the initial random solution. It disables Corana adaptation for that group. Discrete step sizes operate on candidate positions, so candidate ordering matters.

`restart=True` uses the last retained solution, loading `archive_file` if needed. Configure the original population and compatible objective/group definitions before resuming. This is not an exact continuation of temperature or random-generator state. For an independent problem use a new `Anneal`, `restart=False`, and a new file path; disabling restart does not prevent output writes.

`archive_save_interval=10` saves after the first completed temperature and at the interval when changed. Zero means save only at the end. `archive_file` defaults to `archive.json`. Keep independent runs in separate paths and capture each with `copyx()`.

For reproducible experiments seed `numpy.random` before each run; randomness in external simulators needs its own control. Keep cached objectives deterministic for a given solution. Progress contains accepted values, not every attempted evaluation, and adds memory cost.
