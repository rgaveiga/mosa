![MOSA](https://raw.githubusercontent.com/rgaveiga/mosa/refs/heads/main/mosa.png)

# Multi-Objective Simulated Annealing (MOSA)

Simulated Annealing (SA) has been initially proposed in 
[*Optimization by Simulated Annealing*](https://doi.org/10.1126/science.220.4598.671) as 
an optimization heuristic. Multi-objective Simulated Annealing (MOSA) extends the original, 
single-objective SA to approximate the Pareto front in multi-objective optimization problems. 

A comprehensive discussion on MOSA and its algorithm variants can be found in 
[*Multi-objective Simulated Annealing: Principles and Algorithm Variants*](https://doi.org/10.1155/2019/8134674).

If you have any questions, corrections, comments or suggestions, just 
[drop a message](mailto:roberto.veiga@ufabc.edu.br).

You can also reach me on [Linkedin](https://www.linkedin.com/in/roberto-gomes-phd-8a718317b/) or 
follow me on [X](https://x.com/rgaveiga). When I have some free time, which is rare, I publish articles 
on [Medium](https://medium.com/@rgaveiga).

If you want to support this and other open source projects that I maintain, become a 
[sponsor on Github](https://github.com/sponsors/rgaveiga).

## Installation

The easiest way to install MOSA is using **pip**:

```
pip install mosa
```

## Documentation

You can access the API documentation for **MOSA** on the 
[project's GitHub Pages site](https://rgaveiga.github.io/mosa).

### Automatic high-temperature calibration

Automatic selection of the high-temperature stage is opt-in:

```python
from mosa import Anneal

anneal = Anneal()
anneal.auto_high_temperature = True
anneal.high_temperature_acceptance_threshold = 0.8
anneal.number_of_temperatures = 20
anneal.number_of_iterations = 1000
anneal.temperature_decrease_factor = 0.9
anneal.evolve(objective_function)
```

The initial objective scale supplies a first trial temperature. MOSA then uses
the proposals already evaluated at that stage to estimate the expected mean
move-acceptance probability; when needed, one higher calibrated stage runs
before geometric quenching. This calibration adds no objective evaluations of
its own.

An explicitly assigned initial temperature always takes precedence, including
an explicit assignment of the numerical default:

```python
anneal.initial_temperature = 100.0
anneal.auto_high_temperature = True  # 100.0 is used; calibration is skipped
```

## Contribution

Contributions are definitely welcome. However, it should be mentioned that this repository uses 
[poetry](https://python-poetry.org/) as a package manager. 

Source code must be formatted using [black](https://github.com/psf/black).

## Disclaimer

The code is provided "as is," with no guarantees regarding the accuracy of its results. The 
author assumes no responsibility for any losses arising from the use of the code. 

Bugs must be reported as issues on the 
[project's GitHub repository](https://github.com/rgaveiga/mosa).
