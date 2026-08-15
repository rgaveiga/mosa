"""Tests for discretized continuous MC step increments."""

import importlib

import mosa
import pytest
from numpy import random

from mosa.__error import MOSAError

mosa_module = importlib.import_module("mosa.mosa")


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)


def configured_optimizer() -> mosa.Anneal:
    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-100.0, 100.0))
    optimizer.mc_step_size = {"X": 0.5}
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 3
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False
    return optimizer


def test_mc_step_increment_api_and_validation() -> None:
    optimizer = mosa.Anneal()

    assert optimizer.mc_step_increment == {}

    optimizer.mc_step_increment = {"X": 0.01}
    optimizer.set_group_params("Y", mc_step_increment=0.02)
    optimizer.set_opt_param("mc_step_increment", Z=0.03)

    assert optimizer.mc_step_increment == {"X": 0.01, "Y": 0.02, "Z": 0.03}

    for invalid in (0.0, -0.01, float("inf"), True, "0.01"):
        with pytest.raises(MOSAError):
            optimizer.mc_step_increment = {"Invalid": invalid}

    with pytest.raises(MOSAError):
        optimizer.mc_step_increment = 0.01


def test_default_continuous_change_uses_uniform(monkeypatch) -> None:
    optimizer = configured_optimizer()
    calls = []
    original_uniform = mosa_module.uniform

    def record_uniform(low, high, *args):
        calls.append((low, high))
        return original_uniform(low, high, *args)

    monkeypatch.setattr(mosa_module, "uniform", record_uniform)
    random.seed(7)
    optimizer.evolve(lambda X: (0.0,))

    assert calls.count((-0.5, 0.5)) == 3


def test_exact_mc_step_increment_grid_includes_both_limits(monkeypatch) -> None:
    optimizer = configured_optimizer()
    optimizer.mc_step_increment = {"X": 0.01}
    selected = iter((0, 50, 100))
    monkeypatch.setattr(mosa_module, "choice", lambda size, *args: next(selected))
    seen = []

    def objective(X):
        seen.append(X)
        return (0.0,)

    random.seed(7)
    optimizer.evolve(objective)

    changes = [current - previous for previous, current in zip(seen, seen[1:])]
    assert changes == pytest.approx([-0.5, 0.0, 0.5])


def test_non_divisible_increment_warns_and_excludes_upper_limit(monkeypatch) -> None:
    optimizer = configured_optimizer()
    optimizer.number_of_iterations = 1
    optimizer.set_opt_param("mc_step_increment", X=0.3)
    monkeypatch.setattr(mosa_module, "choice", lambda size, *args: size - 1)
    seen = []

    with pytest.warns(UserWarning, match="upper limit 0.5 will not be included"):
        optimizer.evolve(lambda X: seen.append(X) or (0.0,))

    assert seen[1] - seen[0] == pytest.approx(0.4)


def test_mc_step_increment_is_ignored_for_discrete_groups() -> None:
    optimizer = mosa.Anneal()
    optimizer.set_population(X=[0, 1, 2])
    optimizer.set_opt_param("mc_step_increment", X="ignored")
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 2
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False

    optimizer.evolve(lambda X: (float(X),))

    assert optimizer.mc_step_increment == {"X": "ignored"}


def test_mc_step_increment_disables_corana_for_its_group(monkeypatch) -> None:
    calls = []

    def record_adjustment(step_length, accepted_moves, attempted_moves):
        calls.append((step_length, accepted_moves, attempted_moves))
        return step_length

    monkeypatch.setattr(mosa_module, "corana_step_length", record_adjustment)
    optimizer = mosa.Anneal()
    optimizer.adaptative_mc_step = True
    optimizer.set_population(Incremented=(-10.0, 10.0), Adaptive=(-10.0, 10.0))
    optimizer.set_opt_param("mc_step_size", Incremented=0.5, Adaptive=1.0)
    optimizer.set_opt_param("mc_step_increment", Incremented=0.1)
    optimizer.set_opt_param("group_selection_weights", Incremented=1.0, Adaptive=0.0)
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 5
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False
    random.seed(7)

    optimizer.evolve(lambda Incremented, Adaptive: (0.0,))

    assert calls == [(1.0, 0, 0)]
    assert optimizer.mc_step_size == {"Incremented": 0.5, "Adaptive": 1.0}
