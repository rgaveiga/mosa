"""Tests for Corana's adaptive continuous step length."""

import importlib

import mosa
import pytest
from numpy import random

from mosa.__error import MOSAError
from mosa.__support import corana_step_length

mosa_module = importlib.import_module("mosa.mosa")


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    ("accepted", "attempted", "expected"),
    [
        (0, 0, 2.0),
        (0, 10, 2.0 / 3.0),
        (4, 10, 2.0),
        (6, 10, 2.0),
        (10, 10, 6.0),
    ],
)
def test_corana_step_length_rule(
    accepted: int, attempted: int, expected: float
) -> None:
    """The documented thresholds apply and an untried step is preserved."""

    assert corana_step_length(2.0, accepted, attempted) == pytest.approx(expected)


def test_corana_flag_defaults_to_false_and_validates_values() -> None:
    optimizer = mosa.Anneal()

    assert optimizer.adaptative_mc_step is False

    optimizer.adaptative_mc_step = True
    assert optimizer.adaptative_mc_step is True

    with pytest.raises(MOSAError):
        optimizer.adaptative_mc_step = 1


def test_corana_adapts_only_continuous_groups(monkeypatch) -> None:
    """Accepted continuous proposals grow their step; discrete steps stay fixed."""

    calls = []

    def record_adjustment(step_length, accepted_moves, attempted_moves):
        calls.append((step_length, accepted_moves, attempted_moves))
        return corana_step_length(step_length, accepted_moves, attempted_moves)

    monkeypatch.setattr(mosa_module, "corana_step_length", record_adjustment)
    optimizer = mosa.Anneal()
    optimizer.adaptative_mc_step = True
    optimizer.set_population(Continuous=(-10.0, 10.0), Discrete=list(range(20)))
    optimizer.set_opt_param("mc_step_size", Continuous=1.0, Discrete=5)
    optimizer.set_opt_param("group_selection_weights", Continuous=1.0, Discrete=0.0)
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 10
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False
    random.seed(7)

    optimizer.evolve(lambda Continuous, Discrete: (0.0,))

    assert calls == [(1.0, 10, 10)]
    assert optimizer.mc_step_size == {"Continuous": 1.0, "Discrete": 5}


def test_disabling_corana_preserves_continuous_step_length(monkeypatch) -> None:
    monkeypatch.setattr(
        mosa_module,
        "corana_step_length",
        lambda *args: pytest.fail("Corana must not run when disabled"),
    )
    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-10.0, 10.0))
    optimizer.mc_step_size = {"X": 1.0}
    optimizer.adaptative_mc_step = False
    optimizer.number_of_temperatures = 2
    optimizer.number_of_iterations = 5
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False
    random.seed(7)

    optimizer.evolve(lambda X: (0.0,))

    assert optimizer.mc_step_size["X"] == 1.0
