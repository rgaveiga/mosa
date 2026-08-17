"""Tests for adaptive solution-group selection."""

import numpy as np
import pytest

import mosa
from mosa.__error import MOSAError
from mosa.mosa import (
    _adaptive_selection_probabilities,
    _adaptive_selection_reward,
)


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch):
    """Keep optimizer persistence isolated from the working tree."""

    monkeypatch.chdir(tmp_path)


def test_adaptive_selection_property_is_opt_in() -> None:
    optimizer = mosa.Anneal()

    assert optimizer.adaptive_selection is False

    optimizer.adaptive_selection = True
    assert optimizer.adaptive_selection is True

    with pytest.raises(MOSAError, match="must be a boolean"):
        optimizer.adaptive_selection = 1


def test_group_selection_alpha_controls_ema_smoothing() -> None:
    optimizer = mosa.Anneal()

    assert optimizer.group_selection_alpha == pytest.approx(0.2)

    optimizer.group_selection_alpha = 0.75
    assert optimizer.group_selection_alpha == pytest.approx(0.75)

    for invalid_alpha in (-0.1, 1.1, float("inf"), True):
        with pytest.raises(MOSAError, match="between zero and one"):
            optimizer.group_selection_alpha = invalid_alpha


def test_reward_is_mean_of_normalized_objective_variations() -> None:
    maximum_deltas = np.zeros(2, dtype=float)

    first_reward = _adaptive_selection_reward(
        [10.0, 100.0], [8.0, 140.0], maximum_deltas
    )
    second_reward = _adaptive_selection_reward(
        [8.0, 140.0], [7.0, 160.0], maximum_deltas
    )

    assert first_reward == pytest.approx(1.0)
    assert second_reward == pytest.approx(0.5)
    assert maximum_deltas.tolist() == pytest.approx([2.0, 40.0])


def test_boltzmann_probabilities_have_exploration_floor() -> None:
    probabilities = _adaptive_selection_probabilities(
        [1.0, 0.0], temperature=0.001, minimum_probability=0.01
    )

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(0.99)
    assert probabilities[1] == pytest.approx(0.01)


def test_evolve_favors_group_that_changes_the_objective(capsys) -> None:
    np.random.seed(7)
    optimizer = mosa.Anneal()
    optimizer.set_population(Impact=(0.0, 1.0), Inert=(0.0, 1.0))
    optimizer.initial_temperature = 0.1
    optimizer.temperature_decrease_factor = 0.1
    optimizer.number_of_temperatures = 2
    optimizer.number_of_iterations = 200
    optimizer.maximum_archive_rejections = 10_000
    optimizer.archive_save_interval = 0
    optimizer.restart = False
    optimizer.adaptive_selection = True
    optimizer.verbose = True

    optimizer.evolve(lambda Impact, Inert: (0.01 * Impact,))

    probabilities = optimizer.group_selection_weights
    output = capsys.readouterr().out
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["Impact"] > probabilities["Inert"]
    assert probabilities["Inert"] >= 0.01
    assert output.count("    Group selection probabilities:\n") == 2
    assert "        Impact: 0.500000\n        Inert: 0.500000\n" in output
