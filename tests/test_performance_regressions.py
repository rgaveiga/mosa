"""Regression tests for performance-oriented archive changes."""

import mosa
import pytest
from numpy import random
from mosa.mosa import MOSAError


def test_archive_removes_solutions_dominated_by_new_entry() -> None:
    """The internal archive remains a Pareto front after every update."""

    optimizer = mosa.Anneal()
    optimizer.archive = {
        "x": [{"X": 1}, {"X": 2}],
        "f": [[1.0, 2.0], [2.0, 1.0]],
    }

    updated = optimizer._Anneal__updatearchive({"X": 0}, [0.0, 0.0])

    assert updated == 1
    assert optimizer.archive == {"x": [{"X": 0}], "f": [[0.0, 0.0]]}


def test_prune_dominated_preserves_input_order_and_values() -> None:
    """Block-wise comparisons return the same ordered public archive shape."""

    optimizer = mosa.Anneal()
    archive = {
        "x": [{"X": 1}, {"X": 2}, {"X": 3}, {"X": 4}],
        "f": [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]],
    }

    result = optimizer.prune_dominated(archive)

    assert result == {
        "x": [{"X": 1}, {"X": 2}, {"X": 3}],
        "f": [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]],
    }
    assert len(archive["x"]) == 4


def test_deferred_persistence_writes_once_at_completion(monkeypatch) -> None:
    """A zero interval skips intermediate writes but never loses the final state."""

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-1.0, 1.0))
    optimizer.number_of_temperatures = 3
    optimizer.number_of_iterations = 2
    optimizer.maximum_archive_rejections = 20
    optimizer.checkpoint_interval = 0
    optimizer.archive_save_interval = 0
    optimizer.restart = False
    writes = {"checkpoint": 0, "archive": 0}

    def record_checkpoint(*args) -> None:
        writes["checkpoint"] += 1

    def record_archive(*args, **kwargs) -> None:
        writes["archive"] += 1

    monkeypatch.setattr(optimizer, "_Anneal__savecheckpoint", record_checkpoint)
    monkeypatch.setattr(optimizer, "savex", record_archive)
    random.seed(42)

    optimizer.evolve(lambda X: (0.0,))

    assert writes == {"checkpoint": 1, "archive": 1}


def test_persistence_intervals_validate_values() -> None:
    """Persistence cadence is explicit and rejects invalid values."""

    optimizer = mosa.Anneal()

    assert optimizer.checkpoint_interval == 1
    assert optimizer.archive_save_interval == 1

    optimizer.checkpoint_interval = 0
    optimizer.archive_save_interval = 5

    assert optimizer.checkpoint_interval == 0
    assert optimizer.archive_save_interval == 5

    with pytest.raises(MOSAError):
        optimizer.checkpoint_interval = -1

    with pytest.raises(MOSAError):
        optimizer.archive_save_interval = 1.5
