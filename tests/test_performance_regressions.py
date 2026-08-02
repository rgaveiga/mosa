"""Regression tests for performance-oriented archive changes."""

import json
from pathlib import Path

import mosa
import numpy as np
import pytest
from numpy import random
from mosa.mosa import (
    MOSAError,
    _GroupState,
    _non_dominated_mask_kernel,
)


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch) -> None:
    """Keep optimizer persistence isolated from the working tree."""

    monkeypatch.chdir(tmp_path)


def test_internal_groups_use_numeric_numpy_arrays() -> None:
    """Continuous and discrete numeric groups avoid Python-object arrays."""

    continuous = _GroupState.create((-1.0, 1.0), number_of_elements=2)
    integers = _GroupState.create([1, 2, 3], number_of_elements=2)
    floating = _GroupState.create([1, 2.5, 3], number_of_elements=2)

    assert continuous.population.dtype == np.float64
    assert continuous.solution.dtype == np.float64
    assert integers.population.dtype == np.int64
    assert floating.population.dtype == np.float64
    assert not continuous.categorical
    assert not integers.categorical


def test_categorical_positions_have_distinct_codes_and_semantic_equality() -> None:
    """Equal category values remain separate population positions."""

    first = ["A", "B"]
    second = ["A", "B"]
    state = _GroupState.create([first, second, ["C", "D"]], number_of_elements=2)

    assert state.population.tolist() == [0, 1, 2]
    assert len(set(state.population.tolist())) == 3
    assert state.equal(state.population[0], state.population[1])
    assert not state.equal(state.population[0], state.population[2])
    assert state.decode_value(state.population[0]) is first
    assert state.decode_value(state.population[1]) is second


def test_evolve_decodes_categorical_arrays_at_objective_boundary() -> None:
    """The public objective contract still receives original category objects."""

    optimizer = mosa.Anneal()
    population = [["A", "B"], ["A", "B"], ["C", "D"]]
    observed: list[list[str]] = []

    def objective(Item: list[str]) -> tuple[float]:
        observed.append(Item)
        return (0.0,)

    optimizer.set_population(Item=population)
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 3
    optimizer.maximum_archive_rejections = 10
    optimizer.restart = False
    random.seed(7)
    optimizer.evolve(objective)

    state = optimizer._group_states["Item"]
    assert state.population.dtype == np.int64
    assert state.categorical
    assert observed
    assert all(item in population for item in observed)


def test_numba_pruning_kernel_compiles_in_nopython_mode() -> None:
    """Pareto pruning executes through a native Numba specialization."""

    archive = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    keep = _non_dominated_mask_kernel(np.vstack((archive, np.asarray([[3.0, 3.0]]))))

    assert keep.tolist() == [True, True, True, False]
    assert _non_dominated_mask_kernel.nopython_signatures


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
    """A zero interval skips intermediate archive writes but saves the final state."""

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-1.0, 1.0))
    optimizer.number_of_temperatures = 3
    optimizer.number_of_iterations = 2
    optimizer.maximum_archive_rejections = 20
    optimizer.archive_save_interval = 0
    optimizer.restart = False
    writes = {"archive": 0}

    def record_archive(*args, **kwargs) -> None:
        writes["archive"] += 1

    monkeypatch.setattr(optimizer, "savex", record_archive)
    random.seed(42)

    optimizer.evolve(lambda X: (0.0,))

    assert writes == {"archive": 1}
    assert not Path("checkpoint.json").exists()


def test_archive_interval_validates_values() -> None:
    """Archive persistence cadence is explicit and rejects invalid values."""

    optimizer = mosa.Anneal()

    assert optimizer.archive_save_interval == 10

    optimizer.archive_save_interval = 5

    assert optimizer.archive_save_interval == 5

    with pytest.raises(MOSAError):
        optimizer.archive_save_interval = 1.5


def test_default_archive_cadence_saves_first_periodic_and_final(monkeypatch) -> None:
    """The dirty archive is persisted at 1, 10, 20, and final temperature."""

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-1.0, 1.0))
    optimizer.number_of_temperatures = 25
    optimizer.number_of_iterations = 1
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False
    writes = []
    objective_calls = 0

    def objective(X):
        nonlocal objective_calls
        objective_calls += 1
        return (-float(objective_calls),)

    monkeypatch.setattr(optimizer, "savex", lambda *args, **kwargs: writes.append(1))
    random.seed(42)

    optimizer.evolve(objective)

    assert len(writes) == 4
    assert not Path("checkpoint.json").exists()


def test_restart_uses_last_archive_solution_and_rebuilds_distinct_pool() -> None:
    """Archive values are canonicalized and subtracted from the original pool."""

    first_a = ["A"]
    second_a = ["A"]
    value_b = ["B"]
    optimizer = mosa.Anneal()
    optimizer.set_population(Item=[first_a, second_a, value_b])
    optimizer.set_group_params("Item", number_of_elements=2, distinct_elements=True)
    optimizer.archive = {
        "x": [{"Item": [["B"], ["A"]]}, {"Item": [["A"], ["B"]]}],
        "f": [[2.0], [1.0]],
    }
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1

    class StopEvolution(Exception):
        pass

    with pytest.raises(StopEvolution):
        optimizer.evolve(lambda Item: (_ for _ in ()).throw(StopEvolution()))

    state = optimizer._group_states["Item"]
    solution = state.decode_solution()
    remaining = state.decode_population()

    assert solution[0] is first_a
    assert solution[1] is value_b
    assert remaining == [second_a]
    assert remaining[0] is second_a


def test_restart_rejects_archive_values_outside_configured_population() -> None:
    """An archive from another optimization problem is not accepted silently."""

    optimizer = mosa.Anneal()
    optimizer.set_population(Item=["A", "B"])
    optimizer.archive = {"x": [{"Item": "Z"}], "f": [[0.0]]}
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1

    with pytest.raises(MOSAError, match="incompatible"):
        optimizer.evolve(lambda Item: (0.0,))


def test_restart_reconstructs_a_variable_length_solution() -> None:
    """The current pool is derived from archive length, not initial group length."""

    optimizer = mosa.Anneal()
    optimizer.set_population(Items=list(range(6)))
    optimizer.set_group_params(
        "Items",
        number_of_elements=4,
        maximum_number_of_elements=6,
        distinct_elements=True,
        change_value_move=0.7,
        insert_or_delete_move=0.3,
    )
    optimizer.archive = {"x": [{"Items": [2]}], "f": [[2.0, 1.0]]}
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1

    class StopEvolution(Exception):
        pass

    with pytest.raises(StopEvolution):
        optimizer.evolve(lambda Items: (_ for _ in ()).throw(StopEvolution()))

    state = optimizer._group_states["Items"]
    assert state.decode_solution() == [2]
    assert state.decode_population() == [0, 1, 3, 4, 5]


def test_early_termination_persists_a_dirty_archive(monkeypatch) -> None:
    """A recoverable snapshot is written before the rejection-limit return."""

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-1.0, 1.0))
    optimizer.number_of_temperatures = 100
    optimizer.number_of_iterations = 100
    optimizer.maximum_archive_rejections = 1
    optimizer.restart = False
    writes = []
    monkeypatch.setattr(optimizer, "savex", lambda *args, **kwargs: writes.append(1))
    random.seed(1)

    optimizer.evolve(lambda X: (0.0,))

    assert len(writes) == 1


def test_atomic_archive_load_falls_back_to_previous_generation() -> None:
    """A truncated primary archive is recovered from its complete backup."""

    archive_file = Path("archive.json")
    first = {"x": [{"X": 1}], "f": [[1.0]]}
    second = {"x": [{"X": 2}], "f": [[2.0]]}
    optimizer = mosa.Anneal()
    optimizer.archive_file = str(archive_file)
    optimizer.archive = first
    optimizer.savex()
    optimizer.archive = second
    optimizer.savex()

    archive_file.write_text("{", encoding="utf-8")
    restored = mosa.Anneal()
    restored.loadx(str(archive_file))

    assert restored.archive == first
    assert not Path(f"{archive_file}.tmp").exists()


def test_restart_loads_the_last_solution_from_a_persisted_archive() -> None:
    """The complete file-loading and warm-start path uses the final x/f pair."""

    archive_file = Path("archive.json")
    source = mosa.Anneal()
    source.archive = {
        "x": [{"X": 0.75}, {"X": 0.25}],
        "f": [[0.5625], [0.0625]],
    }
    source.savex(archive_file=str(archive_file))
    restored = mosa.Anneal()
    restored.set_population(X=(-1.0, 1.0))
    restored.archive_file = str(archive_file)
    restored.number_of_temperatures = 1
    restored.number_of_iterations = 1

    class StopEvolution(Exception):
        pass

    with pytest.raises(StopEvolution):
        restored.evolve(lambda X: (_ for _ in ()).throw(StopEvolution()))

    assert restored._group_states["X"].decode_solution() == pytest.approx(0.25)


def test_legacy_checkpoint_is_read_but_never_rewritten() -> None:
    """Existing checkpoints support migration without creating new generations."""

    checkpoint = Path("checkpoint.json")
    legacy = {
        "x": {"X": 0.25},
        "f": [0.0625],
        "Population": {"X": [-1.0, 1.0]},
        "SampleSpace": {"X": 1},
    }
    original = json.dumps(legacy)
    checkpoint.write_text(original, encoding="utf-8")
    optimizer = mosa.Anneal()
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1
    optimizer.archive_save_interval = 0
    random.seed(2)

    optimizer.evolve(lambda X: (X * X,))

    assert checkpoint.read_text(encoding="utf-8") == original
