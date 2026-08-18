"""Tests for caching previously evaluated solutions."""

import mosa
import numpy as np
import pytest

from mosa._error import MOSAError


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_solution_cache_properties_and_validation() -> None:
    optimizer = mosa.Anneal()

    assert optimizer.solution_cache is False
    assert optimizer.solution_cache_size == 10000

    optimizer.solution_cache = True
    optimizer.solution_cache_size = 2

    assert optimizer.solution_cache is True
    assert optimizer.solution_cache_size == 2

    with pytest.raises(MOSAError):
        optimizer.solution_cache = 1

    for invalid in (0, -1, True, 1.5):
        with pytest.raises(MOSAError):
            optimizer.solution_cache_size = invalid


def test_cache_and_archive_prevent_repeated_objective_calls() -> None:
    optimizer = mosa.Anneal()
    optimizer.solution_cache = True
    calls = []

    def objective(X):
        calls.append(X)
        return (float(X),)

    evaluate = optimizer._Anneal__evaluate_solution
    update_archive = optimizer._Anneal__updatearchive
    solution = {"X": 1}

    assert evaluate(objective, solution) == [1.0]
    assert evaluate(objective, {"X": 1}) == [1.0]
    assert calls == [1]
    assert optimizer._cache == {"x": [{"X": 1}], "f": [[1.0]]}

    assert update_archive(solution, [1.0]) == 1
    assert optimizer._cache == {"x": [], "f": []}
    assert evaluate(objective, {"X": 1}) == [1.0]
    assert calls == [1]


def test_cache_is_unique_and_respects_its_size() -> None:
    optimizer = mosa.Anneal()
    optimizer.solution_cache = True
    optimizer.solution_cache_size = 2
    evaluate = optimizer._Anneal__evaluate_solution

    objective = lambda X: (float(X),)
    evaluate(objective, {"X": 1})
    evaluate(objective, {"X": 1})
    evaluate(objective, {"X": 2})
    evaluate(objective, {"X": 3})

    assert optimizer._cache == {
        "x": [{"X": 2}, {"X": 3}],
        "f": [[2.0], [3.0]],
    }


def test_cache_deduplicates_unhashable_numpy_categories() -> None:
    optimizer = mosa.Anneal()
    optimizer.solution_cache = True
    calls = []

    def objective(X):
        calls.append(X)
        return (float(np.sum(X)),)

    evaluate = optimizer._Anneal__evaluate_solution
    assert evaluate(objective, {"X": np.array([1, 2])}) == [3.0]
    assert evaluate(objective, {"X": np.array([1, 2])}) == [3.0]

    assert len(calls) == 1
    assert len(optimizer._cache["x"]) == 1


def test_solution_removed_from_archive_moves_to_cache() -> None:
    optimizer = mosa.Anneal()
    optimizer.solution_cache = True
    update_archive = optimizer._Anneal__updatearchive

    assert update_archive({"X": 2}, [2.0]) == 1
    assert update_archive({"X": 1}, [1.0]) == 1

    assert optimizer.archive == {"x": [{"X": 1}], "f": [[1.0]]}
    assert optimizer._cache == {"x": [{"X": 2}], "f": [[2.0]]}


def test_cache_warns_when_a_continuous_group_has_no_increment(capsys) -> None:
    optimizer = mosa.Anneal()
    optimizer.solution_cache = True
    optimizer.set_population(X=(-1.0, 1.0))
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1
    optimizer.maximum_archive_rejections = 100
    optimizer.restart = False

    optimizer.evolve(lambda X: (X * X,))

    output = capsys.readouterr().out
    assert "no Monte Carlo step increment was set" in output
    assert "may prevent effective use of the cache" in output
