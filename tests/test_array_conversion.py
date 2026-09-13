"""Tests for objective and storage representations of solution groups."""

import json

import mosa
import numpy as np


def _configured_optimizer(archive_file) -> mosa.Anneal:
    optimizer = mosa.Anneal()
    optimizer.set_population(
        Vector=(-1.0, 1.0),
        Scalar=(-1.0, 1.0),
        DiscreteVector=[1, 2, 3],
        DiscreteScalar=["A", "B"],
    )
    optimizer.set_group_params("Vector", number_of_elements=2)
    optimizer.set_group_params("DiscreteVector", number_of_elements=2)
    optimizer.initial_temperature = 1.0
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1
    optimizer.maximum_archive_rejections = 100
    optimizer.archive_file = str(archive_file)
    return optimizer


def test_evolve_passes_only_continuous_vectors_as_numpy_arrays(tmp_path) -> None:
    optimizer = _configured_optimizer(tmp_path / "archive.json")
    optimizer.restart = False
    received = []

    def objective(Vector, Scalar, DiscreteVector, DiscreteScalar):
        received.append((Vector, Scalar, DiscreteVector, DiscreteScalar))
        return (float(np.sum(Vector)),)

    optimizer.evolve(objective)

    assert received
    for vector, scalar, discrete_vector, discrete_scalar in received:
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float64
        assert isinstance(scalar, float)
        assert isinstance(discrete_vector, list)
        assert isinstance(discrete_scalar, str)


def test_archive_and_cache_store_continuous_vectors_as_lists(tmp_path) -> None:
    optimizer = _configured_optimizer(tmp_path / "archive.json")
    optimizer.restart = False
    optimizer.solution_cache = True
    optimizer.evolve(lambda Vector, **kwargs: (float(np.sum(Vector)),))

    evaluate = optimizer._Anneal__evaluate_solution
    evaluate(
        lambda Vector, **kwargs: (float(np.sum(Vector)),),
        {
            "Vector": np.asarray([0.123, 0.456]),
            "Scalar": 0.5,
            "DiscreteVector": [1, 2],
            "DiscreteScalar": "A",
        },
    )

    assert all(
        isinstance(solution["Vector"], list) for solution in optimizer.archive["x"]
    )
    assert optimizer._cache["x"]
    assert all(
        isinstance(solution["Vector"], list) for solution in optimizer._cache["x"]
    )
    json.dumps(optimizer.archive)
    json.dumps(optimizer._cache)


def test_restart_restores_continuous_vectors_for_the_objective(tmp_path) -> None:
    archive_file = tmp_path / "archive.json"
    source = _configured_optimizer(archive_file)
    source.archive = {
        "x": [
            {
                "Vector": [0.25, -0.25],
                "Scalar": 0.5,
                "DiscreteVector": [1, 2],
                "DiscreteScalar": "A",
            }
        ],
        "f": [[0.0]],
    }
    source.savex()

    restored = _configured_optimizer(archive_file)
    received = []

    def objective(Vector, Scalar, DiscreteVector, DiscreteScalar):
        received.append((Vector, Scalar, DiscreteVector, DiscreteScalar))
        return (float(np.sum(Vector)),)

    restored.evolve(objective)

    assert received
    assert all(isinstance(values[0], np.ndarray) for values in received)
    assert all(isinstance(values[1], float) for values in received)
    assert all(isinstance(values[2], list) for values in received)
    assert all(isinstance(values[3], str) for values in received)
    assert all(
        isinstance(solution["Vector"], list) for solution in restored.archive["x"]
    )
