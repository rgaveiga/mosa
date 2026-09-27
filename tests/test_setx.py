"""Tests for creating an archive from a user-provided initial solution."""

import json

import pytest

from mosa import Anneal
from mosa._error import MOSAError


@pytest.fixture(autouse=True)
def isolate_archive_file(tmp_path, monkeypatch):
    """Keep archive persistence isolated from the working tree."""

    monkeypatch.chdir(tmp_path)


def test_setx_requires_an_empty_archive() -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0))
    optimizer.archive = {"x": [{"X": 0.5}], "f": [[1.0]]}

    with pytest.raises(MOSAError, match="archive already exists"):
        optimizer.setx({"X": 0.25}, 2.0)


def test_setx_rejects_an_existing_archive_file(tmp_path) -> None:
    (tmp_path / "archive.json").write_text("{}", encoding="utf-8")
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0))

    with pytest.raises(MOSAError, match="archive already exists"):
        optimizer.setx({"X": 0.25}, 2.0)


def test_setx_requires_a_population() -> None:
    optimizer = Anneal()

    with pytest.raises(MOSAError, match="population must be provided"):
        optimizer.setx({"X": 0.25}, 2.0)


def test_setx_requires_x_to_be_a_dictionary() -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0))

    with pytest.raises(MOSAError, match="'x' must be a dictionary"):
        optimizer.setx([0.25], 2.0)


def test_setx_requires_exact_population_keys() -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0), Y=[1, 2])

    with pytest.raises(MOSAError, match="Each key in 'x'"):
        optimizer.setx({"X": 0.25, "Z": 1}, 2.0)


@pytest.mark.parametrize("f", [1, "1", {"objective": 1.0}])
def test_setx_rejects_invalid_f_types(f) -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0))

    with pytest.raises(MOSAError, match="'f' must be a tuple, list, or float"):
        optimizer.setx({"X": 0.25}, f)


def test_setx_without_f_stores_initial_solution_without_creating_archive(
    tmp_path,
) -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0), Y=[1, 2])
    solution = {"X": 0.25, "Y": 1}

    optimizer.setx(solution)
    solution["X"] = 0.75

    assert optimizer._inix == {"X": 0.25, "Y": 1}
    assert optimizer.archive == {"x": [], "f": []}
    assert not (tmp_path / "archive.json").exists()


def test_setx_without_f_is_allowed_when_an_archive_exists() -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0))
    optimizer.archive = {"x": [{"X": 0.5}], "f": [[1.0]]}

    optimizer.setx({"X": 0.25})

    assert optimizer._inix == {"X": 0.25}
    assert optimizer.archive == {"x": [{"X": 0.5}], "f": [[1.0]]}


def test_evolve_uses_and_prints_initial_solution_from_setx(capsys) -> None:
    optimizer = Anneal()
    optimizer.set_population(X=[1, 2, 3, 4], Y=[1, 2, 3])
    optimizer.set_group_params("X", number_of_elements=2)
    optimizer.initial_temperature = 1.0
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1
    optimizer.restart = False
    initial_solution = {"X": [1, 3], "Y": 2}
    evaluated = []

    optimizer.setx(initial_solution)

    def objective(X, Y):
        evaluated.append({"X": X, "Y": Y})
        return (float(sum(X) + Y),)

    optimizer.evolve(objective)

    output = capsys.readouterr().out
    assert evaluated[0] == initial_solution
    assert "Starting from the following initial solution:" in output
    assert "    X = [1, 3]\n    Y = 2" in output


def test_evolve_prefers_archive_over_initial_solution_from_setx(capsys) -> None:
    optimizer = Anneal()
    optimizer.set_population(X=[1, 2, 3])
    optimizer.archive = {"x": [{"X": 2}], "f": [[2.0]]}
    optimizer.setx({"X": 1})
    optimizer.initial_temperature = 1.0
    optimizer.number_of_temperatures = 1
    optimizer.number_of_iterations = 1

    optimizer.evolve(lambda X: (float(X),))

    output = capsys.readouterr().out
    assert "Initial solution loaded from the archive..." in output
    assert "Starting from the following initial solution:" not in output


@pytest.mark.parametrize(
    ("f", "expected_f"),
    [
        (2.0, [2.0]),
        ((2.0, 3.0), [2.0, 3.0]),
        ([2.0, 3.0], [2.0, 3.0]),
    ],
)
def test_setx_creates_and_saves_archive(f, expected_f) -> None:
    optimizer = Anneal()
    optimizer.set_population(X=(0.0, 1.0), Y=[1, 2])
    solution = {"X": 0.25, "Y": 1}

    optimizer.setx(solution, f)

    expected = {"x": [solution], "f": [expected_f]}
    assert optimizer.archive == expected
    with open("archive.json", encoding="utf-8") as archive_file:
        assert json.load(archive_file) == expected
