"""Tests for automatic high-temperature calibration."""

from math import exp, inf

import pytest
from numpy import random

from mosa import Anneal
from mosa._error import MOSAError


@pytest.fixture(autouse=True)
def isolate_optimizer_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_automatic_temperature_api_defaults_and_validation() -> None:
    anneal = Anneal()

    assert anneal.auto_high_temperature is True
    assert anneal.high_temperature_acceptance_threshold == pytest.approx(0.8)
    assert anneal.initial_temperature is None
    assert anneal._initempset is False

    anneal.auto_high_temperature = False
    assert anneal.auto_high_temperature is True

    anneal.initial_temperature = 1.0
    assert anneal._initempset is True

    for value in (0, 1, "yes", None):
        with pytest.raises(MOSAError):
            anneal.auto_high_temperature = value

    for value in (True, 0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(MOSAError):
            anneal.high_temperature_acceptance_threshold = value


@pytest.mark.parametrize(
    ("objectives", "expected"),
    [
        ([0.2], 1.0),
        ([0.5], 10.0),
        ([0.8], 10.0),
        ([1.0], 10.0),
        ([1.5], 10.0),
        ([5.0], 10.0),
        ([10.0], 100.0),
        ([15.0], 100.0),
        ([60.0], 100.0),
        ([0.5, 0.7], 10.0),
        ([0.0, 1.0], 10.0),
        ([50.0, 70.0], 100.0),
        ([-50.0, 70.0], 100.0),
        ([0.0, 0.0], 1.0),
    ],
)
def test_bootstrap_temperature(objectives, expected) -> None:
    anneal = Anneal()
    estimate = anneal._Anneal__estimate_initial_temperature
    assert estimate(objectives) == pytest.approx(expected)


@pytest.mark.parametrize("objectives", [[], [float("nan")], [float("inf")], ["x"]])
def test_bootstrap_temperature_rejects_invalid_objectives(objectives) -> None:
    anneal = Anneal()
    with pytest.raises(MOSAError):
        anneal._Anneal__estimate_initial_temperature(objectives)


def test_reduced_deltas_and_acceptance_match_mosa_equation() -> None:
    anneal = Anneal()
    reduced = anneal._Anneal__reduced_objective_deltas
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    assert reduced([1.0, 10.0], [0.5, 14.0], [1.0, 2.0]) == [0.0, 2.0]
    assert reduced([1.0], [1.0], [1.0]) == [0.0]
    assert probability([2.0], 4.0) == pytest.approx(exp(-0.5))
    assert probability([0.0], 4.0) == pytest.approx(1.0)
    assert probability([0.0, 1.0], 2.0) == pytest.approx(exp(-0.5))

    anneal.alpha = 0.25
    p1 = exp(-0.5)
    p2 = exp(-1.0)
    expected = 0.75 * p1 * p2 + 0.25 * max(p1, p2)
    assert probability([1.0, 2.0], 2.0) == pytest.approx(expected)


@pytest.mark.parametrize("penalty", [inf, float("nan")])
def test_non_finite_trial_objective_has_zero_acceptance(penalty) -> None:
    anneal = Anneal()
    reduced = anneal._Anneal__reduced_objective_deltas
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    penalized_trial = reduced([1.0, 2.0], [penalty, 1.0], [1.0, 1.0])
    assert penalized_trial == [inf, 0.0]
    assert probability(penalized_trial, 1000.0) == 0.0
    assert probability([penalty], 1000.0) == 0.0


def test_negative_infinity_trial_is_a_maximal_improvement() -> None:
    anneal = Anneal()
    anneal.alpha = 0.25
    reduced = anneal._Anneal__reduced_objective_deltas
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    trial = reduced([1.0, 2.0], [-inf, 4.0], [1.0, 1.0])
    second_probability = exp(-2.0)

    assert trial == [0.0, 2.0]
    assert probability(trial, 1.0) == pytest.approx(0.75 * second_probability + 0.25)
    assert probability(reduced([1.0], [-inf], [1.0]), 1.0) == 1.0


@pytest.mark.parametrize("penalty", [inf, float("nan")])
def test_finite_trial_can_escape_non_finite_current_objective(penalty) -> None:
    anneal = Anneal()
    reduced = anneal._Anneal__reduced_objective_deltas
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    finite_trial = reduced([penalty], [1.0], [1.0])
    assert finite_trial == [0.0]
    assert probability(finite_trial, 1000.0) == 1.0


def test_finite_trial_cannot_worsen_negative_infinity_current_objective() -> None:
    anneal = Anneal()
    reduced = anneal._Anneal__reduced_objective_deltas
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    finite_trial = reduced([-inf], [1.0], [1.0])

    assert finite_trial == [inf]
    assert probability(finite_trial, 1000.0) == 0.0


def test_pmax_is_local_to_each_proposal() -> None:
    anneal = Anneal()
    anneal.alpha = 0.5
    probability = anneal._Anneal__acceptance_probability_from_reduced_delta

    second_alone = probability([4.0, 5.0], 1.0)
    probability([0.0, 10.0], 1.0)
    second_after_first = probability([4.0, 5.0], 1.0)

    assert second_after_first == pytest.approx(second_alone)


def test_expected_acceptance_is_monotonic_and_high_estimate_meets_target() -> None:
    anneal = Anneal()
    samples = [[0.0, 0.0], [2.0], [0.0, 4.0], [1.0, 3.0]]
    expected = anneal._Anneal__expected_acceptance
    estimate = anneal._Anneal__estimate_high_temperature

    assert expected(samples, 2.0) >= expected(samples, 1.0)
    target = 0.9
    high = estimate(0.1, samples, target)
    assert high >= 0.1
    assert expected(samples, high) >= target
    assert expected(samples, high * (1.0 - 2e-6)) < target


def _configured_anneal() -> Anneal:
    anneal = Anneal()
    anneal.set_population(X=(0.0, 1.0))
    anneal.set_group_params("X", mc_step_size=0.25)
    anneal.restart = False
    anneal.number_of_iterations = 4
    anneal.number_of_temperatures = 3
    anneal.temperature_decrease_factor = 0.5
    anneal.maximum_archive_rejections = 100
    return anneal


def test_automatic_calibration_is_attempted_by_default(monkeypatch) -> None:
    random.seed(5)
    anneal = _configured_anneal()

    def calibrated_temperature(_objective_values):
        return 3.0

    monkeypatch.setattr(
        anneal,
        "_Anneal__estimate_initial_temperature",
        calibrated_temperature,
    )
    anneal.evolve(lambda X: (X,))

    assert anneal._temp == pytest.approx([3.0, 1.5, 0.75])


@pytest.mark.parametrize("penalty", [inf, -inf, float("nan")])
def test_non_finite_constraint_penalty_does_not_abort_evolution(penalty) -> None:
    random.seed(17)
    anneal = _configured_anneal()
    anneal.initial_temperature = 1.0
    evaluations = 0

    def objective(X):
        nonlocal evaluations
        evaluations += 1
        return (0.0,) if evaluations == 1 else (penalty,)

    anneal.evolve(objective)

    assert evaluations > 1


@pytest.mark.parametrize("penalty", [inf, -inf])
def test_automatic_calibration_rejects_infinite_trial_objectives(penalty) -> None:
    random.seed(17)
    anneal = _configured_anneal()
    evaluations = 0

    def objective(X):
        nonlocal evaluations
        evaluations += 1
        return (0.0,) if evaluations == 1 else (penalty,)

    with pytest.raises(MOSAError, match="no valid objective values"):
        anneal.evolve(objective)


@pytest.mark.parametrize("penalty", [inf, -inf])
def test_automatic_calibration_rejects_infinite_initial_objectives(penalty) -> None:
    random.seed(17)
    anneal = _configured_anneal()

    with pytest.raises(MOSAError, match="no valid objective values"):
        anneal.evolve(lambda X: (penalty,))


def test_automatic_schedule_heats_when_t0_is_insufficient() -> None:
    random.seed(7)
    anneal = _configured_anneal()
    anneal.auto_high_temperature = True
    initial_x = None

    def objective(X):
        nonlocal initial_x
        if initial_x is None:
            initial_x = X
            return (0.0,)
        return (0.0 if X == initial_x else 1000.0,)

    anneal.evolve(objective)

    assert len(anneal._temp) == 3
    assert anneal._temp[0] == pytest.approx(1.0)
    assert anneal._temp[1] > anneal._temp[0]
    assert anneal._temp[2] == pytest.approx(anneal._temp[1] * 0.5)
    assert anneal.initial_temperature == pytest.approx(1.0)
    assert anneal._initempset is False


def test_sufficient_t0_and_explicit_temperature_schedules(capsys) -> None:
    random.seed(11)
    automatic = _configured_anneal()
    automatic.auto_high_temperature = True
    automatic.evolve(lambda X: (0.0,))
    assert automatic._temp == pytest.approx([1.0, 0.5, 0.25])

    random.seed(11)
    explicit = _configured_anneal()
    explicit.initial_temperature = 7.5
    explicit.auto_high_temperature = True
    explicit.verbose = True
    explicit.evolve(lambda X: (0.0,))
    assert explicit._temp == pytest.approx([7.5, 3.75, 1.875])
    assert "Explicit initial temperature provided" not in capsys.readouterr().out


def test_verbose_automatic_calibration_has_concise_status(capsys) -> None:
    random.seed(11)
    anneal = _configured_anneal()
    anneal.auto_high_temperature = True
    anneal.verbose = True

    anneal.evolve(lambda X: (0.0,))

    output = capsys.readouterr().out
    assert (
        "Estimating initial high-temperature...\nDone!\n------\n" "TEMPERATURE:"
    ) in output
    for unnecessary_message in (
        "Initial objective scale:",
        "Initial calibration temperature:",
        "Expected acceptance at initial temperature:",
        "Observed acceptance at initial temperature:",
        "Target high-temperature acceptance:",
        "Initial calibration temperature satisfies the target acceptance.",
        "Starting geometric quench",
    ):
        assert unnecessary_message not in output


def test_one_temperature_does_not_add_a_heating_stage() -> None:
    random.seed(13)
    anneal = _configured_anneal()
    anneal.number_of_temperatures = 1
    anneal.auto_high_temperature = True
    anneal.evolve(lambda X: (1000.0 * X,))
    assert len(anneal._temp) == 1


def test_high_temperature_run_count_api() -> None:
    anneal = Anneal()
    assert anneal.number_of_runs_high_temperature_estimate == 100
    anneal.number_of_runs_high_temperature_estimate = 3
    assert anneal.number_of_runs_high_temperature_estimate == 3

    for value in (True, False, 0, -1, 1.0, "2", None):
        with pytest.raises(MOSAError):
            anneal.number_of_runs_high_temperature_estimate = value


def test_random_estimate_uses_successive_valid_differences_and_archive(
    monkeypatch,
) -> None:
    random.seed(23)
    anneal = Anneal()
    anneal.number_of_runs_high_temperature_estimate = 5
    results = iter(
        [(10.0, 5.0), (float("nan"), 0.0), (3.0, 8.0), (2.0, 6.0), (float("inf"), 0.0)]
    )
    solutions = []
    observed_differences = []

    def objective(X):
        solutions.append(X)
        return next(results)

    def estimate(differences):
        observed_differences.append(differences)
        return 10.0

    monkeypatch.setattr(anneal, "_Anneal__estimate_initial_temperature", estimate)
    temperature = anneal._Anneal__estimate_initial_high_temperature(
        objective,
        {"X": [1, 2, 3, 4, 5]},
        {"X": 1},
        {"X": False},
        {"X": False},
    )

    assert temperature == 10.0
    assert len(solutions) == 5
    assert all(value in (1, 2, 3, 4, 5) for value in solutions)
    assert len(set(solutions)) > 1
    assert observed_differences == [[4, 3]]
    assert anneal.archive["f"] == [[10.0, 5.0], [2.0, 6.0]]
    assert anneal.archive["x"] == [{"X": solutions[0]}, {"X": solutions[3]}]


def test_random_estimate_does_not_cancel_opposite_differences(monkeypatch) -> None:
    anneal = Anneal()
    anneal.number_of_runs_high_temperature_estimate = 3
    results = iter([(10.0,), (3.0,), (10.0,)])
    observed_differences = []

    def objective(X):
        return next(results)

    def estimate(differences):
        observed_differences.append(differences)
        return 10.0

    monkeypatch.setattr(anneal, "_Anneal__estimate_initial_temperature", estimate)
    temperature = anneal._Anneal__estimate_initial_high_temperature(
        objective,
        {"X": [1, 2, 3]},
        {"X": 1},
        {"X": False},
        {"X": False},
    )

    assert temperature == 10.0
    assert observed_differences == [[7]]


def test_invalid_initial_result_uses_valid_random_solution() -> None:
    random.seed(19)
    anneal = _configured_anneal()
    anneal.number_of_runs_high_temperature_estimate = 2
    anneal.number_of_iterations = 1
    anneal.number_of_temperatures = 1
    calls = 0

    def objective(X):
        nonlocal calls
        calls += 1
        return (float("nan"),) if calls == 1 else (float(calls),)

    anneal.evolve(objective)

    assert calls >= 3
    assert anneal.initial_temperature == pytest.approx(10.0)
    assert all(value[0] == value[0] for value in anneal.archive["f"])


def test_explicit_initial_temperature_skips_random_estimate() -> None:
    anneal = _configured_anneal()
    anneal.initial_temperature = 2.0
    anneal.number_of_runs_high_temperature_estimate = 5
    anneal.number_of_iterations = 1
    anneal.number_of_temperatures = 1
    calls = 0

    def objective(X):
        nonlocal calls
        calls += 1
        return (X,)

    anneal.evolve(objective)

    assert calls == 2
    assert anneal.initial_temperature == 2.0


def test_random_estimate_samples_all_groups_and_preserves_shapes() -> None:
    random.seed(31)
    anneal = Anneal()
    anneal.set_population(X=(0.0, 1.0), Y=["a", "b", "c", "d"])
    anneal.set_group_params("X", number_of_elements=2)
    anneal.set_group_params("Y", number_of_elements=2, distinct_elements=True)
    anneal.restart = False
    anneal.number_of_temperatures = 1
    anneal.number_of_iterations = 1
    anneal.number_of_runs_high_temperature_estimate = 4
    observed = []

    def objective(X, Y):
        assert X.shape == (2,)
        assert isinstance(Y, list) and len(Y) == len(set(Y)) == 2
        assert all(0.0 <= value <= 1.0 for value in X)
        observed.append((tuple(X), tuple(Y)))
        return (float(sum(X)),)

    anneal.evolve(objective)

    assert len(observed) >= 5
    assert len({sample[0] for sample in observed[:5]}) > 1
    assert len({sample[1] for sample in observed[:5]}) > 1
    assert all(isinstance(item["X"], list) for item in anneal.archive["x"])
