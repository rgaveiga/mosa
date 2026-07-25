"""Regression tests that reproduce the example notebook optimizations."""

import io
from math import cos, pi, sqrt

import pytest
from numpy import arange, asarray, random

import mosa
import mosa.mosa as mosa_module


@pytest.fixture(autouse=True)
def prevent_optimizer_files(monkeypatch):
    """Keep checkpoints and archives in memory during the tests."""

    monkeypatch.setattr(
        mosa_module,
        "open",
        lambda *args, **kwargs: io.StringIO(),
        raising=False,
    )


def test_alloy_optimization_topsis_result() -> None:
    """The alloy notebook's final TOPSIS selection remains stable."""

    random.seed(0)
    component = asarray(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    k = arange(component.shape[0]) + 1
    x = 0.5 * k + 12.6 + random.uniform(0.0, 5.0, component.shape[0])
    cost = random.uniform(5.0, 500.0, component.shape[0])

    def fobj(Component: list, Concentration: float) -> tuple:
        first, second = Component
        x1 = float(x[component == first][0])
        cost1 = float(cost[component == first][0])
        x2 = float(x[component == second][0])
        cost2 = float(cost[component == second][0])

        concentration_x = x1 * (1.0 - Concentration) + x2 * Concentration
        concentration_cost = cost1 * (1.0 - Concentration) + cost2 * Concentration

        return -concentration_x, concentration_cost

    optimizer = mosa.Anneal()
    optimizer.set_population(Component=component.tolist(), Concentration=(0.0, 0.1))
    optimizer.initial_temperature = 1.0
    optimizer.number_of_temperatures = 100
    optimizer.number_of_iterations = 200
    optimizer.objective_weights = [
        x.max() - x.min(),
        cost.max() - cost.min(),
    ]
    optimizer.archive_size = 1000
    optimizer.maximum_archive_rejections = 1000
    optimizer.set_opt_param("number_of_elements", Component=2, Concentration=1)
    optimizer.set_opt_param("group_selection_weights", Component=1.0, Concentration=4.0)
    optimizer.set_opt_param("distinct_elements", Component=True)
    optimizer.set_opt_param("change_value_move", Component=1.0, Concentration=1.0)
    optimizer.set_opt_param("swap_move", Component=1.0)
    optimizer.set_opt_param("mc_step_size", Concentration=0.05)
    optimizer.restart = False

    archives = []
    for seed in (1, 2, 3):
        random.seed(seed)
        optimizer.evolve(fobj)
        archives.append(optimizer.copyx())

    merged = optimizer.mergex(archives)
    pruned = optimizer.prune_dominated(xset=merged)
    trimmed = optimizer.trimx(xset=pruned, thresholds=(-27.0, None))
    result = optimizer.bestx(xset=trimmed)

    assert result["x"][0]["Component"] == ["V", "R"]
    assert result["x"][0]["Concentration"] == pytest.approx(0.09997473981079008)
    assert result["f"][0] == pytest.approx([-27.412569755936158, 65.41868547183883])


def test_rastrigin_last_solution() -> None:
    """The final Rastrigin solution has the notebook's x and f values."""

    random.seed(0)

    def fobj(X):
        x1, x2 = X
        f = 20.0 + x1**2 - 10.0 * cos(2 * pi * x1)
        f += x2**2 - 10.0 * cos(2 * pi * x2)
        return (f,)

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-5.12, 5.12))
    optimizer.archive_size = 100
    optimizer.maximum_archive_rejections = 10000
    optimizer.initial_temperature = 10.0
    optimizer.number_of_iterations = 100
    optimizer.number_of_temperatures = 100
    optimizer.temperature_decrease_factor = 0.9
    optimizer.set_group_params("X", number_of_elements=2, mc_step_size=1.0)
    optimizer.restart = False
    optimizer.evolve(fobj)

    result = optimizer.prune_dominated()

    assert result["x"][0]["X"] == pytest.approx(
        [0.0013720747713694692, -0.0002757008832727781]
    )
    assert result["f"][0] == pytest.approx([0.00038856846911095033])


def test_rosenbrock_last_solution() -> None:
    """The final Rosenbrock solution has the notebook's x and f values."""

    random.seed(0)

    def fobj(X: list) -> tuple:
        f = 0
        for i in range(2):
            f += 100 * ((X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2)
        return (f,)

    optimizer = mosa.Anneal()
    optimizer.set_population(X=(-100, 100))
    optimizer.archive_size = 100
    optimizer.maximum_archive_rejections = 10000
    optimizer.initial_temperature = 100.0
    optimizer.number_of_iterations = 100
    optimizer.number_of_temperatures = 1000
    optimizer.temperature_decrease_factor = 0.9
    optimizer.set_group_params("X", number_of_elements=3, mc_step_size=1.0)
    optimizer.restart = False
    optimizer.evolve(fobj)

    result = optimizer.prune_dominated()

    assert result["x"][0]["X"] == pytest.approx(
        [1.0010230081413158, 1.002265827636082, 1.0044758281864978]
    )
    assert result["f"][0] == pytest.approx([0.0006232094825663027])


def test_thief_in_the_treasure_room_topsis_result() -> None:
    """The treasure-room notebook's final TOPSIS selection remains stable."""

    random.seed(0)
    weight = random.uniform(0.5, 5.0, 1000)
    weight.sort()
    value = weight * random.uniform(100.0, 300.0, 1000)

    def fobj(Items):
        return -sum(value[Items]), sum(weight[Items])

    optimizer = mosa.Anneal()
    optimizer.set_population(Items=list(range(1000)))
    optimizer.initial_temperature = 1.0
    optimizer.number_of_temperatures = 100
    optimizer.number_of_iterations = 2000
    optimizer.objective_weights = [
        value.max() - value.min(),
        weight.max() - weight.min(),
    ]
    optimizer.archive_size = 1000
    optimizer.maximum_archive_rejections = 1000
    optimizer.set_opt_param("distinct_elements", Items=True)
    optimizer.set_opt_param("sort_elements", Items=True)
    optimizer.set_opt_param("mc_step_size", Items=50)
    optimizer.set_opt_param("change_value_move", Items=0.7)
    optimizer.set_opt_param("insert_or_delete_move", Items=0.3)
    optimizer.set_opt_param("number_of_elements", Items=5)
    optimizer.set_opt_param("maximum_number_of_elements", Items=20)
    optimizer.restart = False
    optimizer.evolve(fobj)

    pruned = optimizer.prune_dominated()
    trimmed = optimizer.trimx(xset=pruned, thresholds=(None, 20))
    result = optimizer.bestx(xset=trimmed, weights=(1.0, 0.25))

    assert result["x"][0]["Items"] == [248, 405, 412, 551, 552, 581, 838]
    assert result["f"][0] == pytest.approx([-5348.113939327506, 19.390611314790082])


def test_travelling_salesman_last_solution() -> None:
    """The final Fruitland route and its objective match the notebook."""

    random.seed(0)
    list_of_cities = [
        "Apple City",
        "Banana City",
        "Strawberry City",
        "BlueBerry City",
        "Pineapple City",
        "Blackberry City",
        "Kiwi City",
        "Cherry City",
        "Star Fruit City",
        "Passion Fruit City",
        "Avocado City",
        "Pomegranate City",
        "Orange City",
        "Lemon City",
        "Tangerine City",
        "Pear City",
        "Tomato City",
    ]
    cities = {"Airport": (0.0, 0.0)}
    for city in list_of_cities:
        cities[city] = (random.uniform(0.0, 20.0), random.uniform(0.0, 20.0))

    def fobj(Stops: list) -> tuple:
        coords1 = cities["Airport"]
        coords2 = cities[Stops[0]]
        total_dist = sqrt(
            (coords2[0] - coords1[0]) ** 2 + (coords2[1] - coords1[1]) ** 2
        )

        for i in range(1, len(Stops)):
            coords1 = cities[Stops[i - 1]]
            coords2 = cities[Stops[i]]
            total_dist += sqrt(
                (coords2[0] - coords1[0]) ** 2 + (coords2[1] - coords1[1]) ** 2
            )

        coords1 = cities[Stops[-1]]
        coords2 = cities["Airport"]
        total_dist += sqrt(
            (coords2[0] - coords1[0]) ** 2 + (coords2[1] - coords1[1]) ** 2
        )
        return (total_dist,)

    optimizer = mosa.Anneal()
    optimizer.set_population(Stops=list_of_cities)
    optimizer.initial_temperature = 10.0
    optimizer.number_of_temperatures = 100
    optimizer.number_of_iterations = 1000
    optimizer.archive_size = 100
    optimizer.maximum_archive_rejections = 50000
    optimizer.set_group_params(
        "Stops",
        number_of_elements=len(list_of_cities),
        distinct_elements=True,
        change_value_move=0.0,
        swap_move=1.0,
    )
    optimizer.restart = False
    optimizer.evolve(fobj)

    result = optimizer.prune_dominated()

    assert result["x"][0]["Stops"] == [
        "Cherry City",
        "Tangerine City",
        "Banana City",
        "Blackberry City",
        "Pineapple City",
        "Avocado City",
        "Passion Fruit City",
        "Kiwi City",
        "BlueBerry City",
        "Pomegranate City",
        "Apple City",
        "Tomato City",
        "Strawberry City",
        "Pear City",
        "Lemon City",
        "Star Fruit City",
        "Orange City",
    ]
    assert result["f"][0] == pytest.approx([82.91878251421768])
