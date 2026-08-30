"""Tests for plotting Pareto fronts."""

import matplotlib
import pytest

from mosa import Anneal
from mosa._error import MOSAError

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def disable_plot_display(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


@pytest.fixture
def three_objective_archive():
    return {
        "x": [{"X": 0.0}, {"X": 1.0}],
        "f": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }


@pytest.mark.parametrize("label", [["Cost", "Risk", "Time"], ("Cost", "Risk", "Time")])
def test_plot_front_accepts_list_or_tuple_labels(three_objective_archive, label):
    anneal = Anneal()

    anneal.plot_front(
        xset=three_objective_archive,
        index1=2,
        index2=0,
        index3=1,
        label=label,
    )

    axes = plt.gcf().axes[0]
    assert axes.get_xlabel() == "Time"
    assert axes.get_ylabel() == "Cost"
    assert axes.get_zlabel() == "Risk"


def test_plot_front_uses_default_labels(three_objective_archive):
    anneal = Anneal()

    anneal.plot_front(
        xset=three_objective_archive,
        index1=2,
        index2=0,
        index3=1,
    )

    axes = plt.gcf().axes[0]
    assert axes.get_xlabel() == "f2"
    assert axes.get_ylabel() == "f0"
    assert axes.get_zlabel() == "f1"


@pytest.mark.parametrize("label", [["Cost", "Risk"], ("Cost",), "Cost"])
def test_plot_front_rejects_invalid_labels(three_objective_archive, label):
    anneal = Anneal()

    with pytest.raises(MOSAError):
        anneal.plot_front(xset=three_objective_archive, label=label)
