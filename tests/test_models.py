import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from palmer_penguins.models import (
    candidates,
    compare_models,
    mcnemar_exact,
    nested_cv_predict,
)


@pytest.fixture
def easy_data():
    """Три далеко разнесённых облака точек, хорошая модель не ошибётся ни разу."""
    X, y = make_blobs(
        n_samples=90,
        centers=[[0, 0], [10, 10], [0, 10]],
        cluster_std=0.5,
        random_state=0,
    )
    return pd.DataFrame(X, columns=["a", "b"]), pd.Series(y)


def test_every_grid_key_is_a_real_parameter():
    for name, (pipeline, grid) in candidates().items():
        params = pipeline.get_params()
        for key in grid:
            assert key in params, f"{name}: нет параметра {key}"


@pytest.mark.parametrize("name", ["logreg", "knn", "svm"])
def test_scaling_happens_inside_pipeline(name):
    pipeline, _ = candidates()[name]
    first_step = pipeline.steps[0][1]
    assert isinstance(first_step, StandardScaler)


def test_nested_cv_is_reproducible(easy_data):
    X, y = easy_data
    pipeline, grid = candidates()["logreg"]
    first = nested_cv_predict(pipeline, grid, X, y, seed=0)
    second = nested_cv_predict(pipeline, grid, X, y, seed=0)
    assert np.array_equal(first, second)


def test_easy_problem_is_solved(easy_data):
    X, y = easy_data
    pipeline, grid = candidates()["logreg"]
    pred = nested_cv_predict(pipeline, grid, X, y)
    assert np.mean(pred == y) == 1.0


def test_compare_models_table(easy_data):
    X, y = easy_data
    table, predictions = compare_models(X, y)
    assert set(table.index) == set(candidates())
    assert list(table.columns) == ["nested", "naive", "optimism"]
    assert table["nested"].between(0, 1).all()
    assert table.index[-1] == "baseline"  # константа хуже всех
    assert all(len(p) == len(y) for p in predictions.values())


def test_mcnemar_identical_models():
    y = [0, 1, 2, 1]
    assert mcnemar_exact(y, y, y) == (0, 0, 1.0)


def test_mcnemar_by_hand():
    # A угадала все 5 объектов, B ошиблась на всех: b = 5, c = 0.
    # При H0 вероятность такого или более крайнего исхода: 2 * (1/2)^5 = 0.0625
    y = [0, 0, 0, 0, 0]
    b, c, p = mcnemar_exact(y, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    assert (b, c) == (5, 0)
    assert p == pytest.approx(0.0625)
