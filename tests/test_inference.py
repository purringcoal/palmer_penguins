import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from palmer_penguins.inference import bootstrap_ci, diff_means, ols, permutation_test


def test_diff_means():
    assert diff_means([3, 5], [1, 1]) == 3.0


def test_permutation_test_far_apart_groups():
    # Группы не пересекаются: ни одна перестановка не даст разность
    # больше наблюдаемой, и p-значение равно минимально возможному 1/(B+1).
    a = [10, 11, 12, 13, 14, 15, 16, 17]
    b = [0, 1, 2, 3, 4, 5, 6, 7]
    observed, p, null = permutation_test(a, b, n_resamples=999)
    assert observed == 10.0
    assert p == pytest.approx(1 / 1000)
    assert len(null) == 999


def test_permutation_test_same_distribution():
    # Обе группы из одного распределения: различие не должно быть значимым.
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    _, p, _ = permutation_test(a, b, n_resamples=999)
    assert p > 0.05


def test_permutation_test_is_reproducible():
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
    assert permutation_test(a, b, seed=7)[1] == permutation_test(a, b, seed=7)[1]


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=5.0, size=200)
    b = rng.normal(loc=2.0, size=200)  # истинная разность средних = 3
    low, high = bootstrap_ci(a, b, n_resamples=2000)
    assert low < 3.0 < high
    assert low < diff_means(a, b) < high


def test_ols_recovers_exact_coefficients():
    # Без шума МНК обязан найти коэффициенты точно: y = 1 + 2*x1 - 3*x2.
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0, 4.0], "x2": [1.0, 0.0, 2.0, 5.0, 3.0]})
    y = 1 + 2 * X["x1"] - 3 * X["x2"]
    result = ols(X, y)
    assert result.loc["intercept", "coef"] == pytest.approx(1.0)
    assert result.loc["x1", "coef"] == pytest.approx(2.0)
    assert result.loc["x2", "coef"] == pytest.approx(-3.0)


def test_ols_matches_sklearn():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = pd.Series(0.5 + X["a"] - 2 * X["b"] + rng.normal(scale=0.3, size=100))
    reference = LinearRegression().fit(X, y)
    result = ols(X, y)
    assert result.loc["intercept", "coef"] == pytest.approx(reference.intercept_)
    assert result.loc[["a", "b"], "coef"].to_numpy() == pytest.approx(reference.coef_)
    assert (result["low"] < result["coef"]).all()
    assert (result["coef"] < result["high"]).all()
