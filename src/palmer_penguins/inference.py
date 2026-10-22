"""Статистические выводы через перестановки, бутстреп и метод наименьших квадратов."""

import numpy as np
import pandas as pd
from scipy import stats


def diff_means(a, b) -> float:
    """Разность выборочных средних."""
    return float(np.mean(a) - np.mean(b))


def permutation_test(a, b, n_resamples: int = 9999, seed: int = 42):
    """Двусторонний перестановочный тест для разности средних двух групп.

    Если H0 верна (группы распределены одинаково), метки групп можно
    произвольно переставлять. Перемешиваем объединённую выборку много раз
    и смотрим, как часто разность средних получается не меньше наблюдаемой.

    Возвращает наблюдаемую разность, p-значение и все разности при H0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = diff_means(a, b)
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)

    null_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        shuffled = rng.permutation(pooled)
        null_diffs[i] = diff_means(shuffled[: len(a)], shuffled[len(a) :])

    extreme = np.sum(np.abs(null_diffs) >= abs(observed))
    p_value = (extreme + 1) / (n_resamples + 1)
    return observed, p_value, null_diffs


def bootstrap_ci(a, b, n_resamples: int = 9999, level: float = 0.95, seed: int = 42):
    """Процентильный бутстреп-интервал для разности средних двух групп.

    Каждая группа заново выбирается из самой себя с возвращением,
    разность средних пересчитывается; квантили этих разностей и дают интервал.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)

    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        a_star = rng.choice(a, size=len(a), replace=True)
        b_star = rng.choice(b, size=len(b), replace=True)
        boot_diffs[i] = diff_means(a_star, b_star)

    alpha = 1 - level
    low, high = np.quantile(boot_diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def ols(X: pd.DataFrame, y: pd.Series, level: float = 0.95) -> pd.DataFrame:
    """Линейная регрессия методом наименьших квадратов.

    Возвращает таблицу: коэффициент, стандартная ошибка и доверительный
    интервал для свободного члена и каждого признака.
    """
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    target = y.to_numpy(dtype=float)
    names = ["intercept", *X.columns]

    beta, *_ = np.linalg.lstsq(design, target, rcond=None)

    residuals = target - design @ beta
    n, k = design.shape
    sigma2 = residuals @ residuals / (n - k)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))

    t_crit = stats.t.ppf(1 - (1 - level) / 2, df=n - k)
    return pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "low": beta - t_crit * se,
            "high": beta + t_crit * se,
        },
        index=names,
    )
