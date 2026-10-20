"""Сравнение моделей, предсказывающих вид пингвина по размерам клюва"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["bill_length_mm", "bill_depth_mm"]
TARGET = "species"
RANDOM_STATE = 42


def candidates() -> dict[str, tuple[Pipeline, dict]]:
    """Модели-кандидаты"""
    baseline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    logreg = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    knn = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])
    tree = Pipeline([("model", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    forest_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    forest = Pipeline([("model", forest_model)])
    svm = Pipeline([("scale", StandardScaler()), ("model", SVC())])

    return {
        "baseline": (baseline, {}),
        "logreg": (logreg, {"model__C": [0.01, 0.1, 1, 10, 100]}),
        "knn": (knn, {"model__n_neighbors": [1, 3, 5, 9, 15, 25]}),
        "tree": (tree, {"model__max_depth": [1, 2, 3, 4, 6, None]}),
        "forest": (forest, {"model__max_depth": [2, 4, None]}),
        "svm": (svm, {"model__C": [0.1, 1, 10, 100], "model__gamma": [0.01, 0.1, 1]}),
    }


def make_folds(seed: int = RANDOM_STATE) -> StratifiedKFold:
    """Разбиение на 5 частей с сохранением долей видов и перемешиванием"""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def naive_cv_score(pipeline, grid, X, y, seed: int = RANDOM_STATE) -> float:
    """Наивная оценка. Лучший результат подбора гиперпараметров

    Одна и та же кросс-валидация и выбирает параметры, и оценивает
    качество, поэтому оценка получается завышенной.
    """
    search = GridSearchCV(pipeline, grid, cv=make_folds(seed), n_jobs=-1)
    search.fit(X, y)
    return search.best_score_


def nested_cv_predict(pipeline, grid, X, y, seed: int = RANDOM_STATE) -> np.ndarray:
    """Честные предсказания вложенной кросс-валидации.

    Внешний цикл откладывает 1/5 данных. На остальных 4/5 внутренний цикл
    (GridSearchCV) подбирает гиперпараметры и обучает модель. Затем модель
    предсказывает отложенную часть, которую не видела ни на одном этапе.
    """
    search = GridSearchCV(pipeline, grid, cv=make_folds(seed), n_jobs=-1)
    return cross_val_predict(search, X, y, cv=make_folds(seed + 1))


def compare_models(X, y, seed: int = RANDOM_STATE):
    """Таблица точности всех кандидатов и их честные предсказания"""
    rows = []
    predictions = {}
    for name, (pipeline, grid) in candidates().items():
        pred = nested_cv_predict(pipeline, grid, X, y, seed)
        predictions[name] = pred
        nested = np.mean(pred == y)
        naive = naive_cv_score(pipeline, grid, X, y, seed)
        rows.append(
            {
                "model": name,
                "nested": nested,
                "naive": naive,
                "optimism": naive - nested,
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("nested", ascending=False), predictions


def mcnemar_exact(y, pred_a, pred_b) -> tuple[int, int, float]:
    """Точный тест Макнемара. Значимо ли различаются две модели.

    b — сколько объектов модель A угадала, а B нет; c — наоборот.
    Если модели одинаково точны, то b ~ Bin(b + c, 1/2).
    """
    y = np.asarray(y)
    a_right = np.asarray(pred_a) == y
    b_right = np.asarray(pred_b) == y
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    if b + c == 0:
        return b, c, 1.0
    return b, c, binomtest(b, n=b + c, p=0.5).pvalue


def fit_final(name: str, X, y, seed: int = RANDOM_STATE):
    """Подбор гиперпараметров и обучение на всех данных лучшей модели."""
    pipeline, grid = candidates()[name]
    search = GridSearchCV(pipeline, grid, cv=make_folds(seed), n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
