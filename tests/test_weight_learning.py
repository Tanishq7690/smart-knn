import numpy as np
import pytest
from smart_knn.weight_learning import (
    _safe_normalize,
    _non_constant_mask,
    _univariate_mse_weights,
    _mutual_info_weights,
    _random_forest_weights,
    learn_feature_weights,
)

def test_safe_normalize_basic():
    w = np.array([1.0, 2.0, 3.0])
    out = _safe_normalize(w)

    assert out.shape == (3,)
    assert np.isclose(out.sum(), 1.0)
    assert np.all(out >= 0)


def test_safe_normalize_all_zero_nonneg_finite():
    w = np.zeros(5)
    out = _safe_normalize(w)


    assert np.all(out >= 0)
    assert np.all(np.isfinite(out))

   
    assert not np.allclose(out, 0.0)



def test_non_constant_mask():
    X = np.array([
        [1, 5, 9],
        [1, 6, 9],
        [1, 7, 9],
    ])
    mask = _non_constant_mask(X)
    assert np.array_equal(mask, np.array([False, True, False]))


def test_non_constant_mask_all_constant():
    X = np.ones((10, 3))
    mask = _non_constant_mask(X)
    assert not np.any(mask)



def test_univariate_mse_weights_shape():
    X = np.random.rand(50, 4)
    y = np.random.rand(50)
    w = _univariate_mse_weights(X, y)

    assert w.shape == (4,)
    assert np.isclose(w.sum(), 1.0)


def test_univariate_mse_constant_feature():
    X = np.column_stack([
        np.ones(50),        
        np.random.rand(50)  
    ])
    y = X[:, 1]

    w = _univariate_mse_weights(X, y)

    assert w[1] > w[0]    


def test_mutual_info_basic():
    X = np.random.rand(100, 3)
    y = X[:, 0] * 2 + 0.1 * np.random.randn(100)

    w = _mutual_info_weights(X, y)

    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]


def test_mutual_info_all_constant():
    X = np.ones((100, 4))
    y = np.random.rand(100)

    w = _mutual_info_weights(X, y)

    assert np.allclose(w, 1 / 4)



def test_random_forest_weights_basic():
    X = np.random.rand(200, 3)
    y = X[:, 1] * 5 + np.random.randn(200) * 0.1

    w = _random_forest_weights(X, y)

    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]


def test_random_forest_weights_exception_fallback(monkeypatch):

    def bad_fit(*args, **kwargs):
        raise ValueError("force error")

    monkeypatch.setattr("sklearn.ensemble.RandomForestRegressor.fit", bad_fit)

    X = np.random.rand(50, 3)
    y = np.random.rand(50)

    w = _random_forest_weights(X, y)

    assert np.allclose(w, 1 / 3)


def test_learn_feature_weights_shape():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    w = learn_feature_weights(X, y)

    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)


def test_learn_feature_weights_informative_feature():
    X = np.random.rand(200, 3)
    y = 10 * X[:, 2] + np.random.randn(200) * 0.01

    w = learn_feature_weights(X, y)

    assert w[2] > w[0]
    assert w[2] > w[1]
    assert np.isclose(w.sum(), 1.0)


def test_learn_feature_weights_nan_inf_handling():
    X = np.array([
        [1, np.nan, np.inf],
        [2, -np.inf, 5],
        [3, 8,  9],
    ], dtype=np.float32)

    y = np.array([1, 2, 3], dtype=np.float32)

    w = learn_feature_weights(X, y)

    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)
