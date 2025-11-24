import numpy as np
import pytest
from smart_knn import SmartKNN


def _cast_classification_preds(preds, y):
    if np.issubdtype(y.dtype, np.integer):
       
        if preds.dtype == object:
            return np.array([int(round(p)) for p in preds], dtype=y.dtype)
        else:
            return np.round(preds).astype(y.dtype)
    else:
        return np.array(preds, dtype=y.dtype)


def test_end_to_end_classification_basic():
    np.random.seed(42)

    X = np.random.rand(200, 5)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0.6).astype(int)

    model = SmartKNN(k=5)
    model.fit(X, y)

    preds = model.predict(X)
    preds = _cast_classification_preds(preds, y)

    acc = (preds == y).mean()

    assert preds.shape == y.shape
    assert acc > 0.75
    assert preds.dtype == y.dtype


def test_end_to_end_regression_basic():
    np.random.seed(42)

    X = np.random.rand(200, 4)
    y = 3 * X[:, 2] + 2 * X[:, 1] + 0.1 * np.random.randn(200)

    model = SmartKNN(k=5)
    model.fit(X, y)

    preds = model.predict(X).astype(float)
    mse = np.mean((preds - y) ** 2)

    assert preds.shape == y.shape
    assert mse < 0.2
    assert np.isfinite(mse)


def test_end_to_end_nan_inf_handling():
    X = np.array([
        [1.0, np.nan, 5.0],
        [2.0, np.inf, 6.0],
        [3.0, -np.inf, 7.0]
    ])
    y = np.array([0, 1, 1])

    model = SmartKNN(k=2)
    model.fit(X, y)

    q = np.array([np.nan, np.inf, -np.inf])
    pred = model.predict(q)[0]

    assert isinstance(pred, (int, float, np.floating))
    assert np.isfinite(pred)


def test_feature_filtering_threshold():
    np.random.seed(42)

    X = np.random.rand(100, 6)
    y = (X[:, 0] > 0.5).astype(int)

    model = SmartKNN(k=3, weight_threshold=0.2)
    model.fit(X, y)

    assert model.X_.shape[1] >= 1               
    assert model.feature_mask_.sum() == model.X_.shape[1]
    assert model.feature_mask_.dtype == bool


def test_query_mask_matching():
    np.random.seed(42)

    X = np.random.rand(50, 4)
    y = (X[:, 2] > 0.4).astype(int)

    model = SmartKNN()
    model.fit(X, y)

    q = np.random.rand(4)
    q_prepared = model._prepare_query(q)

    assert q_prepared.shape[0] == model.n_features_
    assert q_prepared.dtype == np.float32
    assert np.isfinite(q_prepared).all()


def test_kneighbors_returns_sorted_distances():
    np.random.seed(42)

    X = np.random.rand(40, 3)
    y = np.random.randint(0, 2, 40)

    model = SmartKNN(k=5)
    model.fit(X, y)

    q = np.random.rand(3)
    idx, dists = model.kneighbors(q)

    assert len(idx) == len(dists)
    assert all(dists[i] <= dists[i+1] for i in range(len(dists) - 1))
    assert idx.dtype == int


def test_predict_not_fitted():
    model = SmartKNN()
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])


def test_query_dim_mismatch():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    model = SmartKNN()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict([1, 2, 3])


def test_predict_batch_queries():
    np.random.seed(42)

    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    model = SmartKNN(k=3)
    model.fit(X, y)

    Q = np.random.rand(8, 4)
    preds = model.predict(Q)
    preds = _cast_classification_preds(preds, y)

    assert preds.shape == (8,)
    assert preds.dtype == y.dtype
