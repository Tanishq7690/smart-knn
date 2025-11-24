import numpy as np
import pytest
import pandas as pd
from smart_knn.distance import (
    _ensure_float32,
    _sanitize_features_keep_scale,
    _validate_weights,
    weighted_euclidean,
    weighted_euclidean_batch,
    weighted_euclidean_multiquery,
)


def test_ensure_float32_dtype():
    x = np.array([1, 2, 3], dtype=np.float64)
    result = _ensure_float32(x)
    assert result.dtype == np.float32


def test_ensure_float32_list_input():
    x = [1, 2, 3]
    result = _ensure_float32(x)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_ensure_float32_pandas_series():
    s = pd.Series([1, 2, 3])
    out = _ensure_float32(s)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32


def test_sanitize_replaces_nan_inf():
    X = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    out = _sanitize_features_keep_scale(X)

    assert out[0] == 0.0
    assert out[1] == 1e9
    assert out[2] == -1e9


def test_sanitize_keeps_valid_values():
    X = np.array([1.0, 5.0, -3.0], dtype=np.float32)
    out = _sanitize_features_keep_scale(X)
    assert np.allclose(out, X)


def test_validate_weights_basic():
    w = _validate_weights([1, 2, 3], n_features=3, eps=1e-8)
    assert w.shape == (3,)
    assert np.all(w >= 1e-8)


def test_validate_weights_wrong_shape():
    with pytest.raises(ValueError):
        _validate_weights([1, 2], n_features=3, eps=1e-8)


def test_validate_weights_negative():
    with pytest.raises(ValueError):
        _validate_weights([1, -2, 3], n_features=3, eps=1e-8)


def test_validate_weights_nans_infs_raise():
    with pytest.raises(ValueError):
        _validate_weights([np.nan, np.inf, -np.inf], n_features=3, eps=1e-8)


def test_validate_weights_zero_replaced_with_eps():
    w = _validate_weights([0, 1, 2], n_features=3, eps=1e-6)
    assert w[0] == 1e-6
    assert np.all(w >= 1e-6)


def test_weighted_euclidean_correctness():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])
    w = np.array([1, 1, 1])
    d = weighted_euclidean(a, b, w)
    assert np.isclose(d, 1.0)


def test_weighted_euclidean_zero_distance():
    a = np.array([2, 3, 4])
    w = np.ones(3)
    d = weighted_euclidean(a, a, w)
    assert d == 0.0


def test_weighted_euclidean_dtype_mix():
    a = np.array([1, 2], dtype=np.float32)
    b = np.array([2, 3], dtype=np.float64)
    w = np.array([1, 1], dtype=np.float32)
    d = weighted_euclidean(a, b, w)
    assert np.isclose(d, np.sqrt(2))


def test_weighted_euclidean_shape_mismatch():
    with pytest.raises(ValueError):
        weighted_euclidean([1,2], [1,2,3], [1,1])


def test_weighted_euclidean_weight_applies():
    a = np.array([0])
    b = np.array([10])
    w = np.array([100])
    d = weighted_euclidean(a, b, w)
    assert np.isclose(d, 100)


def test_batch_basic():
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    q = np.array([1, 2], dtype=np.float32)
    w = np.array([1, 1])

    d = weighted_euclidean_batch(X, q, w)
    assert d.shape == (2,)
    assert d[0] == 0.0


def test_batch_multiple_distances():
    X = np.array([[1,2], [4,6]], dtype=np.float32)
    q = np.array([1,2], dtype=np.float32)
    w = np.ones(2)

    d = weighted_euclidean_batch(X, q, w)
    assert np.isclose(d[0], 0.0)
    assert np.isclose(d[1], 5.0)


def test_batch_dim_mismatch():
    X = np.random.rand(5, 3)
    q = np.random.rand(2)
    w = np.ones(3)

    with pytest.raises(ValueError):
        weighted_euclidean_batch(X, q, w)


def test_multiquery_basic():
    X = np.array([[1, 1], [4, 5]], dtype=np.float32)
    Q = np.array([[1, 1], [4, 5]], dtype=np.float32)
    w = np.array([1, 1])

    D = weighted_euclidean_multiquery(X, Q, w)
    assert D.shape == (2, 2)
    assert D[0, 0] == 0.0
    assert D[1, 1] == 0.0


def test_multiquery_dim_mismatch():
    X = np.random.rand(10, 4)
    Q = np.random.rand(3, 3)
    w = np.ones(4)

    with pytest.raises(ValueError):
        weighted_euclidean_multiquery(X, Q, w)


def test_multiquery_noncontiguous_arrays():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)[::2]
    Q = np.arange(40, dtype=np.float32).reshape(20, 2)[::4]
    w = np.ones(2)

    D = weighted_euclidean_multiquery(X, Q, w)
    assert D.shape == (Q.shape[0], X.shape[0])


def test_multiquery_memory_error_trigger():
    X = np.random.rand(100, 20).astype(np.float32)
    Q = np.random.rand(1000, 20).astype(np.float32)
    w = np.ones(20)

    with pytest.raises(MemoryError):
        weighted_euclidean_multiquery(X, Q, w, max_mem_bytes=1000)
