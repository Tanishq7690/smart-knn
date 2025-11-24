import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


def _safe_normalize(w, eps=1e-8):

    w = np.asarray(w, dtype=np.float32)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


    if np.any(w < 0):
        w = np.clip(w, eps, None)

    w = np.clip(w, eps, None)

    s = np.sum(w)
    if s < eps:
        return np.ones_like(w, dtype=np.float32) / len(w)

    return (w / (s + eps)).astype(np.float32)


def _non_constant_mask(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    var = np.nanstd(X, axis=0)
    return var > eps


def _univariate_mse_weights(X, y, eps=1e-8, n_jobs=1):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n_features = X.shape[1]

    def _compute(j):
        col = X[:, j]
        if np.std(col) < 1e-12:
            return eps
        try:
            lr = LinearRegression()
            lr.fit(col.reshape(-1, 1), y)
            pred = lr.predict(col.reshape(-1, 1))
            mse = np.mean((y - pred) ** 2)
            return 1.0 / (mse + eps)
        except Exception:
            return eps

    scores = Parallel(n_jobs=n_jobs)(delayed(_compute)(j) for j in range(n_features))
    return _safe_normalize(scores, eps)


def _mutual_info_weights(X, y, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    mask = _non_constant_mask(X)
    if not mask.any():
        return np.ones(X.shape[1], dtype=np.float32) / X.shape[1]

    mi = np.zeros(X.shape[1], dtype=np.float32)
    try:
        mi_vals = mutual_info_regression(X[:, mask], y)
        mi[mask] = mi_vals
    except Exception:
        return np.ones(X.shape[1], dtype=np.float32) / X.shape[1]

    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    return _safe_normalize(mi + eps, eps)


def _random_forest_weights(X, y, n_estimators=150, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    try:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        imp = rf.feature_importances_
        imp = np.nan_to_num(imp, nan=0.0)

        if np.sum(imp) < eps:
            return np.ones(X.shape[1], dtype=np.float32) / X.shape[1]

        return _safe_normalize(imp + eps, eps)

    except Exception:
        return np.ones(X.shape[1], dtype=np.float32) / X.shape[1]


def learn_feature_weights(
    X,
    y,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    n_jobs=1,
    eps=1e-8
):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)


    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

   
    w_mse = _univariate_mse_weights(X, y, eps=eps, n_jobs=n_jobs)
    w_mi  = _mutual_info_weights(X, y, eps=eps)
    w_rf  = _random_forest_weights(X, y, eps=eps)


    w_mse = np.nan_to_num(w_mse, nan=0.0)
    w_mi  = np.nan_to_num(w_mi,  nan=0.0)
    w_rf  = np.nan_to_num(w_rf,  nan=0.0)

    weights = alpha * w_mse + beta * w_mi + gamma * w_rf

    return _safe_normalize(weights, eps)
