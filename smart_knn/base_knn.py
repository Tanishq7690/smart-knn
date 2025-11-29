import logging
import numpy as np
from collections import Counter

from .weight_learning import learn_feature_weights
from .distance import weighted_euclidean_batch
from .utils import (
    normalize,
    apply_normalization,
    clip_weights,
    ensure_numpy,
)
from .data_processing import filter_low_weights

logger = logging.getLogger("smartknn")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | SmartKNN | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SmartKNN:
    def __init__(self, k=5, weight_threshold=0.0, alpha=0.4, beta=0.3, gamma=0.3):
        self.k = int(k)
        self.weight_threshold = float(weight_threshold)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.fitted = False

    def _validate_schema_array(self, X):
        if hasattr(X, "dtypes"):
            non_numeric = [c for c, dt in zip(X.columns, X.dtypes) if not np.issubdtype(dt, np.number)]
            if non_numeric:
                raise TypeError(
                    f"SmartKNN requires numeric input. Found non-numeric columns: {non_numeric}."
                )

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array for X, got {arr.shape}.")
        if not np.isfinite(arr).all():
            logger.warning("Input contains NaN/Inf — will sanitize during fit().")

    def fit(self, X, y):
        self._validate_schema_array(X)

        X = ensure_numpy(X)
        y = ensure_numpy(y).reshape(-1)

        X = np.nan_to_num(X, nan=np.nan, posinf=1e9, neginf=-1e9)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Row count mismatch between X and y.")

        self.expected_n_features_ = X.shape[1]

        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

        inds = np.where(np.isnan(X))
        if inds[0].size > 0:
            X[inds] = col_medians[inds[1]]

        X = np.clip(X, -1e9, 1e9)

        X_norm, self.mean_, self.std_ = normalize(X)

        w = learn_feature_weights(
            X_norm,
            y,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        w = clip_weights(w)

        try:
            X_f, w_f, mask = filter_low_weights(X_norm, w, threshold=self.weight_threshold, return_mask=True)
        except TypeError:
            X_f, w_f = filter_low_weights(X_norm, w, threshold=self.weight_threshold)
            mask = w > self.weight_threshold

        if X_f.shape[1] == 0:
            raise RuntimeError("All features filtered out — lower weight_threshold.")

        self.feature_mask_ = np.asarray(mask, dtype=bool)
        self.X_ = X_f.astype(np.float32)
        self.y_ = y
        self.weights_ = w_f.astype(np.float32)
        self.n_features_ = self.X_.shape[1]

        self.fitted = True
        return self

    def _prepare_query(self, q):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        q_arr = np.asarray(q)
        if q_arr.ndim > 1:
            q_arr = q_arr.reshape(-1)

        if q_arr.shape[0] != self.expected_n_features_:
            raise ValueError("Query feature length mismatch.")

        q_arr = np.nan_to_num(q_arr, nan=np.nan, posinf=1e9, neginf=-1e9)

        mean_fallback = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        q_arr = np.where(np.isnan(q_arr), mean_fallback, q_arr)

        q_norm = apply_normalization(q_arr, self.mean_, self.std_)
        q_masked = q_norm[self.feature_mask_]

        if q_masked.shape[0] != self.n_features_:
            raise ValueError("Masked query length mismatch.")

        return q_masked.astype(np.float32)

    def kneighbors(self, query):
        q_f = self._prepare_query(query)
        dists = weighted_euclidean_batch(self.X_, q_f, self.weights_)
        k = min(self.k, len(dists))

        idx = np.argpartition(dists, k - 1)[:k]
        idx = idx[np.argsort(dists[idx])]
        return idx, dists[idx]

    def predict(self, Xq):
        Xq = np.asarray(Xq)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)

        if not hasattr(self, "_cls_warn"):
            logger.warning(
                "Classification mode is temporarily disabled in this version. "
                "SmartKNN will output regression predictions (mean of neighbors)."
            )
            self._cls_warn = True

        preds = []
        for q in Xq:
            idx, _ = self.kneighbors(q)
            preds.append(float(np.mean(self.y_[idx])))

        return np.array(preds, dtype=float)
