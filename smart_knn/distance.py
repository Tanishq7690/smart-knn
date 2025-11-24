import logging
import numpy as np

logger = logging.getLogger("smartknn.distance")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | smartknn.distance | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


def _ensure_float32(x):
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=np.float32)


def _sanitize_features_keep_scale(X):

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X


def _validate_weights(weights, n_features, eps):

    w = np.asarray(weights, dtype=np.float32)

    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")

    if w.shape[0] != n_features:
        raise ValueError(f"Weight length {w.shape[0]} does not match feature dimension {n_features}")

    if np.any(w < 0):
        neg_count = int(np.sum(w < 0))
        raise ValueError(f"Found {neg_count} negative weight(s). Weights must be non-negative.")

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    w = np.maximum(w, eps).astype(np.float32)

    s = float(np.sum(w))
    if s <= eps * len(w):

        logger.warning("Effective weight sum is very small (sum=%.3e). Distance may be dominated by numeric noise.", s)

    return w



def weighted_euclidean(a, b, weights, eps=1e-8):

    a = _ensure_float32(a)
    b = _ensure_float32(b)


    a = a.reshape(-1)
    b = b.reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same shape: {a.shape} vs {b.shape}")

    a = _sanitize_features_keep_scale(a)
    b = _sanitize_features_keep_scale(b)


    w = _validate_weights(weights, a.shape[0], eps)

    diff = a - b
    dist_sq = np.sum(w * diff * diff, dtype=np.float64)
    return float(np.sqrt(dist_sq))


def weighted_euclidean_batch(X, query, weights, eps=1e-8):

    X = np.atleast_2d(_sanitize_features_keep_scale(_ensure_float32(X)))
    q = _ensure_float32(query).reshape(-1)

    d = X.shape[1]
    if q.shape[0] != d:
        raise ValueError(f"Query dimension {q.shape[0]} does not match X features {d}")

    w = _validate_weights(weights, d, eps)

    diff = X - q 
    dist_sq = np.sum(w[None, :] * (diff * diff), axis=1, dtype=np.float64)
    return np.sqrt(dist_sq).astype(np.float32)


def weighted_euclidean_multiquery(X, Q, weights, eps=1e-8, max_mem_bytes=1_000_000_000):

    X = np.atleast_2d(_sanitize_features_keep_scale(_ensure_float32(X)))
    Q = np.atleast_2d(_sanitize_features_keep_scale(_ensure_float32(Q)))

    n_x, d = X.shape
    n_q = Q.shape[0]

    if Q.shape[1] != d:
        raise ValueError(f"Query features {Q.shape[1]} do not match X features {d}")

    w = _validate_weights(weights, d, eps)

    projected_bytes = int(n_q) * int(n_x) * int(d) * 4  
    if projected_bytes > max_mem_bytes:
   
        raise MemoryError(
            f"Projected memory for multiquery is too large: {projected_bytes} bytes. "
            f"Reduce Q or X, or call function in batches. (max_mem_bytes={max_mem_bytes})"
        )

    diff = X[None, :, :] - Q[:, None, :]  
    dist_sq = np.sum((w[None, None, :] * (diff * diff)), axis=2, dtype=np.float64)
    return np.sqrt(dist_sq).astype(np.float32)
