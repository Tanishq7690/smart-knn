import numpy as np

def filter_low_weights(
    X, 
    weights, 
    threshold=0.0, 
    min_features=1, 
    return_mask=False
):
  

    if hasattr(X, "values"):
        X = X.values

    X = np.asarray(X, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if weights.shape[0] != X.shape[1]:
        raise ValueError(
            f"weights length {weights.shape[0]} != number of features {X.shape[1]}"
        )

    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    X = np.clip(X, -1e9, 1e9).astype(np.float32)

    weights_clean = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights_clean = np.clip(weights_clean, 0.0, None).astype(np.float32)


    if np.all(weights_clean == 0):

        sorted_idx = np.arange(len(weights_clean))
        mask = np.zeros_like(weights_clean, dtype=bool)
        mask[sorted_idx[:min_features]] = True

        X_f = X[:, mask]
        w_f = weights_clean[mask]

        if return_mask:
            return X_f, w_f, mask
        return X_f, w_f


    if threshold <= 0:
        mask = weights_clean > 0.0
    else:
        mask = weights_clean >= threshold

    if mask.sum() < min_features:
        sorted_idx = np.argsort(weights_clean)[::-1]
        top_idx = sorted_idx[:min_features]

        mask = np.zeros_like(weights_clean, dtype=bool)
        mask[top_idx] = True

    X_f = X[:, mask]
    w_f = weights_clean[mask]

    if X_f.shape[1] == 0:
        raise RuntimeError(
            "Filtered out ALL features — lower threshold or adjust weights."
        )

    if return_mask:
        return X_f.astype(np.float32), w_f.astype(np.float32), mask.astype(bool)

    return X_f.astype(np.float32), w_f.astype(np.float32)
