#  SmartKNN Usage Guide

This guide explains **how to use SmartKNN** in real-world projects — from loading data to training, prediction, tuning, and handling edge cases.

SmartKNN works with:

* **NumPy arrays**
* **Pandas DataFrames**
* **Both classification and regression tasks**

---

# 1. Installation

```bash
pip install smart-knn
```

Or editable development mode:

```bash
pip install -e .
```

---

# 2. Basic Usage

This is the standard use-case for training and predicting.

```python
from smart_knn import SmartKNN
import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv")

# Split features + target
X = df.drop(columns=["target"])
y = df["target"]

# Create model
model = SmartKNN(k=5, weight_threshold=0.01)

# Train
model.fit(X, y)

# Predict
pred = model.predict([5.1, 3.4, 1.5, 0.2])
print(pred)
```

---

# 3. Using With Pandas DataFrames

SmartKNN fully supports DataFrames.

```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = SmartKNN(k=7)
model.fit(X, y)
```

---

# 4. Predicting with Multiple Samples

```python
queries = [
    [5.0, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3],
]

preds = model.predict(queries)
print(preds)
```

---

# 5. Classification vs Regression

Prediction type depends on the target:

* **If `y` is numeric** → Regression (outputs mean of neighbors)
* **If `y` is categorical / string / int labels** → Classification (majority vote)

You don't have to configure anything.

---

# 6. Handling NaN / Inf Automatically

SmartKNN automatically:

* replaces NaNs with feature medians
* clips Infs to ±1e9
* normalizes features
* filters noisy/low-weight features

No preprocessing required.

---

# 7. Important Hyperparameters

## `k` (default: 5)

Number of neighbors.

* Larger k → smoother predictions, less noise
* Smaller k → sharper boundaries, more variance

## `weight_threshold` (default: 0.0)

Minimum weight required for a feature to stay.

* Higher → fewer features → noise reduction
* Lower → more features → better pattern capture

## Weight blend parameters (internal but tunable):

* `alpha` — linear (MSE importance)
* `beta` — nonlinear (MI importance)
* `gamma` — interaction-based (RF importance)

Example:

```python
from smart_knn.weight_learning import learn_feature_weights
w = learn_feature_weights(X, y, alpha=0.5, beta=0.2, gamma=0.3)
```

---

# 8. End-to-End Example

```python
import pandas as pd
from smart_knn import SmartKNN

# Load
df = pd.read_csv("bank.csv")
X = df.drop("salary", axis=1)
y = df["salary"]

# Create model
model = SmartKNN(k=10, weight_threshold=0.02)

# Fit\model.fit(X, y)

# Predict one example
new_person = df.iloc[100, :-1]
pred = model.predict(new_person)

print("Predicted salary:", pred)
```

---

# 9. When SmartKNN Works Best

SmartKNN is strong when:

* dataset is **tabular**
* patterns are **local** and not globally linear
* there are **irrelevant/noisy features**
* dataset size is **< 200k rows**
* features have **mixed scales**

---

# 10. Limitations

* Slower than trees/MLPs on very large datasets (due to exact KNN)
* Memory cost is O(N·D)
* No ANN acceleration (planned in future)

---

# 11. Tips for Best Performance

* Normalize? **Not required** (SmartKNN does it internally)
* One-hot encoding? Yes if categorical columns exist
* Remove ID-like columns
* Tune k (5–50) and weight_threshold (0–0.1)

---

# 12. Troubleshooting

###  ValueError: Query feature length mismatch

You must pass **exact number of features** as training.

###  RuntimeError: All features filtered out

Lower the weight threshold:

```python
SmartKNN(weight_threshold=0.0)
```

###  Predictions are slow on large datasets

Use batching or wait for ANN version (FAISS/HNSW).

---

# 13. Summary

SmartKNN provides:

* Automatic preprocessing
* Learning feature importance
* Robust weighted distances
* Dimensionality reduction
* Clean API similar to scikit-learn

It is designed to be **simple**, **powerful**, and **production-ready**.

---
