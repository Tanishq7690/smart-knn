#  SmartKNN Theory & Mathematical Foundations (Updated Full Version)

SmartKNN is a **modern, enhanced, and intelligent K-Nearest Neighbors algorithm** that improves on classical KNN through feature weighting, normalization, robust preprocessing, feature filtering, and stable distance computation. This document provides the **complete theoretical foundation** of SmartKNN, including the mathematics, intuition, and hyperparameter tuning guidelines.

---

#  0. SmartKNN Hyperparameters (Main Performance-Impacting Ones)

Only **4 hyperparameters directly influence model accuracy and performance**. These are the main ones users should tune.

## 1️ `k` — Number of Neighbors (Default: 5)

**Controls:** Bias–variance trade-off, smoothing, noise sensitivity.

### How changing k affects performance:

* **Increase k →** smoother predictions, less noise, more bias
* **Decrease k →** sharper predictions, more variance, more risk of overfitting

### Recommended ranges:

* Small datasets: **3–15**
* Medium datasets: **5–50**
* Large datasets: **20–150**

---

## 2️ `weight_threshold` — Minimum Weight to Keep Feature (Default: 0.0)

Features with weights lower than this are removed.

**Controls:** dimensionality, noise filtering, stability.

### Effects:

* Higher threshold → fewer features → faster + less noise
* Lower threshold → more features → better for complex data

### Recommended ranges:

* Clean datasets: **0.0–0.01**
* Noisy datasets: **0.02–0.1**
* High-dimensional data: **0.05–0.2**

---

## 3️ `alpha`, `beta`, `gamma` — Weight Blending Strengths

These belong to the weight-learning module but directly change performance.

They blend:

* **MSE importance** (linear sensitivity)
* **Mutual Information** (non-linear sensitivity)
* **Random Forest importance** (interaction sensitivity)

### Default:

```
alpha = 0.4
beta  = 0.3
gamma = 0.3
```

### Effects:

* More **alpha** → better for linear/tabular patterns
* More **beta** → better for non-linear relationships
* More **gamma** → better for interaction-heavy, tree-friendly data

### Recommended ranges:

Sum should be 1.0.

* alpha: **0.2 – 0.6**
* beta: **0.1 – 0.5**
* gamma: **0.1 – 0.5**

---

# 1. Introduction

Classical KNN struggles with:

* treating all features equally
* scale sensitivity
* noise and outliers
* curse of dimensionality
* instability on real-world datasets

SmartKNN solves this by:

* learning feature importance
* normalizing all data
* filtering unimportant features
* applying weighted Euclidean distance
* handling NaN/Inf robustly

---

# 2. Weighted Euclidean Distance

SmartKNN uses this improved distance:

```
d(x, y) = sqrt( Σ w_i (x_i - y_i)^2 )
```

### Why this helps

* irrelevant features → low weight → ignored
* important features → high weight → dominate distance
* reduces noise influence
* improves decision boundary alignment with target

This is a valid metric when weights ≥ 0.

---

# 3. Feature Weight Learning

SmartKNN learns weights using **three complementary measures**.

## 3.1 MSE-Based Weighting

Linear predictive power:

```
w_i ∝ 1 / (MSE_i + ε)
```

---

## 3.2 Mutual Information

Captures **non-linear relationships** between feature and target.

---

## 3.3 Random Forest Importances

Captures **interactions and non-linear splits**.

---

## 3.4 Blended Final Weight

```
W = α W_mse + β W_mi + γ W_rf
W = normalize(W)
```

---

# 4. Normalization Theory

SmartKNN normalizes inputs:

```
x' = (x - mean) / (std + ε)
```

This reduces scale issues and improves stability.

---

# 5. Feature Filtering

Features below `weight_threshold` are removed, reducing noise.

---

# 6. Prediction Theory

### Regression

```
pred = mean( neighbor_y )
```

### Classification

```
pred = most_common_label
```

---

# 7. Stability Theory

SmartKNN includes median imputation, sanitization, normalization, clipping, and filtering.

---
---

##  Classification Status (Temporary — Stability Release)

SmartKNN supports both regression and classification internally, but to guarantee strong stability across all data pipelines, the current version follows a **regression-first output policy**:

| Case | SmartKNN Behavior |
|------|------------------|
| Target is continuous | regression (mean of neighbors) |
| Target is integer labels | internally votes like classification, but returns numeric output |

 Why this update:

Classic KNN classification works correctly, but auto-detecting between integer regression and integer classification can confuse external metrics (sklearn, OpenML, Kaggle) and cause errors such as:

```
Classification metrics can't handle a mix of binary and continuous targets
```

To prevent breaking pipelines, SmartKNN now returns **only numeric predictions** by design.

### Classification Users (for now)
If your dataset is classification, simply map predictions back to labels:

```python
classes = np.unique(y_train)
pred_labels = classes[sknn_pred.astype(int)]
```

This ensures **100% reliability** while keeping the API backward compatible.

### Planned upgrade
The next major release (v2.0) will include:

| Feature | Status |
|--------|--------|
| Explicit classification mode |  In progress |
| Probability voting | 🔄 |
| Metrics safety | 🔄 |
| Probability thresholding | 🔄 |

---



---

# 8. Time & Space Complexity 

Understanding the computational complexity of SmartKNN helps users anticipate performance across small, medium, and large datasets.

##  Training Complexity (Fit)

SmartKNN training mainly involves:

1. Normalization → **O(N·D)**
2. Feature weight learning →

   * MSE weights: **O(N·D)** (linear regressions per feature)
   * Mutual Information: **O(N·D)**
   * Random Forest: **O(Trees · N · log N)**

### **Overall Training Complexity:**

```
O(N·D + Trees·N·log(N))
```

With default `n_estimators = 150`, this is manageable for medium-sized tabular data.

### **Training Memory:**

```
O(N·D)
```

Only the normalized training matrix and weights are stored.

---

##  Inference Complexity (Predict)

Prediction is the most expensive part of KNN-style models.

For each query:

1. Normalize query → **O(D)**
2. Weighted Euclidean distance to all training samples → **O(N·D)**
3. Partial sort to get k neighbors → **O(N)**

### **Overall Inference Complexity (per query):**

```
O(N·D)
```

This is the expected cost unless ANN methods (FAISS/HNSW) are added in future versions.

### **Inference Memory:**

```
O(N·D)
```

You store all filtered features of the training set.

---

##  Practical Notes

* SmartKNN is **fast for up to ~200k rows**, very competitive on tabular tasks.
* Beyond that, exact KNN becomes heavy → future ANN acceleration will improve this dramatically.
* Feature filtering (`weight_threshold`) significantly reduces dimension D, improving both speed and stability

---

# 9. Limitations

* O(N) search
* global metric
* no embeddings
* scalability limits

---

# 10. Future Potential

* adaptive-K
* local metrics
* neural embeddings
* FAISS/HNSW ANN
* prototype compression
* full distance matrix learning

---

# 11. Mathematical Guarantees

Distance is stable and normalization reduces dimensionality issues.

---

SmartKNN forms the base for a **next-generation tabular ML algorithm**.
