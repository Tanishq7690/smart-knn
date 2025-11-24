# SmartKNN

A **smarter, weighted, feature-selective KNN algorithm** that automatically learns feature importance, filters weak features, handles missing values, normalizes data, and provides a significant improvement over classic KNN — all with a **plug-and-play sklearn-like API**.

SmartKNN works for both **classification and regression** with no additional settings.

---

##  Key Features

* **Automatic feature weighting** using:

  * Univariate MSE scoring
  * Mutual Information
  * Random Forest importance
* **Automatic normalization** of all input data
* **NaN / Inf handling** (both training and prediction)
* **Automatic feature filtering** using learned weights
* **Weighted Euclidean distance** for more accurate neighbor selection
* **Works out-of-the-box for classification & regression**
* **Scikit-learn style API** (`fit`, `predict`, `kneighbors`)
* **Supports NumPy arrays and Pandas DataFrames**
* **Fast batch distance computation**

---

##  Installation

```
pip install smart-knn
```

(If installing locally)

```
pip install .
```

---

##  Quick Start (Most Common Usage)

```python
import pandas as pd
from smart_knn import SmartKNN

# Load your dataset
# Replace "target" with your actual label column
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train the model
model = SmartKNN(k=5)
model.fit(X, y)

# Predict for a single sample
sample = X.iloc[0]
pred = model.predict(sample)
print("Prediction:", pred)
```

SmartKNN automatically:

* Normalizes features
* Learns weights
* Filters useless features
* Cleans NaN / Inf values
* Prepares optimized distance functions

---

## 🔮 Predict on Multiple Samples

```python
# Predict on first 10 rows
preds = model.predict(X.iloc[:10])
print(preds)
```

---

##  How It Works (Simple Explanation)

SmartKNN improves KNN by:

1. **Finding which features matter** using MSE, MI, and Random Forest scoring.
2. **Removing useless features** based on weights.
3. **Normalizing** everything to prevent scale bias.
4. **Applying weighted Euclidean distance** instead of plain distance.
5. Using NumPy-optimized batch computations for fast inference.

This results in:

* Higher accuracy
* Faster predictions
* Lower noise sensitivity
* Adaptive feature selection

---

## 🔬 API Overview

### **Initialize**

```python
model = SmartKNN(k=5, weight_threshold=0.05)
```

### **Fit**

```python
model.fit(X, y)
```

### **Predict**

```python
pred = model.predict(sample)
```

### **Neighbors**

```python
idx, dists = model.kneighbors(sample)
```

### **Inspect internals**

```python
model.weights_        # Final feature weights
model.feature_mask_   # Which features were kept
model.X_.shape        # Reduced feature matrix
```

---

##  Project Structure

```
smart_knn/
 ├── base_knn.py
 ├── distance.py
 ├── weight_learning.py
 ├── data_processing.py
 ├── utils.py
 ├── evaluation.py
 ├── adaptive_k.py (future)
 ├── prototypes.py (future)
 └── signatures.py (future)
```

Additional documentation in:

* `docs/design.md` — internal architecture
* `docs/theory.md` — math and algorithms
* `docs/usage.md` — extended usage examples
* `docs/roadmap.md` — future improvements

---

##  Roadmap

* Adaptive-K optimization
* Prototype compression
* Distance signatures
* GPU acceleration
* Incremental learning support
* Batch offline inference

---

##  License

This project is licensed under the MIT License. See `LICENSE` file.

---

##  Contributing

PRs, suggestions, and feature requests are welcome! If you like the project, star it on GitHub.

---

##  Support

Have issues or questions? Open an issue on GitHub or message your friendly AI assistant 
