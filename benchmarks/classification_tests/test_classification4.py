import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from smart_knn import SmartKNN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.25
KNN_K = 5

MAX_DATASET_ROWS = 20000
SMARTKNN_MAX_ROWS = 15000


def make_ohe():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def ensure_numpy(X):
    if hasattr(X, "values"):
        return X.values
    return np.asarray(X)


def limit_rows(X, y, limit=MAX_DATASET_ROWS):
    if len(X) > limit:
        idx = np.random.RandomState(SEED).choice(len(X), limit, replace=False)
        return X[idx], y[idx]
    return X, y


def preprocess_dataframe(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    X_num = df[num_cols].copy()

    if len(cat_cols) > 0:
        enc = make_ohe()
        X_cat = enc.fit_transform(df[cat_cols])
        X_cat = pd.DataFrame(
            X_cat, index=df.index, columns=enc.get_feature_names_out(cat_cols)
        )
        X = pd.concat([
            X_num.reset_index(drop=True),
            X_cat.reset_index(drop=True)
        ], axis=1)
    else:
        X = X_num.reset_index(drop=True)

    return X.astype(np.float32).values


def preprocess_pipeline(X_raw, y_raw):
    if isinstance(X_raw, pd.DataFrame):
        X_arr = preprocess_dataframe(X_raw)
    else:
        X_arr = ensure_numpy(X_raw).astype(np.float32)


    y_raw_np = ensure_numpy(y_raw).astype(str)
    y = LabelEncoder().fit_transform(y_raw_np)

    X_arr, y = limit_rows(X_arr, y, MAX_DATASET_ROWS)

    X_imp = SimpleImputer(strategy="most_frequent").fit_transform(X_arr)
    X_scaled = StandardScaler().fit_transform(X_imp).astype(np.float32)

    return X_scaled, y


def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_models(name, X_raw, y_raw, results):
    print(f"\n DATASET: {name}")

    X, y = preprocess_pipeline(X_raw, y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "SVM_RBF": SVC(kernel="rbf", C=2.0, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=KNN_K),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    for model_name, model in models.items():
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)
            pred_time = time.time() - t0

            scores = evaluate_classification(y_test, y_pred)
            print(f"{model_name:15s} | ACC={scores['accuracy']:.4f} F1={scores['f1']:.4f}")

            results.append({
                "dataset": name,
                "model": model_name,
                **scores,
                "train_time": train_time,
                "pred_time": pred_time,
            })

        except Exception as e:
            print(f"[ERROR {model_name}] {e}")

    try:
        if len(X_train) > SMARTKNN_MAX_ROWS:
            print("[SmartKNN SKIPPED] Dataset too large")
            return

        sm = SmartKNN(k=KNN_K)

        t0 = time.time()
        sm.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_raw = sm.predict(X_test)
        pred_time = time.time() - t0

        classes = np.unique(y_train)
        y_pred = np.array([classes[np.argmin(np.abs(classes - p))] for p in y_pred_raw])

        scores = evaluate_classification(y_test, y_pred)
        print(f"{'SmartKNN':15s} | ACC={scores['accuracy']:.4f} F1={scores['f1']:.4f}")

        results.append({
            "dataset": name,
            "model": "SmartKNN",
            **scores,
            "train_time": train_time,
            "pred_time": pred_time,
        })

    except Exception as e:
        print(f"[ERROR SmartKNN] {e}")


dataset_list = [
    ("Blood Transfusion", "blood-transfusion-service-center"),
    ("Car Evaluation", "car"),
    ("Yeast", "yeast"),
    ("Mammographic Mass", "mammographic-mass"),
    ("Ozone Level", "ozone-level-8hr"),
    ("Magic Telescope", "magic"),
    ("SPECT Heart", "spect-heart"),
    ("Spam Base", "spambase"),
    ("Shuttle", "shuttle"),
    ("Page Blocks", "page-blocks"),
]

results = []

for ds_name, openml_name in dataset_list:
    try:
        data = fetch_openml(openml_name, as_frame=True)
        run_models(ds_name, data.data, data.target, results)
    except Exception as e:
        print(f"[ERROR loading {ds_name}] {e}")

df = pd.DataFrame(results)
df.to_csv("classification_results_batch4.csv", index=False)

print("\nBatch-4 complete -> Saved to classification_results_batch4.csv")
