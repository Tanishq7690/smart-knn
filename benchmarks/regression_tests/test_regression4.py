import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_openml, make_regression
from sklearn import datasets as skd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

from smart_knn import SmartKNN

SEED = 42
TEST_SIZE = 0.25
N_JOBS = -1
RANDOM_STATE = SEED

RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 150
KNN_K = 5

SMARTKNN_MAX_TRAIN = 20000 

warnings.filterwarnings("ignore")


def ensure_numpy(X):
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X)

def to_float32(X):
    return ensure_numpy(X).astype(np.float32, copy=False)

def preprocess_dataframe(df):
   
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    X_num = df[num_cols].copy() if len(num_cols) > 0 else pd.DataFrame(index=df.index)
    if len(cat_cols) > 0:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(df[cat_cols])
        X_cat = pd.DataFrame(X_cat, index=df.index, columns=enc.get_feature_names_out(cat_cols))
        X_full = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_full = X_num.reset_index(drop=True)

    return X_full.values.astype(np.float32)


def preprocess_pipeline(X_raw, y_raw):
    if isinstance(X_raw, pd.DataFrame):
        X_arr = preprocess_dataframe(X_raw)
    else:
        X_arr = to_float32(X_raw)

    y = ensure_numpy(y_raw).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_arr).astype(np.float32)

    X_imp = np.clip(X_imp, -1e9, 1e9)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

    return X_scaled, y, {"imputer": imputer, "scaler": scaler}


def safe_fetch_openml(name, as_frame=True, max_samples=20000):

    try:
        data = fetch_openml(name, as_frame=as_frame)
        if as_frame:
            X, y = data.data, data.target
        else:
            X, y = data.data, data.target
        if hasattr(X, "shape") and X.shape[0] > max_samples:
            X = X.sample(n=max_samples, random_state=SEED)
            if hasattr(y, "loc"):
                y = y.loc[X.index]
            else:
                y = np.asarray(y)[:len(X)]
        return X, y
    except Exception as e:
        print(f"[WARN] fetch_openml('{name}') failed: {e}. Falling back to synthetic regression.")
        X_synth, y_synth = make_regression(n_samples=min(5000, max_samples), n_features=10, noise=0.5, random_state=SEED)
        return pd.DataFrame(X_synth), y_synth


def run_all_models_on_dataset(name, X_raw, y_raw, results):
    print(f"\n===== DATASET: {name} =====")
    try:
        X, y, _ = preprocess_pipeline(X_raw, y_raw)
    except Exception as e:
        print(f"[ERROR preprocess {name}] {e}")
        return

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        print(f"[SKIP {name}] shape mismatch after preprocess: X={X.shape}, y={getattr(y,'shape',None)}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    models = {
        "LinearRegression": LinearRegression(),
        "SVR_RBF": SVR(kernel="rbf", C=1.0),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsRegressor(n_neighbors=KNN_K, n_jobs=N_JOBS)
    }


    for model_name, mdl in models.items():
        try:
            t0 = time.time()
            mdl.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = mdl.predict(X_test)
            pred_time = time.time() - t0

            y_pred = to_float32(y_pred)

            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            print(f"{model_name:14s} | MSE={mse:.5f}  MAE={mae:.5f}  R2={r2:.5f}  train={train_time:.3f}s pred={pred_time:.3f}s")

            results.append({
                "dataset": name,
                "model": model_name,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "train_time": train_time,
                "pred_time": pred_time
            })
        except Exception as e:
            print(f"[ERROR {name} {model_name}] {e}")
            results.append({
                "dataset": name,
                "model": model_name,
                "mse": None, "mae": None, "r2": None,
                "train_time": None, "pred_time": None,
                "error": str(e)
            })


    try:
        if X_train.shape[0] <= SMARTKNN_MAX_TRAIN:
            sm = SmartKNN(k=KNN_K)
            t0 = time.time()
            sm.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred_sm = sm.predict(X_test)
            pred_time = time.time() - t0

            y_pred_sm = to_float32(y_pred_sm)

            mse = float(mean_squared_error(y_test, y_pred_sm))
            mae = float(mean_absolute_error(y_test, y_pred_sm))
            r2 = float(r2_score(y_test, y_pred_sm))

            print(f"{'SmartKNN':14s} | MSE={mse:.5f}  MAE={mae:.5f}  R2={r2:.5f}  train={train_time:.3f}s pred={pred_time:.3f}s")

            results.append({
                "dataset": name,
                "model": "SmartKNN",
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "train_time": train_time,
                "pred_time": pred_time
            })
        else:
            print(f"[SmartKNN skipped] train_size={X_train.shape[0]} > {SMARTKNN_MAX_TRAIN}")
            results.append({
                "dataset": name,
                "model": "SmartKNN",
                "mse": None, "mae": None, "r2": None,
                "train_time": None, "pred_time": None,
                "note": "skipped_large_dataset"
            })
    except Exception as e:
        print(f"[ERROR SmartKNN {name}] {e}")
        results.append({
            "dataset": name,
            "model": "SmartKNN",
            "mse": None, "mae": None, "r2": None,
            "train_time": None, "pred_time": None,
            "error": str(e)
        })



dataset_names = [
    "sgemm_product",            
    "diamonds",                 
    "houses",                  
    "cpu_act",                 
    "cpu_small",              
    "machine_cpu",          
    "pol",                     
    "eeg_eye_state",          
    "socmob",                  
    "ailerons",                
    "elevators",              
    "krr"                   
]


results = []

for name in dataset_names:
    X_ds, y_ds = safe_fetch_openml(name, as_frame=True, max_samples=15000)

    try:
        if hasattr(y_ds, "astype"):
            y_try = pd.to_numeric(y_ds, errors="coerce")
            if y_try.isna().all():
            
                print(f"[WARN] target for {name} non-numeric → falling back synthetic")
                X_ds, y_ds = make_regression(n_samples=3000, n_features=10, noise=0.5, random_state=SEED)
            else:
                y_ds = y_try
    except Exception:
        pass

    run_all_models_on_dataset(name, X_ds, y_ds, results)


df = pd.DataFrame(results)
out_file = "regression_results_batch4.csv"
df.to_csv(out_file, index=False)
print(f"\nBatch-4 complete — results saved to {out_file}")
