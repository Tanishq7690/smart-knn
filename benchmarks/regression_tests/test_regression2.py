import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from smart_knn import SmartKNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets as skd
from sklearn.datasets import fetch_openml


SEED = 42
TEST_SIZE = 0.25
N_JOBS = -1
RANDOM_STATE = SEED

RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 150
KNN_K = 5

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)



def ensure_numpy(X):
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X)


def to_float32(X):
    X = ensure_numpy(X)
    return X.astype(np.float32, copy=False)


def preprocess_dataframe(X_df):

    if isinstance(X_df, pd.DataFrame):
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X_df[num_cols].copy() if len(num_cols) > 0 else pd.DataFrame(index=X_df.index)
        if len(cat_cols) > 0:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_cat = enc.fit_transform(X_df[cat_cols])
            X_cat = pd.DataFrame(X_cat, index=X_df.index, columns=enc.get_feature_names_out(cat_cols))
            X_full = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
            enc_obj = enc
        else:
            X_full = X_num.reset_index(drop=True)
            enc_obj = None
        return X_full.values.astype(np.float32), {"num_cols": num_cols, "cat_cols": cat_cols, "encoder": enc_obj}
    else:
        X_arr = to_float32(X_df)
        return X_arr, {"num_cols": None, "cat_cols": None, "encoder": None}


def preprocess_pipeline(X_raw, y_raw):
 
    if isinstance(X_raw, pd.DataFrame):
        X_arr, meta = preprocess_dataframe(X_raw)
    else:
        X_arr = ensure_numpy(X_raw)
        meta = {"num_cols": None, "cat_cols": None, "encoder": None}

    y = ensure_numpy(y_raw).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_arr).astype(np.float32)

    X_imp = np.clip(X_imp, -1e9, 1e9)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

    meta.update({"imputer": imputer, "scaler": scaler})
    return X_scaled, y, meta


def safe_fetch_openml(name, as_frame=True):

    try:
        data = fetch_openml(name, as_frame=as_frame)
        if as_frame:
            X = data.data
            y = data.target
        else:
            X = data.data
            y = data.target
        return X, y
    except Exception as e:
        print(f"[WARN] fetch_openml('{name}') failed: {e}. Falling back to synthetic dataset.")
      
        if "energy" in name.lower():
            X, y = skd.make_regression(n_samples=1000, n_features=8, noise=0.2, random_state=SEED)
        elif "elevators" in name.lower():
            X, y = skd.make_regression(n_samples=5000, n_features=18, noise=1.0, random_state=SEED)
        else:
            X, y = skd.make_regression(n_samples=2000, n_features=10, noise=0.5, random_state=SEED)
        return pd.DataFrame(X), y


def run_all_models_on_dataset(ds_name, X_raw, y_raw, results_list):
    print(f"\n DATASET: {ds_name} ")
    try:
        X, y, meta = preprocess_pipeline(X_raw, y_raw)
    except Exception as e:
        print(f"[ERROR preprocess {ds_name}] {e}")
        return

    if X.ndim != 2:
        print(f"[SKIP {ds_name}] X not 2-D after preprocessing: shape={X.shape}")
        return
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        print(f"[SKIP {ds_name}] mismatch: X rows {X.shape[0]} vs y length {y.shape[0]}")
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

    for model_name, model in models.items():
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - t0

            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            print(f"{model_name:12s} | MSE: {mse:.5f}  MAE: {mae:.5f}  R2: {r2:.5f}  "
                  f"train: {train_time:.3f}s pred: {predict_time:.3f}s")

            results_list.append({
                "dataset": ds_name,
                "model": model_name,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "train_time": train_time,
                "predict_time": predict_time
            })
        except Exception as e:
            print(f"[ERROR {ds_name} {model_name}] {e}")
            results_list.append({
                "dataset": ds_name,
                "model": model_name,
                "mse": None,
                "mae": None,
                "r2": None,
                "train_time": None,
                "predict_time": None,
                "error": str(e)
            })

 
    try:
        t0 = time.time()
        sm = SmartKNN(k=KNN_K)
        sm.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred_sm = sm.predict(X_test)
        predict_time = time.time() - t0

        y_pred_sm = np.nan_to_num(y_pred_sm, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

        mse = float(mean_squared_error(y_test, y_pred_sm))
        mae = float(mean_absolute_error(y_test, y_pred_sm))
        r2 = float(r2_score(y_test, y_pred_sm))

        print(f"{'SmartKNN':12s} | MSE: {mse:.5f}  MAE: {mae:.5f}  R2: {r2:.5f}  "
              f"train: {train_time:.3f}s pred: {predict_time:.3f}s")

        results_list.append({
            "dataset": ds_name,
            "model": "SmartKNN",
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "predict_time": predict_time
        })
    except Exception as e:
        print(f"[ERROR {ds_name} SmartKNN] {e}")
        results_list.append({
            "dataset": ds_name,
            "model": "SmartKNN",
            "mse": None,
            "mae": None,
            "r2": None,
            "train_time": None,
            "predict_time": None,
            "error": str(e)
        })


datasets = []


candidates = [
    ("Energy Efficiency", "energy_efficiency"),
    ("Elevators", "elevators"),
    ("kin8nm", "kin8nm"),
    ("YearPredictionMSD_sample", "YearPredictionMSD"),  
    ("CPU small", "cpu_small"),
    ("Slump Test", "slump_test"),       
    ("Parkinsons Telemonitoring", "parkinsons_updrs"),
    ("Friedman Synthetic", None),       
    ("Skillcraft (skillcraft-2014)", "skillcraft-2014"),
    ("CASP-like (re-use small protein)", "CASP"),       
    ("Energies (another energy)", "ENB2012"),
    ("Concrete (another)", "Concrete_Compressive_Strength")  
]

for title, openml_name in candidates:
    if openml_name is None:
       
        X_syn, y_syn = skd.make_friedman1(n_samples=2000, n_features=10, random_state=SEED)
        datasets.append((title, X_syn.astype(np.float32), y_syn.astype(np.float32)))
        continue

    try:
        X_load, y_load = safe_fetch_openml(openml_name, as_frame=True)
       
        if openml_name.lower() == "yearpredictionmsd" or title.lower().startswith("year"):
           
            try:
                X_df = X_load.sample(n=min(10000, len(X_load)), random_state=SEED) if isinstance(X_load, pd.DataFrame) else pd.DataFrame(X_load).sample(n=10000, random_state=SEED)
                y_sample = y_load.loc[X_df.index] if hasattr(y_load, "loc") else np.asarray(y_load)[:len(X_df)]
                datasets.append((title + " (sampled)", X_df, y_sample))
            except Exception:
              
                X_s, y_s = skd.make_regression(n_samples=2000, n_features=90, noise=1.0, random_state=SEED)
                datasets.append((title + " (synthetic)", X_s, y_s))
        else:
            datasets.append((title, X_load, y_load))
    except Exception as e:
        print(f"[WARN adding {title}] fallback to synthetic due to: {e}")
        X_f, y_f = skd.make_regression(n_samples=2000, n_features=10, noise=0.5, random_state=SEED)
        datasets.append((title + " (fallback)", X_f, y_f))


datasets = datasets[:12]


results = []
for ds_name, X_ds, y_ds in datasets:
    try:
        run_all_models_on_dataset(ds_name, X_ds, y_ds, results)
    except Exception as e:
        print(f"[FATAL {ds_name}] {e}")

# Save
df_res = pd.DataFrame(results)
out_file = "regression_results_batch2.csv"
df_res.to_csv(out_file, index=False)
print(f"\nBatch 2 done — results saved to {out_file}")
