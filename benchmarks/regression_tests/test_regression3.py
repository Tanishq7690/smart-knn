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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_openml, make_regression

from smart_knn import SmartKNN

SEED = 42
TEST_SIZE = 0.25
N_JOBS = -1
RANDOM_STATE = SEED

RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 150
KNN_K = 5

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

    X_num = df[num_cols].copy()
    enc_obj = None
    if len(cat_cols) > 0:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(df[cat_cols])
        X_cat = pd.DataFrame(X_cat, index=df.index, columns=enc.get_feature_names_out(cat_cols))
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        enc_obj = enc
    else:
        X = X_num.reset_index(drop=True)

    return X.values.astype(np.float32), enc_obj


def preprocess_pipeline(X_raw, y_raw):
    if isinstance(X_raw, pd.DataFrame):
        X_arr, enc = preprocess_dataframe(X_raw)
    else:
        X_arr = to_float32(X_raw)
        enc = None

    y = to_float32(y_raw).reshape(-1)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_arr).astype(np.float32)
    X_imp = np.clip(X_imp, -1e9, 1e9)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

    return X_scaled, y, {"imputer": imputer, "scaler": scaler, "encoder": enc}


def safe_fetch_openml(name, as_frame=True, max_samples=10000):
    try:
        data = fetch_openml(name, as_frame=as_frame)
        X, y = data.data, data.target
        
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=SEED)
            y = y.loc[X.index] if hasattr(y, "loc") else np.asarray(y)[:len(X)]
        return X, y
    except Exception as e:
        print(f"[WARN] Failed to fetch {name}: {e}. Falling back to synthetic data.")
        X, y = make_regression(n_samples=min(max_samples, 1000), n_features=10, noise=0.1, random_state=SEED)
        return X, y


def run_all_models_on_dataset(name, X_raw, y_raw, results):
    print(f"\n DATASET: {name} ")
    try:
        X, y, _ = preprocess_pipeline(X_raw, y_raw)
    except Exception as e:
        print(f"[ERROR preprocess {name}] {e}")
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
            random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=KNN_K, n_jobs=N_JOBS)
    }

    for name_model, mdl in models.items():
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

            print(f"{name_model:12s} | MSE={mse:.5f}  MAE={mae:.5f}  R2={r2:.5f}  Train={train_time:.3f}s  Pred={pred_time:.3f}s")

            results.append({
                "dataset": name,
                "model": name_model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "train_time": train_time,
                "pred_time": pred_time
            })
        except Exception as e:
            print(f"[ERROR {name_model}]: {e}")

    try:
        if X_train.shape[0] <= 10000: 
            sm = SmartKNN(k=KNN_K)
            t0 = time.time()
            sm.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred_sm = sm.predict(X_test)
            pred_time = time.time() - t0

            mse = float(mean_squared_error(y_test, y_pred_sm))
            mae = float(mean_absolute_error(y_test, y_pred_sm))
            r2 = float(r2_score(y_test, y_pred_sm))

            print(f"{'SmartKNN':12s} | MSE={mse:.5f}  MAE={mae:.5f}  R2={r2:.5f}  Train={train_time:.3f}s  Pred={pred_time:.3f}s")

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
            print("[SmartKNN skipped] Dataset too large")
    except Exception as e:
        print(f"[ERROR SmartKNN]: {e}")



datasets = [
    "house_prices_nominal",
    "OnlineNewsPopularity",
    "sml",
    "bikesharing",
    "yacht_hydrodynamics",
    "naval-propulsion-plant",
    "real_estate_valuation",
    "servo",
    "automobile",
    "qsar_aquatic_toxicity",
]

results = []

for ds_name in datasets:
    X_ds, y_ds = safe_fetch_openml(ds_name)
    run_all_models_on_dataset(ds_name, X_ds, y_ds, results)

df = pd.DataFrame(results)
df.to_csv("regression_results_batch3.csv", index=False)
print("\nBatch-3 complete! Saved → regression_results_batch3.csv")
