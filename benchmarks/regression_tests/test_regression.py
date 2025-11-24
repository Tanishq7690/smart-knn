import time
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
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

    y = ensure_numpy(y_raw)
    y = y.reshape(-1)

    y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

 
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_arr).astype(np.float32)

    X_imp = np.clip(X_imp, -1e9, 1e9)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

    return X_scaled, y.astype(np.float32), {"imputer": imputer, "scaler": scaler, **(meta if 'meta' in locals() else {})}


def run_all_models_on_dataset(name, X_raw, y_raw, results_list):
    print(f"\n DATASET: {name}")
    try:
        X, y, prep_meta = preprocess_pipeline(X_raw, y_raw)
    except Exception as e:
        print(f"[ERROR preprocess {name}] {e}")
        return

    if X.ndim != 2:
        print(f"[SKIP {name}] X not 2-D after preprocessing: shape={X.shape}")
        return
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        print(f"[SKIP {name}] y mismatch shape X rows {X.shape[0]} vs y shape {y.shape}")
        return


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    preds = {}

    models = {}

    models["LinearRegression"] = LinearRegression()

    models["SVR_RBF"] = SVR(kernel="rbf", C=1.0)

    models["DecisionTree"] = DecisionTreeRegressor(random_state=RANDOM_STATE)

    models["RandomForest"] = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)

    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    models["KNN"] = KNeighborsRegressor(n_neighbors=KNN_K, n_jobs=N_JOBS)

    for name_model, mdl in models.items():
        try:
            t0 = time.time()
            mdl.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = mdl.predict(X_test)
            predict_time = time.time() - t0

          
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

            mse = float(mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            print(f"{name_model:12s} | MSE: {mse:.5f}  MAE: {mae:.5f}  R2: {r2:.5f}  "
                  f"train_time: {train_time:.3f}s predict_time: {predict_time:.3f}s")

            results_list.append({
                "dataset": name,
                "model": name_model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "train_time": train_time,
                "predict_time": predict_time
            })

        except Exception as e:
            print(f"[ERROR {name} {name_model}] {e}")
            results_list.append({
                "dataset": name,
                "model": name_model,
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
              f"train_time: {train_time:.3f}s predict_time: {predict_time:.3f}s")

        results_list.append({
            "dataset": name,
            "model": "SmartKNN",
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "predict_time": predict_time
        })

    except Exception as e:
        print(f"[ERROR {name} SmartKNN] {e}")
        results_list.append({
            "dataset": name,
            "model": "SmartKNN",
            "mse": None,
            "mae": None,
            "r2": None,
            "train_time": None,
            "predict_time": None,
            "error": str(e)
        })


datasets = []

datasets.append(("California Housing", fetch_california_housing().data, fetch_california_housing().target))

datasets.append(("Diabetes", load_diabetes().data, load_diabetes().target))


bh = fetch_openml(name="boston", version=1, as_frame=False)
datasets.append(("Boston Housing", bh.data, bh.target.astype(np.float32)))

abalone = fetch_openml("abalone", as_frame=True)
X_ab = abalone.data.copy()
y_ab = abalone.target.astype(np.float32).values
datasets.append(("Abalone", X_ab, y_ab))


bike = fetch_openml("Bike_Sharing_Demand", as_frame=True)
X_bike = bike.data.select_dtypes(include=[np.number])
if isinstance(bike.target, pd.Series) and bike.target.dtype != object:
    y_bike = bike.target.astype(np.float32).values
else:
    if "count" in bike.data.columns:
        y_bike = bike.data["count"].astype(np.float32).values
    elif "cnt" in bike.data.columns:
        y_bike = bike.data["cnt"].astype(np.float32).values
    else:
        raise ValueError("Bike dataset missing count/cnt")
datasets.append(("Bike Sharing Demand", X_bike, y_bike))

airfoil = fetch_openml("airfoil_self_noise", as_frame=False)
datasets.append(("Airfoil Self Noise", airfoil.data, airfoil.target.astype(np.float32)))


concrete = fetch_openml("Concrete_Compressive_Strength", as_frame=False)
datasets.append(("Concrete Strength", concrete.data, concrete.target.astype(np.float32)))

wine_red = fetch_openml("wine-quality-red", as_frame=False)
datasets.append(("Wine Quality - Red", wine_red.data, wine_red.target.astype(np.float32)))

wine_white = fetch_openml("wine-quality-white", as_frame=False)
datasets.append(("Wine Quality - White", wine_white.data, wine_white.target.astype(np.float32)))



superc = fetch_openml("superconductivity", as_frame=True)
datasets.append(("Superconductivity", superc.data.values, superc.target.values.astype(np.float32)))

try:
    insurance = fetch_openml("insurance", as_frame=True)
    X_ins = insurance.data.copy()
    y_ins = insurance.target.astype(np.float32).values
    datasets.append(("Medical Insurance", X_ins, y_ins))
except Exception:
    import sklearn.datasets as skd
    X_ins, y_ins = skd.make_regression(n_samples=1000, n_features=8, noise=0.2, random_state=SEED)
    datasets.append(("Medical Insurance (synthetic)", X_ins, y_ins.astype(np.float32)))


results = []
for ds_name, X_ds, y_ds in datasets:
    try:
        run_all_models_on_dataset(ds_name, X_ds, y_ds, results)
    except Exception as e:
        print(f"[FATAL {ds_name}] {e}")

df_res = pd.DataFrame(results)
out_file = "regression_results_batch1.csv"
df_res.to_csv(out_file, index=False)
print(f"\nAll done — results saved to {out_file}")
