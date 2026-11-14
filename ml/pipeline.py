import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import TrafficFeaturizer
from ml.regressors import SeasonalMeanRegressor, TorchMLPRegressor

SEED = 42
SPLITS = {"train": ("2015-11-01", "2017-03-01"),
          "validation": ("2017-03-01", "2017-05-01"),
          "test": ("2017-05-01", "2017-07-01")}

MODEL_LABELS = {"seasonal_mean": "Seasonal mean (baseline)", "ridge": "Ridge regression",
                "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
                "mlp": "MLP (PyTorch)"}

GRIDS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [100, 200], "max_depth": [12, 20], "min_samples_leaf": [1, 3]},
    "gradient_boosting": {"learning_rate": [0.05, 0.1], "max_depth": [6, 10], "max_iter": [200, 400]},
    "mlp": {"hidden": [(64, 32), (128, 64)], "lr": [1e-3, 3e-3]},
}


def split_by_time(df):
    out = {}
    for name, (a, b) in SPLITS.items():
        out[name] = df[(df["datetime"] >= a) & (df["datetime"] < b)].sort_values("datetime").reset_index(drop=True)
    return out


def make_model(name, params, n_jobs=1):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    if name == "random_forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=n_jobs, **params)
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=SEED, **params)
    if name == "mlp":
        return TorchMLPRegressor(seed=SEED, **params)
    if name == "seasonal_mean":
        return SeasonalMeanRegressor()
    raise ValueError(name)


def metrics(y, p):
    mse = float(mean_squared_error(y, p))
    return {"mse": round(mse, 3), "rmse": round(mse ** 0.5, 3),
            "mae": round(float(mean_absolute_error(y, p)), 3), "r2": round(float(r2_score(y, p)), 4)}


def fit_predict(name, params, train_df, eval_df, n_jobs=1):
    feat = TrafficFeaturizer(random_state=SEED).fit(train_df)
    Xtr, ytr = feat.transform(train_df), feat.cap_target(train_df)
    Xev = feat.transform(eval_df)
    model = make_model(name, params, n_jobs)
    t = time.time()
    if name == "seasonal_mean":
        jcols = list(range(len(feat.feature_names) - len(feat.junctions_), len(feat.feature_names)))
        model.fit(Xtr, ytr, jcols)
    else:
        model.fit(Xtr, ytr)
    secs = time.time() - t
    return feat, model, np.clip(model.predict(Xev), 0, None), np.clip(model.predict(Xtr), 0, None), secs


def time_series_cv(name, params, train_df, n_splits=5, n_jobs=1):
    tss = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for k, (a, b) in enumerate(tss.split(train_df)):
        tr, va = train_df.iloc[a], train_df.iloc[b]
        _, _, p, _, secs = fit_predict(name, params, tr, va, n_jobs)
        m = metrics(va["vehicles"].values, p)
        m.update({"fold": k + 1, "train_rows": int(len(tr)), "val_rows": int(len(va)),
                  "val_from": str(va["datetime"].min().date()), "val_to": str(va["datetime"].max().date()),
                  "seconds": round(secs, 2)})
        folds.append(m)
    return folds


def grid(name):
    keys = list(GRIDS[name])
    for values in itertools.product(*(GRIDS[name][k] for k in keys)):
        yield dict(zip(keys, values))
