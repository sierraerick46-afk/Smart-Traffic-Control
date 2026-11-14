import argparse
import csv
import json
import os
import time
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.features import TrafficFeaturizer
from ml.pipeline import (GRIDS, MODEL_LABELS, SEED, SPLITS, fit_predict, grid, make_model,
                         metrics, split_by_time, time_series_cv)
from models.data_loader import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
FIG = os.path.join(ROOT, "static", "img", "crisp")
COLORS = {1: "#00482B", 2: "#E0A526", 3: "#D64545", 4: "#3A8FB7"}
FAMILY_COLORS = {"seasonal_mean": "#A3AEA8", "ridge": "#3A8FB7", "random_forest": "#00482B",
                 "gradient_boosting": "#2E9E5B", "mlp": "#E0A526"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def data_quality(raw_path, df, splits, feat_train):
    raw = pd.read_csv(raw_path)
    raw_dt = pd.to_datetime(raw["DateTime"], errors="coerce")
    before = {
        "rows": int(len(raw)), "columns": list(raw.columns),
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "invalid_dates": int(raw_dt.isna().sum()),
        "negative_counts": int((pd.to_numeric(raw["Vehicles"], errors="coerce") < 0).sum()),
        "duplicated_date_junction": int(raw.assign(d=raw_dt).duplicated(["d", "Junction"]).sum()),
        "date_from": str(raw_dt.min()), "date_to": str(raw_dt.max()),
        "vehicles_describe": {k: round(float(v), 2) for k, v in raw["Vehicles"].describe().items()},
        "sample": raw.head(5).to_dict(orient="records"),
    }
    train = splits["train"]
    capped = feat_train.cap_target(train)
    outliers = {}
    for j, g in train.groupby("junction"):
        cap = feat_train.caps_[int(j)]
        outliers[int(j)] = {"cap_p99_5": round(cap, 1), "rows_above_cap": int((g["vehicles"] > cap).sum()),
                            "max_before": int(g["vehicles"].max()), "max_after": round(cap, 1)}
    X = feat_train.transform(train.head(5))
    after = {
        "rows": int(len(df)), "feature_names": feat_train.feature_names,
        "train_target_describe_before_cap": {k: round(float(v), 2) for k, v in train["vehicles"].describe().items()},
        "train_target_describe_after_cap": {k: round(float(v), 2) for k, v in pd.Series(capped).describe().items()},
        "outliers_by_junction": outliers,
        "sample_features": [dict(zip(feat_train.feature_names, [round(float(v), 3) for v in row])) for row in X],
        "sample_raw_for_features": train.head(5)[["datetime", "junction", "vehicles"]].astype(str).to_dict(orient="records"),
    }
    return {"before": before, "after": after}


def figures(df, splits, feat, results, final, test_pred):
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 11})

    m = df.set_index("datetime").groupby("junction")["vehicles"].resample("MS").mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, g in m.groupby("junction"):
        ax.plot(g["datetime"], g["vehicles"], color=COLORS[j], lw=2, label=f"Junction {j}")
    ax.set_title("Monthly average vehicles per hour")
    ax.set_ylabel("vehicles / hour"); ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "monthly_trend.png"), dpi=150); plt.close(fig)

    p = df.groupby(["junction", "hour"])["vehicles"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j in p.columns:
        ax.plot(p.index, p[j], color=COLORS[j], lw=2, label=f"Junction {j}")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day"); ax.set_ylabel("vehicles / hour")
    ax.set_title("Average vehicles by hour of day"); ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hourly_profile.png"), dpi=150); plt.close(fig)

    tr = splits["train"]
    capped = feat.cap_target(tr)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    data_b = [tr.loc[tr["junction"] == j, "vehicles"].values for j in feat.junctions_]
    data_a = [capped[(tr["junction"] == j).values] for j in feat.junctions_]
    for ax, data, title in [(axes[0], data_b, "Before (raw training target)"), (axes[1], data_a, "After (capped at P99.5)")]:
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops={"markersize": 2, "alpha": 0.4})
        for patch, j in zip(bp["boxes"], feat.junctions_):
            patch.set_facecolor(COLORS[j]); patch.set_alpha(0.75)
        ax.set_xticklabels([f"J{j}" for j in feat.junctions_]); ax.set_title(title)
    axes[0].set_ylabel("vehicles / hour")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "outliers_before_after.png"), dpi=150); plt.close(fig)

    daily = df.set_index("datetime")["vehicles"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(daily.index, daily.values, color="#5D6862", lw=0.8)
    for k, ((name, (a, b)), col) in enumerate(zip(SPLITS.items(), ["#00482B", "#B7791F", "#D64545"])):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.15)
        n = len(splits[name])
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, daily.max() * (1.12 - 0.13 * k),
                f"{name} · {n:,} rows", ha="center", va="top", fontsize=8.5, color=col, fontweight="bold")
    ax.set_ylim(top=daily.max() * 1.15)
    ax.set_title("Time-ordered partitions (total vehicles per day)"); ax.set_ylabel("vehicles / day")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "splits_timeline.png"), dpi=150); plt.close(fig)

    fams = [f for f in results["families"]]
    labels = [MODEL_LABELS[f].replace(" (baseline)", "\n(baseline)").replace(" (PyTorch)", "\n(PyTorch)") for f in fams]
    cv_mean = [results["families"][f]["cv_rmse_mean"] for f in fams]
    cv_std = [results["families"][f]["cv_rmse_std"] for f in fams]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, cv_mean, yerr=cv_std, capsize=4, color=[FAMILY_COLORS[f] for f in fams])
    for b, v, e in zip(bars, cv_mean, cv_std):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.4, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (vehicles / hour)"); ax.set_title("Time-series cross-validation (5 folds, best configuration)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_results.png"), dpi=150); plt.close(fig)

    tr_rmse = [results["families"][f]["train"]["rmse"] for f in fams]
    va_rmse = [results["families"][f]["validation"]["rmse"] for f in fams]
    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, tr_rmse, w, label="train", color="#A3AEA8")
    ax.bar(x + w / 2, va_rmse, w, label="validation", color="#00482B")
    for i, (a, b) in enumerate(zip(tr_rmse, va_rmse)):
        ax.text(i - w / 2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8); ax.text(i + w / 2, b + 0.3, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("RMSE (vehicles / hour)")
    ax.set_title("Train vs validation error (overfitting check)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "overfitting.png"), dpi=150); plt.close(fig)

    te = splits["test"].assign(pred=test_pred)
    week = te[(te["datetime"] >= "2017-06-05") & (te["datetime"] < "2017-06-12")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    for ax, j in zip(axes.ravel(), feat.junctions_):
        g = week[week["junction"] == j]
        ax.plot(g["datetime"], g["vehicles"], color="#5D6862", lw=1.2, label="actual")
        ax.plot(g["datetime"], g["pred"], color=COLORS[j], lw=1.6, label="predicted")
        ax.set_title(f"Junction {j}"); ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Independent test set: actual vs predicted (5–11 June 2017)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "test_actual_vs_pred.png"), dpi=150); plt.close(fig)

    pj = final["test_by_junction"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    js = sorted(pj, key=int)
    ax.bar([f"J{j}" for j in js], [pj[j]["mae"] for j in js], color=[COLORS[int(j)] for j in js])
    for i, j in enumerate(js):
        ax.text(i, pj[j]["mae"] + 0.1, f"MAE {pj[j]['mae']:.1f}\nR² {pj[j]['r2']:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("MAE (vehicles / hour)"); ax.set_title("Test error by intersection (bias check)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "test_by_junction.png"), dpi=150); plt.close(fig)

    imp = final.get("feature_importance")
    if imp:
        items = sorted(imp.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh([k for k, _ in items], [v for _, v in items], color="#00482B")
        ax.set_xlabel("increase in MSE when the feature is shuffled (validation)")
        ax.set_title("Permutation importance of the selected model")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train and select the traffic forecasting model (CRISP-ML).")
    ap.add_argument("--quick", action="store_true", help="smaller grids for a fast check")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    t_start = time.time()

    loader = DataLoader()
    df = loader.get_data()
    if loader.is_synthetic:
        raise SystemExit("data/raw/traffic.csv not found: the model must be trained with the real data")
    df = df[["datetime", "junction", "vehicles", "hour", "dayofweek", "month"]].copy()
    splits = split_by_time(df)
    log("partitions: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    feat_train = TrafficFeaturizer(random_state=SEED).fit(splits["train"])
    quality = data_quality(os.path.join(ROOT, "data", "raw", "traffic.csv"), df, splits, feat_train)

    experiments, families = [], {}
    order = ["seasonal_mean", "ridge", "random_forest", "gradient_boosting", "mlp"]
    for fam in order:
        configs = [{}] if fam == "seasonal_mean" else list(grid(fam))
        if args.quick and len(configs) > 2:
            configs = configs[:2]
        best = None
        for params in configs:
            t = time.time()
            folds = time_series_cv(fam, params, splits["train"], n_jobs=args.jobs)
            rmse = [f["rmse"] for f in folds]
            row = {"family": fam, "model": MODEL_LABELS[fam], "params": json.dumps(params, default=list),
                   "cv_rmse_mean": round(float(np.mean(rmse)), 3), "cv_rmse_std": round(float(np.std(rmse)), 3),
                   "cv_mae_mean": round(float(np.mean([f["mae"] for f in folds])), 3),
                   "cv_r2_mean": round(float(np.mean([f["r2"] for f in folds])), 4),
                   "seconds": round(time.time() - t, 1)}
            experiments.append(row)
            log(f"{fam} {params} cv_rmse={row['cv_rmse_mean']} r2={row['cv_r2_mean']}")
            if best is None or row["cv_rmse_mean"] < best[0]["cv_rmse_mean"]:
                best = (row, params, folds)
        row, params, folds = best
        _, _, p_val, p_tr, secs = fit_predict(fam, params, splits["train"], splits["validation"], args.jobs)
        families[fam] = {"label": MODEL_LABELS[fam], "best_params": params,
                         "cv_rmse_mean": row["cv_rmse_mean"], "cv_rmse_std": row["cv_rmse_std"],
                         "cv_r2_mean": row["cv_r2_mean"], "cv_folds": folds,
                         "train": metrics(splits["train"]["vehicles"].values, p_tr),
                         "validation": metrics(splits["validation"]["vehicles"].values, p_val),
                         "train_seconds": round(secs, 2)}
        log(f"  best {params} -> validation {families[fam]['validation']}")

    candidates = [f for f in order if f != "seasonal_mean"]
    selected = min(candidates, key=lambda f: families[f]["validation"]["mse"])
    params = families[selected]["best_params"]
    log(f"selected: {selected} {params}")

    trval = pd.concat([splits["train"], splits["validation"]]).sort_values("datetime").reset_index(drop=True)
    feat_final, model_final, p_test, _, secs = fit_predict(selected, params, trval, splits["test"], args.jobs)
    _, base_model, p_base, _, _ = fit_predict("seasonal_mean", {}, trval, splits["test"])
    y_test = splits["test"]["vehicles"].values
    test_m = metrics(y_test, p_test)
    by_j = {str(j): metrics(y_test[(splits["test"]["junction"] == j).values], p_test[(splits["test"]["junction"] == j).values])
            for j in feat_final.junctions_}
    log(f"TEST {selected}: {test_m}")

    feat_v, model_v, _, _, _ = fit_predict(selected, params, splits["train"], splits["validation"], args.jobs)
    Xv, yv = feat_v.transform(splits["validation"]), splits["validation"]["vehicles"].values
    base_mse = float(np.mean((np.clip(model_v.predict(Xv), 0, None) - yv) ** 2))
    rng = np.random.default_rng(SEED)
    importance = {}
    for i, name in enumerate(feat_v.feature_names):
        Xp = Xv.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
        importance[name] = round(float(np.mean((np.clip(model_v.predict(Xp), 0, None) - yv) ** 2)) - base_mse, 3)

    final = {"selected": selected, "label": MODEL_LABELS[selected], "params": params,
             "trained_on": "train + validation (2015-11-01 to 2017-04-30)",
             "test_period": "2017-05-01 to 2017-06-30", "test": test_m, "test_by_junction": by_j,
             "test_baseline_seasonal_mean": metrics(y_test, p_base), "train_seconds": round(secs, 2),
             "feature_importance": importance}

    results = {"generated_at": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
               "splits": {k: {"from": SPLITS[k][0], "to_exclusive": SPLITS[k][1], "rows": int(len(v)),
                              "rows_by_junction": {str(j): int(n) for j, n in v.groupby("junction").size().items()}}
                          for k, v in splits.items()},
               "cv": {"method": "TimeSeriesSplit", "n_splits": 5, "on": "training partition only"},
               "grids": {k: {kk: [list(x) if isinstance(x, tuple) else x for x in vv] for kk, vv in g.items()} for k, g in GRIDS.items()},
               "data_quality": quality,
               "silhouette_train": {str(j): round(s, 3) for j, s in feat_train.silhouette_.items()},
               "families": families, "final": final, "success_criteria": {"r2_min": 0.8, "beats_baseline": True},
               "runtime_seconds": round(time.time() - t_start, 1)}

    figures(df, splits, feat_train, results, final, p_test)
    with open(os.path.join(ART, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    with open(os.path.join(ART, "experiments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(experiments[0]))
        w.writeheader(); w.writerows(experiments)
    joblib.dump({"featurizer": feat_final, "model": model_final, "name": selected,
                 "label": MODEL_LABELS[selected], "params": params,
                 "version": datetime.now().strftime("%Y%m%d-%H%M"), "test_metrics": test_m,
                 "trained_on": final["trained_on"]}, os.path.join(ART, "model.joblib"), compress=3)
    log(f"saved artifacts/model.joblib, report.json, experiments.csv and figures in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
