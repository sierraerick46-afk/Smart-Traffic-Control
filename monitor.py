import json
import os
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.pipeline import SPLITS
from models.data_loader import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
LOGS = os.path.join(ROOT, "logs")
FIG = os.path.join(ROOT, "static", "img", "crisp")
DRIFT_START = "2017-06-10"
DRIFT_JUNCTION = 1
DRIFT_FACTOR = 1.30
PSI_ALERT = 0.25
WARMUP_DAYS = 7
MAE_FACTOR = 1.5


def psi(reference, current, bins):
    r = np.histogram(reference, bins=bins)[0] / max(len(reference), 1)
    c = np.histogram(current, bins=bins)[0] / max(len(current), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    art = joblib.load(os.path.join(ART, "model.joblib"))
    feat, model = art["featurizer"], art["model"]
    test_mae = art["test_metrics"]["mae"]
    mae_alert = round(MAE_FACTOR * test_mae, 2)

    df = DataLoader().get_data()[["datetime", "junction", "vehicles"]]
    ref = df[(df["datetime"] < SPLITS["test"][0]) & (df["datetime"] >= pd.Timestamp(SPLITS["test"][0]) - pd.Timedelta(weeks=8))]
    stream = df[df["datetime"] >= SPLITS["test"][0]].copy()
    mask = (stream["datetime"] >= DRIFT_START) & (stream["junction"] == DRIFT_JUNCTION)
    stream.loc[mask, "vehicles"] = np.round(stream.loc[mask, "vehicles"] * DRIFT_FACTOR)
    stream["pred"] = np.clip(model.predict(feat.transform(stream)), 0, None)
    stream["abs_err"] = (stream["vehicles"] - stream["pred"]).abs()
    stream["day"] = stream["datetime"].dt.normalize()

    bins = {j: np.unique(np.quantile(g["vehicles"], np.linspace(0, 1, 11))) for j, g in ref.groupby("junction")}
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    records, alerts = [], []
    log_path = os.path.join(LOGS, "monitoring_test.jsonl")
    days = sorted(stream["day"].unique())
    with open(log_path, "w", encoding="utf-8") as f:
        for day in days:
            win = stream[(stream["day"] > day - pd.Timedelta(days=7)) & (stream["day"] <= day)]
            today = stream[stream["day"] == day]
            rec = {"simulated": True, "day": str(pd.Timestamp(day).date()),
                   "predictions": int(len(today)), "mae_day": round(float(today["abs_err"].mean()), 2),
                   "mae_7d": round(float(win["abs_err"].mean()), 2),
                   "bias_7d": round(float((win["vehicles"] - win["pred"]).mean()), 2), "by_junction": {}}
            warm = (pd.Timestamp(day) - pd.Timestamp(days[0])).days >= WARMUP_DAYS - 1
            for j, g in win.groupby("junction"):
                rec["by_junction"][str(j)] = {
                    "mae_7d": round(float(g["abs_err"].mean()), 2),
                    "psi_7d": round(psi(ref[ref["junction"] == j]["vehicles"], g["vehicles"], bins[j]), 3) if warm else None}
            rec["warmup"] = not warm
            rec["alerts"] = []
            if warm and rec["mae_7d"] > mae_alert:
                rec["alerts"].append({"level": "retrain", "message": f"MAE 7d {rec['mae_7d']} > {mae_alert}: schedule retraining"})
            for j, v in rec["by_junction"].items():
                if warm and v["psi_7d"] > PSI_ALERT:
                    rec["alerts"].append({"level": "drift_warning", "message": f"Data drift at junction {j}: PSI {v['psi_7d']} > {PSI_ALERT}"})
            if rec["alerts"]:
                alerts.append({"day": rec["day"], "alerts": rec["alerts"]})
            records.append(rec)
            f.write(json.dumps(rec) + "\n")

    retrain_days = [a["day"] for a in alerts if any(x["level"] == "retrain" for x in a["alerts"])]
    drift_by_j = {}
    for a in alerts:
        for x in a["alerts"]:
            if x["level"] == "drift_warning":
                j = x["message"].split("junction ")[1].split(":")[0]
                drift_by_j[j] = drift_by_j.get(j, 0) + 1
    summary = {"label": "SIMULATED OPERATIONAL DATA",
               "description": (f"Replay of the real hourly counts of May–June 2017 (not used for training) as if they "
                               f"arrived day by day. From {DRIFT_START} a synthetic +{int((DRIFT_FACTOR - 1) * 100)}% "
                               f"increase is injected at junction {DRIFT_JUNCTION} to test the alerts."),
               "generated_at": datetime.now().isoformat(timespec="seconds"),
               "model": art["label"], "model_version": art["version"],
               "thresholds": {"mae_7d": mae_alert, "psi": PSI_ALERT, "reference_test_mae": test_mae},
               "days": len(records), "predictions_logged": int(len(stream)),
               "warmup_days": WARMUP_DAYS, "days_with_alerts": len(alerts),
               "first_retrain_alert_day": retrain_days[0] if retrain_days else None,
               "retrain_alert_days": len(retrain_days), "drift_warning_days_by_junction": drift_by_j,
               "mae_7d_before_drift": round(float(np.mean([r["mae_7d"] for r in records if not r["warmup"] and r["day"] < DRIFT_START])), 2),
               "mae_7d_last_week": records[-1]["mae_7d"],
               "drift_injected": {"from": DRIFT_START, "junction": DRIFT_JUNCTION, "factor": DRIFT_FACTOR},
               "alerts_sample": alerts[:6], "log_file": "logs/monitoring_test.jsonl"}
    with open(os.path.join(ART, "monitoring.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    x = [pd.Timestamp(r["day"]) for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    axes[0].plot(x, [r["mae_7d"] for r in records], color="#00482B", lw=2, label="MAE (7-day window)")
    axes[0].axhline(mae_alert, color="#D64545", ls="--", lw=1.2, label=f"retraining threshold ({mae_alert})")
    axes[0].set_ylabel("vehicles / hour"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("SIMULATED operation: replay of May–Jun 2017 with drift injected at junction 1", fontweight="bold", fontsize=10)
    for j, col in zip(["1", "2", "3", "4"], ["#00482B", "#E0A526", "#D64545", "#3A8FB7"]):
        axes[1].plot(x, [r["by_junction"].get(j, {}).get("psi_7d") if r["by_junction"].get(j, {}).get("psi_7d") is not None else np.nan for r in records], color=col, lw=1.6, label=f"J{j}")
    axes[1].axhline(PSI_ALERT, color="#D64545", ls="--", lw=1.2)
    axes[1].set_ylabel("PSI (7-day window)"); axes[1].legend(frameon=False, ncol=4, fontsize=8)
    for ax in axes:
        ax.axvline(pd.Timestamp(DRIFT_START), color="#5D6862", lw=1, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].text(pd.Timestamp(DRIFT_START), axes[1].get_ylim()[1] * 0.9, "  drift injected", fontsize=8, color="#5D6862")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "monitoring_test.png"), dpi=150); plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("days", "predictions_logged", "days_with_alerts", "first_retrain_alert_day", "retrain_alert_days", "drift_warning_days_by_junction", "mae_7d_before_drift", "mae_7d_last_week", "thresholds")}, indent=1))


if __name__ == "__main__":
    main()
