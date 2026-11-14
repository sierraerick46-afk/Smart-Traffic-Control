import json
import os
import threading
import time

import joblib
import numpy as np
import pandas as pd

from ml.pipeline import MODEL_LABELS, fit_predict, metrics, split_by_time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
FAMILIES = ["seasonal_mean", "ridge", "random_forest", "gradient_boosting", "mlp"]
LABELS = {**MODEL_LABELS, "historical": "Historical average"}


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class PredictionService:
    def __init__(self, data_loader, clustering_service, seed=42):
        self.loader = data_loader
        self.clustering = clustering_service
        self.seed = seed
        self.models = {}
        self.featurizers = {}
        self.best_model = None
        self.metrics = None
        self.version = 0
        self.training = False
        self.train_error = None
        self._baseline = None
        self._lock = threading.Lock()
        df = self.loader.get_data()
        self.ref_date = df["datetime"].max().strftime("%Y-%m-%d")
        self.report = _load_json(os.path.join(ART, "report.json"))
        self.artifact = None
        path = os.path.join(ART, "model.joblib")
        if os.path.exists(path):
            self.artifact = joblib.load(path)
            name = self.artifact["name"]
            self.models[name] = self.artifact["model"]
            self.featurizers[name] = self.artifact["featurizer"]
            self.best_model = name
            self.version = 1
        if self.report:
            self.metrics = self._metrics_from_report(self.report, source="saved")

    def _metrics_from_report(self, rep, source):
        rows = []
        for f in FAMILIES:
            fam = rep["families"].get(f)
            if not fam:
                continue
            v = fam["validation"]
            rows.append({"key": f, "label": LABELS[f], "cv_rmse": fam["cv_rmse_mean"], "mse": v["mse"],
                         "mae": v["mae"], "r2": v["r2"], "train_seconds": fam["train_seconds"]})
        fin = rep["final"]
        return {"rows": rows, "best_model": fin["selected"], "test": fin["test"],
                "test_label": fin["label"], "n_train": rep["splits"]["train"]["rows"],
                "n_validation": rep["splits"]["validation"]["rows"], "n_test": rep["splits"]["test"]["rows"],
                "source": source, "generated_at": rep.get("generated_at")}

    def _train_job(self):
        try:
            df = self.loader.get_data()[["datetime", "junction", "vehicles", "hour", "dayofweek", "month"]]
            sp = split_by_time(df)
            rep = self.report or {}
            params = {f: rep.get("families", {}).get(f, {}).get("best_params", {}) for f in FAMILIES}
            rows, models, feats = [], {}, {}
            for f in FAMILIES:
                if f != "seasonal_mean" and not params[f]:
                    continue
                t = time.time()
                feat, model, p_val, _, secs = fit_predict(f, params[f], sp["train"], sp["validation"])
                m = metrics(sp["validation"]["vehicles"].values, p_val)
                cv = rep.get("families", {}).get(f, {}).get("cv_rmse_mean")
                rows.append({"key": f, "label": LABELS[f], "cv_rmse": cv, **{k: m[k] for k in ("mse", "mae", "r2")},
                             "train_seconds": round(secs, 2)})
                if f != "seasonal_mean":
                    models[f], feats[f] = model, feat
            best = min((r for r in rows if r["key"] != "seasonal_mean"), key=lambda r: r["mse"])["key"]
            trval = pd.concat([sp["train"], sp["validation"]]).sort_values("datetime").reset_index(drop=True)
            feat, model, p_test, _, _ = fit_predict(best, params[best], trval, sp["test"])
            models[best], feats[best] = model, feat
            test = metrics(sp["test"]["vehicles"].values, p_test)
            with self._lock:
                self.models, self.featurizers, self.best_model = models, feats, best
                self.metrics = {"rows": rows, "best_model": best, "test": test, "test_label": LABELS[best],
                                "n_train": len(sp["train"]), "n_validation": len(sp["validation"]),
                                "n_test": len(sp["test"]), "source": "retrained",
                                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds")}
                self.version += 1
        except Exception as e:
            self.train_error = str(e)
        finally:
            self.training = False

    def train(self):
        if self.training:
            return {"started": False, "training": True}
        self.training, self.train_error = True, None
        threading.Thread(target=self._train_job, daemon=True).start()
        return {"started": True, "training": True}

    def _baseline_table(self):
        if self._baseline is None:
            df = self.loader.get_data()
            t = df.groupby(["junction", "dayofweek", "hour"])["vehicles"].mean()
            tj = df.groupby("junction")["vehicles"].mean()
            self._baseline = np.array([[[float(t.get((j, d, h), tj[j])) for h in range(24)]
                                        for d in range(7)] for j in self.clustering.junctions])
        return self._baseline

    def _jindex(self, junctions):
        m = {j: i for i, j in enumerate(self.clustering.junctions)}
        return np.array([m.get(int(j), 0) for j in np.atleast_1d(junctions)])

    def resolved_name(self, model=None):
        if model in self.models or model == "historical":
            return model
        return self.best_model or "historical"

    def model_label(self, model=None):
        return LABELS[self.resolved_name(model)]

    def models_info(self):
        return [{"key": k, "label": LABELS[k], "available": k in self.models}
                for k in ("ridge", "random_forest", "gradient_boosting", "mlp")]

    @staticmethod
    def _at(anchor, dows, hours):
        anchor = pd.Timestamp(anchor).normalize()
        monday = anchor - pd.Timedelta(days=anchor.dayofweek)
        return monday + pd.to_timedelta(np.asarray(dows) % 7, unit="D") + pd.to_timedelta(np.asarray(hours) % 24, unit="h")

    def predict_datetimes(self, datetimes, junctions, model=None):
        name = self.resolved_name(model)
        dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
        junctions = np.atleast_1d(np.asarray(junctions, int))
        if name == "historical":
            return self._baseline_table()[self._jindex(junctions), dt.dayofweek % 7, dt.hour % 24]
        feat = self.featurizers[name]
        return np.clip(self.models[name].predict(feat.transform_parts(dt, junctions)), 0, None)

    def predict_many(self, hours, dows, clusters=None, junctions=None, model=None, reference="latest"):
        hours, dows = np.asarray(hours, int), np.asarray(dows, int)
        if junctions is None:
            return np.mean([self.predict_many(hours, dows, None, np.full(len(hours), j), model, reference)
                            for j in self.clustering.junctions], axis=0)
        junctions = np.atleast_1d(np.asarray(junctions, int))
        name = self.resolved_name(model)
        if reference == "typical" and name != "historical":
            feat = self.featurizers[name]
            preds = []
            for k in range(12):
                anchors = []
                for j in junctions:
                    lo, hi = feat.periods_.get(int(j), (feat.t0_, feat.last_date_))
                    anchors.append(lo + (hi - lo) * (k / 11))
                dts = [self._at(a, [d], [h])[0] for a, d, h in zip(anchors, dows, hours)]
                preds.append(self.predict_datetimes(dts, junctions, name))
            return np.mean(preds, axis=0)
        return self.predict_datetimes(self._at(self.ref_date, dows, hours), junctions, name)

    def predict(self, hora, dia, cluster=None, junction=None, model=None):
        j = None if junction is None else [junction]
        return float(self.predict_many([hora], [dia], None, j, model)[0])

    def status(self):
        return {"trained": self.best_model is not None, "best_model": self.best_model,
                "training": self.training, "error": self.train_error,
                "metrics": self.metrics, "models": self.models_info(), "reference_date": self.ref_date}
