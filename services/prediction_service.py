import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from models.network_mlp import TrafficMLP

torch.set_num_threads(1)

MODEL_LABELS = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
                "mlp": "MLP (PyTorch)", "historical": "Historical average"}


class PredictionService:
    def __init__(self, data_loader, clustering_service, seed=42, epochs=40, max_rows=60000):
        self.loader = data_loader
        self.clustering = clustering_service
        self.seed, self.epochs, self.max_rows = seed, epochs, max_rows
        self.models = {}
        self.metrics = None
        self.best_model = None
        self.version = 0
        self._baseline = None
        self._lock = threading.Lock()
        df = self.loader.get_data()
        self._t0 = df["datetime"].min()
        last = df["datetime"].max()
        self.ref_days = float((last - self._t0).total_seconds() / 86400)
        self.ref_month = int(last.month)
        self.ref_date = last.strftime("%Y-%m-%d")
        g = df.assign(d=(df["datetime"] - self._t0).dt.total_seconds() / 86400).groupby("junction")["d"]
        self.typical_days = {j: np.linspace(lo, hi, 12) for j, lo, hi in zip(g.min().index, g.min().values, g.max().values)}

    def _jindex(self, junctions):
        m = {j: i for i, j in enumerate(self.clustering.junctions)}
        return np.array([m.get(int(j), 0) for j in np.atleast_1d(junctions)])

    def _features(self, hours, dows, clusters, junctions, days=None, months=None):
        hours = np.atleast_1d(np.asarray(hours, dtype=float))
        dows = np.atleast_1d(np.asarray(dows, dtype=float))
        clusters = np.atleast_1d(np.asarray(clusters, dtype=float))
        n = len(hours)
        days = np.full(n, self.ref_days) if days is None else np.asarray(days, dtype=float)
        months = np.full(n, self.ref_month) if months is None else np.asarray(months, dtype=float)
        ang = 2 * np.pi * hours / 24
        onehot = np.eye(len(self.clustering.junctions))[self._jindex(junctions)]
        base = np.column_stack([hours, dows, (dows >= 5).astype(float), clusters,
                                np.sin(ang), np.cos(ang), days, months])
        return np.hstack([base, onehot]).astype(np.float32)

    def _dataset(self):
        df = self.loader.get_data()
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=self.seed)
        h, d, j = df["hour"].values, df["dayofweek"].values, df["junction"].values
        cl = self.clustering.cluster_array(h, d, j)
        days = (df["datetime"] - self._t0).dt.total_seconds().values / 86400
        return self._features(h, d, cl, j, days, df["month"].values), df["vehicles"].values.astype(np.float32)

    @staticmethod
    def _metrics(y, p, secs):
        return {"mse": round(float(mean_squared_error(y, p)), 3),
                "mae": round(float(mean_absolute_error(y, p)), 3),
                "r2": round(float(r2_score(y, p)), 4),
                "train_seconds": round(secs, 2)}

    def _train_rf(self, Xtr, ytr, Xte, yte):
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2,
                                   n_jobs=1, random_state=self.seed).fit(Xtr, ytr)
        return rf, self._metrics(yte, rf.predict(Xte), time.time() - t0)

    def _train_gb(self, Xtr, ytr, Xte, yte):
        t0 = time.time()
        gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=8,
                                           random_state=self.seed).fit(Xtr, ytr)
        return gb, self._metrics(yte, gb.predict(Xte), time.time() - t0)

    def _mlp_forward(self, model, params, X):
        with torch.no_grad():
            xt = torch.from_numpy(((X - params["x_mean"]) / params["x_std"]).astype(np.float32))
            return model(xt).numpy().ravel() * params["y_std"] + params["y_mean"]

    def _train_mlp(self, Xtr, ytr, Xte, yte):
        t0 = time.time()
        torch.manual_seed(self.seed)
        params = {"x_mean": Xtr.mean(0), "x_std": Xtr.std(0) + 1e-6,
                  "y_mean": float(ytr.mean()), "y_std": float(ytr.std() + 1e-6)}
        xt = torch.from_numpy(((Xtr - params["x_mean"]) / params["x_std"]).astype(np.float32))
        yt = torch.from_numpy(((ytr - params["y_mean"]) / params["y_std"]).astype(np.float32)).view(-1, 1)
        model = TrafficMLP(input_dim=Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        n = len(xt)
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss_fn(model(xt[idx]), yt[idx]).backward()
                opt.step()
        model.eval()
        return (model, params), self._metrics(yte, self._mlp_forward(model, params, Xte),
                                              time.time() - t0)

    def train(self):
        with self._lock:
            X, y = self._dataset()
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            jobs = {"random_forest": self._train_rf, "gradient_boosting": self._train_gb,
                    "mlp": self._train_mlp}
            with ThreadPoolExecutor(max_workers=3) as ex:
                futures = {k: ex.submit(fn, Xtr, ytr, Xte, yte) for k, fn in jobs.items()}
                results = {k: f.result() for k, f in futures.items()}
            self.models = {k: r[0] for k, r in results.items()}
            metrics = {k: r[1] for k, r in results.items()}
            self.best_model = min(metrics, key=lambda k: metrics[k]["mse"])
            self.metrics = {**metrics, "best_model": self.best_model,
                            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
            self.version += 1
            return self.metrics

    def _baseline_table(self):
        if self._baseline is None:
            df = self.loader.get_data()
            t = df.groupby(["junction", "dayofweek", "hour"])["vehicles"].mean()
            tj = df.groupby("junction")["vehicles"].mean()
            self._baseline = np.array([[[float(t.get((j, d, h), tj[j])) for h in range(24)]
                                        for d in range(7)] for j in self.clustering.junctions])
        return self._baseline

    def resolved_name(self, model=None):
        if model in self.models or model == "historical":
            return model
        return self.best_model or "historical"

    def model_label(self, model=None):
        return MODEL_LABELS[self.resolved_name(model)]

    def models_info(self):
        return [{"key": k, "label": MODEL_LABELS[k], "available": k in self.models}
                for k in ("random_forest", "gradient_boosting", "mlp")]

    def _predict_with(self, name, X):
        m = self.models[name]
        if name == "mlp":
            return self._mlp_forward(m[0], m[1], X)
        return m.predict(X)

    def predict_many(self, hours, dows, clusters=None, junctions=None, model=None, reference="latest"):
        hours, dows = np.asarray(hours, int), np.asarray(dows, int)
        if junctions is None:
            return np.mean([self.predict_many(hours, dows, None, np.full(len(hours), j), model, reference)
                            for j in self.clustering.junctions], axis=0)
        junctions = np.atleast_1d(np.asarray(junctions, int))
        if clusters is None:
            clusters = self.clustering.cluster_array(hours, dows, junctions)
        name = self.resolved_name(model)
        if name == "historical":
            out = self._baseline_table()[self._jindex(junctions), dows % 7, hours % 24]
        else:
            if reference == "typical":
                preds = []
                for k in range(12):
                    days = np.array([self.typical_days.get(int(j), [self.ref_days] * 12)[k] for j in junctions])
                    months = np.array([(self._t0 + pd.Timedelta(days=float(d))).month for d in days])
                    preds.append(self._predict_with(name, self._features(hours, dows, clusters, junctions, days, months)))
                out = np.mean(preds, axis=0)
            else:
                out = self._predict_with(name, self._features(hours, dows, clusters, junctions))
        return np.clip(out, 0, None)

    def predict(self, hora, dia, cluster=None, junction=None, model=None):
        c = None if (cluster is None or junction is None) else [cluster]
        j = None if junction is None else [junction]
        return float(self.predict_many([hora], [dia], c, j, model)[0])

    def status(self):
        return {"trained": self.best_model is not None, "best_model": self.best_model,
                "metrics": self.metrics, "models": self.models_info(), "reference_date": self.ref_date}
