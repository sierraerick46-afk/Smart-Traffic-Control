import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from models.network_mlp import TrafficMLP

torch.set_num_threads(1)


class PredictionService:
    def __init__(self, data_loader, clustering_service, seed=42, epochs=40, max_rows=60000):
        self.loader = data_loader
        self.clustering = clustering_service
        self.seed, self.epochs, self.max_rows = seed, epochs, max_rows
        self.rf = None
        self.mlp = None
        self._mlp_params = None
        self.metrics = None
        self.best_model = None
        self.version = 0
        self._baseline = None
        self._lock = threading.Lock()

    @staticmethod
    def make_features(hour, dow, cluster):
        hour = np.atleast_1d(np.asarray(hour, dtype=float))
        dow = np.atleast_1d(np.asarray(dow, dtype=float))
        cluster = np.atleast_1d(np.asarray(cluster, dtype=float))
        ang = 2 * np.pi * hour / 24
        return np.column_stack([hour, dow, (dow >= 5).astype(float), cluster,
                                np.sin(ang), np.cos(ang)]).astype(np.float32)

    def _dataset(self):
        df = self.loader.get_data()
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=self.seed)
        cl = self.clustering.cluster_array(df["hour"].values, df["dayofweek"].values)
        return self.make_features(df["hour"].values, df["dayofweek"].values, cl), \
            df["vehicles"].values.astype(np.float32)

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
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_rf = ex.submit(self._train_rf, Xtr, ytr, Xte, yte)
                f_mlp = ex.submit(self._train_mlp, Xtr, ytr, Xte, yte)
                rf, rf_m = f_rf.result()
                (mlp, params), mlp_m = f_mlp.result()
            self.rf, self.mlp, self._mlp_params = rf, mlp, params
            self.best_model = "random_forest" if rf_m["mse"] <= mlp_m["mse"] else "mlp"
            self.metrics = {"random_forest": rf_m, "mlp": mlp_m, "best_model": self.best_model,
                            "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
            self.version += 1
            return self.metrics

    def _baseline_table(self):
        if self._baseline is None:
            df = self.loader.get_data()
            t = df.groupby(["dayofweek", "hour"])["vehicles"].mean()
            glob = float(df["vehicles"].mean())
            self._baseline = np.array([[float(t.get((d, h), glob)) for h in range(24)]
                                       for d in range(7)])
        return self._baseline

    def predict_many(self, hours, dows, clusters=None):
        hours, dows = np.asarray(hours, int), np.asarray(dows, int)
        if clusters is None:
            clusters = self.clustering.cluster_array(hours, dows)
        if self.best_model is None:
            out = self._baseline_table()[dows % 7, hours % 24]
        else:
            X = self.make_features(hours, dows, clusters)
            if self.best_model == "random_forest":
                out = self.rf.predict(X)
            else:
                out = self._mlp_forward(self.mlp, self._mlp_params, X)
        return np.clip(out, 0, None)

    def predict(self, hora, dia, cluster=None):
        c = None if cluster is None else [cluster]
        return float(self.predict_many([hora], [dia], c)[0])

    def status(self):
        return {"trained": self.best_model is not None, "best_model": self.best_model,
                "metrics": self.metrics}