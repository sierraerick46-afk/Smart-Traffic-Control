import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from models.data_loader import DIAS


class ClusteringService:
    def __init__(self, data_loader, n_clusters=3, random_state=42):
        self.loader = data_loader
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._table = np.zeros((7, 24), dtype=int)   
        self._profile = None
        self.silhouette = None
        self.fitted = False

    def label(self, c):
        if self.n_clusters == 3:
            return ["Off-peak", "Moderate", "Peak"][int(c)]
        return f"Cluster {int(c) + 1}"

    def color(self, c):
        palette = (["success", "warning", "danger"] if self.n_clusters == 3
                   else ["success", "info", "primary", "warning", "danger"])
        return palette[min(int(c), len(palette) - 1)]

    def fit(self):
        df = self.loader.get_data()
        prof = (df.groupby(["dayofweek", "hour"])["vehicles"].agg(["mean", "std"])
                  .fillna(0).reset_index())
        X = StandardScaler().fit_transform(prof[["mean", "std"]].values)
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)

        raw = km.labels_
        order = np.argsort([prof.loc[raw == c, "mean"].mean() for c in range(self.n_clusters)])
        remap = {int(old): new for new, old in enumerate(order)}
        prof["cluster"] = [remap[int(c)] for c in raw]

        table = -np.ones((7, 24), dtype=int)
        for r in prof.itertuples():
            table[int(r.dayofweek), int(r.hour)] = int(r.cluster)
        for d in range(7):                                 
            for h in range(24):
                if table[d, h] < 0:
                    others = table[:, h][table[:, h] >= 0]
                    table[d, h] = int(others[0]) if len(others) else 0

        self._table, self._profile = table, prof
        self.silhouette = float(silhouette_score(X, raw)) if len(prof) > self.n_clusters else None
        self.fitted = True
        return self

    def get_cluster(self, hora, dia_semana):
        """Etiqueta numerica del momento (hora 0-23, dia 0=lunes ... 6=domingo)."""
        return int(self._table[int(dia_semana) % 7, int(hora) % 24])

    def cluster_array(self, hours, dows):
        return self._table[np.asarray(dows, int) % 7, np.asarray(hours, int) % 24]

    def describe(self, hora, dia_semana):
        c = self.get_cluster(hora, dia_semana)
        return {"cluster": c, "label": self.label(c), "color": self.color(c),
                "hora": int(hora), "dia": int(dia_semana), "dia_nombre": DIAS[int(dia_semana) % 7]}

    def hourly_profile(self, dia_semana):
        rows = self._profile[self._profile["dayofweek"] == int(dia_semana)].set_index("hour")
        out = []
        for h in range(24):
            c = self.get_cluster(h, dia_semana)
            out.append({"hour": h, "cluster": c, "color": self.color(c),
                        "mean": round(float(rows["mean"].get(h, 0.0)), 1)})
        return out

    def summary(self):
        res = []
        for c in range(self.n_clusters):
            sub = self._profile[self._profile["cluster"] == c]
            wk = sorted({int(h) for d, h in zip(sub.dayofweek, sub.hour) if d < 5})
            we = sorted({int(h) for d, h in zip(sub.dayofweek, sub.hour) if d >= 5})
            res.append({"cluster": c, "label": self.label(c), "color": self.color(c),
                        "mean_vehicles": round(float(sub["mean"].mean()), 1) if len(sub) else 0.0,
                        "slots": int(len(sub)), "weekday_hours": wk, "weekend_hours": we})
        return res
