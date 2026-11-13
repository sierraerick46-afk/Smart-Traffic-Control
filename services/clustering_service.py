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
        self.junctions = []
        self._jidx = {}
        self._global = None
        self._by_junction = {}
        self._tables = None
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

    def _fit_profile(self, df):
        prof = (df.groupby(["dayofweek", "hour"])["vehicles"].agg(["mean", "std"])
                  .fillna(0).reset_index())
        X = StandardScaler().fit_transform(prof[["mean", "std"]].values)
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        raw = km.labels_
        means = [prof.loc[raw == c, "mean"].mean() for c in range(self.n_clusters)]
        order = np.argsort(np.nan_to_num(means))
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
        sil = float(silhouette_score(X, raw)) if len(prof) > self.n_clusters else None
        return table, prof, sil

    def fit(self):
        df = self.loader.get_data()
        self._global = self._fit_profile(df)
        self.silhouette = self._global[2]
        self.junctions = sorted(int(j) for j in df["junction"].unique())
        self._jidx = {j: i for i, j in enumerate(self.junctions)}
        for j in self.junctions:
            sub = df[df["junction"] == j]
            if sub.groupby(["dayofweek", "hour"]).ngroups >= self.n_clusters * 2:
                self._by_junction[j] = self._fit_profile(sub)
            else:
                self._by_junction[j] = self._global
        self._tables = np.stack([self._by_junction[j][0] for j in self.junctions])
        self.fitted = True
        return self

    def _entry(self, junction):
        if junction is None or int(junction) not in self._by_junction:
            return self._global
        return self._by_junction[int(junction)]

    def get_cluster(self, hora, dia_semana, junction=None):
        return int(self._entry(junction)[0][int(dia_semana) % 7, int(hora) % 24])

    def cluster_array(self, hours, dows, junctions=None):
        hours, dows = np.asarray(hours, int) % 24, np.asarray(dows, int) % 7
        if junctions is None:
            return self._global[0][dows, hours]
        idx = np.array([self._jidx.get(int(j), 0) for j in np.atleast_1d(junctions)])
        return self._tables[idx, dows, hours]

    def describe(self, hora, dia_semana, junction=None):
        c = self.get_cluster(hora, dia_semana, junction)
        return {"cluster": c, "label": self.label(c), "color": self.color(c),
                "hora": int(hora), "dia": int(dia_semana), "dia_nombre": DIAS[int(dia_semana) % 7]}

    def silhouette_for(self, junction=None):
        return self._entry(junction)[2]

    def hourly_profile(self, dia_semana, junction=None):
        prof = self._entry(junction)[1]
        rows = prof[prof["dayofweek"] == int(dia_semana)].set_index("hour")
        out = []
        for h in range(24):
            c = self.get_cluster(h, dia_semana, junction)
            out.append({"hour": h, "cluster": c, "color": self.color(c),
                        "mean": round(float(rows["mean"].get(h, 0.0)), 1)})
        return out

    def summary(self, junction=None):
        prof = self._entry(junction)[1]
        res = []
        for c in range(self.n_clusters):
            sub = prof[prof["cluster"] == c]
            wk = sorted({int(h) for d, h in zip(sub.dayofweek, sub.hour) if d < 5})
            we = sorted({int(h) for d, h in zip(sub.dayofweek, sub.hour) if d >= 5})
            res.append({"cluster": c, "label": self.label(c), "color": self.color(c),
                        "mean_vehicles": round(float(sub["mean"].mean()), 1) if len(sub) else 0.0,
                        "slots": int(len(sub)), "weekday_hours": wk, "weekend_hours": we})
        return res
