import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["hour", "dayofweek", "is_weekend", "cluster", "hour_sin", "hour_cos",
                 "days_since_start", "month"]


class TrafficFeaturizer:
    def __init__(self, n_clusters=3, cap_quantile=0.995, random_state=42):
        self.n_clusters = n_clusters
        self.cap_quantile = cap_quantile
        self.random_state = random_state

    def _profile_clusters(self, df):
        prof = (df.groupby(["dayofweek", "hour"])["vehicles"].agg(["mean", "std"])
                  .fillna(0).reset_index())
        X = StandardScaler().fit_transform(prof[["mean", "std"]].values)
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        means = [prof.loc[km.labels_ == c, "mean"].mean() for c in range(self.n_clusters)]
        remap = {int(old): new for new, old in enumerate(np.argsort(np.nan_to_num(means)))}
        table = np.zeros((7, 24), dtype=int)
        for r, lab in zip(prof.itertuples(), km.labels_):
            table[int(r.dayofweek), int(r.hour)] = remap[int(lab)]
        sil = float(silhouette_score(X, km.labels_)) if len(prof) > self.n_clusters else None
        return table, sil

    def fit(self, df):
        self.t0_ = df["datetime"].min()
        self.last_date_ = df["datetime"].max()
        self.junctions_ = sorted(int(j) for j in df["junction"].unique())
        self.jindex_ = {j: i for i, j in enumerate(self.junctions_)}
        self.tables_, self.silhouette_ = {}, {}
        for j in self.junctions_:
            self.tables_[j], self.silhouette_[j] = self._profile_clusters(df[df["junction"] == j])
        self.caps_ = {int(j): float(g["vehicles"].quantile(self.cap_quantile))
                      for j, g in df.groupby("junction")}
        span = df.groupby("junction")["datetime"].agg(["min", "max"])
        self.periods_ = {int(j): (r["min"], r["max"]) for j, r in span.iterrows()}
        return self

    @property
    def feature_names(self):
        return FEATURE_NAMES + [f"junction_{j}" for j in self.junctions_]

    def clusters(self, hours, dows, junctions):
        hours, dows = np.asarray(hours, int) % 24, np.asarray(dows, int) % 7
        return np.array([self.tables_.get(int(j), self.tables_[self.junctions_[0]])[d, h]
                         for h, d, j in zip(hours, dows, junctions)])

    def transform_parts(self, datetimes, junctions):
        dt = pd.DatetimeIndex(pd.to_datetime(datetimes))
        junctions = np.atleast_1d(np.asarray(junctions, int))
        hours, dows = dt.hour.values, dt.dayofweek.values
        days = (dt - self.t0_).total_seconds().values / 86400
        ang = 2 * np.pi * hours / 24
        idx = np.array([self.jindex_.get(int(j), 0) for j in junctions])
        onehot = np.eye(len(self.junctions_))[idx]
        base = np.column_stack([hours, dows, (dows >= 5).astype(float),
                                self.clusters(hours, dows, junctions),
                                np.sin(ang), np.cos(ang), days, dt.month.values])
        return np.hstack([base, onehot]).astype(np.float32)

    def transform(self, df):
        return self.transform_parts(df["datetime"].values, df["junction"].values)

    def cap_target(self, df):
        caps = df["junction"].map(self.caps_).fillna(np.inf).values
        return np.minimum(df["vehicles"].values.astype(np.float32), caps)
