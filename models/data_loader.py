import os
import unicodedata
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def now_bogota():
    """Hora actual de Colombia (UTC-5, sin horario de verano)."""
    return datetime.now(timezone(timedelta(hours=-5)))


def _norm_col(c):
    c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
    return c.strip().upper().replace(" ", "").replace("_", "")


class DataLoader:
    def __init__(self, raw_dir=RAW_DIR):
        self.raw_dir = raw_dir
        self.is_synthetic = False
        self._data = None

    def _read_csv(path):
        last = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                df.columns = [_norm_col(c) for c in df.columns]
                return df
            except UnicodeDecodeError as e:
                last = e
        raise last

    def _discover(self):
        inventory, traffic = None, None
        if not os.path.isdir(self.raw_dir):
            return None, None
        for name in sorted(os.listdir(self.raw_dir)):
            if not name.lower().endswith(".csv"):
                continue
            try:
                df = self._read_csv(os.path.join(self.raw_dir, name))
            except Exception:
                continue
            cols = set(df.columns)
            if {"DATETIME", "VEHICLES"} <= cols:
                traffic = df
            elif cols & {"DIRECCION", "NUMEROFASES"}:
                inventory = df
        return inventory, traffic

    @staticmethod
    def _clean_traffic(df):
        df = df.copy()
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df["VEHICLES"] = pd.to_numeric(df["VEHICLES"], errors="coerce")
        df = df.dropna(subset=["DATETIME", "VEHICLES"])
        df = df[df["VEHICLES"] >= 0]
        if "JUNCTION" not in df.columns:
            df["JUNCTION"] = 1
        df["JUNCTION"] = pd.to_numeric(df["JUNCTION"], errors="coerce").fillna(1).astype(int)
        df = df.drop_duplicates(subset=["DATETIME", "JUNCTION"]).sort_values("DATETIME")
        return df.reset_index(drop=True)

    @staticmethod
    def _add_time_features(df):
        dt = df["DATETIME"]
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["month"] = dt.dt.month
        return df

    @staticmethod
    def _merge(traffic, inventory):
        inventory = inventory.copy()
        if "ID" not in inventory.columns:
            inventory["ID"] = range(1, len(inventory) + 1)
        for col, default in (("DIRECCION", "No address"), ("NUMEROFASES", 2)):
            if col not in inventory.columns:
                inventory[col] = default
        inv_ids = inventory["ID"].dropna().unique()

        traffic = traffic.copy()
        if "ID" in traffic.columns and np.isin(traffic["ID"], inv_ids).any():
            traffic["ID_INTERSECCION"] = traffic["ID"]      
        else:                                              
            order = sorted(traffic["JUNCTION"].unique())
            mapping = {j: inv_ids[i % len(inv_ids)] for i, j in enumerate(order)}
            traffic["ID_INTERSECCION"] = traffic["JUNCTION"].map(mapping)

        inv = inventory.drop_duplicates(subset="ID").rename(columns={"ID": "ID_INTERSECCION"})
        keep = ["ID_INTERSECCION", "DIRECCION", "NUMEROFASES"]
        merged = traffic.merge(inv[keep], on="ID_INTERSECCION", how="left")
        merged["DIRECCION"] = merged["DIRECCION"].fillna("No address")
        merged["NUMEROFASES"] = merged["NUMEROFASES"].fillna(2)
        return merged

    @staticmethod
    def _synthetic():
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=24 * 120, freq="h")
        h, wd = idx.hour.values, idx.dayofweek.values
        g = lambda x, m, s: np.exp(-((x - m) ** 2) / (2 * s ** 2))
        base = 12 + 55 * g(h, 7.5, 1.4) + 62 * g(h, 17.5, 1.8) + 28 * g(h, 12.5, 2.2)
        base = np.where(wd >= 5, 8 + 0.45 * base, base)
        frames = []
        for j, scale in enumerate([1.0, 0.7, 0.5, 0.3], start=1):
            v = np.clip(np.round(base * scale * rng.normal(1, 0.12, len(idx))), 0, None)
            frames.append(pd.DataFrame({"DATETIME": idx, "JUNCTION": j, "VEHICLES": v.astype(int)}))
        traffic = pd.concat(frames, ignore_index=True)
        traffic["ID"] = np.arange(len(traffic))
        inventory = pd.DataFrame({
            "ID": [f"CHIA-00{i}" for i in range(1, 5)],
            "DIRECCION": ["Cra 9 con Calle 11", "Av. Pradilla con Cra 7",
                          "Cra 2 con Calle 15", "Calle 7 con Cra 10"],
            "NUMEROFASES": [2, 2, 2, 2],
        })
        return inventory, traffic

    def get_data(self):
        if self._data is None:
            inventory, traffic = self._discover()
            if traffic is None:
                self.is_synthetic = True
                inventory, traffic = self._synthetic()
            if inventory is None:
                inventory = pd.DataFrame({"ID": sorted(traffic["JUNCTION"].unique())})
            traffic = self._clean_traffic(traffic)
            merged = self._merge(traffic, inventory)
            merged = self._add_time_features(merged)
            merged = merged.rename(columns={"DATETIME": "datetime", "JUNCTION": "junction",
                                            "VEHICLES": "vehicles"})
            self._data = merged
        return self._data

    def summary(self):
        df = self.get_data()
        return {
            "rows": int(len(df)),
            "intersections": int(df["ID_INTERSECCION"].nunique()),
            "date_from": str(df["datetime"].min()),
            "date_to": str(df["datetime"].max()),
            "is_synthetic": bool(self.is_synthetic),
        }
