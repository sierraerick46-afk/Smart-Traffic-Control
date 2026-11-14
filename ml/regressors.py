import numpy as np
import pandas as pd
import torch
from torch import nn

from models.network_mlp import TrafficMLP


class TorchMLPRegressor:
    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=40, batch_size=256, dropout=0.05,
                 weight_decay=1e-5, seed=42):
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed

    def get_params(self):
        return {"hidden": self.hidden, "lr": self.lr, "epochs": self.epochs,
                "batch_size": self.batch_size, "dropout": self.dropout,
                "weight_decay": self.weight_decay}

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-6
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-6)
        xt = torch.from_numpy((X - self.x_mean_) / self.x_std_)
        yt = torch.from_numpy((y - self.y_mean_) / self.y_std_).view(-1, 1)
        self.net_ = TrafficMLP(input_dim=X.shape[1], hidden=self.hidden, dropout=self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        n = len(xt)
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(xt[idx]), yt[idx]).backward()
                opt.step()
        self.net_.eval()
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        with torch.no_grad():
            out = self.net_(torch.from_numpy((X - self.x_mean_) / self.x_std_)).numpy().ravel()
        return out * self.y_std_ + self.y_mean_


class SeasonalMeanRegressor:
    def fit(self, X, y, junction_cols, hour_col=0, dow_col=1):
        self.junction_cols, self.hour_col, self.dow_col = junction_cols, hour_col, dow_col
        j = np.asarray(X)[:, junction_cols].argmax(1)
        key = pd.DataFrame({"j": j, "d": np.asarray(X)[:, dow_col].astype(int),
                            "h": np.asarray(X)[:, hour_col].astype(int), "y": y})
        self.table_ = key.groupby(["j", "d", "h"])["y"].mean().to_dict()
        self.jmean_ = key.groupby("j")["y"].mean().to_dict()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        j = X[:, self.junction_cols].argmax(1)
        d, h = X[:, self.dow_col].astype(int), X[:, self.hour_col].astype(int)
        return np.array([self.table_.get((a, b, c), self.jmean_.get(a, self.global_))
                         for a, b, c in zip(j, d, h)])
