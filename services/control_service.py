import random
import threading
from collections import deque
from datetime import datetime, time, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models.data_loader import DIAS
from models.network_dqn import QNetwork
from models.traffic_env import TrafficEnv

ACTION_NAMES = ["Keep green", "Change phase"]


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
        return f(s), torch.tensor(a).long().unsqueeze(1), f(r).unsqueeze(1), f(s2), f(d).unsqueeze(1)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.97, batch=64, target_sync=200):
        self.n_actions, self.gamma, self.batch, self.target_sync = n_actions, gamma, batch, target_sync
        self.policy = QNetwork(state_dim, n_actions)
        self.target = QNetwork(state_dim, n_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.learn_steps = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    def learn(self):
        if len(self.buffer) < self.batch * 2:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch)
        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            tgt = r + self.gamma * self.target(s2).max(1, keepdim=True)[0] * (1 - d)
        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())


class ControlService:
    CONTEXT_STEPS = 60
    SPEEDS = [1, 2, 5, 10]
    MODELS = ["auto", "random_forest", "gradient_boosting", "mlp", "historical"]
    SPLITS = [[0.28, 0.22, 0.28, 0.22], [0.34, 0.16, 0.30, 0.20],
              [0.22, 0.28, 0.20, 0.30], [0.30, 0.30, 0.20, 0.20]]

    def __init__(self, clustering, prediction, data_loader):
        self.clustering, self.prediction, self.loader = clustering, prediction, data_loader
        self.inters = {i["junction"]: i for i in data_loader.intersections()}
        self.junctions = sorted(self.inters)
        self.n_clusters = clustering.n_clusters
        self.date_min, self.date_max = data_loader.date_bounds()
        self.agent = DQNAgent(TrafficEnv.STATE_DIM, TrafficEnv.N_ACTIONS)
        self.trained, self.report = False, None
        self.lock = threading.RLock()
        self._ctx = {}
        self.cfg = None
        self.configure({})

    def _new_env(self, junction, seed=None, max_steps=None):
        split = self.SPLITS[self.junctions.index(junction) % len(self.SPLITS)]
        ref = max(self.inters[junction]["p95"], 1.0)
        return TrafficEnv(self.n_clusters, ref, seed=seed, max_steps=max_steps, split=split)

    def _context(self, model="auto"):
        name = self.prediction.resolved_name(model)
        key = (name, self.prediction.version)
        if key not in self._ctx:
            self._ctx = {k: v for k, v in self._ctx.items() if k[1] == key[1]}
            n = len(self.junctions)
            hours = np.tile(np.arange(24), 7 * n)
            dows = np.tile(np.repeat(np.arange(7), 24), n)
            js = np.repeat(self.junctions, 168)
            cl = self.clustering.cluster_array(hours, dows, js)
            pr = self.prediction.predict_many(hours, dows, cl, js, name, reference="typical")
            self._ctx[key] = {(int(j), int(d), int(h)): (int(c), float(p))
                              for j, d, h, c, p in zip(js, dows, hours, cl, pr)}
        return self._ctx[key]

    @staticmethod
    def fixed_policy(env):
        return 1 if env.time_in_phase >= 30 else 0

    def train(self, episodes=60, steps=200):
        with self.lock:
            ctx = self._context("auto")
            envs = {j: self._new_env(j, seed=100 + j, max_steps=steps) for j in self.junctions}
            eps_decay = max(1, int(episodes * 0.7))
            rewards, losses = [], []
            for ep in range(episodes):
                eps = max(0.05, 1.0 - 0.95 * ep / eps_decay)
                j = random.choice(self.junctions)
                cl = random.randrange(self.n_clusters)
                keys = [k for k, v in ctx.items() if k[0] == j and v[0] == cl] or \
                       [k for k in ctx if k[0] == j]
                _, dow, hour = random.choice(keys)
                env = envs[j]
                c, p = ctx[(j, dow, hour)]
                env.set_context(hour, dow, c, p)
                s, total = env.reset(), 0.0
                for t in range(steps):
                    a = self.agent.act(s, eps)
                    s2, r, done, _ = env.step(a)
                    self.agent.buffer.push(s, a, r, s2, 0.0)
                    if t % 2 == 0:
                        l = self.agent.learn()
                        if l is not None:
                            losses.append(l)
                    s, total = s2, total + r
                    if done:
                        break
                rewards.append(round(total, 2))
            self.trained = True
            n = max(1, min(5, episodes // 2))
            self.report = {"episodes": episodes, "steps": steps,
                           "reward_first": round(float(np.mean(rewards[:n])), 2),
                           "reward_last": round(float(np.mean(rewards[-n:])), 2),
                           "mean_loss": round(float(np.mean(losses[-200:])), 4) if losses else None,
                           "evaluation": self.evaluate()}
            return self.report

    def evaluate(self, episodes=12, steps=300):
        ctx = self._context("auto")
        res = {"dqn": [], "fixed": []}
        by_j = {j: {"dqn": [], "fixed": []} for j in self.junctions}
        for i in range(episodes):
            rnd = random.Random(i)
            j = self.junctions[i % len(self.junctions)]
            cl = (i // len(self.junctions)) % self.n_clusters
            keys = [k for k, v in ctx.items() if k[0] == j and v[0] == cl] or \
                   [k for k in ctx if k[0] == j]
            _, dow, hour = rnd.choice(keys)
            for name in ("dqn", "fixed"):
                env = self._new_env(j, seed=1000 + i, max_steps=steps)
                c, p = ctx[(j, dow, hour)]
                env.set_context(hour, dow, c, p)
                s, done = env.reset(), False
                while not done:
                    a = self.agent.act(s, 0.0) if name == "dqn" else self.fixed_policy(env)
                    s, _, done, _ = env.step(a)
                row = (env.total_wait / steps, env.total_departed, env.overflow)
                res[name].append(row)
                by_j[j][name].append(row[0])
        out = {}
        for name, rows in res.items():
            arr = np.array(rows, dtype=float)
            out[name] = {"avg_queue": round(float(arr[:, 0].mean()), 2),
                         "throughput": round(float(arr[:, 1].mean()), 1),
                         "overflow": round(float(arr[:, 2].mean()), 1)}
        base = out["fixed"]["avg_queue"]
        out["improvement_pct"] = round(100 * (base - out["dqn"]["avg_queue"]) / base, 1) if base else 0.0
        out["by_intersection"] = {str(j): {"dqn": round(float(np.mean(v["dqn"])), 2),
                                           "fixed": round(float(np.mean(v["fixed"])), 2)}
                                  for j, v in by_j.items() if v["dqn"]}
        return out

    def options(self):
        models = [{"key": "auto", "label": "Best model (auto)", "available": True}]
        models += self.prediction.models_info()
        models.append({"key": "historical", "label": "Historical average", "available": True})
        return {"date_min": self.date_min, "date_max": self.date_max,
                "intersections": [{"junction": j, "id": i["id"], "address": i["address"]}
                                  for j, i in self.inters.items()],
                "models": models, "speeds": self.SPEEDS, "config": self.cfg,
                "trained": self.trained}

    def configure(self, cfg=None):
        with self.lock:
            cfg = cfg or {}
            try:
                day = datetime.strptime(str(cfg.get("date", "")), "%Y-%m-%d").date()
            except ValueError:
                day = datetime.strptime(self.date_max, "%Y-%m-%d").date()
            lo = datetime.strptime(self.date_min, "%Y-%m-%d").date()
            hi = datetime.strptime(self.date_max, "%Y-%m-%d").date()
            day = min(max(day, lo), hi)
            try:
                hour = min(max(int(cfg.get("hour", 7)), 0), 23)
            except (TypeError, ValueError):
                hour = 7
            try:
                junction = int(cfg.get("junction"))
                junction = junction if junction in self.inters else "all"
            except (TypeError, ValueError):
                junction = "all"
            model = cfg.get("model") if cfg.get("model") in self.MODELS else "auto"
            policy = cfg.get("policy") if cfg.get("policy") in ("dqn", "fixed") else "dqn"
            source = cfg.get("source") if cfg.get("source") in ("forecast", "observed") else "forecast"
            try:
                speed = int(cfg.get("speed", 1))
            except (TypeError, ValueError):
                speed = 1
            speed = speed if speed in self.SPEEDS else 1
            self.cfg = {"date": day.isoformat(), "hour": hour, "junction": junction, "model": model,
                        "policy": policy, "source": source, "speed": speed}
            self.active = self.junctions if junction == "all" else [junction]
            self.sim_dt = datetime.combine(day, time(hour))
            self.live_steps = 0
            self.envs = {j: self._new_env(j, seed=7 + j) for j in self.active}
            self.forecast, self.observed, self.src = {}, {}, {}
            self._apply_context()
            self.states = {j: env.reset() for j, env in self.envs.items()}
            self.last = {j: {"action": 0, "reward": 0.0} for j in self.active}
            return self.cfg

    def reset_live(self):
        return self.configure(self.cfg)

    def _apply_context(self):
        ctx = self._context(self.cfg["model"])
        dow, hour, day = self.sim_dt.weekday(), self.sim_dt.hour, self.sim_dt.strftime("%Y-%m-%d")
        for j, env in self.envs.items():
            c, p = ctx[(j, dow, hour)]
            obs = self.loader.observed(j, day, hour)
            self.forecast[j], self.observed[j] = p, obs
            if self.cfg["source"] == "observed" and obs is not None:
                demand, self.src[j] = obs, "observed"
            else:
                demand = p
                self.src[j] = "forecast (no record)" if self.cfg["source"] == "observed" else "forecast"
            env.set_context(hour, dow, c, demand)

    def simulation_step(self):
        with self.lock:
            use_dqn = self.cfg["policy"] == "dqn" and self.trained
            for _ in range(self.cfg["speed"]):
                if self.live_steps > 0 and self.live_steps % self.CONTEXT_STEPS == 0:
                    self.sim_dt += timedelta(hours=1)
                    self._apply_context()
                for j, env in self.envs.items():
                    s = self.states[j]
                    a = self.agent.act(s, 0.0) if use_dqn else self.fixed_policy(env)
                    s2, r, _, info = env.step(a)
                    self.agent.buffer.push(s, a, r, s2, 0.0)
                    self.states[j] = s2
                    self.last[j] = {"action": int(a), "reward": r}
                self.live_steps += 1
            if use_dqn:
                self.agent.learn()
            return self._payload(use_dqn)

    def _payload(self, use_dqn):
        items, tot_q, tot_dep, tot_avg = [], 0, 0, 0.0
        for j, env in self.envs.items():
            snap = env.snapshot()
            last = self.last[j]
            avg = env.total_wait / max(self.live_steps, 1)
            snap.update({
                "junction": j, "id": self.inters[j]["id"], "address": self.inters[j]["address"],
                "cluster": env.cluster, "cluster_label": self.clustering.label(env.cluster),
                "cluster_color": self.clustering.color(env.cluster),
                "forecast": round(self.forecast[j], 1),
                "observed": None if self.observed[j] is None else round(self.observed[j], 1),
                "demand_source": self.src[j], "demand_used": round(env.predicted, 1),
                "action": last["action"], "action_name": ACTION_NAMES[last["action"]],
                "reward": round(last["reward"], 3), "total_queue": int(env.queues.sum()),
                "avg_queue": round(avg, 2), "departed_total": int(env.total_departed),
                "access_names": TrafficEnv.ACCESS_NAMES})
            items.append(snap)
            tot_q += snap["total_queue"]
            tot_dep += snap["departed_total"]
            tot_avg += avg
        return {"t": self.live_steps, "datetime": self.sim_dt.strftime("%Y-%m-%d %H:00"),
                "day_name": DIAS[self.sim_dt.weekday()], "sim_hour": self.sim_dt.hour,
                "seconds_to_next_hour": self.CONTEXT_STEPS - (self.live_steps % self.CONTEXT_STEPS),
                "policy": "DQN agent" if use_dqn else (
                    "Fixed time (30 s)" if self.cfg["policy"] == "fixed" else "Fixed time (DQN not trained)"),
                "trained": bool(self.trained), "config": self.cfg,
                "model_label": self.prediction.model_label(self.cfg["model"]),
                "totals": {"total_queue": tot_q, "departed_total": tot_dep,
                           "avg_queue": round(tot_avg / max(len(items), 1), 2)},
                "intersections": items}
