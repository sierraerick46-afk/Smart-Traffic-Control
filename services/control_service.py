
import random
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models.data_loader import DIAS, now_bogota
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

    def __init__(self, clustering, prediction, data_loader):
        self.clustering, self.prediction = clustering, prediction
        df = data_loader.get_data()
        self.ref_demand = float(max(np.percentile(df["vehicles"], 95), 1.0))
        self.n_clusters = clustering.n_clusters
        self.agent = DQNAgent(TrafficEnv.STATE_DIM, TrafficEnv.N_ACTIONS)
        self.trained, self.report = False, None
        self.lock = threading.RLock()
        self._ctx, self._ctx_version = None, -1
        self.live_env = TrafficEnv(self.n_clusters, self.ref_demand, seed=7)
        self.reset_live()

    
    def _context(self):
        
        if self._ctx is None or self._ctx_version != self.prediction.version:
            hours, dows = np.tile(np.arange(24), 7), np.repeat(np.arange(7), 24)
            cl = self.clustering.cluster_array(hours, dows)
            pr = self.prediction.predict_many(hours, dows, cl)
            self._ctx = {(int(d), int(h)): (int(c), float(p)) for h, d, c, p in zip(hours, dows, cl, pr)}
            self._ctx_version = self.prediction.version
        return self._ctx

    def _apply(self, env, hour, dow):
        c, p = self._context()[(dow, hour)]
        env.set_context(hour, dow, c, p)

    @staticmethod
    def fixed_policy(env):
        return 1 if env.time_in_phase >= 30 else 0       

    
    def train(self, episodes=60, steps=200):
        with self.lock:
            ctx = self._context()
            by_cluster = {c: [k for k, v in ctx.items() if v[0] == c] for c in range(self.n_clusters)}
            by_cluster = {c: ks for c, ks in by_cluster.items() if ks}
            env = TrafficEnv(self.n_clusters, self.ref_demand, seed=123, max_steps=steps)
            eps_decay = max(1, int(episodes * 0.7))
            rewards, losses = [], []
            for ep in range(episodes):
                eps = max(0.05, 1.0 - 0.95 * ep / eps_decay)
                dow, hour = random.choice(by_cluster[random.choice(list(by_cluster))])
                self._apply(env, hour, dow)
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

    def evaluate(self, episodes=9, steps=300):
        
        res = {"dqn": [], "fixed": []}
        ctx = self._context()
        for i in range(episodes):
            rnd = random.Random(i)
            keys = [k for k, v in ctx.items() if v[0] == i % self.n_clusters] or list(ctx)
            dow, hour = rnd.choice(keys)
            for name in ("dqn", "fixed"):
                env = TrafficEnv(self.n_clusters, self.ref_demand, seed=1000 + i, max_steps=steps)
                self._apply(env, hour, dow)
                s, done = env.reset(), False
                while not done:
                    a = self.agent.act(s, 0.0) if name == "dqn" else self.fixed_policy(env)
                    s, _, done, _ = env.step(a)
                res[name].append((env.total_wait / steps, env.total_departed, env.overflow))
        out = {}
        for name, rows in res.items():
            arr = np.array(rows, dtype=float)
            out[name] = {"avg_queue": round(float(arr[:, 0].mean()), 2),
                         "throughput": round(float(arr[:, 1].mean()), 1),
                         "overflow": round(float(arr[:, 2].mean()), 1)}
        base = out["fixed"]["avg_queue"]
        out["improvement_pct"] = round(100 * (base - out["dqn"]["avg_queue"]) / base, 1) if base else 0.0
        return out

    
    def reset_live(self, hour=None, dow=None):
        with self.lock:
            now = now_bogota()
            self.sim_hour = now.hour if hour is None else int(hour)
            self.sim_dow = now.weekday() if dow is None else int(dow)
            self.live_steps, self.last = 0, {"action": 0, "reward": 0.0}
            self._apply(self.live_env, self.sim_hour, self.sim_dow)
            self.state = self.live_env.reset()

    def simulation_step(self):
        with self.lock:
            env = self.live_env
            if self.live_steps > 0 and self.live_steps % self.CONTEXT_STEPS == 0:
                self.sim_hour = (self.sim_hour + 1) % 24
                if self.sim_hour == 0:
                    self.sim_dow = (self.sim_dow + 1) % 7
            self._apply(env, self.sim_hour, self.sim_dow)   

            use_dqn = self.trained
            action = self.agent.act(self.state, 0.02) if use_dqn else self.fixed_policy(env)
            s2, reward, _, info = env.step(action)
            self.agent.buffer.push(self.state, action, reward, s2, 0.0)   
            if use_dqn and self.live_steps % 4 == 0:
                self.agent.learn()                                       
            self.state = s2
            self.live_steps += 1

            snap = env.snapshot()
            snap.update({
                "t": self.live_steps,
                "sim_hour": self.sim_hour, "sim_day": self.sim_dow, "day_name": DIAS[self.sim_dow],
                "cluster": env.cluster, "cluster_label": self.clustering.label(env.cluster),
                "cluster_color": self.clustering.color(env.cluster),
                "predicted": round(env.predicted, 1),
                "action": int(action), "action_name": ACTION_NAMES[int(action)],
                "reward": round(reward, 3), "total_queue": info["total_queue"],
                "avg_queue": round(env.total_wait / self.live_steps, 2),
                "departed_total": int(env.total_departed),
                "policy": "DQN" if use_dqn else "Fixed time (DQN not trained)",
                "trained": bool(self.trained),
                "access_names": TrafficEnv.ACCESS_NAMES,
                "seconds_to_next_hour": self.CONTEXT_STEPS - (self.live_steps % self.CONTEXT_STEPS),
            })
            return snap
