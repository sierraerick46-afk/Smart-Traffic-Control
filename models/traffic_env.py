import numpy as np


class TrafficEnv:
    N_ACCESSES = 4
    ACCESS_NAMES = ["North", "South", "East", "West"]
    PHASE_ACCESSES = {0: [0, 1], 1: [2, 3]}
    PHASE_NAMES = {0: "North-South", 1: "East-West"}
    N_ACTIONS = 2
    STATE_DIM = 9

    MIN_GREEN = 8        
    MAX_GREEN = 60       
    YELLOW = 3           
    QUEUE_CAP = 40       
    SAT_PROB = 0.55     
    SPLIT = np.array([0.28, 0.22, 0.28, 0.22])   

    def __init__(self, n_clusters=3, ref_demand=60.0, seed=None, max_steps=None):
        self.n_clusters = n_clusters
        self.ref_demand = max(float(ref_demand), 1.0)
        self.max_steps = max_steps
        self.rng_arr = np.random.default_rng(seed)                       
        self.rng_srv = np.random.default_rng(None if seed is None else seed + 1)  
        self.hour, self.dow, self.cluster, self.predicted = 12, 0, 0, self.ref_demand * 0.5
        self.rate = 0.5
        self.reset()

    
    def set_context(self, hour, dow, cluster, predicted):
        self.hour, self.dow = int(hour), int(dow)
        self.cluster, self.predicted = int(cluster), float(predicted)
        intensity = min(self.predicted / self.ref_demand, 1.5)
        self.rate = 0.12 + 0.88 * intensity          
    def reset(self):
        self.queues = self.rng_arr.integers(0, 7, size=self.N_ACCESSES).astype(int)
        self.phase, self.time_in_phase, self.yellow_left = 0, 0, 0
        self.t, self.total_wait, self.total_departed, self.overflow = 0, 0, 0, 0
        return self._state()

    
    def _state(self):
        return np.array([
            *(self.queues / self.QUEUE_CAP),
            float(self.phase),
            min(self.time_in_phase / self.MAX_GREEN, 1.0),
            1.0 if self.yellow_left > 0 else 0.0,
            self.cluster / max(1, self.n_clusters - 1),
            min(self.predicted / self.ref_demand, 1.5) / 1.5,
        ], dtype=np.float32)

    def signals(self):
        out = []
        for a in range(self.N_ACCESSES):
            if a in self.PHASE_ACCESSES[self.phase]:
                out.append("yellow" if self.yellow_left > 0 else "green")
            else:
                out.append("red")
        return out

    
    def step(self, action):
        changed, in_transition = False, False
        if self.yellow_left > 0:                       
            in_transition = True
            self.yellow_left -= 1
            if self.yellow_left == 0:
                self.phase = 1 - self.phase
                self.time_in_phase = 0
        else:
            can_change = self.time_in_phase >= self.MIN_GREEN
            must_change = self.time_in_phase >= self.MAX_GREEN
            if must_change or (action == 1 and can_change):
                self.yellow_left = self.YELLOW - 1
                in_transition, changed = True, True
            else:
                self.time_in_phase += 1

        departed = 0
        if not in_transition:                          
            for a in self.PHASE_ACCESSES[self.phase]:
                if self.queues[a] > 0 and self.rng_srv.random() < self.SAT_PROB:
                    self.queues[a] -= 1
                    departed += 1

        arrivals = self.rng_arr.poisson(self.rate * self.SPLIT)   
        for a in range(self.N_ACCESSES):
            new = int(self.queues[a]) + int(arrivals[a])
            self.overflow += max(0, new - self.QUEUE_CAP)
            self.queues[a] = min(new, self.QUEUE_CAP)

        total_q, max_q = int(self.queues.sum()), int(self.queues.max())
        
        reward = -total_q / 40.0 - max_q / 80.0 + 0.2 * departed - (0.2 if changed else 0.0)

        self.t += 1
        self.total_wait += total_q                     
        self.total_departed += departed
        done = self.max_steps is not None and self.t >= self.max_steps
        info = {"departed": departed, "total_queue": total_q, "max_queue": max_q, "changed": changed}
        return self._state(), float(reward), done, info

    def snapshot(self):
        return {
            "queues": [int(q) for q in self.queues],
            "signals": self.signals(),
            "phase": int(self.phase),
            "phase_name": self.PHASE_NAMES[self.phase],
            "yellow": bool(self.yellow_left > 0),
            "time_in_phase": int(self.time_in_phase),
            "queue_cap": self.QUEUE_CAP,
        }
