import json
import os
import threading
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(ROOT, "logs"))


class MonitoringService:
    def __init__(self, log_dir=LOG_DIR, keep=200):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.recent_predictions = deque(maxlen=keep)
        self.recent_feedback = deque(maxlen=keep)
        self.counters = {"predictions": 0, "errors": 0, "feedback": 0, "incidents": 0}
        self.latencies = deque(maxlen=keep)
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def _append(self, name, record):
        with self._lock:
            with open(os.path.join(self.log_dir, name), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_prediction(self, record):
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        self.recent_predictions.appendleft(record)
        self.counters["predictions"] += 1
        if "latency_ms" in record:
            self.latencies.append(record["latency_ms"])
        self._append("predictions.jsonl", record)
        return record

    def log_error(self, record):
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "level": "error", **record}
        self.counters["errors"] += 1
        self._append("errors.jsonl", record)

    def log_feedback(self, record):
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        self.recent_feedback.appendleft(record)
        self.counters["feedback"] += 1
        if record.get("type") == "incident":
            self.counters["incidents"] += 1
        self._append("feedback.jsonl", record)
        return record

    def summary(self):
        lat = sorted(self.latencies)
        p95 = lat[int(0.95 * (len(lat) - 1))] if lat else None
        return {"since": self.started_at, **self.counters,
                "latency_ms_p95": round(p95, 1) if p95 is not None else None,
                "recent_predictions": list(self.recent_predictions)[:10],
                "recent_feedback": list(self.recent_feedback)[:10]}
