"""Indicators for the reports page and monitoring checks (model performance and input changes)."""
import json
import math

from models.database import q, q1

LATENCY_ALERT_MS = 1000
MIN_OUTCOMES = 10          # outcomes needed before the live error is judged
MAE_ALERT_FACTOR = 1.5     # alert when the live MAE exceeds 1.5 x the MAE of the independent test
RANGE_SHARE_ALERT = 0.30   # alert when this share of requests asks for dates beyond the training range


def _p95(values):
    values = sorted(values)
    return round(values[int(0.95 * (len(values) - 1))], 1) if values else None


def indicators(user):
    admin = user["role"] == "admin"
    scope_sql = "" if admin else " AND user_id = %d" % int(user["id"])
    inf = q1("SELECT COUNT(*) AS n, AVG(latency_ms) AS lat, SUM(CASE WHEN observed_vehicles IS NOT NULL THEN 1 ELSE 0 END) AS obs, "
             "AVG(abs_error) AS mae, SUM(observed_vehicles) AS tot_obs, SUM(abs_error) AS tot_err "
             "FROM inferences WHERE 1 = 1" + scope_sql)
    lat = [r["latency_ms"] for r in q("SELECT latency_ms FROM inferences WHERE latency_ms IS NOT NULL" + scope_sql)]
    by_inter = q("SELECT i.code AS code, i.name AS name, COUNT(f.id) AS n, AVG(f.predicted_vehicles) AS avg_pred, "
                 "AVG(f.abs_error) AS mae FROM intersections i LEFT JOIN inferences f ON f.intersection_id = i.id"
                 + ("" if admin else " AND f.user_id = %d" % int(user["id"])) + " GROUP BY i.code, i.name ORDER BY i.code")
    plans = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) AS n FROM timing_plans GROUP BY status")}
    decided = sum(plans.get(s, 0) for s in ("approved", "rejected", "applied"))
    adj = q1("SELECT AVG(ABS(final_green_s - model_green_s)) AS d FROM timing_plans")
    incs = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) AS n FROM incidents GROUP BY status")}
    sev = q("SELECT severity, COUNT(*) AS n FROM incidents WHERE status <> 'closed' GROUP BY severity")
    per_day = q("SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*) AS n FROM inferences WHERE 1 = 1" + scope_sql +
                " GROUP BY SUBSTR(created_at, 1, 10) ORDER BY day DESC LIMIT 14")
    return {
        "inferences": inf["n"], "with_outcome": inf["obs"] or 0,
        "mae": round(inf["mae"], 2) if inf["mae"] is not None else None,
        "wape": round(100 * inf["tot_err"] / inf["tot_obs"], 1) if inf["tot_obs"] else None,
        "latency_mean": round(inf["lat"], 1) if inf["lat"] is not None else None, "latency_p95": _p95(lat),
        "by_intersection": [{**r, "avg_pred": round(r["avg_pred"], 1) if r["avg_pred"] is not None else None,
                             "mae": round(r["mae"], 2) if r["mae"] is not None else None} for r in by_inter],
        "plans": plans, "plans_total": sum(plans.values()),
        "approval_rate": round(100 * (plans.get("approved", 0) + plans.get("applied", 0)) / decided, 1) if decided else None,
        "mean_adjustment": round(adj["d"], 1) if adj and adj["d"] is not None else None,
        "incidents": incs, "incidents_open_by_severity": {r["severity"]: r["n"] for r in sev},
        "per_day": list(reversed(per_day)),
    }


def monitoring():
    """Checks for model performance and input changes (administrators)."""
    version = q1("SELECT * FROM model_versions WHERE is_current = 1")
    versions = q("SELECT * FROM model_versions ORDER BY id DESC")
    out = q("SELECT abs_error, observed_vehicles FROM inferences WHERE observed_vehicles IS NOT NULL "
            "ORDER BY observed_at DESC LIMIT 50")
    live_mae = round(sum(r["abs_error"] for r in out) / len(out), 2) if out else None
    threshold = round(MAE_ALERT_FACTOR * version["test_mae"], 2) if version and version["test_mae"] else None
    total = q1("SELECT COUNT(*) AS n FROM inferences")["n"]
    far = 0
    for r in q("SELECT notes_json FROM inferences ORDER BY id DESC LIMIT 200"):
        try:
            if any("more than 60 days" in n for n in json.loads(r["notes_json"] or "[]")):
                far += 1
        except ValueError:
            pass
    recent = min(total, 200)
    far_share = round(far / recent, 3) if recent else 0
    errors7 = q1("SELECT COUNT(*) AS n FROM app_logs WHERE level = 'error'")["n"]
    lat = [r["latency_ms"] for r in q("SELECT latency_ms FROM app_logs WHERE event = 'inference' AND latency_ms IS NOT NULL "
                                      "ORDER BY id DESC LIMIT 200")]
    p95 = _p95(lat)
    alerts = []
    if live_mae is not None and len(out) >= MIN_OUTCOMES and threshold and live_mae > threshold:
        alerts.append(f"Model performance: live MAE {live_mae} veh/h is above the threshold {threshold}. Consider retraining.")
    if far_share > RANGE_SHARE_ALERT:
        alerts.append(f"Input change: {round(100 * far_share)}% of the latest requests ask for dates well beyond the "
                      f"training data (more than 60 days).")
    if p95 is not None and p95 > LATENCY_ALERT_MS:
        alerts.append(f"Response time: p95 latency {p95} ms is above {LATENCY_ALERT_MS} ms.")
    return {"version": version, "versions": versions, "live_mae": live_mae, "outcomes": len(out),
            "min_outcomes": MIN_OUTCOMES, "threshold": threshold, "far_share": far_share, "latency_p95": p95,
            "errors": errors7, "alerts": alerts,
            "recent_logs": q("SELECT * FROM app_logs ORDER BY id DESC LIMIT 40"),
            "recent_errors": q("SELECT * FROM app_logs WHERE level = 'error' ORDER BY id DESC LIMIT 15")}
