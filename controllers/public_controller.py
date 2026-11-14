import time

import pandas as pd
from flask import Blueprint, jsonify, render_template, request

CLUSTER_LABELS = ["Off-peak", "Moderate", "Peak"]


def create_public_blueprint(services):
    bp = Blueprint("public", __name__)
    prediction, loader, monitor = services["prediction"], services["loader"], services["monitor"]

    def intersections():
        return {i["junction"]: i["address"] for i in loader.intersections()}

    def forecast(payload, source):
        t = time.perf_counter()
        inters = intersections()
        try:
            dt = pd.Timestamp(payload.get("datetime") or f'{payload.get("date")} {int(payload.get("hour", 0)):02d}:00')
        except (ValueError, TypeError):
            raise ValueError("Invalid date or hour. Use YYYY-MM-DD and an hour between 0 and 23.")
        dt = dt.floor("h")
        try:
            junction = int(payload.get("junction"))
        except (TypeError, ValueError):
            raise ValueError("Missing or invalid junction (1-4).")
        if junction not in inters:
            raise ValueError(f"Unknown junction {junction}. Valid values: {sorted(inters)}.")
        name = prediction.best_model
        if name is None:
            raise ValueError("No trained model available. Run python train.py first.")
        feat = prediction.featurizers[name]
        day = pd.date_range(dt.normalize(), periods=24, freq="h")
        curve = prediction.predict_datetimes(day, [junction] * 24, name)
        value = float(curve[dt.hour])
        cluster = int(feat.clusters([dt.hour], [dt.dayofweek], [junction])[0])
        notes = []
        if dt > feat.last_date_ + pd.Timedelta(days=60):
            notes.append("The date is more than 60 days after the training data: tree models keep the last "
                         "learned traffic level, so the forecast may underestimate growth.")
        if dt < feat.t0_:
            notes.append("The date is before the training data.")
        art = prediction.artifact or {}
        out = {"datetime": dt.strftime("%Y-%m-%d %H:00"), "junction": junction,
               "intersection": inters[junction], "vehicles_per_hour": round(value, 1),
               "group": CLUSTER_LABELS[cluster], "model": prediction.model_label(name),
               "model_version": art.get("version", f"runtime-{prediction.version}"),
               "hourly_curve": [round(float(v), 1) for v in curve], "notes": notes}
        monitor.log_prediction({"source": source, "input": {"datetime": out["datetime"], "junction": junction},
                                "output": out["vehicles_per_hour"], "model": out["model"],
                                "model_version": out["model_version"],
                                "latency_ms": round((time.perf_counter() - t) * 1000, 1)})
        return out

    @bp.route("/demo")
    def demo():
        art = prediction.artifact or {}
        return render_template("demo.html", active="demo", intersections=intersections(),
                               model_label=prediction.model_label(), version=art.get("version", "–"),
                               test=(art.get("test_metrics") or {}), trained_on=art.get("trained_on", "–"))

    @bp.post("/v1/forecast")
    def api_forecast():
        payload = request.get_json(silent=True) or request.form.to_dict()
        try:
            return jsonify({"ok": True, **forecast(payload, "api")})
        except ValueError as e:
            monitor.log_error({"endpoint": "/v1/forecast", "input": payload, "error": str(e)})
            return jsonify({"ok": False, "error": str(e)}), 400

    @bp.post("/v1/feedback")
    def api_feedback():
        p = request.get_json(silent=True) or request.form.to_dict()
        kind = p.get("type", "feedback")
        if kind not in ("feedback", "incident"):
            kind = "feedback"
        msg = (p.get("message") or "").strip()[:1000]
        if not msg:
            return jsonify({"ok": False, "error": "Write a message."}), 400
        rec = monitor.log_feedback({"type": kind, "message": msg, "name": (p.get("name") or "")[:80],
                                    "context": p.get("context")})
        return jsonify({"ok": True, "logged": rec})

    @bp.get("/v1/monitoring")
    def api_monitoring():
        return jsonify(monitor.summary())

    return bp
