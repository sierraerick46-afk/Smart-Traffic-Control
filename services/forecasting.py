"""Server-side inference: validates the input, runs the saved model and stores the result in the history."""
import json
import time
from datetime import datetime

import pandas as pd

from models.database import insert, log_event, now_iso, q1

CLUSTER_LABELS = ["Off-peak", "Moderate", "Peak"]
MIN_DATE, MAX_DATE = datetime(2015, 1, 1), datetime(2030, 12, 31)


class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


class ForecastService:
    def __init__(self, prediction, loader, monitor):
        self.prediction, self.loader, self.monitor = prediction, loader, monitor

    # -- input ------------------------------------------------------------------------------------
    def _parse(self, data):
        errors, inter, dt = {}, None, None
        raw_i, raw_j = data.get("intersection_id"), data.get("junction")
        try:
            if raw_i not in (None, ""):
                inter = q1("SELECT * FROM intersections WHERE id = ?", (int(raw_i),))
            elif raw_j not in (None, ""):
                inter = q1("SELECT * FROM intersections WHERE junction = ?", (int(raw_j),))
            else:
                errors["intersection_id"] = "Choose an intersection."
        except (TypeError, ValueError):
            errors["intersection_id"] = "Invalid intersection."
        if inter is not None:
            if not inter["is_active"]:
                errors["intersection_id"] = "That intersection is disabled."
            elif inter["junction"] is None:
                errors["intersection_id"] = "That intersection has no historical series, so the model cannot forecast it."
        elif "intersection_id" not in errors:
            errors["intersection_id"] = "Intersection not found."
        try:
            if data.get("datetime"):
                dt = pd.Timestamp(data["datetime"])
            else:
                hour = int(data.get("hour"))
                if not 0 <= hour <= 23:
                    raise ValueError
                dt = pd.Timestamp(f'{str(data.get("date")).strip()} {hour:02d}:00')
            dt = dt.floor("h")
            if not MIN_DATE <= dt.to_pydatetime() <= MAX_DATE:
                errors["date"] = "The date must be between 2015-01-01 and 2030-12-31."
        except (TypeError, ValueError):
            errors["date"] = "Write a valid date (YYYY-MM-DD) and an hour from 0 to 23."
        if errors:
            raise ValidationError(errors)
        return inter, dt

    # -- inference ----------------------------------------------------------------------------------
    def run(self, user, data, source="web"):
        t0 = time.perf_counter()
        try:
            inter, dt = self._parse(data)
            name = self.prediction.best_model
            if name is None:
                raise ValidationError({"model": "No trained model is available. Run python train.py first."})
            feat = self.prediction.featurizers[name]
            junction = int(inter["junction"])
            day = pd.date_range(dt.normalize(), periods=24, freq="h")
            curve = self.prediction.predict_datetimes(day, [junction] * 24, name)
            value = float(curve[dt.hour])
            cluster = int(feat.clusters([dt.hour], [dt.dayofweek], [junction])[0])
            notes = []
            if dt > feat.last_date_ + pd.Timedelta(days=60):
                notes.append("The date is more than 60 days after the training data: tree models keep the last "
                             "learned traffic level, so the forecast may underestimate growth.")
            if dt < feat.t0_:
                notes.append("The date is before the training data.")
            art = self.prediction.artifact or {}
            version = art.get("version") or "unversioned"
            latency = round((time.perf_counter() - t0) * 1000, 1)
            row = {"user_id": user["id"], "intersection_id": inter["id"],
                   "target_datetime": dt.strftime("%Y-%m-%d %H:00"), "predicted_vehicles": round(value, 1),
                   "group_label": CLUSTER_LABELS[cluster], "model_name": self.prediction.model_label(name),
                   "model_version": version, "latency_ms": latency, "source": source,
                   "curve_json": json.dumps([round(float(v), 1) for v in curve]),
                   "notes_json": json.dumps(notes), "created_at": now_iso()}
            new_id = insert("inferences", row)
            log_event("info", "inference", user_id=user["id"], endpoint=f"/{source}/forecast", status_code=200,
                      latency_ms=latency, detail=f'{inter["code"]} {row["target_datetime"]} -> {row["predicted_vehicles"]}')
            self.monitor.log_prediction({"source": source, "user": user["username"],
                                         "input": {"datetime": row["target_datetime"], "junction": junction},
                                         "output": row["predicted_vehicles"], "model": row["model_name"],
                                         "model_version": version, "latency_ms": latency})
            return {**row, "id": new_id, "intersection": inter["name"], "intersection_code": inter["code"],
                    "curve": json.loads(row["curve_json"]), "notes": notes}
        except ValidationError as e:
            log_event("warning", "inference_rejected", user_id=user["id"], endpoint=f"/{source}/forecast",
                      status_code=400, detail=str(e))
            raise
        except Exception as e:  # unexpected failure while predicting
            log_event("error", "inference_error", user_id=user["id"], endpoint=f"/{source}/forecast",
                      status_code=500, detail=repr(e))
            self.monitor.log_error({"endpoint": f"/{source}/forecast", "input": data, "error": repr(e)})
            raise

    # -- real outcome -----------------------------------------------------------------------------
    def dataset_value(self, inference):
        """Value present in the historical counts for that intersection and hour (None if the date is not in it)."""
        inter = q1("SELECT junction FROM intersections WHERE id = ?", (inference["intersection_id"],))
        if not inter or inter["junction"] is None:
            return None
        date_str, hour = inference["target_datetime"][:10], int(inference["target_datetime"][11:13])
        return self.loader.observed(inter["junction"], date_str, hour)


def record_outcome(inference, observed, user):
    """Stores the real count and the absolute error so predictions can be compared with outcomes."""
    from models.database import run
    try:
        observed = float(observed)
    except (TypeError, ValueError):
        return "Write the observed vehicles per hour as a number."
    if not 0 <= observed <= 2000:
        return "The observed value must be between 0 and 2000 vehicles per hour."
    run("UPDATE inferences SET observed_vehicles = ?, observed_by = ?, observed_at = ?, abs_error = ? WHERE id = ?",
        (observed, user["id"], now_iso(), round(abs(observed - inference["predicted_vehicles"]), 2), inference["id"]))
    log_event("info", "outcome_recorded", user_id=user["id"], detail=f'inference {inference["id"]}: {observed}')
    return None
