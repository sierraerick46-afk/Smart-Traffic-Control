from flask import Blueprint, jsonify, request

from models.data_loader import DIAS, now_bogota
from controllers.auth_controller import api_login_required


def create_api_blueprint(svc):
    bp = Blueprint("api", __name__, url_prefix="/api")
    bp.before_request(api_login_required)
    loader, clustering = svc["loader"], svc["clustering"]

    def _slot():
        now = now_bogota()
        h = request.args.get("hora", default=now.hour, type=int)
        d = request.args.get("dia", default=now.weekday(), type=int)
        return max(0, min(23, h)), max(0, min(6, d))

    def _junction():
        j = request.args.get("junction", type=int)
        return j if j in clustering.junctions else None

    @bp.get("/status")
    def status():
        date_min, date_max = loader.date_bounds()
        return jsonify({"data": loader.summary(), "clusters_ready": clustering.fitted,
                        "dias": DIAS, "intersections": loader.intersections(),
                        "date_min": date_min, "date_max": date_max})

    @bp.get("/clusters")
    def clusters():
        h, d, j = *_slot(), _junction()
        return jsonify({"current": clustering.describe(h, d, j),
                        "profile": clustering.hourly_profile(d, j),
                        "summary": clustering.summary(j),
                        "silhouette": clustering.silhouette_for(j)})

    prediction = svc["prediction"]

    @bp.route("/train_prediction", methods=["GET", "POST"])
    def train_prediction():
        try:
            return jsonify({"ok": True, "metrics": prediction.train()})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.get("/prediction_status")
    def prediction_status():
        return jsonify(prediction.status())

    @bp.get("/predict")
    def predict():
        h, d, j = *_slot(), _junction()
        c = clustering.get_cluster(h, d, j)
        return jsonify({"hora": h, "dia": d, "cluster": c, "cluster_label": clustering.label(c),
                        "vehicles": round(prediction.predict(h, d, c, j), 1),
                        "model": prediction.model_label()})

    control = svc["control"]

    @bp.route("/train_control", methods=["POST"])
    def train_control():
        body = request.get_json(silent=True) or {}
        episodes = max(5, min(int(body.get("episodes", 60)), 300))
        try:
            return jsonify({"ok": True, "report": control.train(episodes=episodes)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.get("/simulation_options")
    def simulation_options():
        return jsonify(control.options())

    @bp.post("/simulation_config")
    def simulation_config():
        cfg = control.configure(request.get_json(silent=True) or {})
        return jsonify({"ok": True, "config": cfg})

    @bp.get("/simulation_step")
    def simulation_step():
        return jsonify(control.simulation_step())

    @bp.post("/simulation_reset")
    def simulation_reset():
        control.reset_live()
        return jsonify({"ok": True})

    return bp
