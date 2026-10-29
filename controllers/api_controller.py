from flask import Blueprint, jsonify, request

from models.data_loader import DIAS, now_bogota


def create_api_blueprint(svc):
    bp = Blueprint("api", __name__, url_prefix="/api")
    loader, clustering = svc["loader"], svc["clustering"]

    def _slot():
        now = now_bogota()
        h = request.args.get("hora", default=now.hour, type=int)
        d = request.args.get("dia", default=now.weekday(), type=int)
        return max(0, min(23, h)), max(0, min(6, d))

    @bp.get("/status")
    def status():
        return jsonify({"data": loader.summary(), "clusters_ready": clustering.fitted,
                        "dias": DIAS})

    @bp.get("/clusters")
    def clusters():
        h, d = _slot()
        return jsonify({"current": clustering.describe(h, d),
                        "profile": clustering.hourly_profile(d),
                        "summary": clustering.summary(),
                        "silhouette": clustering.silhouette})
    return bp