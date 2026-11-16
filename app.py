import logging
import os

from flask import Flask

from controllers.api_controller import create_api_blueprint
from controllers.auth_controller import create_auth_blueprint
from controllers.ops_controller import create_ops_blueprint
from controllers.web_controller import create_web_blueprint
from models import database
from models.data_loader import DataLoader
from services import seed
from services.clustering_service import ClusteringService
from services.forecasting import ForecastService
from services.monitoring_service import MonitoringService
from services.plans import PlanService
from services.prediction_service import PredictionService

log = logging.getLogger("smart_traffic")


def create_app(db=None, light=False):
    """Application factory. `db` lets tests use a temporary database; `light` skips the DQN control service."""
    app = Flask(__name__)
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        log.warning("SECRET_KEY is not set: using an insecure development key. Set it in production.")
    app.secret_key = secret or "dev-only-key-change-me"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE") == "1"
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

    database.init_app(app, db)

    loader = DataLoader()
    services = {"loader": loader}
    clustering = ClusteringService(loader).fit()
    services["clustering"] = clustering
    prediction = PredictionService(loader, clustering)
    services["prediction"] = prediction
    if not light:
        from services.control_service import ControlService
        services["control"] = ControlService(clustering, prediction, loader)
    else:
        services["control"] = None
    monitor = MonitoringService()
    services["monitor"] = monitor
    services["forecasting"] = ForecastService(prediction, loader, monitor)
    services["plans"] = PlanService(loader)

    with app.app_context():
        seed.seed_all(loader, prediction)

    app.register_blueprint(create_auth_blueprint())
    app.register_blueprint(create_web_blueprint(services))
    app.register_blueprint(create_api_blueprint(services))
    app.register_blueprint(create_ops_blueprint(services))
    return app


app = create_app() if os.environ.get("SKIP_APP_AT_IMPORT") != "1" else None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
