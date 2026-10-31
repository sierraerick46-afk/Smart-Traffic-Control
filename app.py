from flask import Flask
from controllers.api_controller import create_api_blueprint
from controllers.web_controller import create_web_blueprint
from models.data_loader import DataLoader
from services.clustering_service import ClusteringService
from services.prediction_service import PredictionService
from services.control_service import ControlService

def create_app():
    app = Flask(__name__)
    loader = DataLoader()
    services = {"loader": loader}

    clustering = ClusteringService(loader).fit()
    services["clustering"] = clustering

    prediction = PredictionService(loader, clustering)
    services["prediction"] = prediction
    
    services["control"] = ControlService(clustering, prediction, loader)

    app.register_blueprint(create_web_blueprint())
    app.register_blueprint(create_api_blueprint(services))
    return app

app = create_app()          

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)