from flask import Blueprint, render_template

def create_web_blueprint():
    bp = Blueprint("web", __name__)

    @bp.route("/dashboard")
    def dashboard():
        return render_template("index.html")

    return bp