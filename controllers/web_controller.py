from flask import Blueprint, abort, render_template

from controllers.auth_controller import login_required
from controllers.crisp_content import PHASES, REFERENCES, build_phase, get_phase, load


def create_web_blueprint(services):
    bp = Blueprint("web", __name__)
    loader = services["loader"]

    @bp.app_context_processor
    def inject_crisp():
        return {"crisp_phases": PHASES, "phase": None}

    @bp.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("index.html")

    @bp.route("/simula")
    @login_required
    def simula():
        return render_template("simula.html")

    @bp.route("/crisp-ml")
    def crisp_overview():
        return render_template("crisp_overview.html", active="crisp", references=REFERENCES, rep=load("report.json"))

    @bp.route("/crisp-ml/<int:num>")
    def crisp_phase(num):
        if get_phase(num) is None:
            abort(404)
        phase = build_phase(num, loader.intersections())
        prev_p = get_phase(num - 1)
        next_p = get_phase(num + 1)
        return render_template("crisp_phase.html", active="crisp", phase=phase,
                               prev_p=prev_p, next_p=next_p, references=REFERENCES)

    @bp.route("/")
    def home():
        return render_template("home.html")

    return bp
