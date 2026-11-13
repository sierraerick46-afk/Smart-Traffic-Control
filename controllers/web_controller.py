from flask import Blueprint, abort, render_template

from controllers.auth_controller import login_required
from controllers.crisp_content import PHASES, REFERENCES, get_phase


def create_web_blueprint():
    bp = Blueprint("web", __name__)

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
        return render_template("crisp_overview.html", active="crisp", references=REFERENCES)

    @bp.route("/crisp-ml/<int:num>")
    def crisp_phase(num):
        phase = get_phase(num)
        if phase is None:
            abort(404)
        prev_p = get_phase(num - 1)
        next_p = get_phase(num + 1)
        return render_template("crisp_phase.html", active="crisp", phase=phase,
                               prev_p=prev_p, next_p=next_p, references=REFERENCES)

    @bp.route("/")
    def home():
        return render_template("home.html")

    return bp
