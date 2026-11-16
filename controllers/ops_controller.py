"""Operational modules (all of them require authentication; roles are enforced on the server)."""
import json
import time

from flask import (Blueprint, abort, flash, g, jsonify, redirect, render_template, request, url_for)

from controllers.auth_controller import api_login_required, current_user, login_required, roles_required
from models.database import log_event, q, q1
from services import accounts, incidents, intersections, plan_rules, reports
from services.forecasting import ValidationError, record_outcome

STAFF = ("admin", "operator")
ADMIN = ("admin",)

FORECAST_SELECT = """SELECT f.*, i.code AS inter_code, i.name AS inter_name, u.full_name AS author, u.username AS author_username
    FROM inferences f JOIN intersections i ON i.id = f.intersection_id JOIN users u ON u.id = f.user_id"""


def create_ops_blueprint(services):
    bp = Blueprint("ops", __name__)
    loader, forecasting, plans = services["loader"], services["forecasting"], services["plans"]
    valid_junctions = {i["junction"] for i in loader.intersections()}

    # ---------------------------------------------------------------- logging and errors
    @bp.before_app_request
    def start_timer():
        g.t0 = time.perf_counter()

    @bp.after_app_request
    def log_slow_or_failed(response):
        t0 = g.get("t0")
        if t0 is not None and not request.path.startswith("/static"):
            ms = round((time.perf_counter() - t0) * 1000, 1)
            if response.status_code >= 500 or ms > reports.LATENCY_ALERT_MS:
                user = g.get("_user")
                log_event("error" if response.status_code >= 500 else "warning", "request",
                          user_id=user["id"] if user else None, endpoint=request.path,
                          status_code=response.status_code, latency_ms=ms)
        return response

    def _error_page(code, title, text):
        return render_template("error.html", code=code, title=title, text=text), code

    @bp.app_errorhandler(403)
    def forbidden(_e):
        user = g.get("_user")
        log_event("warning", "forbidden", user_id=user["id"] if user else None, endpoint=request.path, status_code=403)
        return _error_page(403, "Access denied", "Your role does not allow this action or this data.")

    @bp.app_errorhandler(404)
    def not_found(_e):
        return _error_page(404, "Page not found", "The page or record you asked for does not exist.")

    @bp.app_errorhandler(400)
    def bad_request(e):
        return _error_page(400, "Invalid request", getattr(e, "description", "The request could not be processed."))

    @bp.app_errorhandler(500)
    def server_error(e):
        user = g.get("_user")
        log_event("error", "server_error", user_id=user["id"] if user else None, endpoint=request.path,
                  status_code=500, detail=repr(getattr(e, "original_exception", e)))
        return _error_page(500, "Something went wrong", "The error was logged. Please report it from the Incidents page.")

    # ---------------------------------------------------------------- workspace
    @bp.route("/app")
    @login_required
    def workspace():
        user = current_user()
        mine = user["id"]
        data = {
            "forecasts": q1("SELECT COUNT(*) AS n FROM inferences" + ("" if user["role"] == "admin" else " WHERE user_id = ?"),
                            () if user["role"] == "admin" else (mine,))["n"],
            "pending": q1("SELECT COUNT(*) AS n FROM timing_plans WHERE status = 'pending'")["n"],
            "applied": q1("SELECT COUNT(*) AS n FROM timing_plans WHERE status IN ('approved', 'applied')")["n"],
            "open_incidents": len([i for i in incidents.list_for(user, "open")]),
        }
        recent_plans = plans.list(user)[:5]
        return render_template("ops_home.html", data=data, recent_plans=recent_plans)

    # ---------------------------------------------------------------- intersections
    @bp.route("/intersections")
    @login_required
    def intersections_list():
        return render_template("intersections_list.html", rows=intersections.list_all())

    def _inter_form(inter=None):
        user = current_user()
        values, errors = (dict(inter) if inter else {"phases": 2, "is_active": 1}), {}
        if request.method == "POST":
            values, errors = intersections.validate(request.form, valid_junctions, inter["id"] if inter else None)
            if not errors:
                if inter:
                    intersections.update(user, inter["id"], values)
                    flash("Intersection updated.", "success")
                else:
                    intersections.create(user, values)
                    flash("Intersection created.", "success")
                return redirect(url_for("ops.intersections_list"))
        return render_template("intersection_form.html", values=values, errors=errors, inter=inter,
                               junctions=sorted(valid_junctions))

    @bp.route("/intersections/new", methods=["GET", "POST"])
    @roles_required(*ADMIN)
    def intersection_new():
        return _inter_form()

    @bp.route("/intersections/<int:inter_id>/edit", methods=["GET", "POST"])
    @roles_required(*ADMIN)
    def intersection_edit(inter_id):
        inter = intersections.get(inter_id) or abort(404)
        return _inter_form(inter)

    @bp.post("/intersections/<int:inter_id>/delete")
    @roles_required(*ADMIN)
    def intersection_delete(inter_id):
        inter = intersections.get(inter_id) or abort(404)
        msg = intersections.delete(current_user(), inter)
        flash(msg or "Intersection deleted.", "danger" if msg else "success")
        return redirect(url_for("ops.intersections_list"))

    # ---------------------------------------------------------------- forecasts (model inference + history)
    @bp.route("/forecasts")
    @roles_required(*STAFF)
    def forecasts_list():
        user = current_user()
        where, params = [], []
        if user["role"] != "admin":
            where.append("f.user_id = ?")
            params.append(user["id"])
        inter_id = request.args.get("intersection_id", type=int)
        if inter_id:
            where.append("f.intersection_id = ?")
            params.append(inter_id)
        rows = q(FORECAST_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY f.id DESC LIMIT 200", params)
        return render_template("forecasts_list.html", rows=rows, inters=intersections.list_all(True), selected=inter_id)

    @bp.route("/forecasts/new", methods=["GET", "POST"])
    @roles_required(*STAFF)
    def forecast_new():
        user = current_user()
        errors, values = {}, request.form if request.method == "POST" else {}
        if request.method == "POST":
            try:
                result = forecasting.run(user, request.form, "web")
                flash("Forecast saved in the history.", "success")
                return redirect(url_for("ops.forecast_detail", inf_id=result["id"]))
            except ValidationError as e:
                errors = e.errors
            except Exception:
                flash("The model could not produce a forecast. The error was logged.", "danger")
        art = forecasting.prediction.artifact or {}
        usable = [i for i in intersections.list_all(True) if i["junction"] is not None]
        return render_template("forecast_new.html", inters=usable, errors=errors, values=values,
                               model={"name": forecasting.prediction.model_label(), "version": art.get("version", "-"),
                                      "test": art.get("test_metrics") or {}, "trained_on": art.get("trained_on", "-")})

    def _forecast_or_403(inf_id):
        user = current_user()
        row = q1(FORECAST_SELECT + " WHERE f.id = ?", (inf_id,)) or abort(404)
        if user["role"] != "admin" and row["user_id"] != user["id"]:
            abort(403)
        return row

    @bp.route("/forecasts/<int:inf_id>")
    @roles_required(*STAFF)
    def forecast_detail(inf_id):
        row = _forecast_or_403(inf_id)
        curve = json.loads(row["curve_json"] or "[]")
        hour = int(row["target_datetime"][11:13])
        plan = q1("SELECT id, status FROM timing_plans WHERE inference_id = ? AND status <> 'rejected'", (inf_id,))
        mv = q1("SELECT test_mae FROM model_versions WHERE version = ?", (row["model_version"],))
        return render_template("forecast_detail.html", f=row, curve=curve, hour=hour,
                               peak=max(curve) if curve else 1, notes=json.loads(row["notes_json"] or "[]"),
                               plan=plan, test_mae=mv["test_mae"] if mv else None, suggestion=plans.suggestion(row), dataset_value=forecasting.dataset_value(row))

    @bp.post("/forecasts/<int:inf_id>/outcome")
    @roles_required(*STAFF)
    def forecast_outcome(inf_id):
        row = _forecast_or_403(inf_id)
        observed = request.form.get("observed")
        if request.form.get("use_dataset"):
            observed = forecasting.dataset_value(row)
            if observed is None:
                flash("That date is not in the historical counts.", "danger")
                return redirect(url_for("ops.forecast_detail", inf_id=inf_id))
        msg = record_outcome(row, observed, current_user())
        flash(msg or "Real outcome saved. It is now used to measure the model error.", "danger" if msg else "success")
        return redirect(url_for("ops.forecast_detail", inf_id=inf_id))

    # ---------------------------------------------------------------- timing plans
    @bp.route("/plans")
    @login_required
    def plans_list():
        status = request.args.get("status")
        rows = plans.list(current_user(), status)
        return render_template("plans_list.html", rows=rows, status=status, labels=plan_rules.STATUS_LABELS)

    @bp.route("/plans/new", methods=["GET", "POST"])
    @roles_required(*STAFF)
    def plan_new():
        user = current_user()
        inf_id = request.values.get("inference_id", type=int) or abort(404)
        inf = _forecast_or_403(inf_id)
        errors, values = {}, {"final_green_s": plans.suggestion(inf), "rationale": ""}
        if request.method == "POST":
            values = request.form
            pid, errors = plans.create(user, inf, request.form.get("final_green_s"), request.form.get("rationale"))
            if pid:
                flash("Draft plan created. Submit it for approval when it is ready.", "success")
                return redirect(url_for("ops.plan_detail", plan_id=pid))
            if "_" in errors:
                flash(errors["_"], "danger")
        return render_template("plan_form.html", f=inf, errors=errors, values=values, plan=None,
                               suggestion=plans.suggestion(inf), limits=(plan_rules.MIN_GREEN, plan_rules.MAX_GREEN))

    def _plan_visible(plan_id):
        plan = plans.get(plan_id) or abort(404)
        if not plans.can_see(current_user(), plan):
            abort(403)
        return plan

    @bp.route("/plans/<int:plan_id>")
    @login_required
    def plan_detail(plan_id):
        plan = _plan_visible(plan_id)
        user = current_user()
        return render_template("plan_detail.html", p=plan, labels=plan_rules.STATUS_LABELS,
                               can_edit=plans.can_edit(user, plan), owner=plans.is_owner_or_admin(user, plan),
                               limits=(plan_rules.MIN_GREEN, plan_rules.MAX_GREEN), errors={}, values=plan)

    @bp.post("/plans/<int:plan_id>/edit")
    @roles_required(*STAFF)
    def plan_edit(plan_id):
        plan = _plan_visible(plan_id)
        errors = plans.update(current_user(), plan, request.form.get("final_green_s"), request.form.get("rationale"))
        if errors:
            for msg in errors.values():
                flash(msg, "danger")
        else:
            flash("Plan updated.", "success")
        return redirect(url_for("ops.plan_detail", plan_id=plan_id))

    @bp.post("/plans/<int:plan_id>/transition")
    @roles_required(*STAFF)
    def plan_transition(plan_id):
        plan = _plan_visible(plan_id)
        target = {"submit": "pending", "approve": "approved", "reject": "rejected", "apply": "applied",
                  "reopen": "draft"}.get(request.form.get("action"))
        msg = plans.transition(current_user(), plan, target, request.form.get("note", "")) if target else "Unknown action."
        flash(msg or "Plan status updated.", "danger" if msg else "success")
        return redirect(url_for("ops.plan_detail", plan_id=plan_id))

    @bp.post("/plans/<int:plan_id>/delete")
    @roles_required(*STAFF)
    def plan_delete(plan_id):
        plan = _plan_visible(plan_id)
        msg = plans.delete(current_user(), plan)
        flash(msg or "Plan deleted.", "danger" if msg else "success")
        return redirect(url_for("ops.plans_list") if not msg else url_for("ops.plan_detail", plan_id=plan_id))

    # ---------------------------------------------------------------- incidents and feedback
    @bp.route("/incidents")
    @login_required
    def incidents_list():
        status = request.args.get("status")
        return render_template("incidents_list.html", rows=incidents.list_for(current_user(), status), status=status)

    @bp.route("/incidents/new", methods=["GET", "POST"])
    @login_required
    def incident_new():
        user = current_user()
        errors, values = {}, request.form if request.method == "POST" else request.args
        if request.method == "POST":
            nid, errors = incidents.create(user, request.form)
            if nid:
                flash("Thank you. The report was registered.", "success")
                return redirect(url_for("ops.incident_detail", inc_id=nid))
        kinds = ("feedback",) if user["role"] == "viewer" else incidents.KINDS
        return render_template("incident_form.html", errors=errors, values=values, kinds=kinds,
                               inters=intersections.list_all(True), severities=incidents.SEVERITIES)

    def _incident_visible(inc_id):
        inc = incidents.get(inc_id) or abort(404)
        if not incidents.can_see(current_user(), inc):
            abort(403)
        return inc

    @bp.route("/incidents/<int:inc_id>", methods=["GET", "POST"])
    @login_required
    def incident_detail(inc_id):
        inc = _incident_visible(inc_id)
        user = current_user()
        if request.method == "POST":
            msg = incidents.update(user, inc, request.form)
            flash(msg or "Incident updated.", "danger" if msg else "success")
            return redirect(url_for("ops.incident_detail", inc_id=inc_id))
        staff = [u for u in accounts.list_users() if u["role"] in STAFF and u["is_active"]] if user["role"] == "admin" else []
        return render_template("incident_detail.html", n=inc, staff=staff, statuses=incidents.STATUSES)

    @bp.post("/incidents/<int:inc_id>/delete")
    @roles_required(*ADMIN)
    def incident_delete(inc_id):
        inc = _incident_visible(inc_id)
        msg = incidents.delete(current_user(), inc)
        flash(msg or "Incident deleted.", "danger" if msg else "success")
        return redirect(url_for("ops.incidents_list"))

    # ---------------------------------------------------------------- reports, monitoring, users
    @bp.route("/reports")
    @login_required
    def reports_page():
        return render_template("reports.html", r=reports.indicators(current_user()))

    @bp.route("/admin/monitoring")
    @roles_required(*ADMIN)
    def monitoring_page():
        return render_template("monitoring.html", m=reports.monitoring(), thresholds={
            "latency": reports.LATENCY_ALERT_MS, "factor": reports.MAE_ALERT_FACTOR,
            "range": int(reports.RANGE_SHARE_ALERT * 100)})

    @bp.route("/admin/users")
    @roles_required(*ADMIN)
    def users_page():
        return render_template("admin_users.html", users=accounts.list_users())

    @bp.post("/admin/users/<int:user_id>/role")
    @roles_required(*ADMIN)
    def user_role(user_id):
        msg = accounts.set_role(current_user(), user_id, request.form.get("role", ""))
        flash(msg or "Role updated.", "danger" if msg else "success")
        return redirect(url_for("ops.users_page"))

    @bp.post("/admin/users/<int:user_id>/active")
    @roles_required(*ADMIN)
    def user_active(user_id):
        msg = accounts.set_active(current_user(), user_id, request.form.get("active") == "1")
        flash(msg or "Account updated.", "danger" if msg else "success")
        return redirect(url_for("ops.users_page"))

    # ---------------------------------------------------------------- JSON API (session + role)
    @bp.get("/health")
    def health():
        ok = True
        try:
            q1("SELECT 1 AS ok")
        except Exception:
            ok = False
        art = forecasting.prediction.artifact or {}
        return jsonify({"status": "ok" if ok else "database_error", "model_version": art.get("version")}), (200 if ok else 503)

    @bp.post("/v1/forecast")
    def api_forecast():
        denied = api_login_required()
        if denied:
            return denied
        payload = request.get_json(silent=True) or request.form.to_dict()
        try:
            r = forecasting.run(current_user(), payload, "api")
        except ValidationError as e:
            return jsonify({"ok": False, "errors": e.errors}), 400
        except Exception:
            return jsonify({"ok": False, "error": "The model could not produce a forecast."}), 500
        return jsonify({"ok": True, "inference_id": r["id"], "intersection": r["intersection"],
                        "datetime": r["target_datetime"], "vehicles_per_hour": r["predicted_vehicles"],
                        "group": r["group_label"], "model": r["model_name"], "model_version": r["model_version"],
                        "hourly_curve": r["curve"], "notes": r["notes"]})

    @bp.get("/v1/monitoring")
    def api_monitoring():
        user = current_user()
        if not user:
            return jsonify({"ok": False, "error": "Login required"}), 401
        if user["role"] != "admin":
            return jsonify({"ok": False, "error": "Administrators only"}), 403
        m = reports.monitoring()
        return jsonify({k: m[k] for k in ("live_mae", "outcomes", "threshold", "far_share", "latency_p95", "errors", "alerts")})

    return bp
