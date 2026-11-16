"""Registration, login, logout, roles and request protection (sessions, CSRF, authorization)."""
import hmac
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import (Blueprint, abort, flash, g, jsonify, redirect, render_template, request, session, url_for)

from services import accounts

ROLE_LABELS = {"admin": "Administrator", "operator": "Operator", "viewer": "Viewer"}


def current_user():
    """The logged-in user, read from the database on every request so role changes apply immediately."""
    if "_user" not in g:
        user = None
        uid = session.get("uid")
        if uid:
            user = accounts.get_user(uid)
            if user is None or not user["is_active"]:
                session.clear()
                user = None
        g._user = user
    return g._user


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def roles_required(*roles):
    """Server-side authorization: the page is refused (403) when the role is not listed."""
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("auth.login", next=request.path))
            if user["role"] not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


def api_login_required():
    """Used as before_request of the JSON API. Anonymous -> 401. Viewers cannot trigger work (POST) -> 403."""
    user = current_user()
    if not user:
        return jsonify({"ok": False, "error": "Login required"}), 401
    if request.method == "POST" and user["role"] not in ("admin", "operator"):
        return jsonify({"ok": False, "error": "Your role cannot run this action"}), 403
    return None


def _safe_next(target):
    if not target:
        return url_for("ops.workspace")
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return url_for("ops.workspace")
    return target


def create_auth_blueprint():
    bp = Blueprint("auth", __name__)

    @bp.app_context_processor
    def inject_user():
        return {"current_user": current_user(), "csrf_token": csrf_token, "role_labels": ROLE_LABELS}

    @bp.before_app_request
    def csrf_protect():
        # Forms must carry the token. JSON requests cannot be forged by another site (they need a CORS
        # preflight), and the /api calls of the dashboard are protected by SameSite cookies and by role.
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and not request.is_json \
                and not request.path.startswith("/api/"):
            sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or ""
            expected = session.get("csrf", "")
            if not sent or not expected or not hmac.compare_digest(sent, expected):
                abort(400, "Invalid or missing security token. Reload the page and try again.")

    @bp.route("/register", methods=["GET", "POST"])
    def register():
        if current_user():
            return redirect(url_for("ops.workspace"))
        values, errors = {}, {}
        if request.method == "POST":
            values, errors = accounts.validate_registration(request.form)
            if not errors:
                # The role is always "viewer": it is never taken from the request.
                uid = accounts.create_user(values["username"], values["full_name"], values["email"],
                                           values["password"], role="viewer")
                session.clear()
                session["uid"] = uid
                flash("Your account was created with the Viewer role. An administrator can grant more access.", "success")
                return redirect(url_for("ops.workspace"))
            values.pop("password", None)
        return render_template("register.html", values=values, errors=errors)

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        next_url = request.values.get("next", "")
        if current_user():
            return redirect(_safe_next(next_url))
        error, username = None, ""
        if request.method == "POST":
            data = request.get_json(silent=True) if request.is_json else request.form
            data = data or {}
            username = (data.get("username") or "").strip()
            user, error = accounts.authenticate(username, data.get("password") or "")
            if user:
                session.clear()
                session["uid"] = user["id"]
                if request.is_json:      # lets scripts and curl obtain a session cookie for the JSON API
                    return jsonify({"ok": True, "role": user["role"]})
                return redirect(_safe_next(next_url))
            if request.is_json:
                return jsonify({"ok": False, "error": error}), 401
        return render_template("login.html", error=error, next_url=next_url, username=username)

    @bp.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        flash("You have logged out.", "info")
        return redirect(url_for("web.home"))

    return bp
