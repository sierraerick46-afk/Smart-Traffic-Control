from functools import wraps
from urllib.parse import urlparse

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash

USERS = {
    "erick": ("Erick", "scrypt:32768:8:1$DrmIP6eeCgo6ixoD$9123562472eb4e8c62146c85ae1ce9660f148b2c16570ecb424b5d77ff23d1ce26c5278c056633a5351bd449f803c4f1c9f8b70730c230d307239488ac81f669"),
    "santiago": ("Santiago", "scrypt:32768:8:1$bC3QeOnGdVu24ijr$f6c9bcc67c741147ba4a52650ae32b27e6fb6b2eb88b6c1d18aaa5849710d278e92ff3a2ec1855354bf625e84c9ef0320f4a3af2db4efbfccf26b2fb0fdc2147"),
    "ivan": ("Ivan", "scrypt:32768:8:1$CaiVTLSGKIMRtxlP$a5b55813eb68cba2648fc8ed50e88a317d3c85fe87cca7002d75669d938983442cedb1e3c26f604c98a25c2cf565868eba2ef6b16ee8bda152daf21183a2e794"),
    "edward": ("Edward", "scrypt:32768:8:1$krF47v6uGje5BOiA$bfb9d9cc7bdca14a9063b9e6759aa6997514e8bd9ae0121184bc11ab4f2497625bc496353f7fdcea932320c9a2dd8eddd8bd5f81b0499ba2d4895c9d58bad9ce"),
}


def current_user():
    return session.get("user")


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def api_login_required():
    if not current_user():
        return jsonify({"ok": False, "error": "Login required"}), 401
    return None


def _safe_next(target):
    if not target:
        return url_for("web.dashboard")
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/"):
        return url_for("web.dashboard")
    return target


def create_auth_blueprint():
    bp = Blueprint("auth", __name__)

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        next_url = request.values.get("next", "")
        if current_user():
            return redirect(_safe_next(next_url))
        error = None
        username = ""
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = USERS.get(username.lower())
            if user and check_password_hash(user[1], password):
                session.clear()
                session["user"] = user[0]
                return redirect(_safe_next(next_url))
            error = "Incorrect username or password."
        return render_template("login.html", error=error, next_url=next_url, username=username)

    @bp.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("web.home"))

    return bp
