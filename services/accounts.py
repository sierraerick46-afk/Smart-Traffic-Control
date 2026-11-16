"""Users, passwords and roles."""
import re

from werkzeug.security import check_password_hash, generate_password_hash

from models.database import ROLES, insert, log_event, now_iso, q, q1, run

USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_FAILED = 5          # failed logins allowed ...
LOCK_MINUTES = 15       # ... inside this window


def validate_password(password):
    if len(password) < 8:
        return "The password must have at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "The password must contain letters and numbers."
    if len(password) > 128:
        return "The password is too long."
    return None


def validate_registration(form):
    """Returns (clean_values, errors). The role is never read from the form: self-registered users are viewers."""
    username = (form.get("username") or "").strip().lower()
    full_name = " ".join((form.get("full_name") or "").split())
    email = (form.get("email") or "").strip().lower()
    password = form.get("password") or ""
    confirm = form.get("confirm") or ""
    errors = {}
    if not USERNAME_RE.match(username):
        errors["username"] = "Use 3 to 30 letters, numbers, dots, dashes or underscores."
    elif q1("SELECT id FROM users WHERE username = ?", (username,)):
        errors["username"] = "That username is already taken."
    if not 2 <= len(full_name) <= 80:
        errors["full_name"] = "Write your full name (2 to 80 characters)."
    if not EMAIL_RE.match(email) or len(email) > 120:
        errors["email"] = "Write a valid email address."
    elif q1("SELECT id FROM users WHERE email = ?", (email,)):
        errors["email"] = "That email is already registered."
    msg = validate_password(password)
    if msg:
        errors["password"] = msg
    elif password != confirm:
        errors["confirm"] = "The passwords do not match."
    return {"username": username, "full_name": full_name, "email": email, "password": password}, errors


def create_user(username, full_name, email, password=None, role="viewer", password_hash=None):
    if role not in ROLES:
        raise ValueError("Unknown role")
    return insert("users", {
        "username": username, "full_name": full_name, "email": email,
        "password_hash": password_hash or generate_password_hash(password),
        "role": role, "is_active": 1, "created_at": now_iso()})


def get_user(user_id):
    return q1("SELECT id, username, full_name, email, role, is_active, created_at, last_login "
              "FROM users WHERE id = ?", (user_id,))


def list_users():
    return q("SELECT id, username, full_name, email, role, is_active, created_at, last_login "
             "FROM users ORDER BY id")


def locked_out(username):
    """True after MAX_FAILED wrong passwords for the same username inside LOCK_MINUTES."""
    from datetime import datetime, timedelta
    since = (datetime.strptime(now_iso(), "%Y-%m-%d %H:%M:%S") - timedelta(minutes=LOCK_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    row = q1("SELECT COUNT(*) AS n FROM app_logs WHERE event = 'login_failed' AND detail = ? AND created_at >= ?",
             (username, since))
    return bool(row and row["n"] >= MAX_FAILED)


def authenticate(username, password):
    """Returns (user, error). Same message for unknown user and wrong password."""
    username = (username or "").strip().lower()
    if locked_out(username):
        return None, f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes."
    row = q1("SELECT * FROM users WHERE username = ?", (username,))
    if row and row["is_active"] and check_password_hash(row["password_hash"], password or ""):
        run("UPDATE users SET last_login = ? WHERE id = ?", (now_iso(), row["id"]))
        log_event("info", "login_ok", user_id=row["id"], endpoint="/login")
        return row, None
    log_event("warning", "login_failed", endpoint="/login", detail=username)
    if row and not row["is_active"]:
        return None, "This account is disabled. Contact an administrator."
    return None, "Incorrect username or password."


def active_admins():
    return q1("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND is_active = 1")["n"]


def set_role(actor, user_id, role):
    """Only administrators change roles (enforced in the controller); nobody changes their own role."""
    if role not in ROLES:
        return "Unknown role."
    target = get_user(user_id)
    if not target:
        return "User not found."
    if target["id"] == actor["id"]:
        return "You cannot change your own role."
    if target["role"] == "admin" and role != "admin" and target["is_active"] and active_admins() <= 1:
        return "At least one active administrator must remain."
    run("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    log_event("info", "role_changed", user_id=actor["id"], detail=f"user {user_id}: {target['role']} -> {role}")
    return None


def set_active(actor, user_id, active):
    target = get_user(user_id)
    if not target:
        return "User not found."
    if target["id"] == actor["id"]:
        return "You cannot disable your own account."
    if not active and target["role"] == "admin" and target["is_active"] and active_admins() <= 1:
        return "At least one active administrator must remain."
    run("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))
    log_event("info", "user_active_changed", user_id=actor["id"], detail=f"user {user_id}: active={active}")
    return None
