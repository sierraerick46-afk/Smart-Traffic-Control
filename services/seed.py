"""Initial data: the intersections of the case study, the model registry and the first accounts.

No password is stored in the code. Administrators and test accounts are created from environment variables
(see .env.example). The four team accounts that existed before keep their original password hashes.
"""
import os

import pandas as pd

from models.database import insert, now_iso, q, q1, run
from services import accounts

TEAM_ACCOUNTS = {
    "erick": ("Erick Javier Sierra Martinez", "scrypt:32768:8:1$DrmIP6eeCgo6ixoD$9123562472eb4e8c62146c85ae1ce9660f148b2c16570ecb424b5d77ff23d1ce26c5278c056633a5351bd449f803c4f1c9f8b70730c230d307239488ac81f669"),
    "santiago": ("David Santiago Mendez Carrilo", "scrypt:32768:8:1$bC3QeOnGdVu24ijr$f6c9bcc67c741147ba4a52650ae32b27e6fb6b2eb88b6c1d18aaa5849710d278e92ff3a2ec1855354bf625e84c9ef0320f4a3af2db4efbfccf26b2fb0fdc2147"),
    "ivan": ("Ivan Felipe Leon Pinilla", "scrypt:32768:8:1$CaiVTLSGKIMRtxlP$a5b55813eb68cba2648fc8ed50e88a317d3c85fe87cca7002d75669d938983442cedb1e3c26f604c98a25c2cf565868eba2ef6b16ee8bda152daf21183a2e794"),
    "edward": ("Edward Felipe Gil Gomez", "scrypt:32768:8:1$krF47v6uGje5BOiA$bfb9d9cc7bdca14a9063b9e6759aa6997514e8bd9ae0121184bc11ab4f2497625bc496353f7fdcea932320c9a2dd8eddd8bd5f81b0499ba2d4895c9d58bad9ce"),
}
TEST_ROLES = ("admin", "operator", "viewer")
TEAM_EXTRA_ROLES = ("operator", "viewer")


def _coords():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "Intersecciones_Chia.csv")
    out = {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, r in df.iterrows():
            out[str(r["ID"])] = (r.get("LATITUD"), r.get("LONGITUD"))
    except Exception:
        pass
    return out


def seed_intersections(loader):
    if q1("SELECT COUNT(*) AS n FROM intersections")["n"]:
        return
    coords = _coords()
    for it in loader.intersections():
        lat, lon = coords.get(it["id"], (None, None))
        insert("intersections", {
            "code": it["id"], "name": it["address"], "junction": it["junction"], "phases": it["phases"],
            "latitude": None if pd.isna(lat) else float(lat) if lat is not None else None,
            "longitude": None if pd.isna(lon) else float(lon) if lon is not None else None,
            "speed_limit_kmh": 50, "notes": "Case-study intersection (model junction %d)." % it["junction"],
            "is_active": 1, "created_at": now_iso()})


def register_model(prediction):
    """Keeps a row per model version so every inference can be traced to the model that produced it."""
    art = prediction.artifact
    if not art:
        return None
    version = art.get("version") or "unversioned"
    tm = art.get("test_metrics") or {}
    if not q1("SELECT id FROM model_versions WHERE version = ?", (version,)):
        run("UPDATE model_versions SET is_current = 0")
        insert("model_versions", {"version": version, "model_name": art.get("label", art.get("name", "model")),
                                  "trained_on": art.get("trained_on"), "test_r2": tm.get("r2"),
                                  "test_mae": tm.get("mae"), "test_rmse": tm.get("rmse"),
                                  "is_current": 1, "registered_at": now_iso()})
    return version


def seed_accounts():
    for username, (full_name, pw_hash) in TEAM_ACCOUNTS.items():
        if not q1("SELECT id FROM users WHERE username = ?", (username,)):
            accounts.create_user(username, full_name, f"{username}@team.local", role="admin", password_hash=pw_hash)
    for username, (full_name, _hash) in TEAM_ACCOUNTS.items():
        team_pw = os.environ.get(f"{username.upper()}_PASSWORD") or f"{username}12345"
        for role in TEAM_EXTRA_ROLES:
            name = f"{username}_{role}"
            if not q1("SELECT id FROM users WHERE username = ?", (name,)):
                accounts.create_user(name, full_name, f"{username}.{role}@team.local", team_pw, role)
    admin_user, admin_pw = os.environ.get("ADMIN_USERNAME"), os.environ.get("ADMIN_PASSWORD")
    if admin_user and admin_pw and not q1("SELECT id FROM users WHERE username = ?", (admin_user.lower(),)):
        accounts.create_user(admin_user.lower(), "System Administrator",
                             os.environ.get("ADMIN_EMAIL", f"{admin_user.lower()}@admin.local"), admin_pw, "admin")
    test_pw = os.environ.get("TEST_USERS_PASSWORD")
    if test_pw:
        for role in TEST_ROLES:
            name = f"{role}_test"
            if not q1("SELECT id FROM users WHERE username = ?", (name,)):
                accounts.create_user(name, f"Test {role.capitalize()}", f"{name}@test.local", test_pw, role)


def seed_all(loader, prediction):
    seed_intersections(loader)
    register_model(prediction)
    seed_accounts()
