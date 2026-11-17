"""Automated tests of the information system: authentication, roles, business process, model integration."""
import json

import pytest

from conftest import csrf, login, post
from models.database import Database

# ---------------------------------------------------------------------------------------- public vs protected
PROTECTED = ["/app", "/intersections", "/forecasts", "/forecasts/new", "/plans", "/incidents", "/incidents/new",
             "/reports", "/admin/users", "/admin/monitoring", "/dashboard", "/simula"]


def test_homepage_and_documentation_are_public(client):
    assert client.get("/").status_code == 200
    assert client.get("/crisp-ml").status_code == 200
    assert client.get("/crisp-ml/3").status_code == 200
    assert client.get("/health").status_code == 200


def test_homepage_has_every_pgc_section_and_resource(client):
    html = client.get("/").get_data(as_text=True)
    for section in ["Project title", "Problem statement and justification", "General objective", "Specific objectives",
                    "Reference framework", "Methodology and development schedule", "Bibliographic references"]:
        assert section in html
    assert "Section under construction" not in html and 'href="/demo"' not in html
    for resource in ["Technical report", "CRISP-ML evidence", "Repository", "Intelligent system", "User documentation",
                     "Presentation video"]:
        assert resource in html
    assert client.get("/user-guide").status_code == 200


def test_video_link_is_embedded_when_configured(client, monkeypatch):
    monkeypatch.setenv("VIDEO_URL", "https://youtu.be/abcdefghijk")
    assert "https://www.youtube.com/embed/abcdefghijk" in client.get("/").get_data(as_text=True)


@pytest.mark.parametrize("url", PROTECTED)
def test_operational_pages_require_login(client, url):
    r = client.get(url)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_api_requires_login(client):
    assert client.get("/api/status").status_code == 401
    assert client.post("/v1/forecast", json={"intersection_id": 1, "date": "2017-06-12", "hour": 8}).status_code == 401
    assert client.get("/v1/monitoring").status_code == 401


def test_old_public_demo_is_gone(client):
    assert client.get("/demo").status_code == 404


# ---------------------------------------------------------------------------------------- registration and login
def test_registration_validation(client):
    r = post(client, "/register", {"username": "a", "full_name": "x", "email": "bad", "password": "short", "confirm": "no"})
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "3 to 30 letters" in body and "valid email" in body and "at least 8 characters" in body


def test_registration_creates_viewer_and_ignores_role(client, app):
    r = post(client, "/register", {"username": "maria", "full_name": "Maria Lopez", "email": "maria@example.com",
                                    "password": "Secret123", "confirm": "Secret123", "role": "admin"})
    assert r.status_code == 302
    from models.database import q1
    with app.app_context():
        u = q1("SELECT * FROM users WHERE username = 'maria'")
    assert u["role"] == "viewer"                       # the role field of the form is ignored
    assert u["password_hash"] != "Secret123" and u["password_hash"].startswith(("scrypt:", "pbkdf2:"))
    assert client.get("/admin/users").status_code == 403


def test_duplicate_username_and_email_rejected(client):
    r = post(client, "/register", {"username": "maria", "full_name": "Other", "email": "maria@example.com",
                                    "password": "Secret123", "confirm": "Secret123"})
    body = r.get_data(as_text=True)
    assert "already taken" in body and "already registered" in body


def test_login_logout_and_wrong_password(client):
    assert "Incorrect username or password" in login(client, "viewer_test", "bad").get_data(as_text=True)
    assert login(client, "viewer_test").status_code == 302
    assert client.get("/app").status_code == 200
    assert post(client, "/logout").status_code == 302
    assert client.get("/app").status_code == 302


def test_lockout_after_repeated_failures(client):
    for _ in range(5):
        login(client, "operator_test", "wrong")
    body = login(client, "operator_test", "Test12345").get_data(as_text=True)
    assert "Too many failed attempts" in body


def test_post_without_csrf_token_is_rejected(client):
    assert client.post("/login", data={"username": "x", "password": "y"}).status_code == 400


# ---------------------------------------------------------------------------------------- roles and permissions
def test_viewer_permissions(client):
    login(client, "viewer_test")
    for url in ("/forecasts", "/forecasts/new", "/intersections/new", "/admin/users", "/admin/monitoring"):
        assert client.get(url).status_code == 403, url
    for url in ("/app", "/plans", "/incidents", "/intersections", "/reports", "/dashboard"):
        assert client.get(url).status_code == 200, url
    assert client.post("/api/train_prediction").status_code == 403


def test_operator_permissions(people):
    op = people["op1"]
    for url in ("/app", "/forecasts", "/forecasts/new", "/plans", "/incidents", "/intersections", "/reports", "/dashboard"):
        assert op.get(url).status_code == 200, url
    for url in ("/admin/users", "/admin/monitoring", "/intersections/new"):
        assert op.get(url).status_code == 403, url


def test_admin_pages_open_for_admin(client):
    login(client, "admin_test")
    for url in ("/admin/users", "/admin/monitoring", "/intersections/new", "/reports", "/forecasts", "/plans"):
        assert client.get(url).status_code == 200, url


# ---------------------------------------------------------------------------------------- helpers for the process
def make_user(app, username, role):
    from services import accounts
    with app.app_context():
        accounts.create_user(username, username.title() + " Tester", f"{username}@example.com", "Test12345", role)


def new_client(app, username):
    c = app.test_client()
    r = login(c, username)
    assert r.status_code == 302, r.get_data(as_text=True)
    return c


def run_forecast(client, inter_id=1, date="2017-06-12", hour=8):
    r = post(client, "/forecasts/new", {"intersection_id": inter_id, "date": date, "hour": hour})
    assert r.status_code == 302, r.get_data(as_text=True)[:500]
    return int(r.headers["Location"].rstrip("/").split("/")[-1])


@pytest.fixture(scope="module")
def people(app):
    for name, role in (("op1", "operator"), ("op2", "operator"), ("adm1", "admin"), ("view1", "viewer")):
        make_user(app, name, role)
    return {n: new_client(app, n) for n in ("op1", "op2", "adm1", "view1")}


# ---------------------------------------------------------------------------------------- model integration
def test_forecast_is_saved_with_user_and_model_version(app, people):
    inf = run_forecast(people["op1"])
    from models.database import q1
    with app.app_context():
        row = q1("SELECT f.*, u.username FROM inferences f JOIN users u ON u.id = f.user_id WHERE f.id = ?", (inf,))
        version = q1("SELECT version FROM model_versions WHERE is_current = 1")["version"]
    assert row["username"] == "op1" and row["model_version"] == version
    assert row["predicted_vehicles"] > 0 and row["latency_ms"] is not None and row["created_at"]
    assert len(json.loads(row["curve_json"])) == 24


def test_server_inference_matches_the_trained_pipeline(app, people):
    """The web result must equal calling the saved featurizer + model directly (same preprocessing)."""
    import pandas as pd
    inf = run_forecast(people["op1"], inter_id=1, date="2017-06-12", hour=8)
    from models.database import q1
    with app.app_context():
        row = q1("SELECT predicted_vehicles FROM inferences WHERE id = ?", (inf,))
    svc = app.extensions  # noqa: F841
    import joblib, os
    art = joblib.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "model.joblib"))
    feat, model = art["featurizer"], art["model"]
    day = pd.date_range("2017-06-12", periods=24, freq="h")
    import numpy as np
    X = feat.transform(pd.DataFrame({"datetime": day, "junction": [1] * 24}))
    direct = float(np.asarray(model.predict(X))[8])
    assert abs(direct - row["predicted_vehicles"]) < 0.06


def test_invalid_forecast_inputs_are_rejected(people):
    c = people["op1"]
    for data in ({"intersection_id": "", "date": "2017-06-12", "hour": 8},
                 {"intersection_id": 1, "date": "not-a-date", "hour": 8},
                 {"intersection_id": 1, "date": "2017-06-12", "hour": 30},
                 {"intersection_id": 1, "date": "1999-01-01", "hour": 8},
                 {"intersection_id": 999, "date": "2017-06-12", "hour": 8}):
        r = post(c, "/forecasts/new", data)
        assert r.status_code == 200, data


def test_json_api_forecast(people):
    c = people["op1"]
    r = c.post("/v1/forecast", json={"intersection_id": 2, "date": "2017-06-13", "hour": 18})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] and len(body["hourly_curve"]) == 24 and body["model_version"]
    bad = c.post("/v1/forecast", json={"intersection_id": 2, "date": "x", "hour": 18})
    assert bad.status_code == 400 and not bad.get_json()["ok"]
    assert people["view1"].post("/v1/forecast", json={"intersection_id": 2, "date": "2017-06-13", "hour": 18}).status_code == 403


def test_history_is_scoped_by_role(people):
    inf = run_forecast(people["op2"], date="2017-06-14")
    assert people["op1"].get(f"/forecasts/{inf}").status_code == 403       # another operator's forecast
    assert people["adm1"].get(f"/forecasts/{inf}").status_code == 200      # administrators see everything
    assert people["op2"].get(f"/forecasts/{inf}").status_code == 200


# ---------------------------------------------------------------------------------------- the business process
def test_plan_workflow_end_to_end(app, people):
    op, adm, viewer = people["op1"], people["adm1"], people["view1"]
    inf = run_forecast(op, inter_id=1, date="2017-06-15", hour=9)
    # the form rejects a change beyond 15 s without justification
    page = op.get(f"/plans/new?inference_id={inf}").get_data(as_text=True)
    import re
    green = int(re.search(r'name="final_green_s"[^>]*value="(\d+)"', page).group(1))
    assert 8 <= green <= 60
    far = 8 if green > 23 else 60                    # more than 15 s away from the proposal
    r = post(op, "/plans/new", {"inference_id": inf, "final_green_s": far, "rationale": ""})
    assert r.status_code == 200 and "Explain the change" in r.get_data(as_text=True)
    r = post(op, "/plans/new", {"inference_id": inf, "final_green_s": 61, "rationale": "x"})
    assert "between 8 and 60" in r.get_data(as_text=True)
    r = post(op, "/plans/new", {"inference_id": inf, "final_green_s": green, "rationale": ""})
    assert r.status_code == 302
    plan_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
    # one active plan per forecast
    assert "already has an active plan" in post(op, "/plans/new", {"inference_id": inf, "final_green_s": green}, follow_redirects=True).get_data(as_text=True)
    # the draft is invisible to viewers and to other operators, visible to the admin
    assert viewer.get(f"/plans/{plan_id}").status_code == 403
    assert people["op2"].get(f"/plans/{plan_id}").status_code == 403
    assert adm.get(f"/plans/{plan_id}").status_code == 200
    # submit, then the operator cannot approve
    post(op, f"/plans/{plan_id}/transition", {"action": "submit"})
    body = post(op, f"/plans/{plan_id}/transition", {"action": "approve"}, follow_redirects=True).get_data(as_text=True)
    assert "Only administrators approve" in body
    # rejecting needs a reason; approving works
    assert "reason for rejecting" in post(adm, f"/plans/{plan_id}/transition", {"action": "reject", "note": "no"}, follow_redirects=True).get_data(as_text=True)
    post(adm, f"/plans/{plan_id}/transition", {"action": "approve", "note": "ok"})
    assert viewer.get(f"/plans/{plan_id}").status_code == 200              # approved plans are public to every role
    # cannot skip steps and only approved plans can be applied
    post(op, f"/plans/{plan_id}/transition", {"action": "apply"})
    from models.database import q1 as one
    with app.app_context():
        assert one("SELECT status FROM timing_plans WHERE id = ?", (plan_id,))["status"] == "applied"
    assert "cannot become" in post(op, f"/plans/{plan_id}/transition", {"action": "submit"}, follow_redirects=True).get_data(as_text=True)


def test_rejected_plan_can_be_reopened_and_drafts_deleted(app, people):
    op, adm = people["op1"], people["adm1"]
    inf = run_forecast(op, inter_id=3, date="2017-06-16", hour=7)
    page = op.get(f"/plans/new?inference_id={inf}").get_data(as_text=True)
    import re
    green = int(re.search(r'name="final_green_s"[^>]*value="(\d+)"', page).group(1))
    r = post(op, "/plans/new", {"inference_id": inf, "final_green_s": green})
    pid = int(r.headers["Location"].rstrip("/").split("/")[-1])
    post(op, f"/plans/{pid}/transition", {"action": "submit"})
    post(adm, f"/plans/{pid}/transition", {"action": "reject", "note": "The hour is not a priority."})
    post(op, f"/plans/{pid}/transition", {"action": "reopen"})
    from models.database import q1
    with app.app_context():
        assert q1("SELECT status FROM timing_plans WHERE id = ?", (pid,))["status"] == "draft"
    post(op, f"/plans/{pid}/delete")
    with app.app_context():
        assert q1("SELECT id FROM timing_plans WHERE id = ?", (pid,)) is None


def test_outcome_recording_feeds_reports_and_monitoring(app, people):
    op, adm = people["op1"], people["adm1"]
    inf = run_forecast(op, inter_id=1, date="2017-06-20", hour=12)
    assert "Write the observed" in post(op, f"/forecasts/{inf}/outcome", {"observed": "abc"}, follow_redirects=True).get_data(as_text=True)
    post(op, f"/forecasts/{inf}/outcome", {"use_dataset": "1"})
    from models.database import q1
    with app.app_context():
        row = q1("SELECT observed_vehicles, abs_error FROM inferences WHERE id = ?", (inf,))
    assert row["observed_vehicles"] is not None and row["abs_error"] is not None
    assert "MAE vs outcomes" in adm.get("/reports").get_data(as_text=True)
    assert "Live MAE" in adm.get("/admin/monitoring").get_data(as_text=True)
    assert adm.get("/v1/monitoring").get_json()["outcomes"] >= 1


def test_incidents_rules(app, people):
    view, op, adm = people["view1"], people["op1"], people["adm1"]
    r = post(view, "/incidents/new", {"kind": "incident", "severity": "low", "intersection_id": 1, "description": "A problem with the plan."})
    assert "only send feedback" in r.get_data(as_text=True)
    r = post(view, "/incidents/new", {"kind": "feedback", "severity": "low", "intersection_id": 1, "description": "The page is clear, thanks."})
    iid = int(r.headers["Location"].rstrip("/").split("/")[-1])
    assert op.get(f"/incidents/{iid}").status_code == 403            # not the reporter
    assert adm.get(f"/incidents/{iid}").status_code == 200
    assert "Write the resolution" in post(adm, f"/incidents/{iid}", {"status": "closed", "resolution": ""}, follow_redirects=True).get_data(as_text=True)
    post(adm, f"/incidents/{iid}", {"status": "closed", "resolution": "Answered to the user by e-mail."})
    from models.database import q1
    with app.app_context():
        assert q1("SELECT status FROM incidents WHERE id = ?", (iid,))["status"] == "closed"
    assert view.post(f"/incidents/{iid}", data={"status": "open", "csrf_token": csrf(view)}).status_code == 302
    with app.app_context():
        assert q1("SELECT status FROM incidents WHERE id = ?", (iid,))["status"] == "closed"   # viewers cannot reopen


def test_intersection_crud_and_rules(app, people):
    adm, op = people["adm1"], people["op1"]
    assert op.get("/intersections/new").status_code == 403
    bad = post(adm, "/intersections/new", {"code": "x", "name": "a", "junction": "2", "phases": "9"})
    body = bad.get_data(as_text=True)
    assert "3 to 20 letters" in body and "already linked" in body and "between 2 and 8" in body
    ok = post(adm, "/intersections/new", {"code": "CHIA-005", "name": "Calle 12 con Cra 8", "phases": "2",
                                          "speed_limit_kmh": "30", "is_active": "1"})
    assert ok.status_code == 302
    from models.database import q1
    with app.app_context():
        new = q1("SELECT * FROM intersections WHERE code = 'CHIA-005'")
    assert new["junction"] is None
    # a catalogue entry without historical counts cannot be forecast
    r = post(op, "/forecasts/new", {"intersection_id": new["id"], "date": "2017-06-12", "hour": 8})
    assert r.status_code == 200 and "no historical series" in r.get_data(as_text=True)
    assert post(adm, f"/intersections/{new['id']}/edit", {"code": "CHIA-005", "name": "Calle 12 con Cra 9", "phases": "3", "is_active": "1"}).status_code == 302
    post(adm, f"/intersections/{new['id']}/delete")
    with app.app_context():
        assert q1("SELECT id FROM intersections WHERE id = ?", (new["id"],)) is None
    # an intersection with history cannot be deleted
    used = post(adm, "/intersections/1/delete", follow_redirects=True).get_data(as_text=True)
    assert "Disable it instead" in used


def test_role_management_rules(app, people):
    adm = people["adm1"]
    from models.database import q1
    with app.app_context():
        me = q1("SELECT id FROM users WHERE username = 'adm1'")["id"]
        target = q1("SELECT id FROM users WHERE username = 'view1'")["id"]
    assert "cannot change your own role" in post(adm, f"/admin/users/{me}/role", {"role": "viewer"}, follow_redirects=True).get_data(as_text=True)
    post(adm, f"/admin/users/{target}/role", {"role": "operator"})
    assert people["view1"].get("/forecasts/new").status_code == 200      # the new role applies immediately
    post(adm, f"/admin/users/{target}/role", {"role": "viewer"})
    assert people["view1"].get("/forecasts/new").status_code == 403
    post(adm, f"/admin/users/{target}/active", {"active": "0"})
    assert people["view1"].get("/app").status_code == 302                # a disabled account loses access
    post(adm, f"/admin/users/{target}/active", {"active": "1"})


def test_last_admin_cannot_be_removed(app):
    from models.database import q1
    from services import accounts
    with app.app_context():
        admins = [u for u in accounts.list_users() if u["role"] == "admin" and u["is_active"]]
        actor = {"id": -1}
        # demote all but one administrator, then try to demote the last one
        for u in admins[:-1]:
            assert accounts.set_role(actor, u["id"], "viewer") is None
        assert "At least one active administrator" in accounts.set_role(actor, admins[-1]["id"], "viewer")
        for u in admins[:-1]:
            accounts.set_role(actor, u["id"], "admin")


# ---------------------------------------------------------------------------------------- persistence
def test_data_persists_across_application_restarts(app):
    import app as app_module
    from models.database import q1
    with app.app_context():
        before = q1("SELECT COUNT(*) AS n FROM inferences")["n"]
        users = q1("SELECT COUNT(*) AS n FROM users")["n"]
    again = app_module.create_app(db=Database(url="", path=app.config["DB_PATH"]), light=True)
    with again.app_context():
        assert q1("SELECT COUNT(*) AS n FROM inferences")["n"] == before > 0
        assert q1("SELECT COUNT(*) AS n FROM users")["n"] == users


def test_errors_and_requests_are_logged(app):
    from models.database import q1
    with app.app_context():
        assert q1("SELECT COUNT(*) AS n FROM app_logs WHERE event = 'inference'")["n"] > 0
        assert q1("SELECT COUNT(*) AS n FROM app_logs WHERE event = 'inference_rejected'")["n"] > 0
        assert q1("SELECT COUNT(*) AS n FROM app_logs WHERE event = 'login_failed'")["n"] > 0


# ---------------------------------------------------------------------------------------- every page renders
def test_all_pages_render_for_each_role(app, people):
    inf = run_forecast(people["op1"], inter_id=4, date="2017-06-21", hour=10)
    pages = ["/", "/crisp-ml", "/crisp-ml/5", "/crisp-ml/6", "/login", "/register", "/app", "/intersections",
             "/intersections/1/edit", "/forecasts", "/forecasts/new", f"/forecasts/{inf}", f"/plans/new?inference_id={inf}",
             "/plans", "/incidents", "/incidents/new", "/reports", "/admin/users", "/admin/monitoring", "/dashboard", "/simula"]
    for url in pages:
        r = people["adm1"].get(url)
        assert r.status_code in (200, 302), (url, r.status_code)
    for url in ("/app", "/reports", "/plans", "/incidents", "/incidents/new", "/intersections", "/crisp-ml/5"):
        assert people["op1"].get(url).status_code == 200, url
    assert people["adm1"].get("/does-not-exist").status_code == 404


def test_json_login_gives_a_session_for_the_api(app):
    c = app.test_client()
    bad = c.post("/login", json={"username": "adm1", "password": "wrong"})
    assert bad.status_code == 401
    ok = c.post("/login", json={"username": "adm1", "password": "Test12345"})
    assert ok.status_code == 200 and ok.get_json()["role"] == "admin"
    r = c.post("/v1/forecast", json={"intersection_id": 1, "date": "2017-06-12", "hour": 8})
    assert r.status_code == 200 and r.get_json()["ok"]


def test_team_members_get_operator_and_viewer_accounts(client, app):
    from models.database import Database, q1
    with app.app_context():
        rows = {n: q1("SELECT role FROM users WHERE username = ?", (n,)) for n in
                ("erick", "erick_operator", "erick_viewer", "ivan_operator", "ivan_viewer")}
    assert rows["erick"]["role"] == "admin"
    assert rows["erick_operator"]["role"] == "operator" and rows["erick_viewer"]["role"] == "viewer"
    assert rows["ivan_operator"]["role"] == "operator" and rows["ivan_viewer"]["role"] == "viewer"
    assert login(client, "ivan_viewer", "ivan12345").status_code == 302
    assert login(client, "erick_operator", "Erick12345").status_code == 302
