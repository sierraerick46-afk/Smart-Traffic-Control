"""Incident and feedback reports (users report problems with the system, the forecasts or the plans)."""
from models.database import insert, log_event, now_iso, q, q1, run

KINDS = ("feedback", "incident", "model_issue")
SEVERITIES = ("low", "medium", "high")
STATUSES = ("open", "in_review", "closed")

BASE_SELECT = """SELECT n.*, i.code AS inter_code, i.name AS inter_name,
        r.full_name AS reporter, a.full_name AS assignee
    FROM incidents n
    JOIN intersections i ON i.id = n.intersection_id
    JOIN users r ON r.id = n.reported_by
    LEFT JOIN users a ON a.id = n.assigned_to"""


def can_see(user, inc):
    return user["role"] == "admin" or inc["reported_by"] == user["id"] or inc["assigned_to"] == user["id"]


def list_for(user, status=None):
    where, params = [], []
    if user["role"] != "admin":
        where.append("(n.reported_by = ? OR n.assigned_to = ?)")
        params += [user["id"], user["id"]]
    if status in STATUSES:
        where.append("n.status = ?")
        params.append(status)
    return q(BASE_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY n.id DESC LIMIT 200", params)


def get(inc_id):
    return q1(BASE_SELECT + " WHERE n.id = ?", (inc_id,))


def create(user, form):
    """Returns (incident_id, errors). Viewers can only send feedback; operators and administrators any kind."""
    errors = {}
    kind = form.get("kind", "incident")
    severity = form.get("severity", "low")
    description = " ".join((form.get("description") or "").split())
    if kind not in KINDS:
        errors["kind"] = "Choose a valid type."
    elif user["role"] == "viewer" and kind != "feedback":
        errors["kind"] = "Your role can only send feedback."
    if severity not in SEVERITIES:
        errors["severity"] = "Choose a valid severity."
    if not 10 <= len(description) <= 1000:
        errors["description"] = "Describe what happened (10 to 1000 characters)."
    inter = None
    try:
        inter = q1("SELECT id FROM intersections WHERE id = ?", (int(form.get("intersection_id")),))
    except (TypeError, ValueError):
        pass
    if not inter:
        errors["intersection_id"] = "Choose an intersection."
    links = {}
    for key, table in (("inference_id", "inferences"), ("plan_id", "timing_plans")):
        raw = form.get(key)
        if raw not in (None, ""):
            try:
                row = q1(f"SELECT id FROM {table} WHERE id = ?", (int(raw),))
            except (TypeError, ValueError):
                row = None
            if not row:
                errors[key] = "The linked record does not exist."
            else:
                links[key] = row["id"]
    if errors:
        return None, errors
    now = now_iso()
    nid = insert("incidents", {"intersection_id": inter["id"], "inference_id": links.get("inference_id"),
                               "plan_id": links.get("plan_id"), "reported_by": user["id"], "kind": kind,
                               "severity": severity, "description": description, "status": "open",
                               "created_at": now, "updated_at": now})
    log_event("info", "incident_created", user_id=user["id"], detail=f"incident {nid} ({kind}, {severity})")
    return nid, {}


def update(user, inc, form):
    """Operators and administrators work on incidents; only administrators assign or reopen them."""
    if user["role"] not in ("admin", "operator") or not can_see(user, inc):
        return "You cannot update this incident."
    status = form.get("status", inc["status"])
    resolution = " ".join((form.get("resolution") or "").split())
    if status not in STATUSES:
        return "Unknown status."
    if inc["status"] == "closed" and status != "closed" and user["role"] != "admin":
        return "Only administrators reopen a closed incident."
    if status == "closed" and len(resolution) < 10:
        return "Write the resolution (at least 10 characters) to close the incident."
    assigned = inc["assigned_to"]
    if user["role"] == "admin" and form.get("assigned_to") not in (None, ""):
        row = q1("SELECT id FROM users WHERE id = ? AND is_active = 1 AND role IN ('admin', 'operator')",
                 (int(form["assigned_to"]),)) if str(form["assigned_to"]).isdigit() else None
        if not row:
            return "The assigned user must be an active operator or administrator."
        assigned = row["id"]
    now = now_iso()
    run("UPDATE incidents SET status = ?, resolution = ?, assigned_to = ?, updated_at = ?, closed_at = ? WHERE id = ?",
        (status, resolution or inc["resolution"], assigned, now, now if status == "closed" else None, inc["id"]))
    log_event("info", "incident_updated", user_id=user["id"], detail=f"incident {inc['id']} -> {status}")
    return None


def delete(user, inc):
    if user["role"] != "admin":
        return "Only administrators delete incidents."
    run("DELETE FROM incidents WHERE id = ?", (inc["id"],))
    log_event("info", "incident_deleted", user_id=user["id"], detail=f"incident {inc['id']}")
    return None
