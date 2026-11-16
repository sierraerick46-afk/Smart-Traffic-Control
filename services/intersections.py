"""Intersection catalogue (create, read, update, delete) with validation and relationships."""
import re

from models.database import insert, log_event, now_iso, q, q1, run

CODE_RE = re.compile(r"^[A-Z0-9-]{3,20}$")


def list_all(active_only=False):
    sql = ("SELECT i.*, (SELECT COUNT(*) FROM inferences f WHERE f.intersection_id = i.id) AS forecasts, "
           "(SELECT COUNT(*) FROM timing_plans p WHERE p.intersection_id = i.id) AS plans, "
           "(SELECT COUNT(*) FROM incidents n WHERE n.intersection_id = i.id) AS incidents FROM intersections i")
    if active_only:
        sql += " WHERE i.is_active = 1"
    return q(sql + " ORDER BY i.code")


def get(inter_id):
    return q1("SELECT * FROM intersections WHERE id = ?", (inter_id,))


def _number(raw, low, high, label, integer=False):
    raw = (raw or "").strip()
    if raw == "":
        return None, None
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        return None, f"{label} must be a number."
    if not low <= value <= high:
        return None, f"{label} must be between {low} and {high}."
    return value, None


def validate(form, valid_junctions, current_id=None):
    """Returns (clean, errors)."""
    errors = {}
    code = (form.get("code") or "").strip().upper()
    name = " ".join((form.get("name") or "").split())
    if not CODE_RE.match(code):
        errors["code"] = "Use 3 to 20 letters, numbers or dashes (for example CHIA-005)."
    else:
        dup = q1("SELECT id FROM intersections WHERE code = ?", (code,))
        if dup and dup["id"] != current_id:
            errors["code"] = "That code already exists."
    if not 3 <= len(name) <= 120:
        errors["name"] = "Write the intersection name (3 to 120 characters)."
    junction, msg = _number(form.get("junction"), 1, 999, "The junction", integer=True)
    if msg:
        errors["junction"] = msg
    elif junction is not None:
        if junction not in valid_junctions:
            errors["junction"] = f"Only junctions with historical data can be linked: {sorted(valid_junctions)}."
        else:
            dup = q1("SELECT id FROM intersections WHERE junction = ?", (junction,))
            if dup and dup["id"] != current_id:
                errors["junction"] = "That junction is already linked to another intersection."
    phases, msg = _number(form.get("phases") or "2", 2, 8, "The number of phases", integer=True)
    if msg:
        errors["phases"] = msg
    lat, msg = _number(form.get("latitude"), -90, 90, "The latitude")
    if msg:
        errors["latitude"] = msg
    lon, msg = _number(form.get("longitude"), -180, 180, "The longitude")
    if msg:
        errors["longitude"] = msg
    speed, msg = _number(form.get("speed_limit_kmh"), 10, 120, "The speed limit", integer=True)
    if msg:
        errors["speed_limit_kmh"] = msg
    notes = (form.get("notes") or "").strip()
    if len(notes) > 300:
        errors["notes"] = "Notes can have up to 300 characters."
    clean = {"code": code, "name": name, "junction": junction, "phases": phases, "latitude": lat,
             "longitude": lon, "speed_limit_kmh": speed, "notes": notes or None,
             "is_active": 1 if form.get("is_active") else 0}
    return clean, errors


def create(user, clean):
    nid = insert("intersections", {**clean, "created_at": now_iso()})
    log_event("info", "intersection_created", user_id=user["id"], detail=f"{clean['code']}")
    return nid


def update(user, inter_id, clean):
    cols = ", ".join(f"{c} = ?" for c in clean)
    run(f"UPDATE intersections SET {cols} WHERE id = ?", (*clean.values(), inter_id))
    log_event("info", "intersection_updated", user_id=user["id"], detail=f"{clean['code']}")


def delete(user, inter):
    """Refuses to delete an intersection that has forecasts, plans or incidents (disable it instead)."""
    used = q1("SELECT (SELECT COUNT(*) FROM inferences WHERE intersection_id = ?) + "
              "(SELECT COUNT(*) FROM timing_plans WHERE intersection_id = ?) + "
              "(SELECT COUNT(*) FROM incidents WHERE intersection_id = ?) AS n", (inter["id"],) * 3)["n"]
    if used:
        return "This intersection has forecasts, plans or incidents linked to it. Disable it instead of deleting it."
    run("DELETE FROM intersections WHERE id = ?", (inter["id"],))
    log_event("info", "intersection_deleted", user_id=user["id"], detail=inter["code"])
    return None
