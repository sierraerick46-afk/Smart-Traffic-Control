"""Timing plans: the business process that uses the forecast.

draft -> pending -> approved -> applied      (administrators approve or reject)
draft -> pending -> rejected -> draft        (the author may reopen a rejected plan)
"""
from models.database import insert, log_event, now_iso, q, q1, run
from services import plan_rules

BASE_SELECT = """SELECT p.*, i.code AS inter_code, i.name AS inter_name, i.junction AS junction,
        u.full_name AS creator, d.full_name AS decider,
        inf.target_datetime AS target_datetime, inf.predicted_vehicles AS predicted_vehicles,
        inf.group_label AS group_label, inf.model_version AS model_version
    FROM timing_plans p
    JOIN intersections i ON i.id = p.intersection_id
    JOIN users u ON u.id = p.created_by
    JOIN inferences inf ON inf.id = p.inference_id
    LEFT JOIN users d ON d.id = p.decided_by"""


class PlanService:
    def __init__(self, loader):
        self.loader = loader

    def reference_peak(self, junction):
        for it in self.loader.intersections():
            if it["junction"] == junction:
                return it["p95"]
        return 0.0

    @staticmethod
    def can_see(user, plan):
        if user["role"] == "admin":
            return True
        if plan["status"] in ("approved", "applied"):
            return True
        return user["role"] == "operator" and plan["created_by"] == user["id"]

    @staticmethod
    def can_edit(user, plan):
        return plan["status"] == "draft" and (user["role"] == "admin" or
                                              (user["role"] == "operator" and plan["created_by"] == user["id"]))

    @staticmethod
    def is_owner_or_admin(user, plan):
        return user["role"] == "admin" or (user["role"] == "operator" and plan["created_by"] == user["id"])

    def list(self, user, status=None, intersection_id=None):
        where, params = [], []
        if user["role"] == "operator":
            where.append("(p.status IN ('approved', 'applied') OR p.created_by = ?)")
            params.append(user["id"])
        elif user["role"] == "viewer":
            where.append("p.status IN ('approved', 'applied')")
        if status in plan_rules.STATUS_LABELS:
            where.append("p.status = ?")
            params.append(status)
        if intersection_id:
            where.append("p.intersection_id = ?")
            params.append(intersection_id)
        sql = BASE_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY p.id DESC LIMIT 200"
        return q(sql, params)

    def get(self, plan_id):
        return q1(BASE_SELECT + " WHERE p.id = ?", (plan_id,))

    def suggestion(self, inference):
        inter = q1("SELECT junction FROM intersections WHERE id = ?", (inference["intersection_id"],))
        return plan_rules.recommend_green(inference["predicted_vehicles"], self.reference_peak(inter["junction"]))

    def create(self, user, inference, final_green, rationale):
        """Returns (plan_id, errors)."""
        errors = {}
        if user["role"] not in ("admin", "operator"):
            return None, {"_": "Your role cannot create plans."}
        if user["role"] == "operator" and inference["user_id"] != user["id"]:
            return None, {"_": "You can only plan from your own forecasts."}
        inter = q1("SELECT * FROM intersections WHERE id = ?", (inference["intersection_id"],))
        if not inter or not inter["is_active"]:
            return None, {"_": "The intersection is disabled."}
        if q1("SELECT id FROM timing_plans WHERE inference_id = ? AND status <> 'rejected'", (inference["id"],)):
            return None, {"_": "This forecast already has an active plan."}
        model_green = self.suggestion(inference)
        final, errors = plan_rules.validate_green(final_green, model_green, rationale)
        if errors:
            return None, errors
        now = now_iso()
        pid = insert("timing_plans", {
            "intersection_id": inference["intersection_id"], "inference_id": inference["id"],
            "created_by": user["id"], "status": "draft", "model_green_s": model_green, "final_green_s": final,
            "rationale": (rationale or "").strip()[:500], "created_at": now, "updated_at": now})
        log_event("info", "plan_created", user_id=user["id"], detail=f"plan {pid} from inference {inference['id']}")
        return pid, {}

    def update(self, user, plan, final_green, rationale):
        if not self.can_edit(user, plan):
            return {"_": "Only the author (or an administrator) can edit a draft plan."}
        final, errors = plan_rules.validate_green(final_green, plan["model_green_s"], rationale)
        if errors:
            return errors
        run("UPDATE timing_plans SET final_green_s = ?, rationale = ?, updated_at = ? WHERE id = ?",
            (final, (rationale or "").strip()[:500], now_iso(), plan["id"]))
        return {}

    def transition(self, user, plan, new_status, note=""):
        """Applies one workflow step. Returns an error message or None."""
        if new_status not in plan_rules.TRANSITIONS.get(plan["status"], ()):
            return f"A {plan['status']} plan cannot become {new_status}."
        note = (note or "").strip()[:500]
        if new_status in ("approved", "rejected"):
            if user["role"] != "admin":
                return "Only administrators approve or reject plans."
            if new_status == "rejected" and len(note) < 10:
                return "Write the reason for rejecting the plan (at least 10 characters)."
            run("UPDATE timing_plans SET status = ?, decided_by = ?, decision_note = ?, decided_at = ?, updated_at = ? "
                "WHERE id = ?", (new_status, user["id"], note, now_iso(), now_iso(), plan["id"]))
        else:
            if not self.is_owner_or_admin(user, plan):
                return "Only the author or an administrator can do this."
            if new_status == "applied":
                run("UPDATE timing_plans SET status = 'applied', applied_at = ?, updated_at = ? WHERE id = ?",
                    (now_iso(), now_iso(), plan["id"]))
            else:
                run("UPDATE timing_plans SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, now_iso(), plan["id"]))
        log_event("info", "plan_" + new_status, user_id=user["id"], detail=f"plan {plan['id']}")
        return None

    def delete(self, user, plan):
        if not self.can_edit(user, plan):
            return "Only a draft plan can be deleted, by its author or an administrator."
        run("DELETE FROM timing_plans WHERE id = ?", (plan["id"],))
        log_event("info", "plan_deleted", user_id=user["id"], detail=f"plan {plan['id']}")
        return None
