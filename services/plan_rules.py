"""Business rules that turn a forecast into a proposed green time.

The limits are the same ones used by the simulator (models/traffic_env.py): minimum green 8 s, maximum green 60 s.
The recommendation grows linearly with the share of the intersection's reference peak (its 95th percentile of
hourly vehicles in the training data) that the forecast represents. It is decision support: a person can adjust it,
and an administrator must approve it before it counts as the plan.
"""

MIN_GREEN = 8
MAX_GREEN = 60
MAX_FREE_ADJUSTMENT = 15  # seconds a user may move the recommendation without writing a justification

STATUS_LABELS = {"draft": "Draft", "pending": "Pending approval", "approved": "Approved",
                 "rejected": "Rejected", "applied": "Applied"}
TRANSITIONS = {"draft": ("pending",), "pending": ("approved", "rejected"), "approved": ("applied",),
               "rejected": ("draft",), "applied": ()}


def recommend_green(forecast, reference_peak):
    """Green seconds proposed for the forecast hour (rule R1: always between MIN_GREEN and MAX_GREEN)."""
    if not reference_peak or reference_peak <= 0:
        return MIN_GREEN
    share = max(0.0, min(1.0, float(forecast) / float(reference_peak)))
    return int(round(MIN_GREEN + (MAX_GREEN - MIN_GREEN) * share))


def validate_green(final, model_green, rationale):
    """Returns (final_int, errors)."""
    errors = {}
    try:
        final = int(final)
    except (TypeError, ValueError):
        return None, {"final_green_s": "Write the green time in whole seconds."}
    if not MIN_GREEN <= final <= MAX_GREEN:
        errors["final_green_s"] = f"The green time must be between {MIN_GREEN} and {MAX_GREEN} seconds."
    if abs(final - model_green) > MAX_FREE_ADJUSTMENT and len((rationale or "").strip()) < 10:
        errors["rationale"] = (f"Explain the change (at least 10 characters) when you move the recommendation "
                               f"by more than {MAX_FREE_ADJUSTMENT} seconds.")
    return final, errors
