# ── Rule-Based Recommendation Engine ──────────────────────
# This module takes the model predictions and generates
# clinical recommendations for orthopedic doctors.

# ── Urgency levels ─────────────────────────────────────────
URGENCY_LOW = "low"
URGENCY_MEDIUM = "medium"
URGENCY_HIGH = "high"

# ── Recommendation rules ───────────────────────────────────
# Format: (body_part, condition) -> (urgency, recommendation_text)

RECOMMENDATIONS = {
    # ── Normal findings ────────────────────────────────────
    ("XR_ELBOW", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the elbow X-ray. "
        "Routine follow-up recommended if symptoms persist. "
        "No immediate intervention required."
    ),
    ("XR_FINGER", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the finger X-ray. "
        "Conservative management advised. "
        "Follow-up if pain or swelling develops."
    ),
    ("XR_FOREARM", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the forearm X-ray. "
        "Routine clinical follow-up recommended. "
        "No immediate intervention required."
    ),
    ("XR_HAND", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the hand X-ray. "
        "Conservative management advised. "
        "Follow-up if symptoms worsen."
    ),
    ("XR_HUMERUS", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the humerus X-ray. "
        "Routine follow-up recommended if symptoms persist. "
        "No immediate intervention required."
    ),
    ("XR_SHOULDER", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the shoulder X-ray. "
        "Physiotherapy may be considered if patient reports pain. "
        "No immediate intervention required."
    ),
    ("XR_WRIST", "normal"): (
        URGENCY_LOW,
        "No abnormality detected in the wrist X-ray. "
        "Conservative management advised. "
        "Follow-up if pain or limited range of motion persists."
    ),

    # ── Abnormal findings ──────────────────────────────────
    ("XR_ELBOW", "abnormal"): (
        URGENCY_HIGH,
        "Abnormality detected in the elbow. "
        "Possible fracture or dislocation. "
        "Immediate orthopedic consultation recommended. "
        "Consider immobilization and pain management. "
        "Additional imaging (CT scan) may be required."
    ),
    ("XR_FINGER", "abnormal"): (
        URGENCY_MEDIUM,
        "Abnormality detected in the finger. "
        "Possible fracture or joint abnormality. "
        "Orthopedic evaluation recommended within 24-48 hours. "
        "Splinting and pain management advised in the interim."
    ),
    ("XR_FOREARM", "abnormal"): (
        URGENCY_HIGH,
        "Abnormality detected in the forearm. "
        "Possible fracture requiring immediate attention. "
        "Urgent orthopedic consultation recommended. "
        "Immobilization and pain management required. "
        "Surgical intervention may be necessary."
    ),
    ("XR_HAND", "abnormal"): (
        URGENCY_MEDIUM,
        "Abnormality detected in the hand. "
        "Possible fracture or soft tissue injury. "
        "Orthopedic evaluation recommended within 24-48 hours. "
        "Splinting and pain management advised."
    ),
    ("XR_HUMERUS", "abnormal"): (
        URGENCY_HIGH,
        "Abnormality detected in the humerus. "
        "Possible fracture or significant pathology. "
        "Immediate orthopedic consultation required. "
        "Immobilization essential. "
        "Surgical evaluation may be necessary."
    ),
    ("XR_SHOULDER", "abnormal"): (
        URGENCY_HIGH,
        "Abnormality detected in the shoulder. "
        "Possible fracture, dislocation, or rotator cuff injury. "
        "Urgent orthopedic consultation recommended. "
        "Immobilization and pain management required. "
        "MRI may be indicated for soft tissue evaluation."
    ),
    ("XR_WRIST", "abnormal"): (
        URGENCY_MEDIUM,
        "Abnormality detected in the wrist. "
        "Possible fracture or ligamentous injury. "
        "Orthopedic evaluation recommended within 24 hours. "
        "Splinting and pain management advised. "
        "Consider CT scan for detailed fracture assessment."
    ),
}

# ── Urgency colors for UI ──────────────────────────────────
URGENCY_COLORS = {
    URGENCY_LOW: "#28a745",      # green
    URGENCY_MEDIUM: "#fd7e14",   # orange
    URGENCY_HIGH: "#dc3545",     # red
}

URGENCY_ICONS = {
    URGENCY_LOW: "✅",
    URGENCY_MEDIUM: "⚠️",
    URGENCY_HIGH: "🚨",
}

# ── Main recommendation function ───────────────────────────
def get_recommendation(body_part, condition):
    """
    Returns urgency level and recommendation text
    based on body part and condition.

    Args:
        body_part: string like "XR_ELBOW", "XR_WRIST", etc.
        condition: string "normal" or "abnormal"

    Returns:
        dict with urgency, recommendation, color, and icon
    """
    key = (body_part, condition)

    if key in RECOMMENDATIONS:
        urgency, recommendation = RECOMMENDATIONS[key]
    else:
        # Default fallback
        if condition == "abnormal":
            urgency = URGENCY_MEDIUM
            recommendation = (
                f"Abnormality detected in {body_part.replace('XR_', '')}. "
                "Clinical evaluation recommended. "
                "Please consult an orthopedic specialist."
            )
        else:
            urgency = URGENCY_LOW
            recommendation = (
                f"No abnormality detected in {body_part.replace('XR_', '')}. "
                "Routine follow-up recommended if symptoms persist."
            )

    return {
        "urgency": urgency,
        "recommendation": recommendation,
        "color": URGENCY_COLORS[urgency],
        "icon": URGENCY_ICONS[urgency],
        "body_part": body_part.replace("XR_", "").title(),
        "condition": condition.title()
    }


def get_urgency_description(urgency):
    """Returns a human-readable description of the urgency level."""
    descriptions = {
        URGENCY_LOW: "Low Priority — Routine follow-up sufficient",
        URGENCY_MEDIUM: "Medium Priority — Evaluation within 24-48 hours",
        URGENCY_HIGH: "High Priority — Immediate attention required"
    }
    return descriptions.get(urgency, "Unknown urgency level")