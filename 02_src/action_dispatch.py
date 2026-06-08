import json
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import numpy as np
import requests

from utils import log_to_json

logger = logging.getLogger(__name__)
LAST_GEMINI_ERROR: str | None = None

STATUS_PRIORITY = {"Critical": 0, "Warning": 1, "Stable": 2}
SMTP_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "smtp_settings.local.json"
SMTP_PROVIDER_PRESETS = {
    "Gmail": {"smtp_host": "smtp.gmail.com", "smtp_port": 587, "smtp_security": "STARTTLS"},
    "Outlook": {"smtp_host": "smtp.office365.com", "smtp_port": 587, "smtp_security": "STARTTLS"},
    "Yahoo": {"smtp_host": "smtp.mail.yahoo.com", "smtp_port": 465, "smtp_security": "SSL"},
    "Custom": {"smtp_host": "", "smtp_port": 587, "smtp_security": "STARTTLS"},
}

REACT_DEFAULTS = {
    "minimum_score": 0.25,
    "borderline_score": 0.45,
    "direct_score": 0.68,
    "high_score": 0.76,
    "reliability_threshold": 0.72,
    "rag_success_min": 3,
    "rag_k": 5,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return _clamp(float(np.dot(left, right) / (left_norm * right_norm)), -1.0, 1.0)


def compute_intelligent_score(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    threshold = float(context.get("threshold", record.get("threshold_last", 0.5)) or 0.5)
    svm_prob = float(record.get("svm_prob_24h", 0.0) or 0.0)
    future_risk = float(record.get("future_risk", 0.0) or 0.0)
    prob_stable = float(record.get("prob_stable", 0.0) or 0.0)
    prob_warning = float(record.get("prob_warning", 0.0) or 0.0)
    prob_critical = float(record.get("prob_critical", 0.0) or 0.0)
    cluster_id = int(record.get("fcm_cluster", np.argmax([prob_stable, prob_warning, prob_critical])) or 0)

    severity_base = [0.18, 0.55, 0.90][max(0, min(2, cluster_id))]
    severity_signal = _clamp((0.72 * severity_base) + (0.28 * future_risk))
    svm_gap_signal = _clamp(0.5 + (svm_prob - threshold))
    membership_signal = _clamp((0.12 * prob_stable) + (0.58 * prob_warning) + (1.0 * prob_critical))

    trend_multiplier = float(record.get("arima_trend_multiplier", context.get("default_trend_multiplier", 1.0)) or 1.0)
    arima_signal = _clamp(0.5 + ((trend_multiplier - 1.0) * 1.4))

    shap_vector = np.asarray(record.get("shap_vector", []), dtype=float)
    cluster_means = context.get("shap_cluster_means", {})
    cluster_mean = np.asarray(cluster_means.get(cluster_id, np.zeros_like(shap_vector)), dtype=float)
    shap_reliability = _clamp((_cosine(shap_vector, cluster_mean) + 1.0) / 2.0) if shap_vector.size else 0.0

    weights = {
        "S": 0.24,
        "Gap(SVM)": 0.22,
        "mu(predicted)": 0.20,
        "ARIMA": 0.16,
        "SHAP": 0.18,
    }
    signals = {
        "S": severity_signal,
        "Gap(SVM)": svm_gap_signal,
        "mu(predicted)": membership_signal,
        "ARIMA": arima_signal,
        "SHAP": shap_reliability,
    }
    score = sum(signals[name] * weights[name] for name in weights)
    return {
        "intelligent_score": round(float(score), 4),
        "signals": {name: round(float(value), 4) for name, value in signals.items()},
        "weights": weights,
        "shap_cosine": round(float(shap_reliability), 4),
        "trend_multiplier": round(float(trend_multiplier), 4),
        "dynamic_severity_justification": (
            f"Severity blends live FCM class {cluster_id} with future-risk forecast {future_risk:.3f}"
        ),
    }


def _dominant_feature(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    shap_vector = np.asarray(record.get("shap_vector", []), dtype=float)
    feature_columns = list(context.get("feature_columns", []))
    if shap_vector.size == 0 or not feature_columns:
        return {"name": "overall physiology", "value": 0.0}
    idx = int(np.argmax(np.abs(shap_vector)))
    name = feature_columns[idx] if idx < len(feature_columns) else "overall physiology"
    return {"name": str(name), "value": round(float(shap_vector[idx]), 4)}


def _adaptive_action(
    route: str,
    record: dict[str, Any],
    score_payload: dict[str, Any],
    dominant: dict[str, Any],
    rag_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    score = float(score_payload["intelligent_score"])
    trend_multiplier = float(score_payload["trend_multiplier"])
    rag_vote = (rag_payload or {}).get("winner", "no comparable majority")
    arima_phrase = "rising" if trend_multiplier >= 1.03 else "declining" if trend_multiplier <= 0.97 else "flat"
    intensity = _clamp((score * 0.72) + (max(0.0, trend_multiplier - 1.0) * 1.8))

    if route in {"score_below_minimum", "rag_zero_success"} and score < 0.35:
        timing_minutes = 240
    elif route in {"direct_dispatch", "shap_recheck_dispatch"}:
        timing_minutes = 0
    elif "human_review" in route or route == "rag_tie":
        timing_minutes = 30
    else:
        timing_minutes = int(max(10, 90 - (score * 80)))

    action_type = "clinical escalation" if intensity >= 0.72 else "targeted reassessment" if intensity >= 0.45 else "watchful monitoring"
    return {
        "action_type": action_type,
        "timing_minutes": int(timing_minutes),
        "intensity": round(float(intensity), 4),
        "personalisation": (
            f"Focus on {dominant['name']} because it is the dominant SHAP driver; "
            f"RAG outcome is {rag_vote}; ARIMA trend is {arima_phrase}."
        ),
        "explanation": (
            f"Route {route} selected from live score {score:.3f}, SHAP reliability "
            f"{score_payload['shap_cosine']:.3f}, trend multiplier {trend_multiplier:.3f}."
        ),
    }


def _query_rag(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    query_fn = context.get("rag_query_fn")
    if query_fn is None:
        return {"status": "zero_success", "neighbors": [], "winner": None, "vote_counts": {}}
    return query_fn(record, int(context.get("rag_k", REACT_DEFAULTS["rag_k"])))


def run_react_inner_loop(
    record: dict[str, Any],
    context: dict[str, Any],
    audit_path: str | None = None,
    case_label: str | None = None,
) -> dict[str, Any]:
    """ReAct inner loop: Observe -> Reason -> Act -> Observe."""
    cfg = {**REACT_DEFAULTS, **context.get("react", {})}
    score_payload = compute_intelligent_score(record, context)
    dominant = _dominant_feature(record, context)
    route = "confidence_too_low"
    rag_payload = None

    score = float(score_payload["intelligent_score"])
    cosine = float(score_payload["shap_cosine"])
    trend_multiplier = float(score_payload["trend_multiplier"])
    high_non_shap_risk = (
        float(score_payload["signals"].get("S", 0.0)) >= 0.82
        and float(score_payload["signals"].get("Gap(SVM)", 0.0)) >= 0.70
        and float(score_payload["signals"].get("mu(predicted)", 0.0)) >= 0.80
    )

    # Step 1: OBSERVE -- receive case, compute all five signals
    observations: list[dict] = [
        {
            "step": "Observe",
            "summary": (
                f"New case received. Intelligent score={score:.3f}, "
                f"SHAP cosine={cosine:.3f}, ARIMA multiplier={trend_multiplier:.3f}."
            ),
            "score": score_payload,
        }
    ]

    # Step 2: REASON -- assess which routing branch is warranted
    if score < float(cfg["minimum_score"]):
        routing_branch = "score_too_low"
        reason_pre = (
            f"Score {score:.3f} is below minimum threshold {cfg['minimum_score']}. "
            "No clinical action warranted; filing as low-risk."
        )
    elif score >= float(cfg["direct_score"]) and cosine >= float(cfg["reliability_threshold"]):
        routing_branch = "direct"
        reason_pre = (
            f"Score {score:.3f} >= direct threshold {cfg['direct_score']} and "
            f"SHAP reliability {cosine:.3f} >= {cfg['reliability_threshold']}. "
            "High confidence -- direct dispatch is safe."
        )
    elif (score >= float(cfg["direct_score"]) or high_non_shap_risk) and cosine < float(cfg["reliability_threshold"]):
        routing_branch = "shap_recheck"
        reason_pre = (
            f"Score is high ({score:.3f}) but SHAP reliability {cosine:.3f} < "
            f"{cfg['reliability_threshold']}. Will run SHAP re-check loop."
        )
    elif float(cfg["borderline_score"]) <= score < float(cfg["direct_score"]):
        routing_branch = "rag"
        reason_pre = (
            f"Score {score:.3f} is in the borderline zone "
            f"[{cfg['borderline_score']}, {cfg['direct_score']}). "
            "Will query explicit RAG for historical evidence."
        )
    else:
        routing_branch = "low_confidence"
        reason_pre = (
            f"Score {score:.3f} is above minimum but SHAP reliability {cosine:.3f} "
            "is insufficient for autonomous dispatch. Routing to human review."
        )

    observations.append(
        {
            "step": "Reason",
            "summary": reason_pre,
            "routing_branch": routing_branch,
            "dominant_feature": dominant,
        }
    )

    # Step 3: ACT -- execute the action for the reasoned branch
    if routing_branch == "score_too_low":
        route = "score_below_minimum"
        reason = "Composite score is below the minimum action threshold."
        observations.append(
            {"step": "Act", "summary": "No action dispatched. Case filed as low-risk.", "route": route}
        )

    elif routing_branch == "direct":
        route = "direct_dispatch"
        reason = "Score and implicit SHAP reliability are both high."
        observations.append(
            {"step": "Act", "summary": "Direct dispatch initiated without further evidence checks.", "route": route}
        )

    elif routing_branch == "shap_recheck":
        shap_vector = np.asarray(record.get("shap_vector", []), dtype=float).copy()
        if shap_vector.size:
            shap_vector[int(np.argmax(np.abs(shap_vector)))] = 0.0
        cluster_id = int(record.get("fcm_cluster", 0) or 0)
        cluster_mean = np.asarray(
            context.get("shap_cluster_means", {}).get(cluster_id, np.zeros_like(shap_vector)),
            dtype=float,
        )
        rechecked = _clamp((_cosine(shap_vector, cluster_mean) + 1.0) / 2.0) if shap_vector.size else 0.0
        if rechecked >= float(cfg["reliability_threshold"]):
            route = "shap_recheck_dispatch"
            reason = "Reliability recovered after SHAP re-check."
        else:
            route = "shap_recheck_human_review"
            reason = "Reliability stayed below threshold after SHAP re-check."
        observations.append(
            {
                "step": "Act",
                "summary": (
                    "SHAP re-check loop removed the dominant feature and recomputed reliability. "
                    f"Rechecked cosine={rechecked:.3f} -> route={route}."
                ),
                "rechecked_cosine": round(float(rechecked), 4),
                "route": route,
            }
        )

    elif routing_branch == "rag":
        rag_payload = _query_rag(record, context)
        if rag_payload.get("status") == "zero_success":
            route = "rag_zero_success"
            reason = "RAG had insufficient neighbors, so routing falls back to SHAP reliability."
            if cosine < float(cfg["reliability_threshold"]):
                route = "rag_zero_success_human_review"
        elif rag_payload.get("status") == "tie":
            route = "rag_tie"
            reason = "Retrieved outcomes tied, so a human review is required."
        elif rag_payload.get("status") == "majority":
            if trend_multiplier >= 1.0:
                route = "rag_majority_arima_full_trust"
                reason = "RAG majority and rising/flat ARIMA trend agree."
            else:
                route = "rag_majority_arima_dampened_review"
                reason = "RAG majority is dampened because ARIMA trend is declining."
        else:
            route = "confidence_too_low_human_review"
            reason = "RAG returned an untrusted evidence state."
        observations.append(
            {
                "step": "Act",
                "summary": (
                    f"Explicit RAG queried (k={context.get('rag_k', 5)}). "
                    f"Status={rag_payload.get('status')}, winner={rag_payload.get('winner')}, "
                    f"ARIMA multiplier={trend_multiplier:.3f} -> route={route}."
                ),
                "rag": rag_payload,
                "route": route,
            }
        )

    else:

        route = "confidence_too_low_human_review"
        reason = "Score is above the minimum but not reliable enough for autonomous dispatch."
        observations.append(
            {
                "step": "Act",
                "summary": "Routed to human review -- insufficient confidence for autonomous action.",
                "route": route,
            }
        )

    # Step 4: OBSERVE (result) -- capture final action and close the loop
    action = _adaptive_action(route, record, score_payload, dominant, rag_payload)
    observations.append(
        {
            "step": "Observe",
            "summary": (
                f"Inner ReAct loop complete. Route={route}. "
                f"Action={action['action_type']} in {action['timing_minutes']} min. "
                f"Reason: {reason}"
            ),
            "route": route,
            "reason": reason,
            "action": action,
        }
    )

    result = {
        "case_label": case_label or str(record.get("patient_id", "case")),
        "patient_id": str(record.get("patient_id", "unknown")),
        "hour": int(record.get("hour", 0) or 0),
        "route": route,
        "reason": reason,
        "react_trace": observations,
        "score": score_payload,
        "dominant_feature": dominant,
        "rag": rag_payload,
        "action": action,
        "audit_context": {
            "thresholds": cfg,
            "raw_model_values": {
                "svm_prob_24h": float(record.get("svm_prob_24h", 0.0) or 0.0),
                "future_risk": float(record.get("future_risk", 0.0) or 0.0),
                "prob_stable": float(record.get("prob_stable", 0.0) or 0.0),
                "prob_warning": float(record.get("prob_warning", 0.0) or 0.0),
                "prob_critical": float(record.get("prob_critical", 0.0) or 0.0),
            },
        },
    }
    if audit_path:
        log_to_json(audit_path, result)
    return result


def derive_patient_state(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    future_risk = float(record.get("future_risk", 0.0) or 0.0)
    svm_prob = float(record.get("svm_prob_24h", 0.0) or 0.0)
    prob_critical = float(record.get("prob_critical", 0.0) or 0.0)
    prob_warning = float(record.get("prob_warning", 0.0) or 0.0)
    anomaly_flag = int(record.get("anomaly_flag", 0) or 0)
    trend_alert = bool(record.get("trend_alert", False))

    critical_reasons = []
    warning_reasons = []

    if svm_prob >= max(threshold, 0.60):
        critical_reasons.append(f"24h deterioration score is high ({svm_prob:.2f})")
    if future_risk >= 0.70:
        critical_reasons.append(f"12h future risk is high ({future_risk:.2f})")
    if prob_critical >= 0.50:
        critical_reasons.append(f"critical cluster membership is elevated ({prob_critical:.2f})")

    if anomaly_flag == 1:
        warning_reasons.append("DBSCAN anomaly detected")
    if trend_alert:
        warning_reasons.append("trend deviation alert raised")
    if future_risk >= 0.40:
        warning_reasons.append(f"future risk is above watch level ({future_risk:.2f})")
    if prob_warning >= 0.40:
        warning_reasons.append(f"warning cluster membership is elevated ({prob_warning:.2f})")

    if critical_reasons:
        state = "Critical"
        action_label = "Escalate now"
        reasons = critical_reasons + warning_reasons
    elif warning_reasons:
        state = "Warning"
        action_label = "Close monitoring"
        reasons = warning_reasons
    else:
        state = "Stable"
        action_label = "Routine observation"
        reasons = ["all live monitoring layers are currently within stable range"]

    return {
        "clinical_state": state,
       # NEW — three distinct statuses
        "binary_status": {"Stable": "Normal", "Warning": "Warning", "Critical": "Dangerous Situation"}.get(state, "Unknown"),
        "dispatch_action": action_label,
        "dispatch_priority": STATUS_PRIORITY[state],
        "alert_reason": "; ".join(reasons),
        "is_actionable": state != "Stable",
    }

# NEW
def _get_gemini_api_key() -> str:
    """Read Gemini API key: env var first, then saved JSON settings file."""
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    # Fallback: read from smtp_settings.local.json (handles Streamlit restarts)
    try:
        if SMTP_SETTINGS_PATH.exists():
            with open(SMTP_SETTINGS_PATH, "r", encoding="utf-8") as _fh:
                key = json.load(_fh).get("gemini_api_key", "").strip()
    except Exception:
        pass
    return key

def _generate_gemini_text(prompt: str, system_prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Generate text with Gemini using the native REST API."""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise RuntimeError(f"Gemini returned an empty response: {payload}")

    return text


def generate_professional_email(record: dict[str, Any]) -> str | None:
    """Use Gemini to write a professional medical alert email body from SHAP results.

    Returns None if Gemini is unavailable so build_email_payload can fall back
    to the static template.
    """
    global LAST_GEMINI_ERROR
    LAST_GEMINI_ERROR = None

    if not _get_gemini_api_key():
        LAST_GEMINI_ERROR = "GEMINI_API_KEY is not set in this Streamlit/Python process."
        logger.info("GEMINI_API_KEY is not set; using static email template.")
        return None

    state = str(record.get("clinical_state", "Unknown"))
    binary_status = str(record.get("binary_status", "Unknown"))
    patient_id = str(record.get("patient_id", "N/A"))
    hour = str(record.get("hour", "N/A"))
    future_risk = float(record.get("future_risk", 0.0) or 0.0)
    svm_prob = float(record.get("svm_prob_24h", 0.0) or 0.0)
    reason = str(record.get("alert_reason", "No reason available"))
    action_label = str(record.get("dispatch_action", "Review dashboard"))

    # SHAP-specific context
    dominant = record.get("dominant_feature", {})
    dominant_name = dominant.get("name", "overall physiology") if isinstance(dominant, dict) else "overall physiology"
    dominant_value = dominant.get("value", 0.0) if isinstance(dominant, dict) else 0.0
    top_features = record.get("top_features", [])
    shap_cosine = float(record.get("shap_cosine", 0.0) or 0.0)
    intelligent_score = float(record.get("intelligent_score", 0.0) or 0.0)

    prompt = (
        "You are a clinical alert notification system. "
        "Write a formal, professional medical alert email body for the healthcare team. "
        "Do NOT include a subject line - only the email body.\n\n"
        f"Patient ID: {patient_id}\n"
        f"Monitoring Hour: {hour}\n"
        f"Clinical State: {state} - {binary_status}\n"
        f"Recommended Action: {action_label}\n"
        f"12h Future Risk Score: {future_risk:.3f}\n"
        f"24h SVM Deterioration Score: {svm_prob:.3f}\n"
        f"Alert Reason: {reason}\n\n"
        f"SHAP Analysis Results:\n"
        f"  Dominant Risk Driver: {dominant_name} (SHAP value: {dominant_value:.4f})\n"
        f"  Top Contributing Features: {', '.join(top_features) if top_features else 'N/A'}\n"
        f"  SHAP Reliability (cosine alignment): {shap_cosine:.3f}\n"
        f"  Composite Intelligent Risk Score: {intelligent_score:.4f}\n\n"
        "Write 3-4 concise paragraphs:\n"
        "1. Opening with urgency level and patient identification\n"
        "2. Clinical interpretation of the SHAP analysis - which physiological parameters "
        "are driving the risk and why they matter\n"
        "3. Specific recommended clinical actions based on the scores\n"
        "4. Closing with monitoring instructions and follow-up guidance\n\n"
        "Use formal medical language. Be precise and actionable."
    )

    try:
        return _generate_gemini_text(
            prompt=prompt,
            system_prompt=(
                "You are a clinical decision support system generating "
                "formal medical alert notifications for ICU and ward staff."
            ),
        )
    except Exception as exc:
        LAST_GEMINI_ERROR = str(exc)
        logger.warning("Gemini email body generation failed: %s", exc)
        return None  # Fall back to static template

def build_email_payload(record: dict[str, Any]) -> dict[str, str]:
    state = str(record.get("clinical_state", "Unknown"))

    if record.get("binary_status"):
        binary_status = str(record.get("binary_status"))
    else:
        binary_status = (
            "Dangerous Situation"
            if state.lower() != "stable"
            else "Normal"
        )

    patient_id = str(record.get("patient_id", "N/A"))
    hour = str(record.get("hour", "N/A"))
    future_risk = float(record.get("future_risk", 0.0) or 0.0)
    svm_prob = float(record.get("svm_prob_24h", 0.0) or 0.0)
    reason = str(record.get("alert_reason", "No reason available"))
    action_label = str(record.get("dispatch_action", "Review dashboard"))

    subject = f"[Patient Monitor] {binary_status} for patient {patient_id}"

    # Try Gemini first for a professional, SHAP-informed email body
    gemini_body = generate_professional_email({**record, "binary_status": binary_status})

    gemini_used = bool(gemini_body)
    if gemini_body:
        body = gemini_body
    else:
        # Fallback: original static template
        body = f"""
====================================

ZONE 3 ACTION DISPATCH

====================================


DISPATCH STATE:

{binary_status}


Clinical State:

{state}


Recommended Action:

{action_label}


------------------------------------


Patient ID: {patient_id}
Hour: {hour}


12h Future Risk: {future_risk:.3f}


24h Deterioration Score:

{svm_prob:.3f}


Reason:

{reason}

====================================

Generated by Chronic Patient
Monitoring Dashboard

====================================

"""

    return {
        "subject": subject,
        "body": body,
        "gemini_used": str(gemini_used),
        "gemini_error": "" if gemini_used else (LAST_GEMINI_ERROR or "Gemini returned no body."),
    }

def get_provider_preset(provider_name: str) -> dict[str, Any]:
    return dict(SMTP_PROVIDER_PRESETS.get(provider_name, SMTP_PROVIDER_PRESETS["Custom"]))


def detect_provider_name(smtp_host: str) -> str:
    host = (smtp_host or "").lower()
    if "gmail" in host:
        return "Gmail"
    if "office365" in host or "outlook" in host or "hotmail" in host:
        return "Outlook"
    if "yahoo" in host:
        return "Yahoo"
    return "Custom"


def load_smtp_settings(defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(defaults or {})

    if SMTP_SETTINGS_PATH.exists():
        with open(SMTP_SETTINGS_PATH, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        settings.update(saved)

    env_overrides = {
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": os.getenv("SMTP_PORT"),
        "smtp_security": os.getenv("SMTP_SECURITY"),
        "smtp_username": os.getenv("SMTP_USERNAME"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "sender_email": os.getenv("SENDER_EMAIL"),
        "recipient_email": os.getenv("RECIPIENT_EMAIL"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    }
    for key, value in env_overrides.items():
        if value not in (None, ""):
            settings[key] = int(value) if key == "smtp_port" else value

    settings.setdefault("smtp_host", "")
    settings.setdefault("smtp_port", 587)
    settings.setdefault("smtp_security", "STARTTLS")
    settings.setdefault("smtp_username", "")
    settings.setdefault("smtp_password", "")
    settings.setdefault("sender_email", "")
    settings.setdefault("recipient_email", "")
    settings.setdefault("gemini_api_key", "")
    settings["email_provider"] = detect_provider_name(str(settings.get("smtp_host", "")))
    return settings


def save_smtp_settings(settings: dict[str, Any]) -> str:
    payload = {
        "smtp_host": str(settings.get("smtp_host", "")).strip(),
        "smtp_port": int(settings.get("smtp_port", 587)),
        "smtp_security": str(settings.get("smtp_security", "STARTTLS")).strip().upper(),
        "smtp_username": str(settings.get("smtp_username", "")).strip(),
        "smtp_password": str(settings.get("smtp_password", "")).strip(),
        "sender_email": str(settings.get("sender_email", "")).strip(),
        "recipient_email": str(settings.get("recipient_email", "")).strip(),
        "gemini_api_key": str(settings.get("gemini_api_key", "")).strip(),
    }
    with open(SMTP_SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(SMTP_SETTINGS_PATH)


def validate_smtp_settings(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    sender_email: str,
    recipient_email: str | None = None,
) -> None:
    if not smtp_host:
        raise ValueError("SMTP host is required.")
    if not smtp_port:
        raise ValueError("SMTP port is required.")
    if not sender_email and not username:
        raise ValueError("Sender email or SMTP username is required.")
    if username and not password:
        raise ValueError("SMTP password is required when SMTP username is provided.")
    if username == "sender@gmail.com" or sender_email == "sender@gmail.com":
        raise ValueError("Replace the demo sender email with your real sender email.")
    if password == "app-password":
        raise ValueError("Replace the demo SMTP password with your real SMTP or app password.")
    if recipient_email is not None and not recipient_email.strip():
        raise ValueError("Recipient email is required.")


def build_smtp_settings_from_state(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "smtp_host": str(source.get("smtp_host", "")).strip(),
        "smtp_port": int(source.get("smtp_port", 587)),
        "smtp_security": str(source.get("smtp_security", "STARTTLS")).strip().upper(),
        "smtp_username": str(source.get("smtp_username", "")).strip(),
        "smtp_password": str(source.get("smtp_password", "")).strip(),
        "sender_email": str(source.get("sender_email", "")).strip(),
        "recipient_email": str(source.get("recipient_email", "")).strip(),
        "gemini_api_key": str(source.get("gemini_api_key", "")).strip(),
    }


# ==================== الدالة المعدلة (الحل النهائي) ====================
def send_email_notification(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    sender_email: str,
    recipient_email: str,
    subject: str,
    body: str,
    security_mode: str = "STARTTLS",
    timeout: int = 20,
) -> None:
    validate_smtp_settings(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=username,
        password=password,
        sender_email=sender_email,
        recipient_email=recipient_email,
    )

    sender = sender_email or username
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient_email
    message.set_content(body)

    # ✅ الحل النهائي لمشكلة الشهادة (CERTIFICATE_VERIFY_FAILED)
    context = ssl._create_unverified_context()

    mode = (security_mode or "STARTTLS").upper()

    if mode == "SSL":
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout, context=context) as server:
            server.ehlo()
            if username:
                server.login(username, password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
        server.ehlo()
        if mode == "STARTTLS":
            server.starttls(context=context)
            server.ehlo()
        if username:
            server.login(username, password)
        server.send_message(message)
# ====================================================================


def send_test_email(settings: dict[str, Any], recipient_email: str | None = None) -> str:
    recipient = (recipient_email or settings.get("recipient_email", "")).strip()
    validate_smtp_settings(
        smtp_host=str(settings.get("smtp_host", "")),
        smtp_port=int(settings.get("smtp_port", 587)),
        username=str(settings.get("smtp_username", "")),
        password=str(settings.get("smtp_password", "")),
        sender_email=str(settings.get("sender_email", "")),
        recipient_email=recipient,
    )

    send_email_notification(
        smtp_host=str(settings.get("smtp_host", "")),
        smtp_port=int(settings.get("smtp_port", 587)),
        username=str(settings.get("smtp_username", "")),
        password=str(settings.get("smtp_password", "")),
        sender_email=str(settings.get("sender_email", "")),
        recipient_email=recipient,
        subject="[Patient Monitor] SMTP test email",
        body=(
            "This is a test email from the Chronic Patient Monitoring dashboard.\n"
            "If you received this, the SMTP action email flow is configured correctly.\n"
        ),
        security_mode=str(settings.get("smtp_security", "STARTTLS")),
    )
    return recipient


def log_dispatch_result(log_path: str, payload: dict[str, Any]) -> None:
    log_to_json(log_path, payload)
