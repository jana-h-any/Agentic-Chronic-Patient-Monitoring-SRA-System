import os
import logging
import requests


logger = logging.getLogger(__name__)


def _generate_gemini_text(prompt: str, system_prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Generate text with Gemini using the native REST API."""
    api_key = os.getenv("GEMINI_API_KEY")
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


def generate_clinical_note(top_features: list[str]) -> str:
    """
    Generate a concise clinical note based on top SHAP features.

    Parameters
    ----------
    top_features : list[str]
        Top SHAP features contributing to patient deterioration risk.

    Returns
    -------
    str
        Clinical narrative note.
    """
    if not top_features:
        return "No significant risk factors identified."

    feature_text = ", ".join(top_features)

    prompt = (
        "Patient status: elevated deterioration risk.\n"
        f"Top risk factors: {feature_text}.\n"
        "Write only 3 sections:\n"
        "1. Clinical Concern\n"
        "2. Recommended Immediate Action\n"
        "3. Priority Monitoring\n"
        "Keep concise and suitable for urgent physician email. "
        "Do not include scores, AI terms, technical details, or explanations."
    )

    try:
        return _generate_gemini_text(
            prompt=prompt,
            system_prompt=(
                "You are an ICU physician assistant. "
                "Generate short actionable clinical summaries to support bedside decisions."
            ),
        )
    except Exception as e:
        logger.warning("Gemini clinical note generation failed: %s", e)
        return (
            f"[Fallback] Patient is at elevated risk of deterioration, primarily associated with abnormalities in "
            f"{feature_text}. Close monitoring of these parameters is recommended."
        )
