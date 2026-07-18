import os
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # fast + cheap, plenty for a short explanation
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
REQUEST_TIMEOUT = 10  # seconds


def summarize_url_signals(url: str) -> dict:
    """A handful of human-readable, easily-verified signals about the URL -
    used only to give Gemini something concrete to reason about."""
    domain = urlparse(url).netloc or url

    return {
        "url_length": len(url),
        "domain": domain,
        "domain_length": len(domain),
        "uses_https": url.startswith("https"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "num_dots_in_domain": domain.count("."),
    }


def build_prompt(url: str, result: str, confidence, signals: dict) -> str:
    confidence_line = f"The model's confidence in this result is {confidence:.0%}." if confidence is not None else ""

    return f"""You are a safety assistant inside a phishing-detection app called ScamSense, aimed at everyday Singaporean users.

A machine learning model just classified this URL as: {result}
URL: {url}
{confidence_line}

Observed signals about the URL:
- Domain: {signals['domain']}
- Uses HTTPS: {signals['uses_https']}
- URL length: {signals['url_length']} characters
- Number of digits in URL: {signals['num_digits']}
- Number of special characters in URL: {signals['num_special_chars']}
- Number of dots in domain: {signals['num_dots_in_domain']}

Write a short explanation (2-3 sentences) in plain, non-technical language for
a general user. If the result is phishing, explain what about the URL looks
suspicious and give one concrete safety tip. If it's legitimate, briefly
reassure the user but remind them to stay cautious with personal details
regardless. Do not use technical jargon like "feature vector" or "classifier".
Do not repeat the raw signal numbers back verbatim - describe them naturally."""


def generate_explanation(url: str, result: str, confidence=None) -> dict:
    """
    Returns a dict: {"explanation": str, "available": bool}

    "available" is False whenever the explanation couldn't be generated for
    any reason (missing key, network error, safety block, etc) - the caller
    should still show the classifier's result either way. A broken Gen AI
    call should never take down the core phishing check.
    """
    if not GEMINI_API_KEY:
        return {
            "explanation": "AI explanation is not configured (missing GEMINI_API_KEY).",
            "available": False,
        }

    signals = summarize_url_signals(url)
    prompt = build_prompt(url, result, confidence, signals)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return {"explanation": "Could not reach the AI explanation service. Please try again.", "available": False}

    if response.status_code != 200:
        return {
            "explanation": f"AI explanation service returned an error (status {response.status_code}).",
            "available": False,
        }

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return {"explanation": "The AI explanation was blocked or empty for this request.", "available": False}

    try:
        text = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return {"explanation": "Could not parse the AI explanation response.", "available": False}

    return {"explanation": text, "available": True}


# ── Deepfake explanation ──────────────────────────────────────────────────────

def _build_deepfake_prompt(verdict: str, prob: float, face_found: bool, frames_in_window: int) -> str:
    confidence_pct = f"{prob * 100:.1f}%"
    face_line = (
        "A face was clearly detected and analysed in the video."
        if face_found else
        "No face was clearly detected; the analysis used the best available frame region."
    )
    return f"""You are a safety assistant inside a deepfake-detection app called ScamSense, aimed at everyday Singaporean users.

A machine learning model just analysed a live video call and produced this result:
- Verdict: {verdict}
- Deepfake likelihood: {confidence_pct}
- {face_line}
- Number of recent frames analysed: {frames_in_window}

Write a short explanation (2–3 sentences) in plain, non-technical language for a general user.
If the verdict is DEEPFAKE, explain what it means and give one concrete safety tip (for example: hang up and re-verify the caller through a different channel such as a phone call or in person).
If the verdict is REAL, briefly reassure the user but remind them to stay alert.
Do not use technical jargon such as "model", "tensor", "classifier", or "probability".
Do not repeat the raw numbers verbatim — describe them naturally."""


def generate_deepfake_explanation(
    verdict: str, prob: float, face_found: bool, frames_in_window: int
) -> str:
    """Return a human-readable explanation string for the current deepfake
    verdict.  Returns an empty string silently if the API is unavailable so
    callers never need to handle exceptions."""
    if not GEMINI_API_KEY:
        return ""

    prompt = _build_deepfake_prompt(verdict, prob, face_found, frames_in_window)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"[deepfake explanation] network error: {e}")
        return ""

    if response.status_code != 200:
        print(f"[deepfake explanation] API error {response.status_code}: {response.text[:300]}")
        return ""

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text
    except (KeyError, IndexError, ValueError):
        return ""
