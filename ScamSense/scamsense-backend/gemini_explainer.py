import os
import re
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash-lite-latest"  # lightest/cheapest tier, plenty for a short explanation
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

def _build_deepfake_prompt(
    verdict: str, prob: float, face_found: bool, frames_in_window: int,
    frame_prob: float, max_prob: float, min_prob: float,
    deepfake_frames: int, real_frames: int,
) -> str:
    avg_pct       = f"{prob * 100:.1f}%"
    frame_pct     = f"{frame_prob * 100:.1f}%"
    max_pct       = f"{max_prob * 100:.1f}%"
    min_pct       = f"{min_prob * 100:.1f}%"
    face_line = (
        "A face was clearly detected and analysed in the video."
        if face_found else
        "No face was clearly detected; the analysis used the best available frame region."
    )
    return f"""You are a safety assistant inside a deepfake-detection app called ScamSense, aimed at everyday Singaporean users.

A deepfake detector analysed a live video call. Below are the real statistics from the detector — use only these facts to explain the result. Do NOT invent observations such as "unnatural eyes" or "strange mouth movements".

Detector statistics:
- Overall verdict: {verdict}
- Frames analysed in current window: {frames_in_window}
- Frames classified as DEEPFAKE: {deepfake_frames}
- Frames classified as REAL: {real_frames}
- Weighted average deepfake likelihood across the window: {avg_pct}
- Highest single-frame deepfake likelihood: {max_pct}
- Lowest single-frame deepfake likelihood: {min_pct}
- Most recent single frame deepfake likelihood: {frame_pct}
- {face_line}

Write 2–3 sentences in plain, non-technical language for a general user that explain the verdict using only the statistics above.
If the verdict is DEEPFAKE, give one concrete safety tip (e.g. hang up and verify the caller through a different channel).
If the verdict is REAL, briefly reassure the user but remind them to stay alert.
Do not use technical jargon. Do not invent visual observations not supported by the statistics."""


def generate_deepfake_explanation(
    verdict: str, prob: float, face_found: bool, frames_in_window: int,
    frame_prob: float = 0.0, max_prob: float = 0.0, min_prob: float = 0.0,
    deepfake_frames: int = 0, real_frames: int = 0,
) -> str:
    """Return a human-readable explanation string for the current deepfake
    verdict.  Returns an empty string silently if the API is unavailable so
    callers never need to handle exceptions."""
    if not GEMINI_API_KEY:
        return ""

    prompt = _build_deepfake_prompt(
        verdict=verdict, prob=prob, face_found=face_found, frames_in_window=frames_in_window,
        frame_prob=frame_prob, max_prob=max_prob, min_prob=min_prob,
        deepfake_frames=deepfake_frames, real_frames=real_frames,
    )
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


# ── Text/SMS scam explanation ───────────────────────────────────────────────
#
# This section provides the explanation on the Text Extractor page.
# It runs after the text_model.pkl file has already classified the message.

def summarize_message_signals(message: str) -> dict:
    """
    A handful of human-readable, verifiable signals about the message which is
    used only to give Gemini something concrete to reason about.

    This computes simple, independently checkable facts such as "does it contain a URL", "how many exclamation
    marks", so that the prompt below can base Gemini's explanation in
    something concrete.
    """
    return {
        "length": len(message),
        "word_count": len(message.split()),
        # Similar to the URLTOKEN step in text_feature_extractor.py, but
        # kept separate as this module has no dependency on nltk.
        "has_url": bool(re.search(r"http[s]?://|www\.", message, re.IGNORECASE)),
        "has_phone": bool(re.search(r"\d{7,}", message)),
        "has_money": bool(re.search(r"[$£€]\s*\d", message)),
        "num_exclamations": message.count("!"),
        "num_uppercase_words": sum(1 for w in message.split() if w.isupper() and len(w) > 1),
    }


def build_text_prompt(message: str, result: str, confidence, signals: dict) -> str:
    """
    Makes the prompt sent to Gemini: the original message, the
    ML model's verdict + confidence, and the cheap signals above, followed
    by explicit writing instructions.
    """
    confidence_line = f"The model's confidence in this result is {confidence:.0%}." if confidence is not None else ""

    return f"""You are a safety assistant inside a scam-message detector app called ScamSense, aimed at everyday Singaporean users.

A machine learning model just classified this SMS/text message as: {result}
Message: "{message}"
{confidence_line}

Observed signals about the message:
- Length: {signals['length']} characters, {signals['word_count']} words
- Contains a URL: {signals['has_url']}
- Contains a long number/phone-like sequence: {signals['has_phone']}
- Contains a currency amount: {signals['has_money']}
- Number of exclamation marks: {signals['num_exclamations']}
- Number of ALL-CAPS words: {signals['num_uppercase_words']}

Write a short explanation (2-3 sentences) in plain, non-technical language for
a general user. If the result is a scam, explain what about the message looks
suspicious (e.g. urgency, prize claims, requests for money or personal details)
and give one concrete safety tip. If it's legitimate, briefly reassure the user
but remind them to stay cautious with personal details regardless. Do not use
technical jargon like "feature vector" or "classifier". Do not repeat the raw
signal numbers back verbatim - describe them naturally."""


def generate_text_explanation(message: str, result: str, confidence=None) -> dict:
    """
    Uses Gemini to explain why it is a scam or legitimate and returns {"explanation": str, "available": bool}.
    """
    if not GEMINI_API_KEY:
        return {
            "explanation": "AI explanation is not configured (missing GEMINI_API_KEY).",
            "available": False,
        }

    signals = summarize_message_signals(message)
    prompt = build_text_prompt(message, result, confidence, signals)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        # Covers connection errors, DNS failures, and the REQUEST_TIMEOUT
        # (10s) being exceeded - any of these mean Gemini couldn't be reached
        return {"explanation": "Could not reach the AI explanation service. Please try again.", "available": False}

    if response.status_code != 200:
        # e.g. 401 (bad API key), 429 (rate limited), 500 (Gemini outage).
        return {
            "explanation": f"AI explanation service returned an error (status {response.status_code}).",
            "available": False,
        }

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        # A 200 response with zero candidates usually means Gemini's safety
        # filters blocked the prompt/response.
        return {"explanation": "The AI explanation was blocked or empty for this request.", "available": False}

    try:
        text = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        # This is in case Gemini changing its response shape
        # without this code being updated to match.
        return {"explanation": "Could not parse the AI explanation response.", "available": False}

    return {"explanation": text, "available": True}


# ── Message translation (for non-English input) ─────────────────────────────
#
# Lets a user paste an SMS in another language such as Chinese or Malay and
# still get an accurate scam/legit verdict. The TF-IDF vectorizer/model
# only understands English, so any non-English message has to be translated
# to English.

def _build_translation_prompt(message: str) -> str:
    """
    Asks Gemini to do language detection AND translation in a single call,
    in a strict, easy-to-parse format so
    translate_to_english() can reliably tell apart "this was already
    English" from "here is the translation" using simple string checks.
    """
    return f"""You are a translation assistant inside a scam-message detector app called ScamSense.

Look at the following SMS/text message:
"{message}"

If the message is already in English, respond with exactly this one line and nothing else:
ALREADY_ENGLISH

Otherwise, respond in exactly this two-line format and nothing else:
LANGUAGE: <name of the detected language, in English>
TRANSLATION: <a natural, faithful English translation of the full message>

Do not add any other commentary, explanation, greeting, or formatting."""


def translate_to_english(message: str) -> dict:
    """
    Returns a dict: {
        "text": str,                    # English text to classify/explain (original if already English)
        "was_translated": bool,         # True only if the message was in another language and got translated
        "detected_language": str|None,
        "available": bool,              # False only if translation was needed but could not be produced
    }

    Always asks Gemini to detect the language, rather than trying to
    shortcut this with a cheap heuristic like `message.isascii()`. 

    If translation is needed but unavailable (missing key, network error, bad response, unparseable response), the
    original message is returned unchanged with was_translated=False so
    classification can still proceed - it just won't be
    accurate for non-English input in that specific failure case. The
    frontend only shows the "Translated to English" box when
    was_translated is True, so a failed translation simply results in no
    translation box being shown.
    """
    if not GEMINI_API_KEY:
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    prompt = _build_translation_prompt(message)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    if response.status_code != 200:
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    try:
        raw = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    if raw.upper().startswith("ALREADY_ENGLISH"):
        return {"text": message, "was_translated": False, "detected_language": "English", "available": True}

    language = None
    translation = None
    for line in raw.splitlines():
        if line.upper().startswith("LANGUAGE:"):
            language = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TRANSLATION:"):
            translation = line.split(":", 1)[1].strip()

    if not translation:
        # Gemini didn't follow the requested format closely enough to
        # extract a translation so it falls back on the fail safe.
        return {"text": message, "was_translated": False, "detected_language": None, "available": False}

    return {"text": translation, "was_translated": True, "detected_language": language, "available": True}
