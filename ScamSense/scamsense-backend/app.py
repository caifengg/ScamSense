import base64
import io

import numpy as np
from flask import Flask, request
from joblib import load
import pymysql
from pathlib import Path
from PIL import Image as PILImage

from database import DB_CONNECTION_ERROR, connection, cursor
from feature_extractor import extract_features
from gemini_explainer import generate_explanation, generate_deepfake_explanation

MODEL_PATH = Path(__file__).resolve().with_name("phishing_model.pkl")
model = load(MODEL_PATH)
app = Flask(__name__)

# ── Deepfake detector (lazy-loaded on first request) ────────────────────────────
_deepfake_detector = None
_deepfake_sessions: dict = {}
_deepfake_reasons: dict = {}  # {session_id: {"verdict": str, "reason": str}}


def _get_deepfake_detector():
    global _deepfake_detector
    if _deepfake_detector is None:
        from jing_model import Detector  # heavy import — only once
        _deepfake_detector = Detector()
    return _deepfake_detector


def ensure_db_ready():
    if connection is None or cursor is None:
        return {
            "error": f"Database unavailable. Check DB credentials/config. Details: {DB_CONNECTION_ERROR}"
        }, 503
    return None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def home():
    return {"message": "Backend Working"}


@app.route("/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}

    required_fields = ["username", "email", "password", "role"]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}, 400

    db_error = ensure_db_ready()
    if db_error:
        return db_error

    username = data["username"]
    email = data["email"]
    password = data["password"]
    role = data["role"]

    try:
        cursor.execute(
            """
            INSERT INTO users
            (username,email,password,role)

            VALUES
            (%s,%s,%s,%s)
            """,
            (username, email, password, role),
        )
        connection.commit()
        return {"message": "success"}
    except pymysql.err.IntegrityError:
        connection.rollback()
        return {"error": "Email already exists."}, 409
    except Exception as exc:
        connection.rollback()
        return {"error": f"Signup failed: {str(exc)}"}, 500


@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return "", 204

    db_error = ensure_db_ready()
    if db_error:
        return db_error

    data = request.json

    email = data["email"]
    password = data["password"]

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE email=%s
        AND password=%s
        """,
        (email, password),
    )

    user = cursor.fetchone()

    if user:
        return {
            "success": True,
            "role": user[4],
        }

    return {
        "success": False,
    }


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    db_error = ensure_db_ready()
    if db_error:
        return db_error

    data = request.json
    url = data["url"]

    features = extract_features(url)
    prediction = model.predict([features])

    try:
        confidence = float(max(model.predict_proba([features])[0]))
    except AttributeError:
        confidence = None

    if prediction[0] == 1:
        result = "Legitimate Website"
    else:
        result = "Phishing Website"

    cursor.execute(
        """
        INSERT INTO detections
        (url,result)

        VALUES
        (%s,%s)
        """,
        (url, result),
    )
    connection.commit()

    explanation_result = generate_explanation(url, result, confidence)

    return {
        "result": result,
        "confidence": confidence,
        "explanation": explanation_result["explanation"],
        "explanation_available": explanation_result["available"],
    }


@app.route("/admin/stats", methods=["GET", "OPTIONS"])
def admin_stats():
    if request.method == "OPTIONS":
        return "", 204

    db_error = ensure_db_ready()
    if db_error:
        return db_error

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM detections
        WHERE result='Phishing Website'
        """
    )
    total_scams = cursor.fetchone()[0]

    return {
        "total_users": total_users,
        "total_scams": total_scams,
    }


@app.route("/deepfake/score", methods=["POST", "OPTIONS"])
def deepfake_score():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    session_id = data.get("session_id", "default")
    reset = data.get("reset", False)

    if not frame_b64:
        return {"error": "No frame provided"}, 400

    try:
        img_bytes = base64.b64decode(frame_b64)
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)
    except Exception as exc:
        return {"error": f"Invalid frame data: {exc}"}, 400

    try:
        detector = _get_deepfake_detector()
    except Exception as exc:
        return {"error": f"Model unavailable: {exc}"}, 503

    try:
        frame_result = detector.score_frame(frame_rgb)
    except Exception as exc:
        return {"error": f"Frame scoring failed: {exc}"}, 500

    from jing_model import RollingVerdict
    if reset or session_id not in _deepfake_sessions:
        _deepfake_sessions[session_id] = RollingVerdict(window=25)

    rolling = _deepfake_sessions[session_id].update(
        frame_result["prob"], frame_result["face_found"]
    )

    # ── Debug logging ────────────────────────────────────────────────────────────
    print(
        f"[score] frame_prob={frame_result['prob']:.4f} frame_verdict={frame_result['verdict']}"
        f" | rolling_prob={rolling['prob']:.4f} rolling_verdict={rolling['verdict']}"
        f" | frames_in_window={rolling['frames_in_window']}"
    )
    # ── End debug ────────────────────────────────────────────────────────────────

    # ── LLM explanation (generated only when the verdict changes) ────────────────
    current_verdict = rolling["verdict"]
    cached = _deepfake_reasons.get(session_id)
    if cached is None or cached["verdict"] != current_verdict:
        reason = generate_deepfake_explanation(
            verdict=current_verdict,
            prob=rolling["prob"],
            face_found=frame_result["face_found"],
            frames_in_window=rolling["frames_in_window"],
        )
        _deepfake_reasons[session_id] = {"verdict": current_verdict, "reason": reason}
    else:
        reason = cached["reason"]
    # ── End LLM explanation ───────────────────────────────────────────────────────

    return {
        "frame": frame_result,
        "rolling": rolling,
        "reason": reason,
    }


@app.route("/deepfake/reset", methods=["POST", "OPTIONS"])
def deepfake_reset():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    _deepfake_sessions.pop(session_id, None)
    _deepfake_reasons.pop(session_id, None)
    return {"message": "session reset"}


if __name__ == "__main__":
    app.run(debug=True)