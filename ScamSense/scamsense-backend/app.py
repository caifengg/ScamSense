from flask import Flask, request
from joblib import load
import pymysql
from pathlib import Path

from database import DB_CONNECTION_ERROR, connection, cursor
from feature_extractor import extract_features
from gemini_explainer import generate_explanation

MODEL_PATH = Path(__file__).resolve().with_name("phishing_model.pkl")
model = load(MODEL_PATH)
app = Flask(__name__)


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


if __name__ == "__main__":
    app.run(debug=True)