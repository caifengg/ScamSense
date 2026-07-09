import os

import pymysql

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

DB_CONNECTION_ERROR = None
connection = None
cursor = None

db_host = os.getenv("DB_HOST", "localhost")
db_user = os.getenv("DB_USER", "scamsense")
db_name = os.getenv("DB_NAME", "scamsense")
env_password = os.getenv("DB_PASSWORD")

# Try environment password first (if provided), then common local defaults.
password_candidates = []
if env_password is not None:
    password_candidates.append(env_password)
password_candidates.extend(["scamsense123", "scamsense"])

seen = set()
unique_password_candidates = []
for candidate in password_candidates:
    if candidate not in seen:
        seen.add(candidate)
        unique_password_candidates.append(candidate)

last_error = None
for candidate in unique_password_candidates:
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=candidate,
            database=db_name,
        )
        cursor = connection.cursor()
        DB_CONNECTION_ERROR = None
        break
    except Exception as exc:
        last_error = exc

if connection is None:
    DB_CONNECTION_ERROR = str(last_error)