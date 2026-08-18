CREATE DATABASE IF NOT EXISTS scamsense;
USE scamsense;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url TEXT NOT NULL,
    result VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Text Extractor's per-user history. Each row is one /detect-text check.
-- `flagged` starts FALSE and can only ever be set to TRUE once per row - the
-- backend enforces "one flag per check" server-side (see /text-checks/<id>/flag
-- in app.py) rather than trusting the frontend alone, so a user can't spam
-- flags on the same check by replaying the request.
CREATE TABLE IF NOT EXISTS text_checks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    result VARCHAR(100) NOT NULL,
    confidence FLOAT,
    flagged BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deepfake_checks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(120) NOT NULL UNIQUE,
    result VARCHAR(20) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
