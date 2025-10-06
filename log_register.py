# log_register.py
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import os

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "users.db"

app = Flask(__name__, static_url_path="", static_folder=".")  # login.html / register.html / index.html aynı klasörde
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# ---------------- DB ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# ------------- Pages ---------------
@app.get("/login")
def login_page():
    return send_from_directory(".", "login.html")

@app.get("/register")
def register_page():
    return send_from_directory(".", "register.html")

# ------------- API -----------------
@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    # CORS preflight veya bazı barındırma ortamlarında otomatik preflight -> 200 dön
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or request.form
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "E-posta ve şifre zorunlu"}), 400

    pwd_hash = generate_password_hash(password)
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, pwd_hash),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Bu e-posta zaten kayıtlı"}), 409
    finally:
        conn.close()

    return jsonify({"ok": True, "message": "Kayıt başarılı"})

@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "Geçersiz e-posta veya şifre"}), 401

    session["user_id"] = user["id"]
    session["email"] = user["email"]
    return jsonify({"ok": True, "message": "Giriş başarılı"})

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ------------- Protected -----------
@app.get("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    return send_from_directory(".", "index.html")

# ------------- Utils ---------------
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/whoami")
def whoami():
    if "user_id" not in session:
        return jsonify({"ok": False, "auth": False})
    return jsonify({"ok": True, "auth": True, "email": session.get("email")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
