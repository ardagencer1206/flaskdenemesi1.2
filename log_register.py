from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

DB_PATH = Path(__file__).parent / "users.db"


# ------------------------
# DB bağlantısı ve tablo
# ------------------------
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


# ------------------------
# Register
# ------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        # register.html sayfasını döndür
        return send_from_directory(".", "register.html")

    # POST -> kullanıcı kaydı
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


# ------------------------
# Login
# ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return send_from_directory(".", "login.html")

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


# ------------------------
# Logout
# ------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------
# Ana sayfa koruması
# ------------------------
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
