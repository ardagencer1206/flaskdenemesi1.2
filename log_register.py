# -*- coding: utf-8 -*-
import os
import sqlite3
from pathlib import Path
from datetime import timedelta

from flask import (
    Flask, request, jsonify, session, redirect, url_for,
    send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------
# Ayarlar
# -------------------------------------------------
APP_DIR = Path(__file__).parent
DB_PATH = Path(os.environ.get("USERS_DB", APP_DIR / "users.db"))
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

app = Flask(__name__, static_url_path="", static_folder=".")
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

# -------------------------------------------------
# DB Yardımcıları
# -------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            """
        )

# Uygulama başlarken şemayı garanti et
ensure_schema()

# -------------------------------------------------
# Yardımcılar
# -------------------------------------------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (uid,),
        ).fetchone()
    return dict(row) if row else None

def login_required_response():
    return jsonify({"ok": False, "error": "Login required"}), 401

# -------------------------------------------------
# Sayfalar (statik HTML dosyalarını döndürür)
# -------------------------------------------------
@app.route("/login", methods=["GET"])
def login_page():
    # login.html aynı klasörde olmalı (frontend dosyası)
    return send_from_directory(".", "login.html")

@app.route("/register", methods=["GET"])
def register_page():
    # register.html aynı klasörde olmalı (frontend dosyası)
    return send_from_directory(".", "register.html")

# -------------------------------------------------
# API: Register / Login / Logout / Me / Guard
# -------------------------------------------------
@app.route("/register", methods=["POST"])
def register_post():
    """JSON veya form POST: {name?, email, password}"""
    data = request.get_json(silent=True) or request.form
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "email ve password zorunlu"}), 400

    pwd_hash = generate_password_hash(password)

    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, pwd_hash),
            )
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Bu email zaten kayıtlı"}), 409

    # Oturum aç
    session.permanent = True
    session["user_id"] = user_id
    return jsonify({"ok": True})

@app.route("/login", methods=["POST"])
def login_post():
    """JSON veya form POST: {email, password}"""
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "email ve password zorunlu"}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"ok": False, "error": "Geçersiz kimlik bilgileri"}), 401

    session.permanent = True
    session["user_id"] = int(row["id"])
    return jsonify({"ok": True})

@app.route("/logout", methods=["POST"])
def logout_post():
    session.clear()
    return jsonify({"ok": True})

@app.route("/me", methods=["GET"])
def me():
    user = current_user()
    if not user:
        return jsonify({"ok": False, "authenticated": False})
    return jsonify({"ok": True, "authenticated": True, "user": user})

@app.route("/guard", methods=["GET"])
def guard():
    """Ön yüzde index.html’e gitmeden önce kontrol amaçlı kullanılabilir."""
    if not current_user():
        return jsonify({"ok": False, "authenticated": False}), 401
    return jsonify({"ok": True, "authenticated": True})

# -------------------------------------------------
# (İsteğe bağlı) Korumalı ana sayfa yönlendirmesi
# -------------------------------------------------
@app.route("/")
def root():
    # Kullanıcı giriş yapmadıysa login’e yönlendir
    if not current_user():
        return redirect(url_for("login_page"))
    # Girişliyse index.html’i döndür
    return send_from_directory(".", "index.html")

# -------------------------------------------------
# Hata Yakalama
# -------------------------------------------------
@app.errorhandler(405)
def handle_405(e):
    return jsonify({"ok": False, "error": "Method Not Allowed"}), 405

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"ok": False, "error": "Internal Server Error"}), 500

# -------------------------------------------------
# Geliştirme Sunucusu
# -------------------------------------------------
if __name__ == "__main__":
    # Railway gibi ortamlarda 0.0.0.0 kullan
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
