# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse

from flask import Flask, request, jsonify, session, redirect, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

# SQLAlchemy ayarı (SQLite varsayılan, Railway'de DATABASE_URL ile Postgres)
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base

APP_SECRET = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///auth.sqlite3")

# Postgres URL fix (sqlalchemy için 'postgres' -> 'postgresql')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True)
    email      = Column(String(255), unique=True, index=True, nullable=False)
    password   = Column(String(255), nullable=False)  # hashed
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = APP_SECRET
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# ---- Basit login koruması (index.html ve / için) ----
PUBLIC_PATHS = {
    "/login.html", "/register.html",
    "/auth/login", "/auth/register", "/auth/logout", "/auth/me",
    "/favicon.ico", "/robots.txt"
}

def is_static_file(path: str) -> bool:
    # index.html dışındaki varlıklar (css/js/png/svg) için engel çıkarma
    return any(path.endswith(ext) for ext in (".css",".js",".png",".jpg",".jpeg",".svg",".ico",".map",".txt",".json",".woff",".woff2",".ttf"))

@app.before_request
def require_login():
    p = request.path
    if p == "/":  # ana sayfa
        # index.html'i koruma altına al
        if not session.get("uid"):
            return redirect("/login.html")
        return
    if p in PUBLIC_PATHS or is_static_file(p):
        return
    # index.html’i doğrudan isteyenler
    if p.lower() == "/index.html":
        if not session.get("uid"):
            return redirect("/login.html")
        return

# ---- Statik sayfaları servis et (login/register) ----
@app.route("/login.html")
def serve_login():
    return send_from_directory(".", "login.html")

@app.route("/register.html")
def serve_register():
    return send_from_directory(".", "register.html")

# ---- Basit sağlık ----
@app.route("/auth/health")
def health():
    return jsonify({"ok": True})

# ---- Kayıt ----
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"ok": False, "error": "E-posta ve şifre zorunlu."}), 400
    if len(password) < 8:
        return jsonify({"ok": False, "error": "Şifre en az 8 karakter olmalı."}), 400
    with SessionLocal() as db:
        exists = db.query(User).filter(User.email == email).first()
        if exists:
            return jsonify({"ok": False, "error": "Bu e-posta ile kayıt zaten var."}), 409
        user = User(email=email, password=generate_password_hash(password))
        db.add(user); db.commit()
    return jsonify({"ok": True})

# ---- Giriş ----
@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"ok": False, "error": "E-posta ve şifre zorunlu."}), 400
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({"ok": False, "error": "Geçersiz e-posta/şifre."}), 401
        session["uid"] = user.id
        session["email"] = user.email
    return jsonify({"ok": True})

# ---- Çıkış ----
@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

# ---- Aktif kullanıcı ----
@app.route("/auth/me", methods=["GET"])
def me():
    if not session.get("uid"):
        return jsonify({"ok": True, "authenticated": False})
    return jsonify({"ok": True, "authenticated": True, "email": session.get("email")})

# ---- Ana sayfayı (korumalı) servis etme örneği ----
@app.route("/")
def home():
    # before_request zaten kontrol etti; sadece dosyayı döndürüyoruz
    return send_from_directory(".", "index.html")

# ---- Opsiyonel: /index.html doğrudan çağrısı (korumalı) ----
@app.route("/index.html")
def index_file():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    # Railway default port
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
