from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Database path
DB_PATH = Path(__file__).parent / "users.db"

# ------------------------
# Database Connection & Init
# ------------------------
def get_db():
    """Veritabanı bağlantısı oluşturur"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Veritabanı tablosunu oluşturur"""
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    except Exception as e:
        print(f"Database init error: {e}")
    finally:
        conn.close()

# Initialize database on startup
init_db()

# ------------------------
# CORS Support (if needed)
# ------------------------
@app.after_request
def after_request(response):
    """CORS headers ekler (gerekirse)"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# ------------------------
# Register Route
# ------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    """Kullanıcı kayıt işlemi"""
    if request.method == "GET":
        try:
            return send_from_directory(".", "register.html")
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "register.html bulunamadı"}), 404
    
    # POST - Kayıt işlemi
    try:
        # JSON veya form data kabul et
        data = request.get_json(silent=True) or request.form
        
        # Verileri al ve temizle
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        
        # Validasyon
        if not name:
            return jsonify({"ok": False, "error": "Ad Soyad gerekli"}), 400
        
        if len(name) < 2:
            return jsonify({"ok": False, "error": "Ad Soyad en az 2 karakter olmalı"}), 400
        
        if not email:
            return jsonify({"ok": False, "error": "E-posta gerekli"}), 400
        
        if "@" not in email or "." not in email:
            return jsonify({"ok": False, "error": "Geçerli bir e-posta adresi girin"}), 400
        
        if not password:
            return jsonify({"ok": False, "error": "Şifre gerekli"}), 400
        
        if len(password) < 8:
            return jsonify({"ok": False, "error": "Şifre en az 8 karakter olmalı"}), 400
        
        # Şifreyi hashle
        pwd_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Veritabanına kaydet
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, pwd_hash)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return jsonify({"ok": False, "error": "Bu e-posta adresi zaten kayıtlı"}), 409
        finally:
            conn.close()
        
        return jsonify({"ok": True, "message": "Kayıt başarılı"}), 201
    
    except Exception as e:
        print(f"Register error: {e}")
        return jsonify({"ok": False, "error": "Sunucu hatası"}), 500

# ------------------------
# Login Route
# ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """Kullanıcı giriş işlemi"""
    if request.method == "GET":
        try:
            return send_from_directory(".", "login.html")
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "login.html bulunamadı"}), 404
    
    # POST - Giriş işlemi
    try:
        data = request.get_json(silent=True) or request.form
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        
        # Validasyon
        if not email or not password:
            return jsonify({"ok": False, "error": "E-posta ve şifre gerekli"}), 400
        
        # Kullanıcıyı bul
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        
        # Kullanıcı kontrolü ve şifre doğrulama
        if not user:
            return jsonify({"ok": False, "error": "Geçersiz e-posta veya şifre"}), 401
        
        if not check_password_hash(user["password_hash"], password):
            return jsonify({"ok": False, "error": "Geçersiz e-posta veya şifre"}), 401
        
        # Session oluştur
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        session["name"] = user["name"]
        
        return jsonify({
            "ok": True, 
            "message": "Giriş başarılı",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
        }), 200
    
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"ok": False, "error": "Sunucu hatası"}), 500

# ------------------------
# Logout Route
# ------------------------
@app.route("/logout")
def logout():
    """Kullanıcı çıkış işlemi"""
    session.clear()
    return redirect(url_for("login"))

# ------------------------
# Protected Routes
# ------------------------
@app.route("/")
def index():
    """Ana sayfa (giriş kontrolü ile)"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    try:
        return send_from_directory(".", "index.html")
    except FileNotFoundError:
        return "<h1>Ana Sayfa</h1><p>Hoş geldiniz!</p><a href='/logout'>Çıkış Yap</a>", 200

@app.route("/dashboard")
def dashboard():
    """Dashboard sayfası (giriş kontrolü ile)"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    return jsonify({
        "ok": True,
        "user": {
            "id": session.get("user_id"),
            "name": session.get("name"),
            "email": session.get("email")
        }
    })

# ------------------------
# API: Check Auth Status
# ------------------------
@app.route("/api/auth/status")
def auth_status():
    """Kullanıcının giriş durumunu kontrol eder"""
    if "user_id" in session:
        return jsonify({
            "ok": True,
            "authenticated": True,
            "user": {
                "id": session.get("user_id"),
                "name": session.get("name"),
                "email": session.get("email")
            }
        })
    else:
        return jsonify({
            "ok": True,
            "authenticated": False
        })

# ------------------------
# Health Check (Railway için)
# ------------------------
@app.route("/health")
def health():
    """Sunucu sağlık kontrolü"""
    return jsonify({"status": "healthy", "ok": True}), 200

# ------------------------
# Error Handlers
# ------------------------
@app.errorhandler(404)
def not_found(e):
    """404 hatası için özel yanıt"""
    if request.path.startswith('/api/'):
        return jsonify({"ok": False, "error": "Endpoint bulunamadı"}), 404
    return "<h1>404</h1><p>Sayfa bulunamadı</p><a href='/'>Ana Sayfa</a>", 404

@app.errorhandler(500)
def server_error(e):
    """500 hatası için özel yanıt"""
    return jsonify({"ok": False, "error": "Sunucu hatası"}), 500

# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
