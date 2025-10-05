from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_PATH = "users.db"

# ---- Veritabanı oluşturma ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()  # 🚀 app başlarken tabloyu garantiler

# ---- Register endpoint ----
@app.route("/register", methods=["POST"])
def register():
    data = request.form
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"ok": False, "error": "Eksik bilgi"}), 400

    hashed_pw = generate_password_hash(password)

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                  (name, email, hashed_pw))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Bu email zaten kayıtlı"}), 400

    return jsonify({"ok": True, "message": "Kayıt başarılı"})

# ---- Login endpoint ----
@app.route("/login", methods=["POST"])
def login():
    data = request.form
    email = data.get("email")
    password = data.get("password")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()

    if row and check_password_hash(row[2], password):
        session["user_id"] = row[0]
        session["user_name"] = row[1]
        return jsonify({"ok": True, "message": "Giriş başarılı"})
    else:
        return jsonify({"ok": False, "error": "Email veya şifre hatalı"}), 401

# ---- Basit çıkış ----
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ---- HTML sayfaları ----
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)
