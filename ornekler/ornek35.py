# Örnek 35 : Flask API
'''
from flask import Flask, request, jsonify
import sqlite3
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return "hoşgeldiniz flask sistemi çalışıyor"


def get_db():
    conn = sqlite3.connect("okul.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/ogrenciler", methods=["GET"])
def ogrencileri_listele_api():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ogrenciler")
    ogrenciler = [dict(row) for row in cursor.fetchall()]
    return jsonify(ogrenciler)

@app.route("/ekle", methods=["POST"])
def ekle():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ogrenciler (ad, soyad, notu) VALUES (?, ?, ?)",
                   (data["ad"], data["soyad"], data["notu"]))
    conn.commit()
    return jsonify({"message": "Öğrenci eklendi."})

@app.route("/guncelle/<int:id>", methods=["PUT"])
def guncelle(id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE ogrenciler SET notu = ? WHERE id = ?", (data["notu"])),
    conn.commit()
    return jsonify({"mesaj": "not guncellendi"})

@app.route("/sil/<int:id>", methods=["DELETE"])
def sil(id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ogrenciler WHERE id = ?", (id,))
    conn.commit()
    return jsonify({"mesaj": "öğrenci silindi"})

if __name__ == "__main__":
    app.run(debug=True)
'''
