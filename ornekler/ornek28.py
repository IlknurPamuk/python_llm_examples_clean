# Örnek 34 : SQLite
from dotenv import load_dotenv
import sqlite3

load_dotenv()
conn = sqlite3.connect("okul.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS ogrenciler(
    id INTEGER PRIMARY KEY AUTOINCREMENT,  --Her öğrenciye otomatik ID ver
    ad TEXT ,                              -- Öğrencinin adı
    soyad TEXT ,                           -- Öğrencinin soyadı
    notu INTEGER                           -- Öğrencinin notu
)
""")
def ogrenci_ekle(ad, soyad, notu):
    cursor.execute("INSERT INTO ogrenciler (ad, soyad, notu) VALUES (?, ?, ?)", (ad, soyad, notu))
    conn.commit()

def ogrencileri_listele():
    cursor.execute("SELECT * FROM ogrenciler")
    ogrenciler = cursor.fetchall()
    for ogrenci in ogrenciler:
        print(ogrenci)

def notları_guncelle(ogrenci_id, yeni_not):
    cursor.execute("UPDATE ogrenciler SET notu = ? WHERE id = ?", (yeni_not, ogrenci_id))
    conn.commit()

def notları_sil(ogrenci_id):
    cursor.execute("DELETE FROM ogrenciler WHERE id = ?", (ogrenci_id,))
    conn.commit()

def listele():
    return ogrenciler

ogrenci_ekle("Ali", "Yılmaz", 85)
ogrenci_ekle("Ayşe", "Kara", 90)
ogrenci_ekle("Mehmet", "Demir", 78)

print("Öğrenciler:", ogrencileri_listele())

notları_guncelle(1, 95)
print("Güncel Liste:", ogrencileri_listele())

ogrenci_ekle("Fatma", "Çelik", 88)
print("Güncellenmiş Öğrenciler:", ogrencileri_listele())

'''


