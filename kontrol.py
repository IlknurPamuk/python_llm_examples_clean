import os

klasor = "ornekler"
dosyalar = sorted(os.listdir(klasor))

for i, dosya in enumerate(dosyalar, start=0):
    eski_yol = os.path.join(klasor, dosya)
    yeni_ad = f"ornek{str(i).zfill(2)}.py"   # örn: ornek00.py, ornek01.py
    yeni_yol = os.path.join(klasor, yeni_ad)
    os.rename(eski_yol, yeni_yol)
    print(f"{dosya} → {yeni_ad}")
