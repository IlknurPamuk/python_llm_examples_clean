import os
import re

def dosya_ayir(girdi_dosyasi, cikti_klasoru="ornekler"):
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)

    with open(girdi_dosyasi, "r", encoding="utf-8") as f:
        satirlar = f.readlines()

    ornek_no = 0
    ornek_satirlari = []

    for satir in satirlar:
        # "örnek" kelimesini büyük/küçük harf duyarsız ara
        match = re.match(r"#\s*örnek\s*:?(\d+)", satir.strip(), re.IGNORECASE)
        if match:
            if ornek_satirlari:
                dosya_adi = os.path.join(cikti_klasoru, f"ornek{ornek_no}.py")
                with open(dosya_adi, "w", encoding="utf-8") as ciktidosya:
                    ciktidosya.writelines(ornek_satirlari)
                print(f"{dosya_adi} oluşturuldu.")
                ornek_satirlari = []

            ornek_no = int(match.group(1))  # numarayı al

        ornek_satirlari.append(satir)

    if ornek_satirlari:
        dosya_adi = os.path.join(cikti_klasoru, f"ornek{ornek_no}.py")
        with open(dosya_adi, "w", encoding="utf-8") as ciktidosya:
            ciktidosya.writelines(ornek_satirlari)
        print(f"{dosya_adi} oluşturuldu.")


if __name__ == "__main__":
    girdi_dosyasi = "main.py"
    dosya_ayir(girdi_dosyasi)
