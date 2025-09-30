# rename_examples.py
import os, re, subprocess, unicodedata, pathlib, sys

EXAMPLES_DIR = "ornekler"
DRY_RUN = "--apply" not in sys.argv

def slugify(s: str) -> str:
    tr_map = str.maketrans({"ç":"c","ğ":"g","ı":"i","İ":"i","ş":"s","ö":"o","ü":"u",
                            "Ç":"c","Ğ":"g","Ş":"s","Ö":"o","Ü":"u"})
    s = s.translate(tr_map)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9\s\-_\.]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-").lower()
    return s or "baslik"

def first_heading(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t.startswith("#"):
                t = re.sub(r"^\s*#\s*", "", t)
                t = re.sub(r"^örnek\s*\d+\s*[:\-–]\s*", "", t, flags=re.I)
                return t.strip() or "icerik"
    return "icerik"

def idx_from_name(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None

root = pathlib.Path(".")
ex_dir = root / EXAMPLES_DIR
files = sorted([p for p in ex_dir.glob("*.py") if p.is_file()])

planned, used = [], set()
for p in files:
    title = first_heading(p)
    idx = idx_from_name(p.name) or len(planned)
    base = slugify(title)
    new_name = f"{idx:02d}-{base}.py"
    k, cand = 2, new_name
    while cand in used or (ex_dir / cand).exists():
        cand = f"{idx:02d}-{base}-v{k}.py"; k += 1
    used.add(cand)
    planned.append((p.name, cand))

print("Planlanan değişiklikler:\n")
for old, new in planned:
    print(f"  {old} -> {new}")

if DRY_RUN:
    print("\n(Önizleme) Uygulamak için:  python rename_examples.py --apply\n")
    sys.exit(0)

def git_mv(src, dst):
    try:
        subprocess.check_call(["git", "mv", src, dst])
        return True
    except Exception:
        return False

for old, new in planned:
    src = str(ex_dir / old)
    dst = str(ex_dir / new)
    if not git_mv(src, dst):
        os.rename(src, dst)

# README indeksini güncelle
lines = ["# LLM Alıştırmaları (Otomatik İndeks)\n\n", "## Örnekler\n"]
for _, new in sorted(planned, key=lambda x: x[1]):
    title = new.split("-", 1)[1].rsplit(".",1)[0].replace("-", " ").title()
    lines.append(f"- [{new}](ornekler/{new}) — {title}\n")

with open("README.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\nYeniden adlandırma tamamlandı ve README.md güncellendi.")

