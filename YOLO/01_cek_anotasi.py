"""Memeriksa kualitas anotasi format YOLO dan menggambar kotaknya untuk dicek visual.

Struktur folder yang diharapkan:
    dataset_yolo/
    ├── classes.txt          # satu nama kelas per baris, urutan = class_id
    ├── images/  *.jpg / *.png
    └── labels/  *.txt       # nama sama dengan gambar

Pemakaian : python 01_cek_anotasi.py [folder_dataset]
Hasil     : laporan di layar + folder cek_visual/ berisi gambar dengan kotak
"""
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_yolo")
if not (ROOT / "classes.txt").exists():
    sys.exit(f"Folder '{ROOT}' atau file '{ROOT / 'classes.txt'}' belum ada.\n"
             "Siapkan dataset dulu: ekspor dari CVAT, atau jalankan 00_siapkan_dari_roboflow.py /\n"
             "00_siapkan_dari_voc.py untuk dataset publik (misalnya dataset jalan berlubang).")
KELAS = [k.strip() for k in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if k.strip()]
MIN_SISI_PX = 8          # kotak lebih kecil dari ini patut dicurigai
WARNA = [(0, 200, 0), (0, 140, 255), (255, 80, 0), (200, 0, 200), (0, 200, 200)]

gambar = sorted(p for p in (ROOT / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
out_dir = ROOT / "cek_visual"
out_dir.mkdir(exist_ok=True)
hitung, masalah = Counter(), []
tanpa_label, kosong = 0, 0

for img_path in gambar:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = ROOT / "labels" / (img_path.stem + ".txt")
    if not lbl.exists():
        tanpa_label += 1
        masalah.append(f"{img_path.name}: tidak ada file label")
        continue
    baris = [b for b in lbl.read_text().splitlines() if b.strip()]
    if not baris:
        kosong += 1                      # boleh: gambar latar tanpa objek
    for i, b in enumerate(baris, 1):
        v = b.split()
        if len(v) != 5:
            masalah.append(f"{lbl.name} baris {i}: harus 5 angka, ada {len(v)}")
            continue
        cid, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
        if not 0 <= cid < len(KELAS):
            masalah.append(f"{lbl.name} baris {i}: class_id {cid} tidak ada di classes.txt")
            continue
        if not all(0 <= t <= 1 for t in (cx, cy, bw, bh)):
            masalah.append(f"{lbl.name} baris {i}: nilai di luar 0-1 (belum dinormalisasi?)")
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        if bw * w < MIN_SISI_PX or bh * h < MIN_SISI_PX:
            masalah.append(f"{lbl.name} baris {i}: kotak sangat kecil ({bw * w:.0f}x{bh * h:.0f} px)")
        hitung[KELAS[cid]] += 1
        warna = WARNA[cid % len(WARNA)]
        cv2.rectangle(img, (x1, y1), (x2, y2), warna, 2)
        cv2.putText(img, KELAS[cid], (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)
    cv2.imwrite(str(out_dir / img_path.name), img)

print(f"Gambar          : {len(gambar)}")
print(f"Tanpa file label: {tanpa_label}")
print(f"Label kosong    : {kosong} (gambar latar tanpa objek)")
print("Jumlah kotak per kelas:")
for k in KELAS:
    print(f"  {k:10s}: {hitung[k]}")
print(f"\n{len(masalah)} masalah ditemukan" + (":" if masalah else "."))
for m in masalah[:20]:
    print("  [!]", m)
print(f"\nGambar dengan kotak tersimpan di {out_dir}/ untuk dicek visual.")
