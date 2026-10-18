"""Mengubah satu file COCO JSON menjadi label format YOLO (satu .txt per gambar).

Pemakaian : python 03_coco_ke_yolo.py <file_coco.json> <folder_keluaran>
Hasil     : <folder_keluaran>/labels/*.txt dan <folder_keluaran>/classes.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

coco = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "hasil_yolo")
(OUT / "labels").mkdir(parents=True, exist_ok=True)

# Urutkan kategori menurut id, lalu beri class_id YOLO 0, 1, 2, ...
kategori = sorted(coco["categories"], key=lambda c: c["id"])
ke_yolo = {c["id"]: i for i, c in enumerate(kategori)}
(OUT / "classes.txt").write_text("\n".join(c["name"] for c in kategori) + "\n")

per_gambar = defaultdict(list)
for a in coco["annotations"]:
    per_gambar[a["image_id"]].append(a)

for img in coco["images"]:
    w, h = img["width"], img["height"]
    baris = []
    for a in per_gambar[img["id"]]:
        x, y, bw, bh = a["bbox"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h           # titik tengah, dinormalisasi
        baris.append(f"{ke_yolo[a['category_id']]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    stem = Path(img["file_name"]).stem
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(baris) + ("\n" if baris else ""))

print(f"{len(coco['images'])} file label dibuat di {OUT / 'labels'}")
