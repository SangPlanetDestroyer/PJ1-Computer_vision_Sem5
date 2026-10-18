"""Mengubah dataset format YOLO menjadi satu file COCO JSON.

YOLO : class_id cx cy w h  (dinormalisasi 0-1, titik tengah)
COCO : bbox = [x_kiri, y_atas, lebar, tinggi]  (piksel, pojok kiri atas)

Pemakaian : python 02_yolo_ke_coco.py [folder_dataset] [file_keluaran]
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_yolo")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "annotations_coco.json")
KELAS = [k.strip() for k in (ROOT / "classes.txt").read_text().splitlines() if k.strip()]

coco = {"images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(KELAS)]}  # COCO: id mulai 1
ann_id = 1
gambar = sorted(p for p in (ROOT / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

for img_id, img_path in enumerate(gambar, 1):
    h, w = cv2.imread(str(img_path)).shape[:2]
    coco["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
    lbl = ROOT / "labels" / (img_path.stem + ".txt")
    if not lbl.exists():
        continue
    for b in lbl.read_text().splitlines():
        if not b.strip():
            continue
        cid, cx, cy, bw, bh = int(b.split()[0]), *map(float, b.split()[1:])
        lebar, tinggi = bw * w, bh * h                    # ukuran dalam piksel
        x_kiri, y_atas = cx * w - lebar / 2, cy * h - tinggi / 2
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": cid + 1,
            "bbox": [round(x_kiri, 2), round(y_atas, 2), round(lebar, 2), round(tinggi, 2)],
            "area": round(lebar * tinggi, 2), "iscrowd": 0})
        ann_id += 1

OUT.write_text(json.dumps(coco, indent=2))
print(f"{len(coco['images'])} gambar, {len(coco['annotations'])} kotak, "
      f"{len(KELAS)} kategori -> {OUT}")
