"""Membagi dataset YOLO menjadi train/val/test dan membuat data.yaml untuk Ultralytics.

Masukan  : dataset_yolo/ (classes.txt, images/, labels/) hasil anotasi
Keluaran : dataset_v1/
           ├── data.yaml
           ├── images/{train,val,test}/
           └── labels/{train,val,test}/

Pemakaian : python 05_split_dataset_yolo.py [folder_masukan] [folder_keluaran]
Jalankan SETELAH 01_cek_anotasi.py tidak menemukan masalah.
"""
import random
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_yolo")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "dataset_v1")
PROPORSI = {"train": 0.7, "val": 0.2, "test": 0.1}
random.seed(42)                      # pembagian bisa diulang dengan hasil sama

kelas = [k.strip() for k in (SRC / "classes.txt").read_text().splitlines() if k.strip()]
gambar = sorted(p for p in (SRC / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
gambar = [g for g in gambar if (SRC / "labels" / f"{g.stem}.txt").exists()]   # hanya yang berlabel
random.shuffle(gambar)

n = len(gambar)
n_train = int(n * PROPORSI["train"])
n_val = int(n * PROPORSI["val"])
bagian = {"train": gambar[:n_train], "val": gambar[n_train:n_train + n_val], "test": gambar[n_train + n_val:]}

for nama, daftar in bagian.items():
    (DST / "images" / nama).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / nama).mkdir(parents=True, exist_ok=True)
    for g in daftar:
        shutil.copy(g, DST / "images" / nama / g.name)
        shutil.copy(SRC / "labels" / f"{g.stem}.txt", DST / "labels" / nama / f"{g.stem}.txt")
    print(f"{nama:5s}: {len(daftar)} gambar")

yaml = [f"path: {DST.resolve().as_posix()}", "train: images/train", "val: images/val",
        "test: images/test", "names:"] + [f"  {i}: {k}" for i, k in enumerate(kelas)]
(DST / "data.yaml").write_text("\n".join(yaml) + "\n")
print(f"\ndata.yaml dibuat di {DST / 'data.yaml'}")
