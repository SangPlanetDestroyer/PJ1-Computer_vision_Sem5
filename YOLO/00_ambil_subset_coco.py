"""Membuat dataset deteksi kecil dari COCO val2017 dengan kelas pilihan sendiri.

Langkah yang dilakukan (inilah praktik persiapan data minggu 4):
  1. Mengunduh val2017 (gambar) dan anotasinya dari cocodataset.org, jika belum ada.
  2. Memilih gambar yang berisi kelas pilihan, plus sebagian kecil gambar latar
     (tanpa objek pilihan) agar model belajar membedakan objek dan latar.
  3. Mengubah anotasi COCO (JSON, bbox piksel [x, y, w, h]) menjadi label YOLO
     (cx cy w h dinormalisasi), dengan nomor kelas baru 0, 1, 2, ...
  4. Menyimpan hasil dalam struktur dataset_yolo/ (classes.txt, images/, labels/).

Pemakaian : python 00_ambil_subset_coco.py --kelas person car bottle --jumlah 200
            (tanpa argumen: kelas person car bottle, 200 gambar)
Lanjutan  : python 01_cek_anotasi.py dataset_yolo
"""
import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL_GAMBAR = "http://images.cocodataset.org/zips/val2017.zip"
URL_ANOTASI = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
UNDUH = Path("coco_unduhan")

p = argparse.ArgumentParser()
p.add_argument("--kelas", nargs="+", default=["person", "car", "bottle"],
               help="nama kelas COCO (default: person car bottle)")
p.add_argument("--jumlah", type=int, default=200, help="jumlah gambar berisi objek (default 200)")
p.add_argument("--latar", type=float, default=0.1, help="proporsi gambar latar tanpa objek (default 0.1)")
p.add_argument("--out", default="dataset_yolo", help="folder keluaran (default dataset_yolo)")
args = p.parse_args()
print(f"Kelas dipilih: {args.kelas} | jumlah gambar: {args.jumlah}")
print("(Ubah dengan: python 00_ambil_subset_coco.py --kelas <nama1> <nama2> --jumlah <n>)\n")


def unduh_dan_ekstrak(url, cek):
    if cek.exists():
        return
    UNDUH.mkdir(exist_ok=True)
    zip_path = UNDUH / url.split("/")[-1]
    if not zip_path.exists():
        print(f"Mengunduh {url} ... (bisa beberapa menit)")
        urllib.request.urlretrieve(url, zip_path)
    print(f"Mengekstrak {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(UNDUH)


unduh_dan_ekstrak(URL_GAMBAR, UNDUH / "val2017")
unduh_dan_ekstrak(URL_ANOTASI, UNDUH / "annotations" / "instances_val2017.json")
coco = json.loads((UNDUH / "annotations" / "instances_val2017.json").read_text())

# Cocokkan nama kelas pilihan dengan kategori COCO
nama_ke_id = {c["name"]: c["id"] for c in coco["categories"]}
salah = [k for k in args.kelas if k not in nama_ke_id]
if salah:
    contoh = ", ".join(sorted(nama_ke_id)[:15])
    raise SystemExit(f"Kelas tidak ada di COCO: {salah}\nContoh nama yang benar: {contoh}, ...")
id_baru = {nama_ke_id[k]: i for i, k in enumerate(args.kelas)}   # id COCO -> class_id YOLO baru

# Kumpulkan anotasi kelas pilihan per gambar (tanpa iscrowd)
per_gambar = {}
for a in coco["annotations"]:
    if a["category_id"] in id_baru and not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0:
        per_gambar.setdefault(a["image_id"], []).append(a)

info = {g["id"]: g for g in coco["images"]}
random.seed(0)
berobjek = random.sample(sorted(per_gambar), min(args.jumlah, len(per_gambar)))
kandidat_latar = sorted(set(info) - set(per_gambar))
latar = random.sample(kandidat_latar, min(int(len(berobjek) * args.latar), len(kandidat_latar)))

OUT = Path(args.out)
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "classes.txt").write_text("\n".join(args.kelas) + "\n", encoding="utf-8")

hitung = {k: 0 for k in args.kelas}
for img_id in berobjek + latar:
    g = info[img_id]
    W, H = g["width"], g["height"]
    baris = []
    for a in per_gambar.get(img_id, []):
        x, y, w, h = a["bbox"]                                   # COCO: piksel, pojok kiri atas
        cid = id_baru[a["category_id"]]
        baris.append(f"{cid} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        hitung[args.kelas[cid]] += 1
    shutil.copy(UNDUH / "val2017" / g["file_name"], OUT / "images" / g["file_name"])
    stem = Path(g["file_name"]).stem
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(baris) + ("\n" if baris else ""))

print(f"Kelas baru      : {', '.join(f'{i} = {k}' for i, k in enumerate(args.kelas))}")
print(f"Gambar berobjek : {len(berobjek)}")
print(f"Gambar latar    : {len(latar)}")
print("Jumlah kotak    : " + ", ".join(f"{k} {n}" for k, n in hitung.items()))
print(f"Siap di {OUT}/ -> lanjutkan dengan: python 01_cek_anotasi.py {OUT}")
