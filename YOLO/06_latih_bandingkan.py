"""Praktik terpadu minggu 4-7, tahap 3-5: melatih, tuning, memilih, dan menguji model YOLO.

Eksperimen yang dijalankan (pengaturan lain SAMA untuk semua):
  E1 pretrained : model pretrained COCO (transfer learning)      -> baseline
  E2 scratch    : arsitektur sama, bobot acak (tanpa transfer)   -> pembanding
  E3 tuning     : E1 + SATU perubahan (default imgsz=640)        -> eksperimen tuning
Model terbaik dipilih dengan mAP50-95 pada data VAL, lalu diuji SEKALI pada data TEST.

Pemakaian:
  python 06_latih_bandingkan.py --nim 4222101001 --nama "Budi Santoso"
  python 06_latih_bandingkan.py --nim ... --nama "..." --epochs 30 --ubah epochs=60
  python 06_latih_bandingkan.py --nim ... --nama "..." --model yolov8n   (jika YOLO26 belum tersedia)

Membutuhkan: pip install ultralytics, dan dataset_v1/data.yaml dari 05_split_dataset_yolo.py.
Disarankan memakai GPU (komputer lab atau Google Colab).
"""
import argparse
import csv
import getpass
import hashlib
import platform
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--nim", required=True)
p.add_argument("--nama", required=True)
p.add_argument("--data", default="dataset_v1/data.yaml")
p.add_argument("--model", default="yolo26n", help="nama model tanpa ekstensi, misal yolo26n atau yolov8n")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--imgsz", type=int, default=416)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default=None, help="0 untuk GPU pertama, cpu untuk CPU (default: otomatis)")
p.add_argument("--ubah", default="imgsz=640", help="SATU perubahan untuk E3, format kunci=nilai")
args = p.parse_args()

# --- Satu perubahan untuk eksperimen tuning (E3) ---
kunci, nilai = args.ubah.split("=", 1)
try:
    nilai = int(nilai)
except ValueError:
    try:
        nilai = float(nilai)
    except ValueError:
        pass                                   # biarkan sebagai teks, misal optimizer=SGD

DATA = Path(args.data)
if not DATA.exists():
    raise SystemExit(f"{DATA} tidak ditemukan. Jalankan dulu 05_split_dataset_yolo.py.")
PROJECT = Path("hasil_praktik").resolve()


def sidik_jari_dataset(root: Path) -> str:
    """Hash isi semua file label: dataset sama -> kode sama (untuk mendeteksi salin-menyalin)."""
    h = hashlib.sha256()
    for f in sorted((root / "labels").rglob("*.txt")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12].upper()


def latih(nama, model_file, pretrained, **ubah):
    """Melatih satu eksperimen, lalu mengevaluasi best.pt di data val."""
    cfg = dict(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
               seed=0, deterministic=True, project=str(PROJECT), name=nama,
               exist_ok=True, pretrained=pretrained, plots=True, verbose=False)
    if args.device is not None:
        cfg["device"] = args.device
    cfg.update(ubah)
    print(f"\n=== {nama}: {model_file} | pretrained={pretrained} | {ubah or 'pengaturan dasar'} ===")
    model = YOLO(model_file)
    t0 = time.time()
    model.train(**cfg)
    menit = (time.time() - t0) / 60
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=cfg["imgsz"], batch=args.batch,
                            plots=False, verbose=False)
    return {"eksperimen": nama, "perubahan": ", ".join(f"{k}={v}" for k, v in ubah.items()) or "-",
            "pretrained": pretrained, "mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4),
            "precision": round(m.box.mp, 4), "recall": round(m.box.mr, 4),
            "inferensi_ms": round(m.speed["inference"], 2), "waktu_latih_menit": round(menit, 2),
            "imgsz": cfg["imgsz"], "bobot": str(best)}


hasil = [
    latih("E1_pretrained", f"{args.model}.pt", True),
    latih("E2_scratch", f"{args.model}.yaml", False),
    latih("E3_tuning", f"{args.model}.pt", True, **{kunci: nilai}),
]

# --- Pilih model terbaik dengan data VAL, lalu uji SEKALI di data TEST ---
terbaik = max(hasil, key=lambda r: r["mAP50-95"])
uji = YOLO(terbaik["bobot"]).val(data=str(DATA), split="test", imgsz=terbaik["imgsz"],
                                  batch=args.batch, plots=False, verbose=False)

garis = "=" * 78
baris = [garis, "  PRAKTIK TERPADU MINGGU 4-7 - RET503", garis,
         f"  NIM / Nama    : {args.nim} / {args.nama}",
         f"  Waktu         : {datetime.now():%d-%m-%Y %H:%M:%S}",
         f"  Komputer/user : {platform.node()} / {getpass.getuser()}",
         f"  Dataset       : {DATA.parent}  (sidik jari label: {sidik_jari_dataset(DATA.parent)})",
         f"  Pengaturan    : model {args.model}, epochs {args.epochs}, imgsz {args.imgsz}, batch {args.batch}",
         garis,
         f"  {'Eksperimen':15s}{'Perubahan':14s}{'mAP50':>8s}{'mAP50-95':>10s}{'P':>7s}{'R':>7s}{'ms':>7s}{'menit':>8s}"]
for r in hasil:
    baris.append(f"  {r['eksperimen']:15s}{r['perubahan']:14s}{r['mAP50']:8.3f}{r['mAP50-95']:10.3f}"
                 f"{r['precision']:7.3f}{r['recall']:7.3f}{r['inferensi_ms']:7.1f}{r['waktu_latih_menit']:8.1f}")
baris += [garis,
          f"  Model terpilih (mAP50-95 val tertinggi): {terbaik['eksperimen']}",
          f"  Hasil di data TEST (sekali saja)       : mAP50 {uji.box.map50:.3f} | mAP50-95 {uji.box.map:.3f}",
          garis]
laporan = "\n".join(baris)
print("\n" + laporan)

Path(f"hasil_praktik_{args.nim}.txt").write_text(laporan + "\n", encoding="utf-8")
with open(f"hasil_praktik_{args.nim}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hasil[0].keys()))
    w.writeheader()
    w.writerows(hasil)
print(f"Tersimpan: hasil_praktik_{args.nim}.txt dan .csv | grafik tiap eksperimen di {PROJECT}/")
