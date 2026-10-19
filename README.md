# PJ1 - Deteksi Tanaman dari UAV

## Ringkasan

Notebook ini menyiapkan dataset Plant Detection and Counting dari format Supervisely ke format YOLO, memeriksa anotasi, membagi data berdasarkan sesi penerbangan, lalu menjalankan eksperimen E1, E2, dan E3. Notebook dirancang untuk Kaggle Notebook dengan accelerator GPU.

## Kesesuaian proyek

Dataset berisi citra tanaman dari UAV dan tiga kelas: `maize`, `sugarbeet`, serta `sunflower`. Karena itu dataset lebih dekat dengan tujuan drone agrikultur dibanding dataset pisang sebelumnya. Dataset ini dapat dipakai untuk latihan deteksi dan penghitungan tanaman dari udara.

Dataset belum memiliki label kondisi tanaman atau keputusan `semprot`/`tidak semprot`. Hasil model dari notebook ini hanya mengukur deteksi jenis tanaman, bukan kemampuan sprayer mengambil keputusan. Data penerbangan drone proyek sendiri tetap dibutuhkan untuk validasi lapangan dan pengembangan keputusan penyemprotan.

## Dataset

- Sumber: [Plant Detection and Counting](https://datasetninja.com/plant-detection-and-counting) dan arsip asli [Zenodo](https://zenodo.org/records/4890370).
- Isi: 189 gambar UAV, 16.122 bounding box, tiga kelas, dan 27 grup `site/year/flight_id`.
- Jumlah objek: `maize` 5.257, `sugarbeet` 5.647, `sunflower` 5.218.
- Format dataset: Supervisely (`ds0/img`, `ds0/ann`, `meta.json`). Dataset tidak menyediakan split train/validation/test.
- Lisensi: CC BY 4.0. Atribusi sumber perlu dicantumkan saat dataset digunakan.

Karena tidak ada split bawaan, notebook membagi data per sesi penerbangan dan menjaga satu grup agar tidak muncul di train, validation, dan test sekaligus. Proporsi dibuat sedekat mungkin dengan 70/15/15 sambil menjaga ketiga kelas ada pada tiap split.

## Menjalankan di Kaggle

1. Buat Kaggle Notebook dan tambahkan dataset Plant Detection and Counting lewat **Add Input**.
2. Pilih accelerator **GPU T4 x2**. Training memakai kedua GPU jika tersedia; validation dan test memakai GPU pertama. Aktifkan Internet bila bobot pretrained `yolo26n.pt` perlu diunduh; alternatifnya, tambahkan file bobot sebagai input Kaggle.
3. Buka [notebook PJ1](PJ1_Plant_Detection_Kaggle.ipynb), lalu jalankan sel berurutan dari atas. Sel awal mencari dataset otomatis dari `meta.json`.
4. Hasil konversi, split, grafik, bobot, TXT, dan CSV disimpan di `/kaggle/working/plant_detection_project/`. Unduh folder itu dari bagian Output setelah notebook selesai.

Notebook hanya memasang paket yang belum tersedia dan tidak mengganti PyTorch bawaan Kaggle. Jika GPU tidak terdeteksi, sel training berhenti dengan pesan agar accelerator GPU diaktifkan; training tidak dilanjutkan di CPU.

## Eksperimen

- E1: YOLO26n pretrained COCO, `imgsz=640`.
- E2: arsitektur YOLO26n dari bobot acak, `imgsz=640`.
- E3: YOLO26n pretrained dengan satu perubahan, `imgsz=896`.

Ketiga eksperimen memakai split yang sama, 30 epoch, batch 8, seed 42, dan device yang sama. Model dipilih dari mAP50-95 validation tertinggi. Hanya model terpilih yang dievaluasi pada test, satu kali. Notebook menyimpan tabel metrik CSV/TXT, `results.png`, dan bobot `best.pt`.

### Hasil Kaggle

Training dijalankan pada dua Tesla T4. Split berdasarkan `site/year/flight_id` menghasilkan 134 gambar train, 27 validation, dan 28 test; tidak ada grup penerbangan yang dibagi ke lebih dari satu split.

| Eksperimen | Pengaturan | mAP50 val | mAP50-95 val | Precision | Recall | Inferensi val | Waktu train |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 pretrained | `imgsz=640` | 0.7060 | 0.3751 | 0.7475 | 0.7137 | 9.96 ms | 4.36 menit |
| E2 scratch | `imgsz=640` | 0.3283 | 0.1353 | 0.3541 | 0.4036 | 3.57 ms | 3.94 menit |
| E3 tuning | `imgsz=896` | **0.7471** | **0.4112** | **0.8058** | **0.7577** | 11.18 ms | 4.34 menit |

Model yang dipilih dari validation adalah E3. Model ini diuji satu kali pada test dan mendapat mAP50 **0.9482**, mAP50-95 **0.5145**, precision **0.9153**, dan recall **0.9220**.

**Analisis:** E1 unggul 0.2398 mAP50-95 dari E2, jadi pada eksperimen ini bobot pretrained memberi hasil jauh lebih baik daripada mulai dari bobot acak. E3 menaikkan mAP50-95 validation 0.0361 dibanding E1, tetapi ukuran input yang lebih besar juga menaikkan waktu inferensi validation dari 9.96 menjadi 11.18 ms.

Maize masih menjadi kelas paling sulit. Pada validation, E3 mendapat mAP50-95 0.0742 untuk maize, dibanding 0.599 untuk sugarbeet dan 0.550 untuk sunflower. Pada test nilainya membaik menjadi 0.330 untuk maize, 0.566 untuk sugarbeet, dan 0.647 untuk sunflower. Confusion matrix validation juga menunjukkan banyak deteksi maize yang tidak cocok dengan objek berlabel dan banyak tanaman maize yang terlewat. Hasil maize perlu diperiksa dengan contoh gambar dan anotasinya, bukan hanya mengandalkan nilai rata-rata.

Skor test keseluruhan lebih tinggi daripada validation walaupun split dipisah berdasarkan flight. Ini bisa terjadi karena tingkat kesulitan flight berbeda; validation hanya 27 gambar dari 3 grup, sedangkan test 28 gambar dari 3 grup. Jadi, hasil test yang lebih tinggi belum membuktikan generalisasi yang kuat.

Ada perbedaan besar pada angka inferensi: validation E3 mencatat 11.18 ms per gambar, sedangkan evaluasi test mencatat 100.56 ms. Angka tersebut belum layak dipakai sebagai klaim FPS/latensi perangkat. Ulangi benchmark dengan warm-up dan beberapa putaran prediksi di GPU yang sama, lalu laporkan median waktu inferensi secara terpisah dari preprocessing dan postprocessing.

Kurva E3 menunjukkan mAP validation masih meningkat hingga mendekati akhir training dan loss kelas terus turun. Karena itu, hasil ini sudah berguna sebagai baseline, tetapi tuning tambahan dan validasi pada flight/lahan yang berbeda masih disarankan.

Artefak: [laporan TXT](plant_detection_project/hasil_praktik_4222401015.txt), [rekap CSV](plant_detection_project/hasil_praktik_4222401015.csv), [kurva E1](plant_detection_project/runs/E1_pretrained/results.png), [kurva E3](plant_detection_project/runs/E3_tuning/results.png), dan [confusion matrix E3](plant_detection_project/runs/E3_tuning/confusion_matrix.png).

## Bagian notebook

Setiap program Python memiliki pengantar Markdown sendiri. Alurnya: konversi Supervisely ke YOLO, pemeriksaan anotasi (Program 01), konversi YOLO ke COCO dan kembali (Program 02/03), split per sesi (Program 05), lalu training dan evaluasi (Program 06). Program pengambil subset COCO dan penyiap banana lama tidak dipakai dalam alur baru.

## Referensi

David, E. (2021). _Plant detection and counting from high-resolution RGB images acquired from UAVs: comparison between deep-learning and handcrafted methods with application to maize, sugar beet, and sunflower crops_. Zenodo. https://doi.org/10.1101/2021.04.27.441631. Lisensi CC BY 4.0.
