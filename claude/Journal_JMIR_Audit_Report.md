# Laporan Audit Jurnal JMIR

**Tanggal:** 2026-10-15
**Auditor:** sesi kerja Claude Code
**Status:** TASK 1 — laporan saja, belum ada file jurnal yang diubah

Cakupan: isi `D:\1.AmadeusKuliah\Skripsi\Journal-JMIR`, dibandingkan terhadap
modul konten di `thesis-project/scripts/`, artefak hasil di
`thesis-project/results/`, dan `HANDOFF.md`.

Metode: kedua `.docx` diekstrak secara *read-only* dengan `python-docx`
(teks, heading, tabel, properti dokumen, bagian media), lalu angka headline
dicocokkan terhadap artefak hasil. Tidak ada file jurnal yang disentuh.

---

## A. Ringkasan temuan

### Temuan utama: `Journal-JMIR` bukan versi terbaru

Ini membalik premis di brief. Brief menyebut `Journal-JMIR` sebagai "sumber
kebenaran konten saat ini". Buktinya menunjukkan sebaliknya — folder itu
**satu commit di belakang** pada isi, dan **metadata dokumennya rusak**.

| | `Journal-JMIR` (14 Okt) | `thesis-project` (8 Okt) | modul konten (sekarang) |
|---|---|---|---|
| Ukuran | 7,70 MB | 2,63 MB | — |
| Menyebut *anterior fibromuscular stroma* | **tidak (0×)** | ya (2×) | ya, versi terkoreksi |
| Menyebut lisensi eksternal *CC BY* | **tidak (0×)** | ya (2×) | ya |
| `author` di properti dokumen | **kosong** | `Benedict Amadeus Sandro` | — |
| `last_modified_by` | **`python-docx`** | `Benedict Amadeus Sandro` | — |
| Resolusi gambar (media terbesar) | 2,29 MB | 0,61 MB | — |
| Abstrak | 449 / 418 kata | 449 kata | 449 / 418 |

Tiga konsekuensi langsung:

1. **Kalau konten `Journal-JMIR` disalin ke skripsi apa adanya, dua koreksi
   hilang** — penamaan ketidaksesuaian definisi zona dan penamaan lisensi
   eksternal, keduanya ditambahkan commit `9f81194`.
2. **Metadata `Journal-JMIR` melanggar aturan keras proyek.** `HANDOFF.md` §8:
   *"No AI/tool attribution anywhere — not in the text, not in document
   metadata."* Field `last_modified_by` berisi `python-docx`, yaitu atribusi
   tool, dan field `author` kosong sehingga nama Mas hilang. Build di
   `thesis-project` benar pada kedua field ini.
3. **Satu hal di `Journal-JMIR` justru lebih baik:** gambarnya beresolusi jauh
   lebih tinggi (2,29 MB vs 0,61 MB untuk figure terbesar). Untuk submission
   jurnal itu menguntungkan. Jadi folder itu bukan "salah total" — ia hasil
   build dengan template berbeda yang kebetulan membawa figure lebih tajam
   tetapi gagal mereset properti dokumen.

**Sumber kebenaran konten yang sebenarnya adalah
`thesis-project/scripts/manuscript_content_en.py` dan `_id.py`.** Keduanya saat
ini **ada perubahan belum ter-commit** yang lebih maju dari kedua `.docx`
(detail di bagian C).

### Yang sudah benar

- **Struktur JMIR patuh.** Abstrak terstruktur (Background / Objective /
  Methods / Results / Conclusions), 38 heading, 6 tabel, 9 figure, referensi
  bernomor urut kemunculan. Abstrak 449 kata (EN) dan 418 (ID), di bawah batas
  450.
- **Semua angka headline terlacak ke artefak.** Dicocokkan langsung: CG Dice
  0.8442, PZ 0.6852, eksternal CG 0.8137, PZ 0.6247 — keempatnya cocok antara
  `.docx` dan artefak hasil. Tidak ditemukan angka yang diketik manual.
- **Integritas statistik dijaga.** Uji pada data validasi dilabeli deskriptif
  dan sirkular; uji per-zona pada data uji dilabeli *post hoc* dengan catatan
  Bonferroni; riwayat pemakaian data uji diungkap (*"not pristine"*), tidak
  diklaim bersih.
- **Round-trip Dice tidak masuk tabel hasil**, dilaporkan sebagai pass/fail —
  konsisten dengan keputusan §5.4 dan menutup pertanyaan "Dice 0.8 bukan 1".
- **4 referensi inti semuanya ada dan dikutip** di kedua bahasa: `adams_cbm`,
  `adams_dib`, `astono`, `cheng`.
- Verifikasi label mapping, inspeksi multi-planar, SHA-256 checkpoint, jumlah
  parameter, Bland–Altman beserta limits of agreement — semuanya sudah
  dilaporkan di naskah.

### Yang keliru atau hilang

- **Judul menyimpang dari judul yang dikunci.** Lihat bagian E.1 — prioritas
  tinggi untuk Task 3.
- **Noise float32 tercetak di naskah.** `"in-plane resolution spans
  0.30000001192092896-0.6000000238418579 mm"`. Ada di **kedua** `.docx`. Sudah
  dikoreksi di modul konten, belum di-rebuild.
- **Satu kalimat mengulang dirinya sendiri** di bagian Comparison With Prior
  Work. Sudah dikoreksi di modul, belum di-rebuild.
- **Tiga dari empat dokumen pembanding yang disebut brief tidak ada** —
  lihat E.5.

---

## B. Tabel penilaian per-section

Status dinilai terhadap modul konten terkini (bukan terhadap `Journal-JMIR`,
yang tertinggal). "Keliru" berarti ada pernyataan yang salah atau tidak
didukung; "Kurang" berarti benar tetapi belum memadai.

| Section | Status | Masalah | Rekomendasi Perbaikan | Prioritas |
|---|---|---|---|---|
| **Title** | **Keliru** | Judul jurnal menambah *"Spatially Validated"* / *"yang Tervalidasi Secara Spasial"*, tidak ada di judul yang dikunci | Putuskan: judul skripsi **wajib** pakai versi terkunci; untuk jurnal, pertahankan atau selaraskan — keputusan Mas | **High** |
| **Abstract** | Lengkap | 449 kata (EN) / 418 (ID), terstruktur, di bawah batas 450. Keyword 10 buah — tepat di batas atas JMIR (3–10) | Tidak ada perubahan wajib. Opsional: buang keyword *"reproducibility"*, karena naskah sendiri menyatakan reproducibility adalah properti, bukan temuan | Low |
| **Introduction** | Lengkap | Latar belakang, celah riset, dan kontribusi tersusun. Pernyataan kontribusi sebelumnya muncul tiga kali; sudah dikonsolidasi di modul | Rebuild agar konsolidasi masuk | Low |
| **Related Work** | **Hilang (sebagai section terpisah)** | JMIR tidak memakai section ini; prior work tersebar di Introduction + Discussion → Comparison With Prior Work. Benar untuk jurnal, **tetapi skripsi butuh BAB II utuh** | Tidak diubah di jurnal. Untuk Task 3, BAB II harus ditulis sebagai bab baru, bukan hasil salin | **High** (untuk skripsi) |
| **Dataset** | Lengkap | Prostate158, split resmi 119/20/19, label mapping terverifikasi | — | — |
| **Methods — Preprocessing** | **Kurang** | Memuat noise float32 yang tercetak | Sudah difix di modul; rebuild | **Med** |
| **Methods — Slicing** | Lengkap | Penyiapan irisan 2D terdokumentasi | — | — |
| **Methods — 2D U-Net** | Lengkap | Arsitektur + 7.762.531 parameter dilaporkan | — | — |
| **Methods — Training** | Lengkap | Empat konfigurasi faktor-tunggal terdokumentasi | — | — |
| **Methods — Model Selection** | Lengkap | Aturan seleksi pra-spesifikasi, `test_metrics_used_for_selection: false` | Opsional: sebut eksplisit epoch 88 sebagai detail reproduktibilitas (saat ini tidak disebut) | Low |
| **Evaluation Protocol (RQ1)** | Lengkap | Metrik, ddof, bootstrap CI terdokumentasi | — | — |
| **3D Reconstruction (RQ2)** | Lengkap | Rantai operasi dan inversinya dijelaskan | — | — |
| **Spatial Verification** | Lengkap | Terbagi di *Geometric and Fidelity Validation* (Methods) dan *What the Reconstruction Steps Are Worth* (Results). Ablasi 19/19 bit-identical | — | — |
| **Results** | Lengkap | Semua angka terlacak ke artefak | — | — |
| **Discussion** | **Keliru** | Memuat klaim bahwa sebagian gap eksternal berasal dari perbedaan anotasi (stroma). **Klaim itu sudah diuji dan terbantahkan** — lihat bagian C.1 | Sudah dikoreksi di modul; rebuild wajib sebelum kirim | **High** |
| **Limitations** | Lengkap | Dataset tunggal, arsitektur standar, riwayat data uji, tidak ada reader study — semuanya diungkap | — | — |
| **Conclusion** | Lengkap | Klaim konservatif, tidak mengaku SOTA | — | — |
| **References** | Lengkap | 20 referensi, bernomor urut kemunculan, 4 referensi inti ada dan dikutip, DOI lengkap | — | — |

---

## C. Sudah dikerjakan tetapi belum dilaporkan di jurnal

### C.1 Analisis anterior stroma — membantah klaim yang ada di naskah

Artefak: `results/external_anterior_stroma/` (204 kasus eksternal + 19 internal
sebagai kontrol). Skrip: `scripts/analyse_anterior_stroma.py`.

Naskah versi sekarang menyatakan sebagian gap eksternal berasal dari perbedaan
definisi zona (rujukan eksternal memuat anterior fibromuscular stroma).
Klaim itu diuji dengan prediksi spasial — kalau benar, recall CG eksternal harus
jatuh ke arah anterior lebih tajam daripada internal. **Prediksinya gagal:**

| | Eksternal | Internal | Gap |
|---|---|---|---|
| CG Dice, paruh **anterior** | 0.8307 | 0.8366 | **−0.0059** |
| CG Dice, paruh **posterior** | 0.7895 | 0.8516 | **−0.0621** |
| Defisit anterior (post − ant) | −0.0436 | +0.0465 | p = **0.0020** |

**91% gap ada di paruh posterior, 9% di anterior.** Paruh anterior — justru
tempat stroma berada — adalah bagian yang paling sepakat antar kohort, dan arah
defisitnya terbalik antar kohort. Ketahanannya diuji dengan bidang pembagi
alternatif (−0.0677 / −0.0071, praktis identik).

Dua gate validasi lolos: sumbu anterior diambil dari affine tiap volume dan
tandanya diuji secara anatomis (sentroid PZ harus posterior terhadap CG) —
**223/223 lolos**; dan skripnya mereproduksi dua angka headline secara
independen dari volume mentah (CG internal 0.8442, eksternal 0.8137).

Status di tiap salinan:
- `Journal-JMIR`: klaim stroma **tidak ada sama sekali** (versinya mendahului commit yang menambahkannya).
- `thesis-project/*.docx`: **memuat klaim yang terbantahkan**.
- Modul konten: **sudah dikoreksi**, belum di-rebuild dan belum di-commit.

Koreksinya mempertahankan fakta bahwa definisi zona memang berbeda — itu benar
dan terdokumentasi penulis dataset — tetapi menghentikan implikasi bahwa
perbedaan itu menjelaskan gap-nya.

### C.2 Perubahan lain yang belum ter-commit di modul konten

Semuanya sudah ditulis dan build-nya bersih, tetapi belum masuk `.docx` mana pun:

- Perbaikan noise float32 (`0.30-0.60 mm`).
- Penghapusan kalimat yang mengulang diri di Comparison With Prior Work.
- Penggantian angka FOV yang diulang verbatim di Results dengan rujukan balik.
- Konsolidasi pernyataan kontribusi di Introduction.
- Penyingkatan Principal Results (RQ2) — **satu edit ini belum tuntas**, sesi
  sebelumnya terputus di tengah.

Dampak panjang EN: 6371 → 6323 kata prosa isi. Versi ID **belum dipangkas sama
sekali**.

### C.3 Catatan atas rencana pemangkasan di HANDOFF

`HANDOFF.md` §6.C menargetkan ~800 kata dari Limitations + Comparison. Target itu
**tidak bisa dipenuhi**: kedua bagian totalnya 644 kata dan keduanya padat —
tiap paragraf membawa temuan. Yang realistis ~250–350 kata dari duplikasi nyata.
Memaksa 800 berarti membuang temuan.

---

## D. Konten yang perlu ditulis atau ditambah

### Untuk jurnal

1. **Rebuild kedua `.docx`** dari modul konten terkini. Ini bukan penambahan
   konten, tetapi tanpanya semua koreksi di atas tidak sampai ke dokumen.
   Rebuild ID wajib memakai `--template` yang menunjuk docx bersih.
2. **Selesaikan pemangkasan EN dan terapkan ke ID.** ID belum disentuh,
   sehingga kedua bahasa saat ini tidak sejajar isinya.
3. **Perbaiki metadata pada salinan yang akan dikirim** — pastikan `author`
   terisi dan `last_modified_by` bukan nama tool.
4. **Putuskan soal figure.** Figure di `Journal-JMIR` lebih tajam. Kalau itu
   hasil regenerasi dengan DPI lebih tinggi, langkah itu perlu dipindahkan ke
   pipeline build supaya tidak hilang saat rebuild.

### Untuk skripsi (Task 3, belum dimulai)

5. **BAB II Tinjauan Pustaka harus ditulis baru.** Jurnal tidak punya section
   Related Work; isinya tersebar di Introduction dan Discussion. Bab ini tidak
   bisa dihasilkan dengan menyalin.
6. **Bagian yang tidak ada padanannya di jurnal** dan harus ditulis dari nol
   untuk BAB I: Batasan Masalah dan Manfaat Penelitian.
7. **Judul skripsi wajib memakai versi terkunci**, bukan versi jurnal.

---

## E. Catatan integritas ilmiah

### E.1 Judul menyimpang dari judul yang dikunci — **High**

Judul terkunci (`HANDOFF.md` §1, *"Title and RQs are fixed. Do not reword
them."*):

> Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat menggunakan 2D
> U-Net dan Rekonstruksi 3D

Judul di jurnal ID:

> Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat Menggunakan 2D
> U-Net dan Rekonstruksi 3D **yang Tervalidasi Secara Spasial**

Tambahan "yang Tervalidasi Secara Spasial" itu sebenarnya *didukung* hasil —
verifikasi spasialnya memang dikerjakan dan merupakan kontribusi utama. Jadi ini
bukan klaim berlebihan. Tetapi ia **tetap penyimpangan dari judul terkunci**, dan
untuk skripsi judul terkunci yang berlaku. Butuh keputusan Mas, bukan keputusan
saya.

### E.2 Atribusi tool di metadata dokumen — **High**

`Journal-JMIR` kedua file: `last_modified_by = "python-docx"`, `author` kosong,
`title = "Word Document"`. Ini melanggar aturan keras "tidak ada penyebutan AI
atau tool di mana pun, termasuk metadata". Salinan `thesis-project` benar.
Jangan kirim file dari `Journal-JMIR` sebelum ini diperbaiki.

### E.3 Klaim yang sudah terbantahkan masih ada di naskah — **High**

Lihat C.1. Ini satu-satunya temuan yang menyentuh kebenaran isi, bukan bentuk.
Perlu dicatat bahwa yang terbantahkan adalah *penjelasan* yang ditawarkan
naskah, bukan angka hasilnya — semua metrik tetap berlaku apa adanya.

### E.4 Inkonsistensi angka — tidak ditemukan

Semua angka headline terlacak ke artefak. Satu-satunya cacat penyajian angka
adalah noise float32, yang bersifat format, bukan nilai. Pemindaian seluruh
naskah untuk desimal panjang lain hanya menghasilkan DOI (positif palsu).

### E.5 Tiga dokumen pembanding yang disebut brief tidak ada — perlu klarifikasi

| Dokumen | Status |
|---|---|
| `Thesis-Write-Jurnal.md` | **ada** — `thesis-project/Thesis-Write-Jurnal.md` |
| `Audit_Report_Exp01_Baseline_2RQ.md` | **tidak ditemukan** |
| `Metodologi_Final_Prostate158_2RQ.md` | **tidak ditemukan** |
| `Research_Strengthening_Plan_LockedRQ.md` | **tidak ditemukan** |

Dicari di seluruh `D:\1.AmadeusKuliah\Skripsi`. Audit ini memakai `HANDOFF.md`
sebagai otoritas metodologi dan keputusan sebagai gantinya. Brief juga menyebut
"Project Instructions (Prostate158 Thesis — Zonal Segmentation)" sebagai dokumen
rujukan; dokumen itu **tidak ada di dalam sesi ini**, jadi saya tidak bisa
memverifikasi apa pun terhadapnya. Kalau tiga dokumen itu dan Project
Instructions memang ada di tempat lain, tolong tunjukkan — kalau ada
pertentangan antara mereka dan `HANDOFF.md`, itu harus diselesaikan **sebelum**
Task 3, karena Task 3 akan menyalin isi ke dokumen yang dinilai.

### E.6 Klaim novelty — proporsional

Naskah secara eksplisit menolak mengklaim arsitektur baru, SOTA, maupun
"pipeline yang reproducible" sebagai temuan. Kontribusi dinyatakan sebagai
(1) perbandingan empiris empat konfigurasi termasuk hasil negatif, dan
(2) demonstrasi terukur bahwa validasi geometris saja tidak cukup. Keduanya
didukung artefak. Akurasi di bawah paper dataset (Adams CG 0.877 / PZ 0.754)
diakui terbuka. **Tidak ada klaim berlebihan.**

### E.7 Risiko data leakage — diungkap, tidak disembunyikan

Data uji 19 kasus pernah dipakai dua kali sebelum keempat konfigurasi dirancang,
dengan model berbeda. Diungkap di Limitations, bukan diklaim bersih. Aturan
seleksi dikunci di muka dan hanya memakai data validasi. Penanganannya benar.

---

## Rekomendasi urutan kerja

1. **Keputusan Mas dulu** atas tiga hal: judul (E.1), salinan mana yang jadi
   basis submission (A), dan lokasi dokumen yang hilang (E.5).
2. Setelah itu: selesaikan pemangkasan EN + ID, rebuild kedua `.docx` dengan
   metadata benar, commit bersama pembaruan `HANDOFF.md`.
3. Baru lanjut Task 2 (template LaTeX) dan Task 3 (sinkronisasi ke skripsi).

Catatan: Task 2 sudah bisa dijalankan paralel — strukturnya sudah saya
inspeksi sekilas dan terbaca penuh (`SKRIPSI.tex`, `report1.cls`,
`_internals/umnthesis.sty`, `src/{1.Awal,2.Isi,3.Akhir}`, bibliografi
IEEEtran). Saya tahan sesuai instruksi "berhenti setelah Task 1".

**Belum ada file jurnal maupun skripsi yang diubah.**
