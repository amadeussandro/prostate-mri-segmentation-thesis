# Audit naskah JMIR v2 (judul tetap)

**Naskah:** JurnalJMIR-BenedictAmadeusSandro-EN-61126.docx
**Judul (tetap):** Segmentation of Anatomical Zones in Prostate MRI Using a 2D U-Net and Spatially Validated 3D Reconstruction
**Tanggal:** 6 November 2026
**Dasar audit:** isi naskah lengkap, file hasil di `results/`, uji pilot pada data yang sudah ada (bagian 3), aturan JMIR terbaru, dan pencarian prior work.
**Melengkapi:** `JMIR_audit_2026-11-06.md`. Daftar 35 pelanggaran format (F1–F35) di sana tetap berlaku.

---

## 1. Kenapa nilainya rendah: akar masalah

Diurutkan dari dampak terbesar.

| # | Akar masalah | Bukti | Akibat di mata reviewer |
|---|---|---|---|
| R1 | **Paper dibingkai sebagai paper akurasi.** Abstrak dan Introduction dibuka dengan Dice, padahal kekuatan judul ada di "Spatially Validated 3D Reconstruction" | Abstrak, Results, dan Conclusions dibuka dengan Dice 0.8442 / 0.6852 | Langsung dibandingkan dengan model 3D (Adams: 0.877 / 0.754), dan kalah |
| R2 | **Akurasi di bawah kesepakatan antar-ahli**, dan ini belum dilaporkan | Pilot (bagian 3): model lebih rendah dari agreement reader 1–reader 2 sebesar 0.029 (CG, P=.006) dan 0.062 (PZ, P=.02) | Kalau reviewer menemukan sendiri, posisinya jadi lebih buruk daripada kalau dilaporkan jujur |
| R3 | **"Spatially validated" baru setengah.** Hanya ada cek header + round-trip. Flip anatomis tidak dicek dengan aturan anatomi, dan sifat 2D (pita antar-slice, komponen nyasar) tidak termasuk dalam validasi | P56–P59, Tabel 4 | Klaim di judul terasa lebih besar dari buktinya |
| R4 | **Klaim novelty keliru.** "We are not aware of prior work…" (P160), padahal flip kiri-kanan di data MRI publik sudah didokumentasikan (Glen et al 2020, Front Neuroinform), kesalahan koordinat saat konversi DICOM→NIfTI dicatat oleh tim MSD (Simpson et al 2019), dan ada laporan label MONAI yang tidak sejajar setelah disimpan ulang | Hasil pencarian (bagian 7) | Reviewer akan menemukannya, dan kredibilitas naskah turun. Padahal literatur ini justru **mendukung** bahwa masalahnya nyata, jadi seharusnya dipakai |
| R5 | **Ablation hanya di 1 dataset**, dan shortcut-nya dibuat sendiri | Tabel 4 | Dianggap "engineering hygiene", atau strawman |
| R6 | **Statistik rapuh.** Uji primer P=.055; hasil negatif hanya bersandar pada P>.05; SD populasi; koreksi multipel disebut tapi tidak diterapkan | P72, P86, P94 | Kesimpulan mudah diserang |
| R7 | **Data sudah "tersentuh".** Test set dipakai 2× sebelumnya; LCC ditemukan dengan melihat hasil PROSTATEx | P129–P130, P168 | PROSTATEx tidak bisa mengonfirmasi LCC; test set tidak bisa dipakai untuk analisis konfirmatori baru |
| R8 | **Klaim klinis hipotetis.** PSA density dihitung tanpa nilai PSA | P117, P164 | Dianggap overclaim |
| R9 | **Angka tidak konsisten** | Tabel 4 hanya memuat centroid CG (8.2 mm), sedangkan teks menyebut "PZ up to 33 mm" (PZ: rata-rata 17.2, maks 33.4). Abstrak "183 mm" vs tabel "182.8". "Volume error 47%" berlaku untuk CG dan PZ, tapi kolomnya tidak menjelaskannya | Copyeditor dan reviewer JMIR mewajibkan angka konsisten di abstrak, teks, dan tabel |
| R10 | **Format dan penulisan.** 35 pelanggaran: abstrak 452 kata, P value, 4 paragraf dobel, RQ1/RQ2 tidak didefinisikan, Supplementary di dalam naskah, tanpa CLAIM, tanpa Funding/CRediT/AI disclosure | Audit v1 | Bisa langsung desk reject |
| R11 | **Salah alamat jurnal.** Scope JMIR utama adalah digital health | — | Peluang di JMIR utama rendah apa pun kualitasnya. Target: JMIR AI / JMIR Biomedical Engineering |

**Kesimpulan:** nilai rendah **bukan karena modelnya**, tapi karena (a) cara membingkai, (b) bukti untuk frasa "spatially validated" masih tipis, (c) statistik dan klaim belum dijaga, dan (d) format. Semuanya bisa diperbaiki **tanpa ganti judul dan tanpa wajib training ulang**.

---

## 2. Nilai sekarang

| Dimensi | Nilai | Alasan singkat |
|---|---|---|
| Novelty | 5 | R4, R5 |
| Rigor desain | 6 | Protokol seleksi bagus; R7, 1 seed |
| Statistik | 5 | R6 |
| Akurasi | 5 | R1, R2 |
| Relevansi klinis | 5 | R8 |
| Format JMIR | 4 | R10 |
| Penulisan | 6 | R9, R10 |
| Reproducibility | 8 | Repo + SHA-256 sudah kuat |
| **Rata-rata** | **5.5** | |

---

## 3. Uji pilot (dari data yang sudah ada, tanpa training)

| Uji | Hasil | Artinya |
|---|---|---|
| TOST E2 vs baseline (validasi, macro Dice, margin ±0.02) | selisih −0.003, 90% CI −0.015 s/d 0.010, **P=.01 → ekuivalen** | Klaim "augmentasi tidak mengubah apa-apa" bisa dibuktikan, bukan cuma P>.05 |
| TOST E3 vs E1 (validasi, macro, ±0.02) | selisih −0.001, 90% CI −0.009 s/d 0.006, **P<.001 → ekuivalen** | Sama untuk class weighting |
| TOST E1 vs baseline (test, macro, ±0.02) | selisih +0.010, 90% CI −0.003 s/d 0.022, P=.08 → **inconclusive** | Jujur: tidak lebih baik dan tidak terbukti setara. Tulis seperti itu |
| Model vs reader 2 (test, CG) | 0.844 vs 0.873; selisih −0.029 (95% CI −0.051 s/d −0.006), Wilcoxon P=.006 | Model di bawah agreement antar-ahli, tapi selisihnya kecil (≈97%) |
| Model vs reader 2 (test, PZ) | 0.685 vs 0.747; selisih −0.062 (95% CI −0.113 s/d −0.011), P=.02 | ≈92% dari agreement antar-ahli |
| Volume PZ antar-reader | reader 2 rata-rata +19.3% dari reader 1 | Bias volume PZ model (−28.5%) harus dibaca terhadap ketidakpastian label sebesar ini |
| SD populasi vs sampel (CG test) | 0.0493 vs 0.0506 | Perubahan kecil, tapi harus diganti ke SD sampel |

Catatan: validasi adalah split tempat seleksi dilakukan, jadi TOST di sana deskriptif. Margin ±0.02 perlu ditetapkan dan dijustifikasi **sebelum** dilaporkan, misalnya sebagai sebagian kecil dari SD agreement antar-reader, dan sebaiknya disepakati dengan dosbing dulu.

---

## 4. Solusi tanpa ganti judul + kenaikan nilai per perubahan

Δ rata-rata = total kenaikan poin dimensi ÷ 8.

| # | Perubahan | Menjawab | Dimensi yang naik | Δ rata-rata | Training? |
|---|---|---|---|---|---|
| 1 | Format JMIR lengkap (F1–F35), termasuk subjudul tipe studi wajib JMIR ": Algorithm Development and Validation Study" (judul inti tetap) | R10 | Format 4→9 | **+0.63** | Tidak |
| 2 | Framing ulang isi (judul tetap): abstrak/Intro/Conclusions dibuka dengan rekonstruksi; RQ didefinisikan; hapus paragraf dobel dan kalimat "cara baca tabel"; pangkas 15–20%; perbaiki angka yang tidak konsisten | R1, R9 | Penulisan 6→8, Novelty +0.5 | **+0.31** | Tidak |
| 3 | Statistik: TOST (margin ditetapkan dulu), Holm, effect size + 95% CI, SD sampel, uji primer dilaporkan sebagai inconclusive | R6 | Statistik 5→7 | **+0.25** | Tidak |
| 4 | Kohort eksternal kedua **NCI-ISBI 2013** (80 kasus, label CG + PZ sama persis dengan model, 1.5T endorectal + 3T): konfirmasi akurasi + uji LCC yang **ditetapkan sebelumnya** | R7, R2 | Rigor +1, Akurasi +0.5 | **+0.19** | Tidak (inference CPU) |
| 5 | Ablation model-free di **MSD Task05 + NCI-ISBI** (mask anotasi sebagai "prediksi") | R5 | Novelty +1, Rigor +0.5 | **+0.19** | Tidak |
| 6 | Rilis round-trip check sebagai tool kecil + DOI Zenodo + versi software lengkap | R5 | Reproducibility 8→9, Novelty +0.5 | **+0.19** | Tidak |
| 7 | **Validasi level 3 (anatomis):** PZ harus di posterior CG, konsistensi antar-slice, jumlah komponen. Masukkan ke ablation sebagai matriks "cek mana menangkap error apa" | R3 | Novelty +0.5, Rigor +0.5 | **+0.13** | Tidak |
| 8 | Pembanding reader 2 (sudah di-pilot) + volume antar-reader | R2 | Akurasi +0.5, Klinis +0.5 | **+0.13** | Tidak |
| 9 | PSA density diubah jadi sensitivity analysis (rentang PSA realistis, proporsi yang melewati 0.15) | R8 | Klinis +0.5 | **+0.06** | Tidak |
| 10 | Posisikan terhadap prior work (Glen 2020, Simpson 2019, RIDGE, issue MONAI); hapus "not aware of prior work"; ganti ref 27–28 yang tidak relevan | R4 | Mencegah penalti (≈ −0.1 s/d −0.2 kalau tidak dikerjakan) | **0 (proteksi)** | Tidak |
| | **Total inti (1–10)** | | | **+2.1 → ≈7.6** | |
| 11 | *(Opsional, kalau dosbing tetap minta)* 4 run training: baseline & E1 × 2 seed baru (Kaggle/Colab dengan resume otomatis) | R6, R7 | Rigor +0.5, Statistik +0.5 | +0.13 | **4 run** |
| 12 | *(Opsional)* Audit literatur 30–50 paper soal verifikasi rekonstruksi | R4, R5 | Novelty +0.5 | +0.06 | Tidak |
| | **Total + opsional** | | | **+2.3 → ≈7.8** | |

Catatan untuk langkah 7: cek "PZ di posterior CG" menangkap flip depan-belakang. Varian "inverse orientation omitted" di data ini membalik dua sumbu (LPS→RAS), jadi tertangkap. Flip kiri-kanan saja **tidak** akan tertangkap aturan anatomi, karena prostat kira-kira simetris. Laporkan keterbatasan ini terang-terangan: justru inilah alasan round-trip test tetap diperlukan.

---

## 5. Nilai sebelum vs sesudah

| Dimensi | Sebelum | Sesudah inti (1–10) | Sesudah + opsional (11–12) |
|---|---|---|---|
| Novelty | 5 | 7.5 | 8 |
| Rigor desain | 6 | 8 | 8.5 |
| Statistik | 5 | 7 | 7.5 |
| Akurasi | 5 | 6 | 6 |
| Relevansi klinis | 5 | 6 | 6 |
| Format JMIR | 4 | 9 | 9 |
| Penulisan | 6 | 8 | 8 |
| Reproducibility | 8 | 9 | 9 |
| **Rata-rata** | **5.5** | **≈7.6** | **≈7.8** |

| Jurnal | Sebelum | Sesudah inti | Sesudah + opsional |
|---|---|---|---|
| JMIR AI (APF USD 1985) | ~20% | ~55–60% | ~60–65% |
| JMIR Biomedical Engineering (APF USD 1500) | ~25% | ~60–65% | ~65–70% |
| JMIR utama (APF USD 3350) | ~5% | ~10% | ~15% |

Peluang adalah perkiraan dari penilaian, bukan data. Akurasi tetap di 6 karena model memang di bawah model 3D dan di bawah agreement antar-ahli. Batas itu hanya bisa dinaikkan dengan arsitektur lain, misalnya 2.5D U-Net, yang di luar rencana ini.

---

## 6. Rencana kerja dan to-do list

Pemilik: **C** = Claude, **S** = Sandro, **D** = dosbing.

### Minggu 1: format, tulisan, statistik (semua CPU)
- [ ] (S+D) Sepakati: target JMIR AI vs Biomedical Engineering, corresponding author, sumber dana APF, subjudul tipe studi, margin TOST
- [ ] (C) Salinan .docx kerja; naskah asli tidak disentuh
- [ ] (C) Perbaikan format F1–F35 (abstrak ≤450 kata, P value, ρ, ×, em dash, US English, section akhir, Abbreviations, referensi + PMID)
- [ ] (C) Hapus 4 paragraf dobel; definisikan RQ; redaksi E3 ("single-factor relative to E1"); jelaskan 8.2 vs 33 mm; samakan 182.8/183; kolom volume error
- [ ] (C) Pindahkan Supplementary → Multimedia Appendix 1–3
- [ ] (C) Hitung ulang dengan SD sampel; TOST + Holm + CI; uji primer ditulis "inconclusive"
- [ ] (C) Analisis reader 2 (Dice, HD95 reader 1 vs reader 2 dihitung sendiri dari masker, volume)
- [ ] (C) PSA density → sensitivity analysis
- [ ] (C) Bagian prior work: Glen 2020, Simpson 2019, RIDGE; hapus "not aware of prior work"; ganti ref 27–28

### Minggu 2: memperkuat "spatially validated" (CPU)
- [ ] (C) Validasi level 3 + matriks cek vs jenis error di ablation
- [ ] (S) Izinkan unduhan MSD Task05 dan NCI-ISBI 2013 (TCIA)
- [ ] (C) Ablation model-free + round-trip di MSD Task05 dan NCI-ISBI
- [ ] (C) Aturan LCC ditulis dulu (pre-specified), divalidasi di split validasi internal
- [ ] (C) Inference model beku di NCI-ISBI: akurasi + LCC sebagai analisis konfirmatori; PROSTATEx hanya sekunder

### Minggu 3: finalisasi
- [ ] (C) Tool round-trip (paket + test + README); (S) rilis Zenodo untuk DOI
- [ ] (C) Framing ulang abstrak/Intro/Conclusions **setelah** semua angka keluar
- [ ] (C) Cek konsistensi semua angka (abstrak = teks = tabel = appendix)
- [ ] (C) Isi checklist CLAIM 2024; draf cover letter (sebut asal skripsi + AI disclosure)
- [ ] (S) Ekspor gambar PNG ≤1200×1200 px dan ≤5 MB; ORCID dosbing
- [ ] (D) Review akhir → (S) submit

### Opsional (paralel)
- [ ] (C) Notebook training 4 run dengan resume otomatis → (S) jalankan di Kaggle/Colab → (C) evaluasi di CPU
- [ ] (C) Audit literatur 30–50 paper → (D) kode ulang 20%, laporkan κ

**Perkiraan waktu:** kerja aktif ±6–7 hari, kalender ±2–3 minggu (tergantung unduhan TCIA dan review dosbing).

---

## 7. Sumber

Aturan JMIR:
- Instructions for Authors — https://www.jmir.org/author-information/instructions-for-authors
- Submission guide — https://support.jmir.org/hc/en-us/articles/37982552280987
- Statistics — https://support.jmir.org/hc/en-us/articles/360019690851
- References — https://support.jmir.org/hc/en-us/articles/115001333067
- Figures/supplementary — https://support.jmir.org/hc/en-us/articles/115002199367
- Generative AI — https://support.jmir.org/hc/en-us/articles/13387268671771
- Fees — https://www.jmir.org/about-journal/article-processing-fees

Prior work dan dataset (verifikasi DOI sebelum disitasi):
- Glen DR et al. Beware (surprisingly common) left-right flips in your MRI data. Front Neuroinform. 2020;14:18 — https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2020.00018/pdf
- Simpson AL et al. A large annotated medical image dataset for the development and evaluation of segmentation algorithms (MSD). arXiv 1902.09063 — https://arxiv.org/pdf/1902.09063
- RIDGE checklist. arXiv 2401.08847 — https://arxiv.org/pdf/2401.08847
- Issue MONAI: label tidak sejajar setelah NiftiSaver — https://lightrun.com/answers/project-monai-monai-misaligned-labels-saved-using-niftisaver-and-viewed-in-3d-slicer
- NCI-ISBI 2013 (TCIA, DOI 10.7937/K9/TCIA.2015.zF0vlOPv) — https://wiki.cancerimagingarchive.net/x/wQZp
- MSD Task05 Prostate (32 kasus, PZ/TZ, Radboud) — https://arxiv.org/pdf/2106.05735
