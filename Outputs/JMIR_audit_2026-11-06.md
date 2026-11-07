# Audit naskah JMIR — JurnalJMIR-BenedictAmadeusSandro-EN-61126.docx

Tanggal audit: 6 November 2026. Sumber aturan: JMIR Instructions for Authors, "Submitting Your Manuscript… A Guide for Authors", Guidelines for Reporting Statistics, House Style, aturan referensi/gambar/AI, dan Fee Schedule (semuanya dibaca langsung dari jmir.org / support.jmir.org hari ini).

Kondisi naskah: 452 kata abstrak, ±9.200 kata isi utama + ±2.000 kata Supplementary di dalam naskah (total ±11.200), 7 tabel utama + 3 tabel S, 10 gambar utama + 11 gambar S, 29 referensi.

---

## 1. Nilai saat ini vs setelah perbaikan

| # | Dimensi | Sekarang | Setelah fase 1–2 | Setelah semua fase | Alasan utama |
|---|---|---|---|---|---|
| 1 | Novelty / kontribusi | 5 | 6 | 7.5–8 | Sekarang mudah dicap "engineering hygiene"; naik kalau klaim "rarely verified" dibuktikan dengan data dan ablation dijalankan di dataset lain |
| 2 | Desain & rigor metodologi | 6 | 7 | 8 | Selection protocol bagus; lemah di 1 seed, n=20 validasi, test set sudah dipakai 2×, LCC ditemukan post hoc |
| 3 | Statistik | 5 | 7 | 7.5 | Hasil negatif diklaim dari P>.05; belum ada equivalence test; SD populasi; koreksi multipel tidak konsisten |
| 4 | Akurasi / hasil | 5 | 6 | 6 | Di bawah referensi 3D; naik karena dibandingkan dengan **agreement antar-reader** (lihat temuan baru) + LCC yang sah |
| 5 | Relevansi klinis | 5 | 6 | 6 | PSA density tanpa PSA nyata → ubah jadi sensitivity analysis |
| 6 | Kepatuhan format JMIR | 4 | 9 | 9 | ±30 pelanggaran, kebanyakan mekanis (daftar di bagian 3) |
| 7 | Kualitas tulisan | 6 | 8 | 8 | 4 paragraf dobel, gaya skripsi, kalimat "cara baca tabel", British spelling |
| 8 | Reproducibility / open science | 8 | 9 | 9 | Repo + SHA-256 sudah kuat; tinggal DOI Zenodo & versi software |
| | **Rata-rata** | **5.5** | **7.3** | **7.8** | |

### Peluang diterima (perkiraan kasar, berdasarkan penilaian, bukan data)

| Jurnal | APF (USD) | Sekarang | Setelah fase 1–2 | Setelah semua | Catatan |
|---|---|---|---|---|---|
| Journal of Medical Internet Research (flagship) | 3350 | ~5% | ~10% | ~15% | Scope digital health/internet; paper metode segmentasi murni rawan desk reject |
| **JMIR AI** | 1985 | ~20% | ~45% | ~60% | Cocok untuk "Algorithm Development and Validation" + tool reusable |
| **JMIR Biomedical Engineering** | 1500 | ~25% | ~50% | ~65% | Paling cocok dengan sisi pipeline/rekonstruksi; APF termurah |
| JMIR Cancer | 1985 | ~10% | ~25% | ~35% | Butuh sudut klinis yang lebih kuat |
| JMIR Medical Informatics | 2750 | ~15% | ~35% | ~45% | |

Rekomendasi target: **JMIR AI** (pilihan 1) atau **JMIR Biomedical Engineering** (pilihan 2). JMIR punya cascading peer review, jadi naskah bisa dipindah antarjurnal JMIR tanpa mulai ulang.

---

## 2. Temuan baru dari audit (tidak ada di rencana yang di-paste)

| # | Temuan | Bukti | Dampak | Usaha |
|---|---|---|---|---|
| N1 | **Data reader-2 test set sudah ada dan belum dipakai.** `G:/My Drive/THESIS_PROSTATE158/dataset/test/extracted/interrater.csv` + 19 masker `t2_anatomy_reader2`. Inter-reader Dice: TZ/CG 0.873 (SD 0.033), PZ 0.747 (SD 0.068). Model: 0.844 / 0.685 → **96.7% dan 91.7% dari agreement manusia**. Volume PZ antar-reader berbeda +19.3% rata-rata. | dihitung dari interrater.csv | Sangat besar: memberi human benchmark, jangkar margin TOST, dan konteks untuk bias volume PZ −28.5% | 0.5–1 hari, CPU |
| N2 | **Kohort eksternal BUKAN lagi "untouched" untuk LCC.** LCC ditemukan dengan melihat HD95 PROSTATEx (20.2→6.5 mm). Menjadikan PROSTATEx sebagai evaluasi konfirmatori LCC = sirkular. Rencana yang di-paste perlu dikoreksi di sini. | naskah P129–P130 | Kalau tidak dikoreksi, reviewer bisa menyebutnya tuning di external | Butuh kohort ketiga yang segar |
| N3 | Kohort ketiga kandidat: **NCI-ISBI 2013** (label PZ + CG, multi-site) untuk konfirmasi LCC; **MSD Task05 Prostate** untuk ablation model-free. Catatan: MSD Task05 dan PROSTATEx sama-sama dari Radboud, jadi cek kemungkinan pasien yang sama sebelum memakainya sebagai kohort independen. | — | Mengganti PROSTATEx sebagai data konfirmatori yang benar-benar segar | 2–4 hari |
| N4 | 1 seed (42) per konfigurasi. Selisih antar-konfigurasi 0.0112 tidak bisa dipisahkan dari noise seed. CV 5-fold juga mengatasi ini, tapi minimal 3 seed di split resmi lebih murah (12 run). | Tabel 1 | Klaim hasil negatif jadi kuat | Colab GPU |
| N5 | Pengungkapan AI generatif wajib (pertanyaan khusus saat submit + transkrip prompt sebagai Multimedia Appendix). Tidak boleh ditaruh di Acknowledgments. | Kebijakan AI JMIR | Risiko integritas jika tidak diungkapkan | 1–2 jam |
| N6 | Tumpang tindih dengan skripsi (DRAFT skripsi PDF). JMIR menganggap overlap ±10% sebagai duplikatif jika tidak diungkap → harus disebut di cover letter (skripsi/repositori UMN). | aturan duplicate publication | Risiko desk reject / masalah etika | 30 menit |
| N7 | DOI 29 referensi dicek ke Crossref: semuanya ada. Tapi ref 27 (HCC microvascular invasion) dan 28 (XAI scoping) tidak relevan dan terlihat seperti padding sitasi JMIR. | Crossref | Reviewer akan menyadarinya | 1 jam |

---

## 3. Pelanggaran format JMIR (checklist)

Prioritas: **W** = wajib/berisiko desk reject, **P** = akan diminta copyeditor, **S** = saran.

| # | Item | Sekarang | Aturan JMIR | Perbaikan | Prio |
|---|---|---|---|---|---|
| F1 | Judul | Tanpa subjudul tipe studi | Format "Issue: Method/Study Design", title case, ≤280 karakter, tanpa membocorkan hasil | mis. "Spatially Verified 3D Reconstruction of 2D U-Net Prostate Zone Segmentation on MRI: Algorithm Development and Validation Study" | W |
| F2 | Gelar penulis | "S.Kom., M.T.I." | Gelar tertinggi, tanpa titik/spasi | "Eunike Endariahna Surbakti, MTI"; mahasiswa S1 boleh tanpa gelar | P |
| F3 | Corresponding author | Tanpa email/telepon/alamat lengkap | Email, telepon, alamat lengkap | Lengkapi; ORCID semua penulis wajib saat diterima | W |
| F4 | Afiliasi | Department, Faculty, University, City, Country | Department, Institute/University, City, Country | Sudah hampir benar; konsisten di metadata | S |
| F5 | Panjang abstrak | 452 kata | Maks 450 | Pangkas ke ±400 | W |
| F6 | Isi abstrak | Tanpa CI/P value; "PSA" tidak dijabarkan | Results memuat n, CI, P value; singkatan dijabarkan | Tambah 95% CI Dice dan P/TOST; jabarkan PSA | P |
| F7 | Keywords | 10, bebas | 5–10, sebaiknya MeSH | Pakai MeSH: Prostatic Neoplasms; Magnetic Resonance Imaging; Deep Learning; Image Processing, Computer-Assisted; … | S |
| F8 | P value | 9× "P=0.0192", "P=0.055" | Tanpa nol di depan, 2 desimal (3 jika <.01 atau jika pembulatan mengubah signifikansi) | P=.02, P=.06 (dari .055), P=.048, P=.39, P=.33, P=.002, P=.72 | P |
| F9 | Korelasi Spearman | "Spearman r=0.766" | Simbol ρ, tidak italic | ρ=0.77 | P |
| F10 | Pearson | "r=-0.36" | *r* italic; minus pakai en dash | *r*=–0.36, sebut uji apa | P |
| F11 | Dash | 30× " - " | Em dash tanpa spasi (—); minus = en dash | Find/replace | P |
| F12 | Dimensi | 36× "442x442", "3x3" | Tanda kali × | 442×442 | P |
| F13 | Statistik di teks | "Mean (SD) Dice was 0.8442 (0.0493)" | Teks: "(mean 0.84, SD 0.05)"; tabel boleh mean (SD) | Ubah format teks | P |
| F14 | SD | SD populasi | Lazim SD sampel | Hitung ulang dengan ddof=1, nyatakan | P |
| F15 | Presisi desimal | 4 desimal di mana-mana; 183 vs 182.8 mm | Konsisten di abstrak/teks/tabel | Dice 3 desimal, mm 1 desimal, % 1 desimal | P |
| F16 | Ejaan | tumours, optimised, centre, licence, neighbour, emphasise | US English (Merriam-Webster) | Ubah ke tumors, optimized, center, license, neighbor, emphasize | P |
| F17 | Hyperlink di teks | URL GitHub di Data Availability | Tanpa hyperlink di teks; URL jadi referensi + tanggal akses | Jadikan referensi; arsipkan di Zenodo (DOI) | P |
| F18 | Supplementary di naskah | Gambar S1–S11, Tabel S1–S3 di dalam naskah | Supplementary = Multimedia Appendix dengan daftar referensinya sendiri | Pindahkan ke Multimedia Appendix 1–3 | W |
| F19 | Jumlah tabel/gambar | 7 tabel, 10 gambar | Soft limit 5 tabel, 8 gambar | Tabel 7 (2 baris) dijadikan teks; Tabel 1 jadi textbox; gabung Fig 9+10 | P |
| F20 | Panjang | ±11.200 kata | >10.000 kena biaya tambahan | Target ±7.500 kata isi utama | W |
| F21 | Paragraf dobel | P78=P79, P110=P111, P121=P122, P135=P136 | — | Hapus satu dari tiap pasang | W |
| F22 | Gaya skripsi | "RQ1/RQ2" tidak didefinisikan; "This extended version adds…"; "Table 4 is the measurement this study exists to make" | Kalimat lengkap, objektif | Definisikan RQ di akhir Introduction atau hapus; buang paragraf meta | W |
| F23 | Section akhir | Funding masuk Acknowledgments; tanpa Authors' Contributions; tanpa AI disclosure; "Data and Code Availability" | Funding (wajib, terpisah), CoI, Data Availability, Authors' Contributions (CRediT), Abbreviations, AI disclosure di form submit | Tambah section; ganti judul jadi "Data Availability" | W |
| F24 | Abbreviations | 8 entri | Semua yang dipakai ≥3×, alfabetis | Tambah: CLAIM, LCC, NIfTI, PI-RADS, RAS, TOST, TRIPOD+AI, TZ, dll | P |
| F25 | Referensi | Tanpa PMID; judul title case; arXiv tanpa tanggal posting | DOI **dan** PMID; sentence case; preprint dengan tanggal | Pakai style EndNote JMIR; tambah PMID; ubah ke sentence case | P |
| F26 | Klaim tanpa sitasi | "at least one reports the reverse" | — | Sitasi atau hapus | W |
| F27 | Gambar | PNG 6–12 MB tertanam; Fig S9–S11 hanya dibedakan warna (oranye/biru) | Tetap ada di naskah saat review **dan** diunggah terpisah sebagai PNG, maks 1200×1200 px dan 5 MB; caption tanpa "Figure N"; jangan kodekan kategori hanya dengan warna | Resize/ekspor ulang; tambah pola/kontur; caption sentence case | P |
| F28 | Jenis gambar | Fig 5A kemungkinan bar graph sederhana | Bar graph sederhana tidak diizinkan | Jadikan grouped/complex dengan nilai eksak, atau cukup Tabel 4 | P |
| F29 | Tabel | Baris "Macro Dice" dan "Background Dice" diulang di 2 kolom | Header jelas, tanpa shading, portrait | Pindah ke footnote/baris gabungan | S |
| F30 | Software | "PyTorch" tanpa versi | Nama (versi; pengembang) | PyTorch (version x; Meta), MONAI x, nibabel x, Python x | P |
| F31 | Unit | "ng/mL/cc" | SI | ng/mL/mL (atau ng/mL²) | P |
| F32 | Etika | "IRB not required" | Jelaskan alasan dengan merujuk kebijakan institusi/lokal | Rujuk kebijakan UMN; nomor exemption jika ada | P |
| F33 | Checklist pelaporan | Tidak ada | Checklist diunggah sebagai Multimedia Appendix | Isi CLAIM 2024 (kutip baris dari naskah) | W |
| F34 | Diagram alur data | Fig 1 tanpa jumlah kasus | CLAIM: flow data dengan n dan eksklusi | Tambah n per tahap; jelaskan kenapa PROSTATEx = 204 | P |
| F35 | Cover letter | — | Ungkap preprint/skripsi, kesesuaian dengan jurnal | Tulis; sebut overlap dengan skripsi | W |

---

## 4. Audit isi (sains dan statistik)

| # | Masalah | Lokasi | Perbaikan | Fase |
|---|---|---|---|---|
| S1 | Klaim "widely performed and rarely verified" masih asersi | Abstract, Intro P15, Disc P160 | Audit literatur 30–50 paper (search string ditetapkan dulu; 20% dikode ulang oleh dosbing, laporkan κ) → Multimedia Appendix | 3 |
| S2 | Shortcut di ablation adalah bug yang dibuat sendiri, rawan dibilang strawman | Tabel 4 | Tunjukkan bug ini nyata (hasil audit S1, issue tracker MONAI/nnU-Net, literatur konversi DICOM→NIfTI) | 3 |
| S3 | Hasil negatif diklaim dari P>.05 | P86, P151 | TOST dengan margin ditentukan di awal; margin dijangkarkan ke agreement antar-reader (N1) | 2 |
| S4 | 1 seed, n=20 validasi | Tabel 2 | ≥3 seed × 4 konfigurasi atau 5-fold CV di 139 kasus | 2 (Colab) |
| S5 | E3 bukan single-factor terhadap baseline (E3 = E1 + bobot) | P43, Tabel 2 | Tulis "single-factor relative to E1"; perbandingan E3 dengan E1 | 1 |
| S6 | Koreksi multipel tidak konsisten (Bonferroni 0.0167 disebut tapi tidak diterapkan) | P72, P94 | Holm untuk uji per-zona; tonjolkan effect size + 95% CI | 2 |
| S7 | Centroid 8.2 mm (Tabel 4) vs "up to 33 mm" (teks) | P102 | Sebut mean vs max secara eksplisit | 1 |
| S8 | LCC: diagnosis post hoc di external | P129–130 | Tetapkan dulu, validasi di split validasi internal, lalu konfirmasi di **kohort ketiga yang segar** (N2/N3); PROSTATEx hanya sekunder | 2–3 |
| S9 | PSA density tanpa nilai PSA | P117, P164 | Sensitivity analysis di rentang PSA realistis (mis. 2–20 ng/mL); hitung proporsi pasien yang melewati 0.15 | 2 |
| S10 | Bias volume PZ −28.5% tanpa pembanding manusia | Tabel 5 | Bandingkan dengan selisih reader1–reader2 (+19.3%) dan LoA antar-reader | 1–2 |
| S11 | "Regressed" padahal Spearman | P140 | "correlated" | 1 |
| S12 | Prior work cuma 1 pembanding (Adams) | Tabel 7 | Tambah inter-reader + PROMISE12/MSD/nnU-Net dari literatur; opsional: nnU-Net 2D di split yang sama | 2 (opsional) |
| S13 | Test set sudah dipakai 2× | P168 | Tetap diungkap; evaluasi konfirmatori analisis baru di kohort segar | 2–3 |
| S14 | Klaim "safer direction" (lebih banyak biopsi) bersifat normatif | P117, P164 | Hapus "safer"; cukup "toward overestimation" | 1 |
| S15 | Hausdorff antar-reader di interrater.csv kemungkinan HD maksimum, bukan HD95 | N1 | Hitung HD95 reader1 vs reader2 sendiri dari masker | 1–2 |

---

## 5. Rencana kerja

| Fase | Kapan | Isi | Usaha | Mesin | Naik ke |
|---|---|---|---|---|---|
| **1 — Format & tulisan** | Minggu ini | F1–F35 semua; hapus 4 paragraf dobel; definisikan RQ; S5, S7, S11, S14; pindah Supplementary ke Multimedia Appendix; pangkas 15–20%; isi CLAIM 2024; cover letter + AI disclosure + overlap skripsi | 3–4 hari | Lokal | 6.3 |
| **2a — Analisis ulang tanpa training** | Minggu 2 | N1/S10/S15 inter-reader benchmark; TOST + Holm + CI (S3, S6); SD sampel; sensitivity analysis PSA (S9); LCC ditetapkan dulu, divalidasi di validasi internal (S8) | 3–4 hari | Lokal (CPU) | 7.0 |
| **2b — Power** | Minggu 2–3 | 3 seed × 4 konfigurasi, atau 5-fold CV (S4) | 2–5 hari GPU | Colab (notebook .ipynb) | 7.3 |
| **3 — Generalisasi novelty** | Minggu 3–5 | Audit literatur (S1, S2); ablation model-free + round-trip di MSD Task05 dan ≥1 organ lain; konfirmasi LCC di NCI-ISBI 2013; rilis `roundtrip-check` (pip + Zenodo DOI) | 1.5–2 minggu | Lokal | 7.8 |

Kalau waktu terbatas, kerjakan **fase 1 + 2a saja** (±1.5 minggu, tanpa GPU) → ±7.0, sudah layak submit ke JMIR Biomedical Engineering.

---

## 6. Yang perlu dibahas dengan dosbing

1. Target jurnal: JMIR AI vs JMIR Biomedical Engineering (APF 1985 vs 1500 USD; siapa yang membayar → Funding statement).
2. Siapa corresponding author (biasanya dosbing untuk paper mahasiswa).
3. Status test set: tetap diungkap; kohort mana yang jadi konfirmatori untuk analisis baru (rekomendasi: NCI-ISBI 2013, karena PROSTATEx sudah dipakai untuk menemukan LCC).
4. Dosbing jadi coder kedua untuk audit literatur (κ).
5. Pengungkapan AI generatif dan overlap dengan skripsi.

## Sumber
- https://www.jmir.org/author-information/instructions-for-authors
- https://support.jmir.org/hc/en-us/articles/37982552280987 (submission guide)
- https://support.jmir.org/hc/en-us/articles/360019690851 (statistics)
- https://support.jmir.org/hc/en-us/articles/360019504191 (house style)
- https://support.jmir.org/hc/en-us/articles/115001333067 (references)
- https://support.jmir.org/hc/en-us/articles/115002199367 (figures/supplementary)
- https://support.jmir.org/hc/en-us/articles/360021623072 (table/figure limits)
- https://support.jmir.org/hc/en-us/articles/13387268671771 (generative AI)
- https://support.jmir.org/hc/en-us/articles/115002943791 (titles)
- https://support.jmir.org/hc/en-us/articles/360020210151 (degrees)
- https://support.jmir.org/hc/en-us/articles/115001575267 (reporting guidelines)
- https://www.jmir.org/about-journal/article-processing-fees
