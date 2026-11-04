# Ceklis Bimbingan Jurnal

Diperbarui 2026-11-04. Status diukur dari isi naskah, bukan dari niat.

**Legenda:** ✅ selesai · 🟡 sebagian · ⛔ terhambat · ⬜ belum

---

## Daftar permintaan pembimbing

| # | Permintaan | Status | Bukti / yang kurang |
|---|---|---|---|
| 1 | 2D U-Net, jelaskan pemilihannya | ✅ | Tiga alasan eksplisit di Metode jurnal + Bab 3 skripsi: geometri data anisotropik, komponen sengaja baku agar pengukuran rekonstruksi tidak tercampur, konfigurasi sudah ditetapkan penelitian terdahulu |
| 2 | Gambar untuk *image preprocessing* | ⬜ | Belum ada. Perlu diagram tahap prapemrosesan |
| 3 | Pengaruh ukuran GT; model konsisten? | ✅ | Analisis baru. Eksternal CG Spearman **+0,766**, selisih **0,147 Dice** antar tertil. Dugaan perlipatan ke densitas PSA **diuji dan tidak terbukti** |
| 4 | Perilaku klinis 2D→3D, keterbatasan | 🟡 | Rantai volume→densitas PSA ada; Keterbatasan lengkap. **Kurang:** framing "AI sebagai opsi dari tidak ada menjadi ada" belum ditulis |
| 5 | Step-by-step prapemrosesan; pisahkan seg vs rek | 🟡 | Skripsi §3.16 memisahkan tegas ("segmentasi berakhir saat model menghasilkan peta label 2D"). **Kurang:** rincian tujuan tiap tahap prapemrosesan |
| 6 | Perjelas gambar study flow | ⛔ | Gambar baru jauh lebih baik, **tapi arm keempat tertulis "+ Sampling (class balancing)"** sedangkan E3 sebenarnya `ce_class_weights`. Ditahan sampai diperbaiki |
| 7 | Semua gambar/tabel benar-benar dilaporkan hasilnya | ⬜ | Audit belum dijalankan |
| 8 | Minimal 5 paper JMIR, 30 total | ✅ | **29 tersitasi** di kedua dokumen, 5 keluarga JMIR + CLAIM 2024 + TRIPOD+AI. Metadata dari CrossRef |
| 9 | Jelaskan gambar/tabel detail, jangan pasif | 🟡 | Gambar 2 dan 3 sudah dapat caption kaya + paragraf pemandu. **Kurang:** 7 gambar dan 7 tabel sisanya |
| 10 | Tabel perbandingan dengan penelitian lain | ✅ | Tabel 7: CG **96,3%**, PZ **90,9%** dari acuan Adams, dengan alasan arsitektural |
| 11 | Kurangi narasi, perbanyak penjelasan hasil | ⬜ | **Justru bertambah** — EN kini ~7,4k kata. Perlu pemangkasan terarah |
| 12 | Buat ORCID iD | ✅ | `0009-0000-1459-9139` di baris penulis korespondensi, EN + ID |
| 13 | ATM struktur penulisan paper JMIR | ⬜ | Belum dipelajari secara sistematis |
| 14 | Hubungi David Agustriawan | — | **Dilewati** atas permintaan Mas |
| 15 | Publish repo ke GitHub | ✅ | [amadeussandro/prostate-mri-segmentation-thesis](https://github.com/amadeussandro/prostate-mri-segmentation-thesis), README ditulis ulang total |
| 16 | Penjelasan kode di laporan skripsi | ✅ | 3 listing dari source asli di §3.16, masuk DAFTAR KODE |

**Rekap:** 7 selesai · 3 sebagian · 1 terhambat · 4 belum · 1 dilewati

---

## Tambahan di luar daftar (dikerjakan sekalian)

| Pekerjaan | Status |
|---|---|
| Spesifikasi laptop + alasan memakai Colab T4 (skripsi saja) | ✅ |
| Alur kerja VS Code → Colab, 6 langkah (skripsi saja) | ✅ |
| BAB II diperkuat dengan konteks bidang | ✅ |
| Gambar arsitektur & alur rekonstruksi diganti versi baru | ✅ |
| Optimasi gambar 27 MB → 2,5 MB tanpa turun kualitas | ✅ |
| Hipotesis stroma diuji dan **terbantahkan** | ✅ |
| Daftar pustaka skripsi lepas dari BibTeX | ✅ |

---

## Yang menghambat, dan apa yang dibutuhkan

### ⛔ Nomor 6 — gambar study flow

Gambar barunya bagus, tapi memuat satu kotak yang salah:

| Gambar menulis | Seharusnya |
|---|---|
| + Sampling — *sampling strategy (class balancing)* | **+ Class weights** — `ce_class_weights: [0.0456, 1.0000, 2.8477]` |

Config E3 menyatakan eksplisit *"only `ce_class_weights` added"* sebagai variabel bebas. `WeightedRandomSampler` memang ada di kode, tapi itu infrastruktur yang dipakai semua arm, bukan intervensi E3.

Kalau dipakai apa adanya, gambar akan bertentangan dengan Metode — dan itu jenis kesalahan yang dilihat penguji.

### ⬜ Nomor 11 — panjang naskah

Ini sekarang kendala mengikat. EN ~7,4k kata vs lazimnya JMIR 3.000–6.000.

Tambahan hari ini substansi, bukan basa-basi, jadi pemangkasan harus dari tempat lain. Perlu diputuskan apa yang dikorbankan.

---

## Urutan yang disarankan

1. **Nomor 6** — perbaiki satu kotak di gambar study flow (Mas)
2. **Nomor 2** — gambar prapemrosesan (saya)
3. **Nomor 9** — lengkapi penjelasan gambar & tabel sisanya (saya)
4. **Nomor 5 & 4** — rincian tahap prapemrosesan + framing klinis (saya)
5. **Nomor 7** — audit setiap gambar/tabel punya hasil (saya)
6. **Nomor 13** — pelajari struktur JMIR (saya)
7. **Nomor 11** — pemangkasan, **paling akhir**, setelah isi final
