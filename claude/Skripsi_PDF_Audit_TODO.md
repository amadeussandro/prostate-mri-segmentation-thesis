# Audit PDF Skripsi + To-Do List

**Sumber:** `2611_Skripsi_Sandro_103840 (1).pdf` — 66 halaman, hasil kompilasi 2026-10-15
**Metode:** diukur dari artefaknya, bukan ditebak dari sumber. Posisi tiap karakter
dibaca dengan `pdfplumber` lalu dibandingkan terhadap blok teks yang didefinisikan
template (A4, `left=4cm`, `right=3cm` → lebar teks **14,00 cm**, batas kanan
510,24 pt).

---

## A. Yang sudah benar

Supaya jelas apa yang **tidak** perlu disentuh:

- **Seluruh 10 tabel dan 9 gambar tercetak** di PDF. Tidak ada yang hilang.
- **DAFTAR TABEL, DAFTAR GAMBAR, DAFTAR RUMUS, DAFTAR ISI** semuanya terisi
  benar, termasuk ketujuh rumus (Rumus 2.1–2.7) dengan nomor halaman tepat.
- **Identitas sudah masuk**: judul terkunci, nama penulis, dan footer berjalan
  di tiap halaman sudah benar.
- **Tidak ada limpahan vertikal.** Satu-satunya yang berada di bawah blok teks
  adalah footer berjalan dan nomor halaman — keduanya memang dirancang template
  untuk berada di margin bawah.
- **Tidak ada limpahan pada 62 dari 66 halaman.** Masalahnya terlokalisasi,
  bukan menyebar.

---

## B. Temuan

### B.1 DAFTAR PUSTAKA kosong — ini yang paling serius

**Bukti terukur:** 23 penanda sitasi tercetak sebagai `[?]`, dan **nol** sitasi
bernomor `[n]` di seluruh dokumen. Halaman 62 hanya memuat judul
"DAFTAR PUSTAKA" lalu langsung teks instruksi template, kemudian lompat ke
Lampiran. Tidak ada satu pun entri referensi tercetak.

**Penyebab: BibTeX tidak pernah menghasilkan `.bbl` yang terbaca dokumen.**

Perlu saya jelaskan kenapa saya menyimpulkan itu, dan bukan sesuatu yang lain.
Saya menemukan satu cacat nyata di `pustaka.bib` (lihat B.2), tetapi cacat itu
**tidak cukup menjelaskan** gejala ini: cacat tersebut ada pada entri
`ronneberger`, yang merupakan rujukan ke-5 dalam urutan sitasi. Kalau BibTeX
berhasil jalan, empat entri sebelumnya akan tetap tercetak. Kenyataannya nol
entri tercetak dan **semua** sitasi `[?]`, yang hanya terjadi bila `.bbl`
memang tidak ada.

Daftar isi dan nomor halaman sudah benar, artinya `pdflatex` jalan minimal dua
kali. Jadi yang terlewat spesifik: **lintasan BibTeX**.

**Cara memastikan:** di Overleaf buka **Logs and output files** → cari bagian
`bibtex`. Di situ akan terlihat apakah BibTeX tidak dijalankan sama sekali,
atau dijalankan tapi gagal beserta pesan errornya.

### B.2 DOI dengan garis bawah tidak di-escape — cacat dari saya

Entri `ronneberger` di `pustaka.bib`:

```
doi = {10.1007/978-3-319-24574-4_28}
```

Karakter `_` di luar mode matematika memicu error `Missing $ inserted` saat
daftar pustaka disusun. Ini **akan** mematahkan daftar pustaka begitu BibTeX
berhasil jalan, jadi harus diperbaiki sekarang meskipun bukan penyebab B.1.

Audit menyeluruh atas `pustaka.bib` lainnya: 30 entri, tidak ada kunci ganda,
kurung seimbang (322 vs 322), tidak ada BOM, semua entri punya field wajib
IEEEtran. Satu karakter non-ASCII (`©`) ada di field `abstract` entri bawaan
template — tidak dicetak IEEEtran, jadi aman.

### B.3 Teks instruksi template tercetak setelah DAFTAR PUSTAKA

Halaman 62, persis di bawah judul daftar pustaka, tercetak:

> IEEE *citation style* dan jumlah referensi paling sedikit 10 yang dapat
> bersumber dari: buku, jurnal, conference proceeding... Each reference must be
> completed with DOI and can be traced online...

Itu petunjuk template untuk penulis, bukan isi skripsi. Sumbernya ada di
`SKRIPSI.tex` tepat setelah `\bibliography{pustaka}`. Tidak boleh ikut
tercetak pada dokumen yang dikumpulkan.

### B.4 Tiga tabel melimpah ke margin kanan

Diukur dari ujung tinta sebenarnya:

| Halaman | Tabel | Limpahan | Baris pemicu |
|---|---|---|---|
| **47** | **Tabel 4.5** (ablasi rekonstruksi) | **+2,90 cm** | baris header 6 kolom |
| 49 | Tabel 4.6 (kesepakatan volume) | +1,63 cm | header + baris data |
| 43 | Tabel 4.1 (hasil 4 konfigurasi) | +0,29 cm | baris header |

Tabel 4.5 yang paling parah: 6 kolom dengan header panjang (`Jalan pintas`,
`Geser sentroid`, `Galat volume`, `Terdeteksi`) pada lebar teks 14 cm. Ia perlu
menyusut sekitar **21%**.

Tabel 4.2, 4.3, 4.4, 4.7, 4.8, 2.1, dan 3.1 **tidak melimpah** — jangan diubah.

### B.5 Judul melimpah di halaman pernyataan AI

Halaman 4, +0,25 cm. Judul skripsi panjang, dan tata letak halaman itu bawaan
template. Ringan, tapi terlihat karena ada di halaman depan.

### B.6 Halaman depan dan lampiran masih berisi teks template

Bagian-bagian ini belum pernah diisi dan isinya masih contoh/petunjuk template:

- **KATA PENGANTAR** (hal. 10) — masih memuat "(Kata Pengantar dapat
  dikembangkan dan harus meliputi ucapan syukur, tujuan pembuatan tugas
  akhir...)" beserta daftar ucapan terima kasih generik.
- **HALAMAN PERSEMBAHAN/MOTO** (hal. 9) — masih moto contoh template.
- **HALAMAN PERSETUJUAN PUBLIKASI** (hal. 8) — masih memuat catatan
  "**Jika tidak bisa membuktikan LoA jurnal/HKI, saya bersedia...".
- **LAMPIRAN** — Lampiran 3 "Transkrip Wawancara (optional)" dan Lampiran 4
  "Judul yang Panjang Melampaui Satu Baris" masih contoh; DAFTAR KODE memuat
  "Kode A1 Script untuk menghitung jumlah dokumen berdasarkan status", juga
  contoh template.

Semuanya bagian yang memang harus Mas isi sendiri — saya tidak mengarangnya.

### B.7 Field identitas yang belum terisi

`NIM` masih tercetak literal "NIM", NIDN pembimbing masih `0012345600`, dan
tanggal-tanggal administratif masih placeholder. Sudah ditandai `% TODO` di
`settings.tex`.

---

## C. TO-DO LIST

Urut prioritas. Kolom "Milik" menunjukkan siapa yang perlu mengerjakan.

| # | Tugas | Lokasi | Perbaikan | Prioritas | Milik |
|---|---|---|---|---|---|
| 1 | Jalankan lintasan BibTeX | Overleaf | Cek **Logs** → bagian `bibtex`. Kalau tidak jalan: Menu → pastikan *Compiler* `pdfLaTeX`, lalu Recompile; bila perlu klik panah di tombol Recompile → **Recompile from scratch**. Sitasi harus berubah dari `[?]` ke `[1]`–`[20]` | **Tinggi** | Mas |
| 2 | Escape garis bawah di DOI | `pustaka.bib` entri `ronneberger` | Ubah `..._28}` → `...\_28}` | **Tinggi** | saya |
| 3 | Matikan teks instruksi daftar pustaka | `SKRIPSI.tex`, setelah `\bibliography{pustaka}` | Komentari blok "IEEE citation style..." sampai akhir `itemize`-nya | **Tinggi** | saya |
| 4 | Rapikan Tabel 4.5 | `src/2.Isi/Bab4.tex` | Perlu menyusut ~21%: bungkus `\footnotesize`, pendekkan header (`Geser sentroid`→`Sentroid`, `Galat volume`→`Volume`, `Terdeteksi`→`Deteksi`), dan buang kolom pemisah yang tidak perlu | **Tinggi** | saya |
| 5 | Rapikan Tabel 4.6 | `src/2.Isi/Bab4.tex` | Perlu ~12%: `\small` + header `Batas kesepakatan 95\%` → `LoA 95\%` | Sedang | saya |
| 6 | Rapikan Tabel 4.1 | `src/2.Isi/Bab4.tex` | Perlu ~2%: cukup `\small` | Sedang | saya |
| 7 | Isi NIM | `settings.tex` `\nim` | NIM tidak tercatat di repo mana pun; nama folder memuat `103840` tapi tidak saya tebak | Sedang | Mas |
| 8 | Isi NIDN pembimbing | `settings.tex` `\pembimbingNIDN` | NIDN Bu Eunike | Sedang | Mas |
| 9 | Tulis KATA PENGANTAR | `src/1.Awal/06-Kata Pengantar.tex` | Ganti teks petunjuk template dengan isi sebenarnya | Sedang | Mas |
| 10 | Isi halaman persembahan/moto | `src/1.Awal/05-Persembahan.tex` | Ganti moto contoh | Rendah | Mas |
| 11 | Bersihkan halaman publikasi | `src/1.Awal/04-Publikasi.tex` | Hapus catatan "**Jika tidak bisa membuktikan LoA..." | Sedang | Mas |
| 12 | Rapikan lampiran | `src/3.Akhir/lampiran.tex` | Hapus Lampiran 3 & 4 contoh dan contoh kode; sisakan Turnitin + Formulir Bimbingan yang wajib | Sedang | Mas |
| 13 | Isi halaman pernyataan AI | `src/1.Awal/02b-HalamanAI.tex` | Pernyataan pribadi Mas | Sedang | Mas |
| 14 | Isi tanggal administratif | `settings.tex` | Bulan, tanggal sidang, pengumpulan, pengesahan | Rendah | Mas |
| 15 | Judul melimpah di halaman AI | `src/1.Awal/02b-HalamanAI.tex` | +0,25 cm; bisa dibiarkan atau dikecilkan fontnya sedikit | Rendah | opsional |

---

## D. Catatan

Nomor halaman di DAFTAR ISI melompat (iii→v, vi→viii). Sudah saya periksa:
halaman iv dan vii **ada isinya** — keduanya halaman kedua dari bagian
sebelumnya. Jadi itu perilaku normal template, **bukan** bug. Tidak masuk
to-do.

Item 2–6 adalah pekerjaan saya dan bisa saya kerjakan sekarang kalau Mas
mengizinkan. Item 1 harus Mas lakukan di Overleaf karena saya tidak punya akses
ke log kompilasinya. Sisanya isi personal yang tidak boleh saya karang.
