# Panduan Template LaTeX Skripsi UMN

Peta file, aturan styling, dan hal-hal yang harus dihormati template.
Lokasi: `D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840`

---

## 1. Peta file

| File | Peran |
|---|---|
| **`SKRIPSI.tex`** | **Entry point.** Mengatur urutan seluruh bagian. Jangan diubah kecuali menambah/menghapus bab. |
| `report1.cls` | Kelas dokumen (turunan `report` standar LaTeX). Jangan disentuh. |
| `_internals/umnthesis.sty` | Konfigurasi utama kampus: margin, heading, caption, daftar-daftar, makro. Jangan disentuh. |
| `_internals/hype.indonesia.tex` | Aturan pemenggalan suku kata Indonesia. |
| **`settings.tex`** | **Metadata dokumen** — judul, penulis, NIM, pembimbing, tahun, nama tiap bab. Di sinilah identitas skripsi diisi. |
| `istilah.tex` | Makro istilah (`\license`, `\bslash`). Bisa ditambah. |
| **`pustaka.bib`** | **Bibliografi aktif.** `SKRIPSI.tex` memanggil `\bibliography{pustaka}`, jadi file di root inilah yang terpakai. |
| `references.bib`, `src/3.Akhir/pustaka.bib` | **Tidak terpakai.** Jangan diisi — mudah tertukar. |
| `IEEEtranDOI.bst` | Style sitasi aktif (`\bibliographystyle{IEEEtranDOI}`). Menampilkan DOI. |
| `src/1.Awal/` | Halaman depan: sampul, judul, plagiat, **pernyataan AI**, persetujuan, publikasi, persembahan, kata pengantar, abstrak ID + EN. |
| `src/2.Isi/Bab1..Bab5.tex` | Isi skripsi. Bawaan template berisi tutorial LaTeX, bukan konten skripsi. |
| `src/3.Akhir/lampiran.tex` | Lampiran. |
| `assets/pics/` | Semua gambar. |
| `Makefile` | Kompilasi lokal. |

---

## 2. Nama bab diatur dari `settings.tex`

Bab di-`\input` tanpa judul literal; judulnya dari variabel:

```latex
\Var{\babSatu}{Pendahuluan}
\Var{\babDua}{Landasan Teori}
\Var{\babTiga}{Metodologi Penelitian}
\Var{\babEmpat}{Hasil dan Diskusi}
\Var{\babLima}{Simpulan dan Saran}
```

Di file bab, judul dipanggil `\chapter{\babSatu}`. **Mengganti nama bab
dilakukan di `settings.tex`, bukan di file bab.**

Catatan: bawaan template menamai BAB II **"Landasan Teori"**, bukan "Tinjauan
Pustaka". Nama itu dipertahankan, dan penelitian terdahulu dimasukkan sebagai
section di dalamnya. Kalau program studi mewajibkan "Tinjauan Pustaka", ubah
satu baris di `settings.tex`.

---

## 3. Aturan styling yang harus dihormati

### Gambar — caption di BAWAH

```latex
\begin{figure}
	\centering
	\includegraphics[width=0.9\textwidth]{assets/pics/fig_nama.png}
	\caption{Judul gambar dengan sentence case, tidak tebal.}
	\label{fig:nama}
\end{figure}
```

- Caption **sentence case**, rata kiri-kanan, **tidak** dicetak tebal.
- Kalau gambar berasal dari sumber lain, tambahkan `{\small Sumber: \cite{Key}}`
  setelah `\label`.
- `\fbox{...}` opsional untuk memberi garis tepi.

### Tabel — caption di ATAS

```latex
\begin{table}
	\centering
	\caption{Judul tabel, sentence case.}
	\label{tab:nama}
	\begin{tabular}{l c c}
		\hline
		... \\
		\hline
	\end{tabular}
\end{table}
```

### Persamaan — wajib dibungkus `\addequation`

```latex
\addequation{Dice}{%
\begin{equation}\label{eq:dice}
	\mathrm{Dice} = \frac{2|P \cap G|}{|P| + |G|}
\end{equation}
}
```

Argumen pertama adalah nama yang muncul di **DAFTAR RUMUS**. Tanpa
`\addequation`, persamaan tidak terdaftar.

### Rujukan silang — pakai makro bahasa

| Makro | Hasil |
|---|---|
| `\pic~\ref{fig:x}` | Gambar 4.1 |
| `\tab~\ref{tab:x}` | Tabel 4.2 |
| `\equ~\ref{eq:x}` | Persamaan 3.1 |

Jangan tulis "Gambar~\ref{...}" manual — pakai makronya supaya konsisten.

### Sitasi

- `\cite{Key}` dengan style IEEEtranDOI → penomoran `[1]`, `[2]`, ...
- Entri masuk ke **`pustaka.bib` di root**.
- Syarat template: **minimal 10 referensi**, tiap entri **wajib ada DOI** dan
  dapat dilacak daring, diutamakan 5 tahun terakhir.

---

## 4. Konvensi penulisan

- **Bahasa Indonesia formal.** Hindari bentuk percakapan.
- Template menyediakan `\saya` → "Penulis". Rujuk diri sebagai *Penulis*, bukan
  "saya"/"kami".
- Istilah asing dicetak miring: `\textit{central gland}`, `\textit{Dice}`.
  Istilah teknis yang sudah lazim (MRI, U-Net, voxel) tidak perlu dimiringkan
  setiap kali.
- Spasi isi `\onehalfspacing` (diatur `SKRIPSI.tex`); jangan diubah per bab.
- `\hyphenpenalty=10000` aktif — LaTeX menghindari pemenggalan kata.

---

## 5. Hal yang TIDAK boleh disentuh

1. `report1.cls` dan `_internals/*.sty` — konfigurasi kampus.
2. Urutan halaman depan di `SKRIPSI.tex`.
3. `src/1.Awal/02b-HalamanAI.tex` — **halaman pernyataan penggunaan bantuan AI.**
   Ini pernyataan pribadi mahasiswa dan harus diisi sendiri oleh Mas, bukan oleh
   pihak lain. Dibiarkan apa adanya.
4. Penomoran roman → arab, diatur `SKRIPSI.tex`.

---

## 6. Yang masih perlu diisi manual di `settings.tex`

Beberapa field tidak bisa ditebak dan harus Mas isi:

- `\nim` — **NIM tidak tercatat di repo mana pun.** Nama folder memuat `103840`,
  tetapi menebak NIM di halaman sampul berisiko, jadi dibiarkan sebagai TODO.
- `\pembimbingNIDN` — NIDN Bu Eunike.
- `\bulan`, `\tanggalPengesahan`, `\tanggalPengumpulan`, `\hariTanggalSidang`,
  `\waktuSidang`, `\tanggalLulus` — tanggal administratif.
- `\ketuaSidang`, `\penguji` beserta NIDN — ditentukan setelah jadwal sidang.

---

## 7. Kompilasi

Urutan standar BibTeX (perlu dua kali `pdflatex` setelah `bibtex` agar nomor
sitasi dan daftar isi stabil):

```bash
pdflatex SKRIPSI && bibtex SKRIPSI && pdflatex SKRIPSI && pdflatex SKRIPSI
```

Ada `Makefile` di root sebagai alternatif. **Belum dikompilasi di sesi ini** —
tidak ada distribusi LaTeX terpasang di mesin ini, jadi kompilasi perlu Mas
lakukan (lokal atau Overleaf).
