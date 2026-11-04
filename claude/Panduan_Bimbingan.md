# Panduan Bimbingan — Apa yang Dijelaskan dan Urutannya

Untuk bimbingan dengan Bu Eunike Endariahna Surbakti, S.Kom., M.T.I.
Disusun 2026-10-21.

---

## 1. Mulai dari mana

**Jangan mulai dari hasil, dan jangan mulai dari draft LaTeX.** Mulai dari
**empat hal yang Bu Eunike minta di bimbingan sebelumnya**, lalu tunjukkan
masing-masing sudah dijawab.

Alasannya sederhana: itu agenda yang beliau tinggalkan, jadi itu yang ada di
kepalanya. Membuka dengan "empat yang Bu minta sudah selesai, saya laporkan
satu-satu" langsung menempatkan pertemuan pada jalur yang beliau harapkan,
dan Mas tidak perlu menebak mau mulai dari mana.

### Kalimat pembuka yang bisa dipakai

> "Bu, di bimbingan terakhir ada empat hal yang Bu minta. Semuanya sudah
> selesai. Saya laporkan satu-satu dulu, lalu masuk ke hasil RQ1 dan RQ2, baru
> draft skripsinya. Ada satu temuan baru yang justru mengoreksi naskah saya
> sendiri, itu saya ceritakan di bagian akhir."

Setelah itu ikuti alur di bagian 3.

---

## 2. Alur 20 menit

| Menit | Bagian | Inti yang disampaikan |
|---|---|---|
| 0–5 | Empat permintaan Bu Eunike | Semua terjawab; satu di antaranya membenarkan keraguan beliau |
| 5–9 | RQ1 — hasil segmentasi | Dice CG 0,8442 / PZ 0,6852; **hasil negatif** sebagai temuan |
| 9–14 | RQ2 — rekonstruksi & ablasi | Validasi geometris saja tidak cukup; ini temuan utamanya |
| 14–17 | Konsekuensi klinis + validasi eksternal | Volume → densitas PSA; 204 kasus, turun hanya 0,0455 |
| 17–19 | Temuan baru: stroma terbantahkan | Menunjukkan kehati-hatian, bukan kelemahan |
| 19–20 | Status draft & yang diminta | 64 halaman, apa yang masih kurang |

---

## 3. Empat permintaan Bu Eunike dan jawabannya

Ini tulang punggung pertemuan. Hafal bagian ini.

### 3.1 "Lanjutkan RQ1 di luar baseline dengan skenario tuning" (21/09)

**Sudah.** Tiga skenario dijalankan di atas baseline, masing-masing mengubah
satu faktor saja:

| Konfigurasi | Intervensi | Dice makro (validasi) |
|---|---|---|
| Baseline | Cross entropy | 0,8008 |
| **E1 ← terpilih** | + SoftDice (CG, PZ) | **0,8094** |
| E2 | + augmentasi | 0,7982 |
| E3 | + pembobotan kelas CE | 0,8081 |

Yang penting disampaikan: **keempatnya hanya terentang 0,0112**, sedangkan
simpangan baku antar pasien di dalam satu konfigurasi 0,0497–0,0661 — beberapa
kali lebih besar. Jadi perbedaan antar pasien jauh melampaui perbedaan antar
intervensi.

E1 dipilih karena **tertinggi**, bukan karena terbukti unggul. Pembedaan ini
penting dan harus diucapkan.

### 3.2 "Referensi paper untuk tiap skenario tuning" (22/09)

**Sudah** — dan di sini ada hal yang sebaiknya Mas sampaikan terus terang:
**E2 dan E3 ternyata hasil nol.** Augmentasi tidak membantu, pembobotan kelas
tidak membantu.

Kalimat yang bisa dipakai:

> "Bu waktu itu agak ragu soal skenario tuning-nya. Hasilnya ternyata
> membenarkan keraguan Bu — dua dari tiga tidak memberi perbedaan berarti. Saya
> laporkan sebagai hasil negatif, bukan dibuang."

Hasil negatif yang dilaporkan terbuka itu justru nilai tambah, bukan kegagalan.
E3 membuktikan intervensi pada tingkat fungsi kerugian memang menggeser titik
operasi PZ tanpa menggeser tumpang tindihnya — artinya batasnya ada di
ketepatan batas, bukan di desain fungsi kerugian.

### 3.3 "RQ2 butuh penjelasan ilmiah yang terkait kondisi medis" (22/09)

**Sudah, dan ini jadi salah satu bagian terkuat.** Rantainya:

galat segmentasi → volume zona (mL) → **densitas PSA** → keputusan biopsi

- Volume seluruh kelenjar diperkirakan terlalu kecil: bias **−6,58 mL
  (−12,3%)**
- PZ paling parah: **−28,5%**, terlalu kecil pada **18 dari 19** kasus
- Densitas PSA = PSA ÷ volume, jadi volume terlalu kecil **menaikkan**
  densitas: rerata **+14,7%**, hingga **+50,3%**
- Terhadap ambang 0,15 ng/mL/cc, densitas sebenarnya **0,131** sudah terbaca
  sebagai 0,15

Arahnya ke **lebih banyak** biopsi — secara klinis lebih aman daripada
sebaliknya, tapi tetap kesalahan kalibrasi yang nyata.

**Yang dilaporkan harus batas kesepakatannya, bukan biasnya.** Untuk satu
pasien, rentang ±14 mL pada kelenjar ~54 mL jauh lebih menentukan daripada
bias rerata.

### 3.4 "Dice 0,8 bukan 1" (22/09)

**Sudah selesai, dan Bu Eunike benar.** Ini perlu dijelaskan hati-hati karena
ada dua besaran berbeda yang dua-duanya disebut "Dice":

| Besaran | Nilainya | Statusnya |
|---|---|---|
| **Round-trip Dice** | 1,0 | Uji **lulus/gagal**, bukan akurasi |
| **Dice segmentasi 3D** | ~0,8 (makro 0,7647) | Ini yang masuk tabel hasil |

Round-trip Dice = 1,0 itu **kepastian matematis**, bukan pengukuran yang
beruntung: jalur mask tidak melibatkan interpolasi sama sekali, jadi pemulihan
eksak adalah konsekuensi transformasi yang lossless. Ia tidak bisa "diganti ke
0,8" tanpa memalsukan.

Kalimat yang bisa dipakai:

> "Naluri Bu benar — yang masuk tabel hasil memang harus yang ~0,8. Ternyata
> ada dua hal berbeda yang dua-duanya saya sebut Dice. Yang 1,0 itu uji
> verifikasi, hasilnya lulus atau gagal, dan sekarang saya keluarkan dari tabel
> hasil. Yang di tabel sekarang Dice segmentasi 3D yang sebenarnya, dan itu
> memang ~0,8."

Dan tambahkan: studi ablasi kini **membuktikan** uji round-trip itu satu-satunya
gerbang yang menangkap volume ter-mirror. Jadi pertanyaan Bu menuntun ke
temuan.

---

## 4. RQ1 — hasil utama

**Angka yang dihafal:**

| Metrik | CG | PZ |
|---|---|---|
| Dice | **0,8442 ± 0,0493** | **0,6852 ± 0,1286** |
| Precision | 0,8646 | 0,8297 |
| Recall | 0,8318 | **0,6090** |

Dice makro **0,7647**. Selang kepercayaan 95% CG: 0,8213–0,8642.

**Temuan kualitatif terpenting** — hafal ini, kemungkinan besar ditanya:

> PZ punya **precision tinggi tapi recall rendah**. Artinya model jarang salah
> menyebut PZ, tapi terlalu sedikit menyebutnya. Zona itu
> **tersegmentasi kurang, bukan salah letak.**

Pola ini berulang di mana-mana: di volume (−28,5%), dan menguat di kohort
eksternal (recall turun ke 0,5076 sementara precision naik ke 0,8555).

### Yang HARUS dinyatakan terbuka

- Uji Dice makro E1 vs baseline: Δ +0,0098, **p = 0,055 — tidak mencapai ambang
  konvensional.** Sebut "arahnya konsisten, lebih baik di 13 dari 19 kasus",
  **jangan** sebut signifikan.
- Uji per-zona (PZ p = 0,018) **tidak ditetapkan di muka**. Dengan tiga
  pembandingan, ambang Bonferroni 0,0167 → laporkan sebagai **eksploratif**.
- Uji pada data validasi **sirkular** (pemilihan model dilakukan di situ) →
  deskriptif semata.

---

## 5. RQ2 — temuan utama skripsi

Ini bagian yang paling layak dibanggakan. Intinya satu kalimat:

> **Validasi geometris saja tidak cukup untuk membuktikan volume hasil
> rekonstruksi itu benar.**

Buktinya dari studi ablasi — prediksi model identik, yang diubah hanya tahap
rekonstruksinya:

| Jalan pintas | Dice CG | Dice PZ | Terdeteksi checklist |
|---|---|---|---|
| *(benar)* | 0,8442 | 0,6852 | — |
| **Tanpa reorientasi** | **0,5960** | **0,1758** | **0/19** |
| **Affine identitas** | 0,8442 | 0,6852 | 19/19 |

Dua baris itu saling melengkapi secara sempurna:

- Volume **ter-mirror**: Dice jatuh drastis, tapi **checklist geometris
  meloloskannya 0 dari 19** — karena checklist memeriksa header, header-nya
  benar, hanya voxelnya yang salah.
- **Affine hilang**: Dice **sama sekali tidak berubah** — karena Dice dihitung
  pada indeks voxel dan tidak bisa melihat affine — padahal kelenjar bergeser
  **182,8 mm** dan volume menggelembung **47%**.

**Tidak ada satu pemeriksaan pun yang menangkap keduanya.** Itulah alasan uji
round-trip ada.

---

## 6. Penjelasan tiap gambar

Sembilan gambar. Kolom terakhir adalah yang diucapkan saat menunjuk gambarnya.

| Gambar | Isi | Apa yang dikatakan |
|---|---|---|
| **3.1** Alur penelitian | Rantai lengkap: Prostate158 (119/20/19) → prapemrosesan (RAS, z-score, crop/pad 442×442, tanpa resampling through-plane) → ekstraksi irisan 2D → 4 konfigurasi → pemilihan model pada 20 kasus validasi → evaluasi sekali pada 19 kasus uji → rekonstruksi 3D | "Ini peta keseluruhannya, Bu. Yang penting: parameter transformasi disimpan per kasus supaya bisa dibalik tepat, dan data uji hanya disentuh **sekali** di akhir." |
| **3.2** Arsitektur 2D U-Net | Jalur encoder–decoder dengan skip connection | "Arsitekturnya baku, tidak saya modifikasi. 7.762.531 parameter. U-Net saya perlakukan sebagai komponen, bukan objek penelitian." |
| **3.3** Alur rekonstruksi | Tahap demi tahap: inferensi per irisan → penyusunan ulang berdasarkan **indeks aksial eksplisit** → pembalikan prapemrosesan **urutan terbalik**, nearest-neighbour → **pemulihan affine sumber**, bukan identitas → tulis NIfTI | "Ini yang biasanya cuma disebut satu kalimat di paper lain. Saya uraikan karena tiap tahap bisa gagal sendiri-sendiri, dan kegagalannya tidak mengumumkan diri." |
| **4.1** Kasus baik (002) | T2, anotasi pakar, prediksi, peta selisih | "Kasus terbaik. Kesalahan terkonsentrasi di batas, bukan di tengah zona." |
| **4.2** Kasus sulit (016) | Susunan panel sama | "Kasus terburuk, Dice makro 0,5077. Perhatikan PZ-nya: wilayah acuan yang **tidak diklaim** model, bukan diklaim di tempat salah." |
| **4.3** Ablasi rekonstruksi | **Panel A**: akurasi terlapor di tiap jalan pintas. **Panel B**: matriks — baris "Overlap metrics" vs "Geometry checklist", kolom tiap jalan pintas, menunjukkan pemeriksaan mana melihat kegagalan mana | "Panel B ini inti RQ2 saya, Bu. Lihat polanya: metrik tumpang tindih melihat yang mirror tapi buta pada affine; checklist sebaliknya. Keduanya saling melengkapi persis." |
| **4.4** Volume ter-mirror | Satu kasus yang sama di bawah rekonstruksi benar vs tanpa reorientasi, **dengan putusan checklist tercetak di masing-masing** | "Ini gambar paling penting di skripsi saya. Dua-duanya **lolos** seluruh checklist geometris. Yang kanan anatominya tercermin. Secara header tidak ada bedanya." |
| **4.5** Bland–Altman volume | Selisih vs rerata untuk CG, PZ, seluruh kelenjar; garis bias dan ±1,96 SD | "Biasnya sistematis ke arah terlalu kecil. Tapi yang penting untuk satu pasien batas kesepakatannya, bukan biasnya — rentangnya ±14 mL." |
| **4.6** Validasi eksternal | **Panel A** Dice internal vs eksternal berdampingan; distribusi HD95 CG yang berekor berat; hubungan HD95–precision | "204 kasus PROSTATEx, inferensi saja, model dibekukan. Macro turun hanya 0,0455. HD95-nya berekor — itu saya uji dan terbukti komponen palsu kecil yang berjarak jauh, bukan batas yang salah merata." |

### Kalau waktunya sempit

Tunjukkan **tiga** saja: **3.1** (peta), **4.4** (temuan utama), **4.5**
(konsekuensi klinis). Tiga itu sudah menceritakan keseluruhan skripsi.

---

## 7. Sepuluh tabel — yang perlu disebut

| Tabel | Isi | Perlu disorot? |
|---|---|---|
| 2.1 | Penelitian terdahulu + posisi penelitian ini | Kalau ditanya novelty |
| 3.1 | Empat konfigurasi pelatihan | Ya, saat bagian 3.1 |
| **4.1** | Hasil 4 konfigurasi pada validasi | **Ya** — tunjukkan rentang 0,0112 |
| 4.2 | Uji Wilcoxon pada validasi | Sebut "deskriptif, sirkular" |
| **4.3** | Performa pada data uji | **Ya** — ini jawaban RQ1 |
| 4.4 | E1 vs baseline pada data uji | Ya — sekaligus akui p = 0,055 |
| **4.5** | Hasil ablasi | **Ya** — ini jawaban RQ2 |
| **4.6** | Kesepakatan volume | **Ya** — jembatan ke densitas PSA |
| 4.7 | Internal vs eksternal | Ya, singkat |
| 4.8 | Dice CG per paruh kelenjar | Bagian stroma (bagian 8) |

---

## 8. Temuan baru: hipotesis stroma terbantahkan

Sampaikan ini **sebagai kekuatan, bukan kesalahan.**

Naskah sebelumnya menyatakan sebagian selisih eksternal berasal dari perbedaan
definisi zona — rujukan eksternal memuat *anterior fibromuscular stroma* yang
tidak pernah dilatihkan. Itu masuk akal, jadi saya uji alih-alih
mengasumsikannya.

Karena stroma ada di tepi **anterior**, atribusi itu memunculkan prediksi
spasial yang bisa diuji, dan kohort internal jadi kontrolnya:

| Dice CG | Eksternal | Internal | Selisih |
|---|---|---|---|
| Seluruh kelenjar | 0,8137 | 0,8442 | −0,0304 |
| Paruh **posterior** | 0,7895 | 0,8516 | **−0,0621** |
| Paruh **anterior** | 0,8307 | 0,8366 | **−0,0059** |

**Prediksinya gagal.** 91% selisih ada di paruh posterior, cuma 9% di anterior.
Paruh anterior — justru tempat stroma berada — adalah yang paling **sepakat**
antar kohort. Arah defisitnya bahkan berkebalikan antar kohort (p = 0,002).

Kalimat yang bisa dipakai:

> "Saya sempat menulis bahwa sebagian selisih eksternal itu karena perbedaan
> definisi zona. Lalu saya uji, dan ternyata salah — selisihnya justru di
> posterior, bukan anterior tempat stroma berada. Jadi saya koreksi naskahnya:
> perbedaan definisinya tetap fakta, tapi ia bukan penjelasan selisihnya."

**Tidak ada satu angka hasil pun yang berubah.** Yang terbantahkan adalah
penjelasan yang saya tawarkan, bukan pengukurannya.

---

## 9. Pertanyaan yang mungkin muncul

**"Kenapa 2D, bukan 3D?"**
Citra MRI prostat sangat anisotropik — resolusi in-plane satu orde lebih halus
daripada ketebalan irisan. 2D tidak memaksa interpolasi di sumbu yang
informasinya paling jarang, dan tiap irisan jadi satu sampel. Konsekuensinya
keluarannya per irisan, dan justru itu yang membuat RQ2 ada.

**"Kenapa hasilnya di bawah paper dataset-nya?"**
Diakui terbuka. Adams dkk. melaporkan CG ~0,877 / PZ ~0,754 dengan U-Net
residual **tiga dimensi**. Selisihnya konsisten dengan perbedaan arsitektur:
jaringan 3D bisa memakai konteks through-plane yang tidak terlihat jaringan 2D.
Penelitian ini **tidak** mengklaim kompetitif terhadap acuan itu.

**"Kenapa tidak ada E4?"**
Empat konfigurasi hanya terentang 0,0112. E3 membuktikan intervensi di tingkat
fungsi kerugian menggeser titik operasi PZ tanpa menggeser tumpang tindih —
jadi batasnya ketepatan batas, bukan desain fungsi kerugian. Yang defensibel
sebenarnya E1+E2 (sel 2×2 yang belum dijalankan), dan itu sudah masuk Saran
di Bab 5.

**"Data ujinya benar-benar belum tersentuh?"**
**Tidak, dan itu diungkap di Keterbatasan.** Data uji 19 kasus pernah dipakai
dua kali saat pengembangan pipeline, **sebelum** keempat konfigurasi ini
dirancang, dan dengan model berbeda. Aturan pemilihan untuk perbandingan ini
dikunci di muka dan hanya memakai data validasi. Diungkap, bukan diklaim murni.

**"RQ2 mengkonfirmasi RQ1?"**
**Tidak.** Metrik segmentasinya **identik bit-per-bit** dengan data uji RQ1,
karena keduanya menempuh jalur kode yang sama dan evaluasi RQ1 memang sudah
dihitung di ruang voxel asli. RQ2 adalah pengukuran yang sama dalam bentuk 3D,
bukan konfirmasi independen. Ini dinyatakan terbuka di Bab 4.

**"Kenapa penyaringan komponen terhubung tidak dipakai saja?"**
Karena memilihnya **setelah** melihat skor eksternal berarti menyetel pada
kohort eksternal. Itu akan mengubah metode. Dilaporkan sebagai diagnosis, dan
masuk Saran: boleh diadopsi kalau ditetapkan di muka dan divalidasi pada data
validasi internal dulu.

---

## 10. Status draft LaTeX

**Yang sudah ada:** 64 halaman. Bab 1–5 lengkap, 39 section, 15 subsection,
10 tabel, 9 gambar, 7 rumus. Abstrak ID + EN. 20 referensi ber-DOI. Semua angka
diturunkan dari artefak hasil, tidak ada yang diketik manual.

**Yang masih kurang dan jujur disampaikan kalau beliau membuka PDF-nya:**

1. **Daftar pustaka belum tampil** — sitasi masih `[?]`. Ini masalah teknis
   LaTeX, bukan isi: file akar proyek Overleaf masih versi lama sehingga
   daftar pustaka yang sudah dibuat belum terpanggil. Isi dan 20 referensinya
   sudah ada.
2. KATA PENGANTAR, halaman persembahan, dan lampiran masih teks contoh
   template.
3. NIDN pembimbing dan tanggal-tanggal administratif belum diisi.

**Kalau Bu Eunike bertanya soal daftar pustaka:** jawab terus terang bahwa itu
persoalan konfigurasi kompilasi yang sedang dibereskan, bukan referensinya
belum ada — tunjukkan `pustaka.bib` kalau perlu.

### Yang sebaiknya Mas minta di bimbingan ini

- **NIDN beliau** untuk halaman persetujuan.
- Konfirmasi **judul**: skripsi memakai judul terkunci "Segmentasi Zona Anatomi
  pada Citra MRI Pasien Kanker Prostat menggunakan 2D U-Net dan Rekonstruksi
  3D", sementara naskah jurnal memakai versi berimbuhan "yang Tervalidasi
  Secara Spasial". Tanyakan mana yang beliau mau untuk skripsi.
- Konfirmasi **BAB II dinamai "Landasan Teori"** (bawaan template) sudah sesuai,
  atau program studi mewajibkan "Tinjauan Pustaka".
- Target jadwal sidang, supaya sisa pekerjaan bisa dijadwalkan.

---

## 11. Satu hal yang jangan sampai salah ucap

Jangan pernah menyebut hasil E1 **signifikan secara statistik**. Uji yang
ditetapkan di muka memberi p = 0,055 dan tidak mencapai ambang. Yang benar
diucapkan: **"arahnya konsisten, lebih baik pada 13 dari 19 kasus."**

Kalau beliau mendorong ke arah "berarti E1 lebih bagus ya", jawab:

> "Arahnya konsisten, Bu, tapi ujinya tidak mencapai ambang. Saya tidak mau
> mengklaim lebih dari yang didukung datanya. Justru itu yang jadi temuan RQ1:
> ketiga intervensi tidak menggeser performa melampaui variasi antar pasien."

Sikap itu akan dinilai lebih tinggi daripada klaim yang tidak tertopang.
