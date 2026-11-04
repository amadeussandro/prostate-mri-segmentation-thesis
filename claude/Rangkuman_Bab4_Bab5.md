# Rangkuman Detail Hasil + Bab IV & V

Dikunci ke **`DRAFT - 2611_Skripsi_BenedictAmadeusSandro_103840.pdf`** (65 halaman).
Nomor halaman di sini = halaman PDF, jadi bisa langsung dibuka saat bimbingan.

**Status draft: daftar pustaka sudah tampil.** 43 sitasi bernomor, nol `[?]`.
Masalah kompilasi sudah beres.

---

## Peta halaman cepat

| Bagian | Halaman PDF |
|---|---|
| BAB 1 Pendahuluan | 18 |
| BAB 2 Landasan Teori | 23 |
| BAB 3 Metodologi | 31 |
| **BAB 4 Hasil dan Diskusi** | **41** |
| 4.1 Pemilihan Model pada Data Validasi | 41 |
| 4.2 Performa Segmentasi pada Data Uji | 42 |
| 4.3 Hasil Kualitatif | 43 |
| 4.4 Rekonstruksi 3D dan Validasi Geometris | 44 |
| 4.5 Nilai Setiap Tahap Rekonstruksi | 44 |
| 4.6 Volume Zona Anatomi | 47 |
| 4.7 Validasi Eksternal | 48 |
| 4.8 Pembahasan | 51 |
| 4.9 Keterbatasan Penelitian | 53 |
| **BAB 5 Simpulan dan Saran** | **56** |
| 5.1 Simpulan | 56 |
| 5.2 Saran | 58 |
| DAFTAR PUSTAKA | 60 |

| Tabel | Hal | | Gambar | Hal |
|---|---|---|---|---|
| 2.1 Penelitian terdahulu | 29 | | 3.1 Alur penelitian | 31 |
| 3.1 Empat konfigurasi | 35 | | 3.2 Arsitektur U-Net | 34 |
| 4.1 Hasil 4 konfigurasi | 41 | | 3.3 Alur rekonstruksi | 36 |
| 4.2 Uji Wilcoxon validasi | 41 | | 4.1 Kasus baik (002) | 43 |
| 4.3 **Performa data uji** | 42 | | 4.2 Kasus sulit (016) | 44 |
| 4.4 E1 vs baseline | 43 | | 4.3 Ablasi rekonstruksi | 45 |
| 4.5 **Hasil ablasi** | 45 | | 4.4 **Volume ter-mirror** | 46 |
| 4.6 Kesepakatan volume | 47 | | 4.5 Bland–Altman volume | 47 |
| 4.7 Internal vs eksternal | 48 | | 4.6 Validasi eksternal | 49 |
| 4.8 Dice CG per paruh | 51 | | | |

---

# BAB IV — HASIL DAN DISKUSI

## 4.1 Pemilihan Model pada Data Validasi (hal. 41)

**Tabel 4.1** — empat konfigurasi pada 20 kasus validasi yang sama:

| Konfigurasi | Intervensi | Epoch | Dice CG | Dice PZ | **Dice makro** |
|---|---|---|---|---|---|
| Baseline | Cross entropy | 77 | 0,8642 | 0,7375 | 0,8008 |
| **E1 ← terpilih** | + SoftDice (CG, PZ) | **88** | 0,8662 | 0,7527 | **0,8094** |
| E2 | + augmentasi | 70 | 0,8636 | 0,7327 | 0,7982 |
| E3 | + pembobotan kelas CE | 99 | 0,8652 | 0,7510 | 0,8081 |

**Yang harus diucapkan, bukan sekadar ditunjuk:**

Rentang antar konfigurasi hanya **0,0112**. Simpangan baku per kasus **di dalam
satu konfigurasi** 0,0497–0,0661 — beberapa kali lebih besar. Jadi perbedaan
antar pasien jauh melampaui perbedaan antar intervensi.

E1 dipilih karena **tertinggi**, bukan karena terbukti unggul. Aturannya
ditetapkan sebelum data uji diperiksa: Dice makro tertinggi pada validasi, dan
Dice makro = rerata CG dan PZ saja (background dilaporkan terpisah supaya
kelas yang mendominasi voxel tidak menyamarkan performa dua zona yang jadi
perhatian).

**Tabel 4.2** — uji Wilcoxon berpasangan pada validasi:

| Pasangan | Δ makro | p |
|---|---|---|
| E1 vs Baseline | +0,0086 | 0,0192 |
| E1 vs E2 | +0,0112 | 0,0484 |
| E3 vs E1 | −0,0013 | 0,3300 |
| E2 vs Baseline | −0,0026 | 0,3884 |

**Sebut eksplisit: ini deskriptif semata, dan sirkular** — validasi adalah data
tempat pemilihan model dilakukan, jadi uji di atasnya tidak boleh dibaca
sebagai bukti keunggulan.

---

## 4.2 Performa Segmentasi pada Data Uji (hal. 42) — JAWABAN RQ1

**Tabel 4.3** — E1 dibekukan, 19 kasus uji, ruang voxel asli:

| Metrik | Central gland | Peripheral zone |
|---|---|---|
| **Dice** | **0,8442 ± 0,0493** | **0,6852 ± 0,1286** |
| IoU | 0,7333 ± 0,0703 | 0,5339 ± 0,1313 |
| HD95 (mm) | 6,381 ± 3,109 | 6,538 ± 4,041 |
| ASD (mm) | 1,560 ± 0,634 | 1,427 ± 0,808 |
| Precision | 0,8646 ± 0,0542 | 0,8297 ± 0,0746 |
| Recall | 0,8318 ± 0,0856 | **0,6090 ± 0,1585** |

- **Dice makro 0,7647**
- Selang kepercayaan 95% CG: **0,8213–0,8642**
- Background 0,9923 ± 0,0036 — dilaporkan terpisah, tidak masuk makro
- Dice makro per kasus: **0,5077 sampai 0,8392**

**Temuan kualitatif terpenting — hafal ini:**

> PZ punya **precision tinggi (0,8297) tapi recall rendah (0,6090)**. Model
> jarang salah menyebut PZ, tapi terlalu sedikit menyebutnya. Zona itu
> **tersegmentasi kurang, bukan salah letak.**

Pola ini berulang di seluruh penelitian: di volume (−28,5%), dan menguat di
kohort eksternal (recall 0,5076, precision naik 0,8555).

**Tabel 4.4** — E1 vs baseline, 19 kasus yang sama:

| Besaran | Δ | W | p | E1 lebih baik |
|---|---|---|---|---|
| Dice makro (**ditetapkan di muka**) | +0,0098 | 47 | **0,055** | 13/19 |
| Dice PZ (*post hoc*) | +0,0177 | 37 | 0,018 | 16/19 |
| Dice CG (*post hoc*) | +0,0019 | 75 | 0,441 | 11/19 |

**Ini bagian paling rawan salah ucap.** Uji yang ditetapkan di muka
**tidak mencapai ambang konvensional**. Yang benar diucapkan: *"arahnya
konsisten, lebih baik pada 13 dari 19 kasus."* Uji per-zona tidak ditetapkan di
muka; dengan tiga pembandingan ambang Bonferroni 0,0167, jadi p = 0,018
dilaporkan sebagai **eksploratif**.

---

## 4.3 Hasil Kualitatif (hal. 43)

Tiga kasus perwakilan dipilih berdasarkan Dice makro per kasus: **002** (baik),
**006** (median), **016** (menantang).

- **Gambar 4.1** (hal. 43) — kasus 002: T2, anotasi pakar, prediksi, peta selisih
- **Gambar 4.2** (hal. 44) — kasus 016, susunan panel sama

Yang dikatakan: pemeriksaan visual sejalan dengan pola kuantitatif. Kesalahan
terkonsentrasi **di batas**, dan pada PZ bentuknya wilayah acuan yang
**tidak diklaim** model — bukan wilayah yang diklaim di tempat salah.

---

## 4.4 Rekonstruksi 3D dan Validasi Geometris (hal. 44)

- **19 dari 19** kasus lolos seluruh butir daftar periksa geometris
- Selisih absolut maksimum matriks affine: **0,0 tepat**
- Orientasi seluruh kasus LPS, himpunan label sesuai
- Uji fidelitas round-trip: **LULUS**, pemulihan eksak

**Dua hal yang harus disampaikan terbuka di bagian ini:**

**Pertama**, round-trip Dice = 1,0 itu **kepastian struktural, bukan pengukuran
beruntung**. Jalur mask tidak melibatkan interpolasi sama sekali, jadi pemulihan
eksak adalah konsekuensi matematis transformasi yang lossless. Inilah jawaban
atas pertanyaan Bu Eunike soal "Dice 0,8 bukan 1": ada **dua besaran berbeda**
yang dua-duanya disebut Dice. Yang 1,0 itu uji lulus/gagal dan sekarang
dikeluarkan dari tabel hasil. Yang di tabel adalah Dice segmentasi 3D
sebenarnya, dan itu memang ~0,8.

**Kedua**, dan ini penting untuk integritas: metrik segmentasi pada rekonstruksi
ini **identik bit-per-bit** dengan data uji RQ1, karena keduanya menempuh jalur
kode yang sama dan evaluasi RQ1 memang sudah dihitung di ruang voxel asli.
**RQ2 bukan konfirmasi independen RQ1** — ia pengukuran yang sama dalam bentuk
3D. Ini dinyatakan terbuka di draft.

---

## 4.5 Nilai Setiap Tahap Rekonstruksi (hal. 44–46) — JAWABAN RQ2, TEMUAN UTAMA

**Gerbang kebenaran lolos dulu:** rekonstruksi acuan yang disusun ulang secara
mandiri terbukti **identik bit-per-bit** dengan volume yang ditulis proses
utama, seluruh 19 kasus. Tanpa itu, perbandingan di bawah tidak punya acuan.

**Tabel 4.5** (hal. 45) — prediksi model **identik** di semua perlakuan, yang
berubah hanya tahap rekonstruksinya:

| Jalan pintas | Dice CG | Dice PZ | Sentroid | Volume | **Deteksi** |
|---|---|---|---|---|---|
| *(benar)* | 0,8442 | 0,6852 | 0,0 mm | 0% | — |
| **Tanpa reorientasi** | **0,5960** | **0,1758** | 8,2 mm | 0% | **0/19** |
| Pemotongan terpusat naif | 0,8442 | 0,6852 | 0,0 mm | 0% | 0/19 |
| **Affine identitas** | 0,8442 | 0,6852 | **182,8 mm** | **47%** | 19/19 |
| Urutan penumpukan loader | 0,8442 | 0,6852 | 0,0 mm | 0% | 0/19 |

### Dua baris yang jadi inti skripsi

**Tanpa reorientasi** → volume **ter-mirror secara anatomis**. Dice jatuh dari
0,8442 ke 0,5960 (CG) dan 0,6852 ke 0,1758 (PZ), PZ bergeser hingga 33,4 mm.
Tapi daftar periksa geometris menangkap **0 dari 19**. Alasannya langsung:
checklist memeriksa **header**, header-nya benar, hanya data voxelnya yang
salah.

**Affine identitas** → pola **berkebalikan**. Voxelnya tidak tersentuh, jadi
Dice **sama sekali tidak berubah** — karena Dice dihitung pada indeks voxel dan
tidak bisa melihat affine. Padahal kelenjar bergeser **182,8 mm** dan setiap
volume zona menggelembung **47%**. Di sini checklist menangkap **19 dari 19**.

> **Kedua mode kegagalan persis saling melengkapi, dan tidak ada satu
> pemeriksaan pun yang menemukan keduanya.**

Itulah argumen untuk menjalankan kedua jenis pemeriksaan, dan itulah alasan
keberadaan uji round-trip yang membandingkan voxel alih-alih header.

- **Gambar 4.3** (hal. 45) — Panel A: akurasi terlapor tiap jalan pintas.
  **Panel B**: matriks baris "Overlap metrics" vs "Geometry checklist" × kolom
  tiap jalan pintas. Panel B inilah visual komplementaritasnya.
- **Gambar 4.4** (hal. 46) — satu kasus di bawah rekonstruksi benar vs tanpa
  reorientasi, **dengan putusan checklist tercetak di masing-masing**.
  **Ini gambar terpenting di skripsi.**

### Dua jalan pintas yang aman — tapi alasannya beda

Jangan sampai terbaca sebagai "berarti tahap itu tidak perlu":

- **Pemotongan terpusat naif aman secara struktural** — pipeline ini selalu
  menaruh padding di tengah, jadi crop terpusat tidak bisa menyimpang dari
  besaran tercatat untuk kasus yang di-pad. Tapi ia **tetap tidak bisa membalik
  pemotongan maju**, yang akan muncul pada kohort dengan citra lebih besar.
- **Urutan penumpukan loader aman hanya secara kebetulan** — loader yang
  dipakai memang mengeluarkan irisan dalam urutan aksial. Itu sifat *loader*,
  bukan sifat rekonstruksi, dan bisa berubah tanpa perubahan apa pun pada kode
  rekonstruksi.

---

## 4.6 Volume Zona Anatomi (hal. 47) — JEMBATAN KE KLINIS

**Tabel 4.6** — kesepakatan Bland–Altman:

| Besaran | Rerata acuan | Bias | % acuan | LoA 95% |
|---|---|---|---|---|
| Central gland | 35,72 mL | −1,48 | −4,2% | −10,82 … +7,85 |
| **Peripheral zone** | 17,88 mL | **−5,10** | **−28,5%** | −16,53 … +6,34 |
| Seluruh kelenjar | 53,60 mL | −6,58 | −12,3% | −20,18 … +7,02 |

Kecenderungan **sistematis ke arah terlalu kecil**, paling kuat di PZ —
terlalu kecil pada **18 dari 19** kasus. Inilah recall rendah Tabel 4.3 yang
dinyatakan dalam mililiter.

**Gambar 4.5** (hal. 47) — diagram Bland–Altman: selisih vs rerata untuk CG, PZ,
dan seluruh kelenjar, dengan garis bias dan ±1,96 SD.

### Konsekuensi klinis — ini yang Bu Eunike minta

Densitas PSA = PSA ÷ volume kelenjar. Volume di **penyebut**, jadi perkiraan
terlalu kecil **menaikkan** densitas:

- Rerata **+14,7%**
- Terburuk **+50,3%**
- Berarah ke atas pada **15 dari 19** kasus
- Terhadap ambang 0,15 ng/mL/cc, densitas sebenarnya **0,131** sudah terbaca
  sebagai 0,15

Arahnya ke **lebih banyak** biopsi — secara klinis lebih aman daripada
sebaliknya, tapi tetap kesalahan kalibrasi yang nyata.

**Yang dilaporkan harus batas kesepakatannya, bukan biasnya.** Untuk satu
pasien, rentang ±14 mL pada kelenjar ~54 mL jauh lebih menentukan daripada
bias rerata −6,58 mL yang berlaku untuk populasi.

---

## 4.7 Validasi Eksternal (hal. 48–51)

**204 pemeriksaan PROSTATEx**, inferensi saja, model dibekukan, tanpa adaptasi
apa pun — tanpa pelatihan, penyetelan halus, pencarian ambang, maupun pemilihan
model.

**Tabel 4.7** (hal. 48):

| Metrik | Internal CG | Eksternal CG | Internal PZ | Eksternal PZ |
|---|---|---|---|---|
| Dice | 0,8442 | 0,8137 | 0,6852 | 0,6247 |
| Precision | 0,8646 | 0,8193 | 0,8297 | **0,8555** |
| Recall | 0,8318 | 0,8229 | 0,6090 | **0,5076** |

- Dice makro eksternal **0,7192** — turun hanya **0,0455** dari 0,7647
- Pola zonal **bertahan dan menguat**: recall PZ turun, precision naik → model
  jadi **lebih konservatif** di luar domain: menyebut lebih sedikit, lebih
  sering benar soal yang disebutnya

**Gambar 4.6** (hal. 49) — Dice internal vs eksternal berdampingan; distribusi
HD95 CG; hubungan HD95–precision.

### Rekonstruksi pada geometri berbeda — uji yang lebih kuat

- **204 dari 204** lolos checklist, affine diff **0,0**
- **7 dari 204** cukup besar sehingga standardisasi in-plane **memotong**
  alih-alih menambah tepi → cabang transformasi yang tidak pernah terpicu data
  pengembangan ikut teruji, dan **nol** kasus kehilangan anatomi teranotasi
- Orientasi **LAS** (vs LPS internal), ditangani transformasi balik yang sama
- Total: rekonstruksi terverifikasi pada **223 pemeriksaan**, dua konvensi
  orientasi, kedua cabang transformasi in-plane

### Ekor berat HD95 — diuji, bukan diklaim

HD95 CG: median 9,0 mm (dekat nilai internal) tapi rerata **20,2 mm**, maksimum
**81,0 mm**, dengan **64 dari 204** kasus melampaui 30 mm — sementara Dice tetap
tinggi. Korelasi HD95 dengan precision **r = −0,36**.

Diuji dengan penguraian komponen terhubung:

- **Seluruh** 64 kasus berekor punya >1 komponen (vs 80% sisanya)
- Komponen di luar yang terbesar hanya **8,3%** volume prediksi tapi berjarak
  rerata **73 mm**
- Menyisakan komponen terbesar: HD95 **20,2 → 6,5 mm** (ekor: **44,5 → 6,8 mm**)
  sementara Dice hanya bergerak 0,8137 → 0,8266

Jadi ekornya **komponen positif-palsu kecil yang berjarak jauh**, bukan batas
yang tergeser merata. Konsisten dengan medan pandang **1,76×** lebih luas
(192 vs 108,8 mm).

**Sengaja TIDAK diterapkan.** Menyisakan komponen terbesar = mengubah metode,
dan memilihnya setelah melihat skor eksternal = menyetel pada kohort eksternal.
Angka yang dilaporkan adalah pipeline tanpa modifikasi.

### Apakah selisih eksternal dari perbedaan anotasi? — TEMUAN BARU

Rujukan eksternal untuk CG turut memuat *anterior fibromuscular stroma* yang
tidak pernah dilatihkan. Wajar diatribusikan ke situ — **dan itu diuji, bukan
diasumsikan.** Karena stroma ada di tepi **anterior**, atribusi itu memunculkan
prediksi spasial, dan kohort internal jadi **kontrolnya**.

**Tabel 4.8** (hal. 51):

| Dice CG | Eksternal | Internal | Selisih |
|---|---|---|---|
| Seluruh kelenjar | 0,8137 | 0,8442 | −0,0304 |
| **Paruh posterior** | 0,7895 | 0,8516 | **−0,0621** |
| **Paruh anterior** | 0,8307 | 0,8366 | **−0,0059** |

**Prediksinya gagal.** 91% selisih di paruh **posterior**, hanya 9% di anterior.
Paruh anterior — justru tempat stroma berada — adalah yang **paling sepakat**
antar kohort. Arah defisit anterior bahkan **berkebalikan** antar kohort
(eksternal −0,0436 vs internal +0,0465, **p = 0,002**). Urutan sama ketika
bidang pembagi diambil pada rentang seluruh kelenjar.

Gerbang orientasi lolos **223/223**: sumbu anterior diambil dari affine tiap
volume dan tandanya diuji secara anatomis (sentroid PZ harus posterior terhadap
sentroid CG).

Bentuk selisihnya ternyata: CG rujukan yang dibiarkan model sebagai **latar** di
ujung posterior kelenjar — **23,2%** voxel terlewat jatuh ke latar vs hanya
**4,3%** ke PZ. Jadi batas kelenjar yang kurang menjangkau di luar domain,
bukan struktur yang gagal dikenali.

**Tidak ada satu angka hasil pun yang berubah.** Yang terbantahkan adalah
*penjelasan* yang ditawarkan naskah, bukan pengukurannya.

---

## 4.8 Pembahasan (hal. 51–53)

### Hasil utama RQ1

2D U-Net baku menyegmentasi CG dengan Dice 0,8442 dan PZ 0,6852, dan tiga
intervensi pelatihan menggeser performa **lebih kecil daripada variasi antar
pasien**. **Paruh kedua kalimat itu yang lebih berguna.**

Pada data dan skala ini, **fungsi kerugian dan kebijakan augmentasi bukan
tempat galat yang tersisa berada**. Faktor pembatasnya ketepatan penarikan batas
pada PZ, dan pola precision tinggi–recall rendah melokalisasinya: model tidak
salah mengira jaringan lain sebagai PZ, melainkan gagal mengklaim cukup
banyak darinya.

### Hasil utama RQ2

Validasi geometris — verifikasi yang paling lazim dilaporkan, itu pun bila
dilaporkan — **tidak memadai**. Masing-masing pemeriksaan buta tepat terhadap
kegagalan yang dideteksi pemeriksaan lainnya.

### Perbandingan dengan penelitian terdahulu

Akurasi di sini **di bawah** implementasi acuan Prostate158: Adams dkk.
melaporkan CG ~0,877 dan PZ ~0,754 dengan U-Net residual **tiga dimensi**.
Selisihnya konsisten dengan perbedaan arsitektur — jaringan 3D bisa memakai
konteks through-plane. **Tidak diajukan sebagai kompetitif.**

Urutan kesulitan antar zona dilaporkan **tidak konsisten** antar penelitian pada
data ini: sebagian melaporkan CG lebih mudah, setidaknya satu sebaliknya. Hasil
di sini menempatkan CG jelas lebih mudah, stabil di kedua kohort. Sebagian
ketidakkonsistenan itu wajar diatribusikan pada **protokol**, bukan model:
agregasi per irisan vs per kasus, evaluasi pada kisi resampling vs ruang voxel
asli, dan pembacaan nilai integer label. Justru itu sebabnya pemetaan label
diverifikasi eksplisit.

Pada sisi rekonstruksi, literatur memperlakukan konsistensi spasial terutama
sebagai sifat model, dan ketika penyusunan ulang dipaparkan, ia umumnya
dipaparkan sebagai langkah alih-alih sesuatu yang **diverifikasi**. Tidak
ditemukan penelitian terdahulu yang mengukur nilai masing-masing operasi dengan
cara menghilangkannya. **Itu kontribusi yang paling bisa dialihkan**, dan tidak
khas prostat: setiap pipeline yang menghasilkan prediksi per irisan lalu
menyusunnya kembali terpapar dua mode kegagalan yang sama.

---

## 4.9 Keterbatasan Penelitian (hal. 53–55)

Enam hal, semuanya dinyatakan terbuka. **Sampaikan sendiri sebelum ditanya** —
itu menunjukkan kendali atas keterbatasan sendiri.

1. **Data dan arsitektur** — satu data publik, 158 pemeriksaan; data uji hanya
   19 kasus. Perkiraan tidak presisi, karena itu selang kepercayaan dilaporkan.
   Arsitektur baku, tidak ada pencarian arsitektur.
2. **Riwayat data uji** — **bukan data yang belum tersentuh.** Pernah dipakai
   dua kali saat pengembangan pipeline, **sebelum** keempat konfigurasi
   dirancang, dengan model berbeda. Aturan pemilihan dikunci di muka dan hanya
   memakai validasi; tidak ada metrik data uji yang berkontribusi. **Diungkap,
   bukan diklaim murni.**
3. **Perbandingan statistik** — uji yang ditetapkan di muka tidak mencapai
   ambang; uji validasi sirkular; uji per-zona post hoc dan tidak akan bertahan
   terhadap koreksi pembandingan berganda.
4. **Pencampuran faktor pada kohort eksternal** — anotator, protokol, pemindai,
   dan medan pandang berbeda sekaligus, tidak dapat diuraikan, di luar satu
   perbedaan yang diuji (stroma). Pipeline juga tidak melakukan resampling
   in-plane, jadi sebagian selisih bisa dari pergeseran skala piksel.
5. **Batas analisis diagnostik** — analisis komponen terhubung menjelaskan ekor
   tapi tidak menetapkan mengapa komponen palsu muncul di tempatnya; analisis
   anterior–posterior melokalisasi selisih tanpa mengukur stroma itu sendiri,
   karena tidak ada mask stroma yang bisa dikurangkan.
6. **Tidak ada reader study** — tidak ada klaim soal kegunaan klinis, kesiapan
   penerapan, maupun dampak pada luaran pasien. Analisis volumetrik menetapkan
   **besar potensi galat, bukan konsekuensi klinisnya dalam praktik**.

---

# BAB V — SIMPULAN DAN SARAN (hal. 56)

## 5.1 Simpulan (hal. 56–58)

Disusun mengikuti kedua pertanyaan penelitian.

### Jawaban RQ1

CG Dice **0,8442 ± 0,0493** (CI 95%: 0,8213–0,8642), PZ **0,6852 ± 0,1286**,
**Dice makro 0,7647**. HD95/ASD: 6,381/1,560 mm (CG) dan 6,538/1,427 mm (PZ).

Tiga simpulan:

1. **CG tersegmentasi andal, PZ kurang andal** — urutan stabil di kedua kohort.
2. **PZ tersegmentasi kurang, bukan salah letak** — precision 0,8297 dengan
   recall 0,6090. Berulang di volume (terlalu kecil pada 18/19 kasus) dan
   menguat di kohort eksternal.
3. **Ketiga intervensi tidak memberi perbedaan berarti** — rentang 0,0112 vs
   simpangan baku per kasus 0,0497–0,0661; uji yang ditetapkan di muka
   Δ +0,0098, **p = 0,055**. Hasil negatif ini menunjukkan fungsi kerugian dan
   augmentasi bukan tempat galat tersisa; faktor pembatasnya ketepatan batas PZ.

### Jawaban RQ2

Rekonstruksi diimplementasikan sebagai pembalikan berurutan setiap transformasi
spasial, diikuti pemulihan affine sumber. Karena jalur mask tanpa interpolasi,
rekonstruksi bersifat **penyusunan ulang voxel yang eksak**.

Verifikasi: **19/19** internal lolos (affine diff 0,0), round-trip lulus eksak;
**204/204** eksternal lolos, mencakup orientasi berbeda dan cabang transformasi
yang tak pernah terpicu — total **223 pemeriksaan**.

Empat simpulan — dan yang terpenting bukan keberhasilan verifikasinya, tapi apa
yang terungkap saat tahapnya dihilangkan:

1. **Validasi geometris semata tidak memadai** — volume ter-mirror lolos seluruh
   butir checklist pada **19/19**, padahal Dice jatuh ke 0,5960/0,1758.
2. **Metrik tumpang tindih semata juga tidak memadai** — affine hilang **sama
   sekali tidak menyentuh** Dice, padahal menggeser kelenjar 182,8 mm dan
   menggelembungkan volume 47%.
3. **Keduanya persis saling melengkapi, dan tidak ada satu pemeriksaan pun yang
   menemukan keduanya** — itulah alasan uji round-trip ada.
4. **Tahap yang tampak aman belum tentu tidak diperlukan** — crop terpusat aman
   *struktural*, urutan loader aman *kebetulan*.

### Konsekuensi klinis terukur

Volume kelenjar terlalu kecil, bias −6,58 mL (−12,3%), LoA −20,18 … +7,02 mL.
Densitas PSA naik rerata **14,7%**, hingga **50,3%**, berarah ke atas 15/19.
Densitas sebenarnya 0,131 sudah terbaca pada ambang 0,15 ng/mL/cc.

### Batas klaim — ucapkan ini

Penelitian ini **tidak** mengusulkan arsitektur baru, **tidak** mengklaim
state-of-the-art, **tidak** mengklaim kesiapan klinis. Akurasi di bawah
implementasi acuan Prostate158 yang memakai jaringan 3D, dan selisihnya
konsisten dengan perbedaan arsitektur. Tidak ada reader study.

## 5.2 Saran (hal. 58–59)

Delapan arah. Yang paling mungkin ditanya Bu Eunike ada di nomor 1, 2, dan 5.

1. **Arahkan perbaikan ke ketepatan batas, bukan fungsi kerugian** — hasil
   negatif RQ1 menunjukkan intervensi tingkat fungsi kerugian sudah mencapai
   batasnya. Yang lebih menjanjikan: fungsi kerugian berbasis jarak permukaan,
   resolusi keluaran lebih tinggi, atau konteks through-plane (2,5D / 3D).
2. **Uji sel 2×2 yang belum dijalankan** — keempat konfigurasi berfaktor
   tunggal, jadi interaksi tidak terukur. **E1+E2 (SoftDice + augmentasi)**
   adalah sel yang belum diuji dan kelanjutan paling wajar secara rancangan
   eksperimen. *(Ini jawaban kalau beliau menanyakan E4.)*
3. **Tangani under-segmentation PZ langsung** — geser titik operasi, misalnya
   kalibrasi ambang keluaran yang ditetapkan dan divalidasi pada data validasi
   **dulu**, bukan setelah melihat skor data uji.
4. **Pisahkan pergeseran skala piksel dari kegagalan model** — proses kedua
   dengan adapter resampling.
5. **Tetapkan penyaringan komponen terhubung di muka, lalu validasi** —
   diagnostik menunjukkan HD95 turun 20,2 → 6,5 mm tanpa merugikan Dice.
   Sengaja tidak diterapkan karena memilihnya setelah melihat skor eksternal =
   menyetel pada kohort eksternal. Bisa diadopsi **secara sah** dengan
   menetapkannya di muka dan memvalidasinya pada validasi internal.
6. **Perluas validasi eksternal dan selaraskan definisi zona** — analisis
   anterior–posterior menunjukkan selisih memusat di posterior; anotasi dengan
   definisi zona yang diselaraskan akan memisahkan generalisasi model dari
   perbedaan anotasi secara lebih tegas.
7. **Lakukan reader study** — penilaian kegunaan klinis menuntut perbandingan
   terhadap pembaca manusia dalam alur kerja sebenarnya.
8. **Jadikan verifikasi rekonstruksi praktik baku** — temuan ini berlaku bagi
   setiap pipeline yang memprediksi per irisan lalu menyusunnya kembali. Uji
   round-trip berbiaya rendah.

---

# Kalau waktunya cuma 10 menit

Buka empat halaman ini, berurutan:

| Urut | Halaman | Yang dikatakan |
|---|---|---|
| 1 | **42** (Tabel 4.3) | "Ini jawaban RQ1, Bu. CG 0,8442, PZ 0,6852. Yang penting PZ-nya: precision tinggi, recall rendah — tersegmentasi kurang, bukan salah letak." |
| 2 | **45** (Tabel 4.5) | "Ini jawaban RQ2 dan temuan utamanya. Prediksi modelnya identik; yang diubah cuma tahap rekonstruksinya." |
| 3 | **46** (Gambar 4.4) | "Dua-duanya lolos seluruh checklist geometris. Yang kanan anatominya tercermin. Secara header tidak ada bedanya." |
| 4 | **47** (Tabel 4.6) | "Dan ini yang Bu minta — kaitan ke kondisi medis. Volume terlalu kecil 12,3%, densitas PSA naik rerata 14,7%." |

Lalu tutup dengan hasil negatif RQ1: rentang antar konfigurasi 0,0112, lebih
kecil daripada variasi antar pasien — **itu temuan, bukan kegagalan.**
