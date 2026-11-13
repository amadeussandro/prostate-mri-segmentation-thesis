# Pengaruh ukuran kelenjar terhadap akurasi segmentasi

Pertanyaan dari pembimbing: apakah kelenjar yang lebih kecil tersegmentasi lebih buruk, dan apakah model konsisten di seluruh rentang ukuran?

Diukur dua cara: korelasi volume acuan terhadap Dice (Spearman sebagai statistik utama, karena hubungannya tidak harus linear), dan Dice rerata per tertil ukuran, supaya besarnya terbaca dalam satuan Dice alih-alih satuan korelasi.

### Kohort internal (Prostate158) (n = 19)

| Zona | Spearman r | p | Pearson r | p |
|---|---|---|---|---|
| Central gland | +0.449 | 0.0537 | +0.309 | 0.1984 |
| Peripheral zone | +0.137 | 0.5764 | -0.008 | 0.9738 |
| Seluruh kelenjar (Dice makro) | -0.254 | 0.2933 | -0.151 | 0.5382 |

**Central gland per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 7 | 14.6–27.2 | 0.8296 ± 0.0414 |
| sedang | 6 | 28.3–41.6 | 0.8288 ± 0.0594 |
| besar | 6 | 41.6–77.4 | 0.8766 ± 0.0251 |

**Peripheral zone per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 7 | 9.2–13.8 | 0.7103 ± 0.0679 |
| sedang | 6 | 14.2–16.8 | 0.6085 ± 0.1712 |
| besar | 6 | 17.5–42.8 | 0.7328 ± 0.0940 |

### Kohort eksternal (PROSTATEx) (n = 204)

| Zona | Spearman r | p | Pearson r | p |
|---|---|---|---|---|
| Central gland | +0.766 | 0.0000 | +0.531 | 0.0000 |
| Peripheral zone | +0.279 | 0.0001 | +0.277 | 0.0001 |
| Seluruh kelenjar (Dice makro) | +0.204 | 0.0034 | +0.067 | 0.3398 |

**Central gland per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 68 | 4.6–26.2 | 0.7287 ± 0.1030 |
| sedang | 68 | 26.2–51.5 | 0.8372 ± 0.0495 |
| besar | 68 | 51.6–252.1 | 0.8754 ± 0.0398 |

**Peripheral zone per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 68 | 4.7–16.6 | 0.5835 ± 0.1686 |
| sedang | 68 | 16.7–22.3 | 0.6277 ± 0.1205 |
| besar | 68 | 22.3–45.1 | 0.6629 ± 0.1337 |

## Selisih Dice antara tertil terbesar dan terkecil

| Kohort | Central gland | Peripheral zone |
|---|---|---|
| Internal | +0.0470 | +0.0225 |
| Eksternal | +0.1467 | +0.0794 |

Nilai positif berarti kelenjar besar tersegmentasi lebih baik.

## Apakah galat volume relatif juga lebih besar pada kelenjar kecil?

Pertanyaan ini penting karena densitas PSA membagi dengan volume kelenjar: bila galat relatif ikut membesar saat penyebutnya mengecil, kedua efek akan berlipat. Diuji, bukan diasumsikan.

Korelasi volume acuan terhadap |galat volume %|: Spearman **+0.088** (p = 0.7210, n = 19).

| Tertil | n | Rentang volume (mL) | \|galat volume\| rerata |
|---|---|---|---|
| kecil | 7 | 27.9–42.6 | 13.9% |
| sedang | 6 | 43.4–58.4 | 7.8% |
| besar | 6 | 60.1–91.2 | 15.9% |

**Tidak terbukti.** Korelasinya praktis nol dan pola tertilnya tidak monoton. Jadi efek ukuran yang nyata pada Dice **tidak** diikuti oleh galat volume relatif yang lebih besar pada kelenjar kecil, setidaknya pada 19 kasus internal. Dengan n sekecil ini pengujiannya berdaya rendah, sehingga yang dapat dinyatakan adalah tidak ditemukan bukti, bukan bahwa efeknya tidak ada.
