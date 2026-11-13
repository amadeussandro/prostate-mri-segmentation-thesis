# Pengaruh ukuran kelenjar terhadap akurasi segmentasi

Pertanyaan dari pembimbing: apakah kelenjar yang lebih kecil tersegmentasi lebih buruk, dan apakah model konsisten di seluruh rentang ukuran?

Diukur dua cara: korelasi volume acuan terhadap Dice (Spearman sebagai statistik utama, karena hubungannya tidak harus linear), dan Dice rerata per tertil ukuran, supaya besarnya terbaca dalam satuan Dice alih-alih satuan korelasi.

### Kohort internal (Prostate158) (n = 19)

| Zona | Spearman r | p | Pearson r | p |
|---|---|---|---|---|
| Central gland | +0.405 | 0.0852 | +0.300 | 0.2117 |
| Peripheral zone | +0.179 | 0.4636 | -0.058 | 0.8121 |
| Seluruh kelenjar (Dice makro) | -0.218 | 0.3710 | -0.264 | 0.2751 |

**Central gland per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 7 | 14.6–27.2 | 0.8475 ± 0.0427 |
| sedang | 6 | 28.3–41.6 | 0.8522 ± 0.0582 |
| besar | 6 | 41.6–77.4 | 0.8893 ± 0.0316 |

**Peripheral zone per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 7 | 9.2–13.8 | 0.7268 ± 0.0844 |
| sedang | 6 | 14.2–16.8 | 0.6429 ± 0.1641 |
| besar | 6 | 17.5–42.8 | 0.7587 ± 0.1048 |

### Kohort eksternal (PROSTATEx) (n = 204)

| Zona | Spearman r | p | Pearson r | p |
|---|---|---|---|---|
| Central gland | +0.738 | 0.0000 | +0.480 | 0.0000 |
| Peripheral zone | +0.248 | 0.0003 | +0.257 | 0.0002 |
| Seluruh kelenjar (Dice makro) | +0.174 | 0.0129 | +0.064 | 0.3638 |

**Central gland per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 68 | 4.6–26.2 | 0.7433 ± 0.1223 |
| sedang | 68 | 26.2–51.5 | 0.8527 ± 0.0510 |
| besar | 68 | 51.6–252.1 | 0.8852 ± 0.0421 |

**Peripheral zone per tertil volume acuan**

| Tertil | n | Rentang volume (mL) | Dice rerata |
|---|---|---|---|
| kecil | 68 | 4.7–16.6 | 0.6005 ± 0.1862 |
| sedang | 68 | 16.7–22.3 | 0.6573 ± 0.1176 |
| besar | 68 | 22.3–45.1 | 0.6794 ± 0.1546 |

## Selisih Dice antara tertil terbesar dan terkecil

| Kohort | Central gland | Peripheral zone |
|---|---|---|
| Internal | +0.0417 | +0.0320 |
| Eksternal | +0.1419 | +0.0789 |

Nilai positif berarti kelenjar besar tersegmentasi lebih baik.

## Apakah galat volume relatif juga lebih besar pada kelenjar kecil?

Pertanyaan ini penting karena densitas PSA membagi dengan volume kelenjar: bila galat relatif ikut membesar saat penyebutnya mengecil, kedua efek akan berlipat. Diuji, bukan diasumsikan.

Korelasi volume acuan terhadap |galat volume %|: Spearman **+0.140** (p = 0.5666, n = 19).

| Tertil | n | Rentang volume (mL) | \|galat volume\| rerata |
|---|---|---|---|
| kecil | 7 | 27.9–42.6 | 12.6% |
| sedang | 6 | 43.4–58.4 | 7.5% |
| besar | 6 | 60.1–91.2 | 15.8% |

**Tidak terbukti.** Korelasinya praktis nol dan pola tertilnya tidak monoton. Jadi efek ukuran yang nyata pada Dice **tidak** diikuti oleh galat volume relatif yang lebih besar pada kelenjar kecil, setidaknya pada 19 kasus internal. Dengan n sekecil ini pengujiannya berdaya rendah, sehingga yang dapat dinyatakan adalah tidak ditemukan bukti, bukan bahwa efeknya tidak ada.
