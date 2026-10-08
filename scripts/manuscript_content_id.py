"""Indonesian JMIR manuscript content.

A translation of `manuscript_content_en`, with the same block structure so the
same builder renders it. Numbers come from `manuscript_numbers`, so the two
language versions cannot disagree with each other or with the artifacts.

Established technical terms are kept in English where translating them would be
unnatural in Indonesian academic writing (Dice, batch size, learning rate,
affine), with the Indonesian term given at first use where one exists.
"""

from __future__ import annotations

from typing import List, Tuple

TITLE = ("Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat "
         "Menggunakan 2D U-Net dan Rekonstruksi 3D yang Tervalidasi Secara Spasial")

AUTHORS = ["Benedict Amadeus Sandro", "Eunike Endariahna Surbakti, S.Kom., M.T.I."]
AFFIL = ["Program Studi Informatika, Fakultas Teknik dan Informatika, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Penulis korespondensi: Benedict Amadeus Sandro, Program Studi Informatika, "
        "Fakultas Teknik dan Informatika, Universitas Multimedia Nusantara, "
        "Tangerang, Banten, Indonesia")

FIG_DIR_RQ2 = "results/figures_rq2"
VIZ = "results/rq2_e1_epoch88/visualizations"


def build(N: dict) -> List[Tuple[str, object]]:
    a = N["arms"]
    vt = N["validation_tests"]
    t = N["test"]
    tb = N["test_vs_baseline"]
    g = N["geometry"]
    ab = N["ablation"]
    v = N["volumes"]
    e = N["external"]
    eg = e["geometry"]

    cg_d = t["per_class"]["cg"]["dice"]
    pz_d = t["per_class"]["pz"]["dice"]
    bg_d = t["per_class"]["background"]["dice"]
    v1, v3 = ab["V1_no_reorientation"], ab["V3_identity_affine"]

    B: List[Tuple[str, object]] = []
    add = B.append

    add(("title", TITLE))
    add(("authors", AUTHORS))
    add(("affil", AFFIL))
    add(("corr", CORR))

    # ------------------------------------------------------------- abstrak
    add(("h2", "Abstrak"))
    add(("abs", ("Latar Belakang",
        "Kanker prostat merupakan salah satu keganasan yang paling sering didiagnosis pada "
        "pria, dan MRI multiparametrik telah menjadi modalitas utama dalam deteksi serta "
        "penentuan stadiumnya. Interpretasi citra bergantung pada anatomi zonal kelenjar, "
        "karena central gland (CG) dan peripheral zone (PZ) berbeda baik dalam penampakan "
        "maupun dalam prevalensi dan signifikansi tumor yang timbul padanya. Segmentasi zonal "
        "otomatis karena itu menjadi kebutuhan praktis bagi pembacaan berbantuan komputer. "
        "Jaringan dua dimensi tetap menarik untuk MRI prostat yang sangat anisotropik, namun "
        "keluarannya berupa potongan per irisan, dan penyusunan kembali keluaran tersebut "
        "menjadi volume 3D yang valid secara spasial merupakan tahap implementasi yang lazim "
        "dilakukan tetapi jarang diverifikasi.")))
    add(("abs", ("Tujuan",
        "Penelitian ini memiliki dua tujuan: mengevaluasi 2D U-Net untuk segmentasi zona "
        "anatomi pada MRI prostat di bawah protokol pemilihan model yang ditetapkan di muka, "
        "serta mengimplementasikan dan memvalidasi rekonstruksi prediksi per irisan tersebut ke "
        "ruang voxel tiga dimensi aslinya. Tujuan tambahan adalah mengukur biaya tahap "
        "rekonstruksi ketika operasi-operasi penyusunnya dihilangkan, dan menyatakan hasilnya "
        "dalam satuan yang bermakna secara klinis.")))
    add(("abs", ("Metode",
        f"Penelitian menggunakan dataset Prostate158 dengan pembagian resmi pada tingkat pasien "
        f"(119 pelatihan, {a['e1']['n']} validasi, {t['n_cases']} uji held-out). Empat "
        "konfigurasi pelatihan faktor-tunggal dibandingkan: baseline cross-entropy, penambahan "
        "suku soft Dice pada CG dan PZ, penambahan augmentasi geometrik dan intensitas, serta "
        "penambahan cross-entropy berbobot kelas. Konfigurasi dengan macro Dice tertinggi pada "
        "data validasi dipilih sebelum data uji diperiksa, dibekukan, dan selanjutnya "
        "diidentifikasi melalui nilai hash SHA-256 dari checkpoint-nya. Prediksi per irisan "
        "disusun kembali berdasarkan indeks aksial eksplisit, ditransformasi balik, dan ditulis "
        "dengan affine sumber dipulihkan, lalu dinilai per kasus di ruang voxel asli terhadap "
        "anotasi pakar yang tidak dimodifikasi. Kebenaran rekonstruksi dinilai melalui dua cara "
        "yang saling bebas: daftar periksa geometris per kasus, dan uji fidelitas round-trip "
        "yang tidak melibatkan model. Sebuah ablasi menyusun ulang prediksi yang sama dengan "
        "operasi rekonstruksi tertentu dihilangkan. Volume zona dihitung dalam mililiter dan "
        "dibandingkan melalui analisis Bland-Altman. Model beku akhirnya diterapkan satu kali "
        f"pada kohort independen berisi {e['n_cases']} pemeriksaan PROSTATEx.")))
    add(("abs", ("Hasil",
        f"Pada data validasi, keempat konfigurasi hanya terentang {N['arm_spread']:.4f} macro "
        f"Dice, lebih kecil daripada simpangan baku antar kasus di dalam konfigurasi mana pun "
        f"({N['arm_sd_range'][0]:.2f}-{N['arm_sd_range'][1]:.2f}). Konfigurasi terpilih mencapai "
        f"rerata (SB) Dice tingkat volume sebesar {cg_d['mean']:.4f} ({cg_d['sd']:.4f}) untuk CG "
        f"dan {pz_d['mean']:.4f} ({pz_d['sd']:.4f}) untuk PZ pada data uji, dengan macro Dice "
        f"{t['macro']['mean']:.4f}. Seluruh {g['n_all_ok']} dari {g['n']} rekonstruksi lolos "
        f"setiap pemeriksaan geometris, dengan selisih absolut affine maksimum "
        f"{g['affine_max_abs_diff']:.1f}. Ablasi menunjukkan bahwa menghilangkan tahap inversi "
        f"orientasi menghasilkan volume yang tercermin secara anatomis namun tetap lolos seluruh "
        f"daftar periksa geometris, sementara Dice yang dilaporkan turun menjadi "
        f"{v1['cg']['dice_vs_gt']:.4f} (CG) dan {v1['pz']['dice_vs_gt']:.4f} (PZ); menghilangkan "
        f"pemulihan affine justru tidak mengubah Dice sama sekali namun menggeser kelenjar "
        f"sejauh {v3['cg']['centroid_shift_mm']:.0f} mm dan menggelembungkan seluruh volume "
        f"sebesar {v3['cg']['volume_pct_error']:.0f}%. Volume kelenjar total diremehkan sebesar "
        f"{abs(v['whole']['bias']):.2f} mL ({abs(v['whole']['pct_of_gt']):.1f}% dari rujukan), "
        f"yang menggelembungkan PSA density sebesar {v['psad']['mean_pct']:.1f}%. Pada kohort "
        f"eksternal, macro Dice sebesar {e['macro']['mean']:.4f}.")))
    add(("abs", ("Simpulan",
        "2D U-Net standar menyegmentasi central gland secara andal dan peripheral zone kurang "
        "demikian, dan tiga intervensi pelatihan faktor-tunggal menggeser performa lebih kecil "
        "daripada variasi antar pasien. Temuan pada sisi rekonstruksi lebih dapat "
        "digeneralisasi: validasi geometris dan metrik tumpang tindih masing-masing buta "
        "terhadap kelas kesalahan rekonstruksi yang berbeda, sehingga tidak satu pun memadai "
        "bila berdiri sendiri, dan uji fidelitas round-trip tanpa model adalah yang mendeteksi "
        "volume dengan header benar tetapi voxel salah. Karena volume zona masuk sebagai "
        "penyebut pada PSA density, kesalahan rekonstruksi yang tidak terlihat oleh Dice tetap "
        "dapat mengubah besaran klinis secara substansial.")))
    add(("abs", ("Kata Kunci",
        "kanker prostat; magnetic resonance imaging; segmentasi citra; deep learning; U-Net; "
        "rekonstruksi 3D; peripheral zone; validasi eksternal; reprodusibilitas; PSA density")))

    # -------------------------------------------------------------- pendahuluan
    add(("h2", "Pendahuluan"))
    add(("p",
        "Kanker prostat merupakan salah satu keganasan yang paling sering didiagnosis pada pria "
        "di seluruh dunia sekaligus penyebab utama kematian akibat kanker. MRI multiparametrik "
        "telah menjadi modalitas sentral dalam deteksi, lokalisasi, dan penentuan stadium, dan "
        "kerangka pelaporan terstruktur telah membuat interpretasi citra lebih konsisten. "
        "Interpretasi tersebut disusun berdasarkan anatomi zonal kelenjar prostat."))
    add(("p",
        "Prostat secara konvensional dibagi menjadi peripheral zone dan central gland, dengan "
        "yang terakhir mencakup transition zone dan central zone. Pembedaan ini penting secara "
        "klinis: sebagian besar karsinoma timbul pada peripheral zone, kriteria penilaian lesi "
        "berbeda antar zona, dan rasio volume zonal turut menjelaskan pembesaran jinak. "
        "Segmentasi zonal karena itu bukan sekadar tugas pemartisian abstrak, melainkan "
        "prasyarat bagi sejumlah besaran klinis turunan."))
    add(("p",
        "Deep learning telah menjadi pendekatan baku untuk persoalan segmentasi ini. MRI prostat "
        "bersifat sangat anisotropik, dengan resolusi in-plane satu orde lebih halus daripada "
        "ketebalan irisan, dan jaringan dua dimensi tetap merupakan pilihan yang wajar pada "
        "geometri tersebut: pendekatan ini menghindari interpolasi pada sumbu yang informasinya "
        "paling jarang, dan murah untuk dilatih. Namun keluarannya berupa potongan per irisan. "
        "Apa pun yang memanfaatkan hasil segmentasi sebagai objek tiga dimensi - pengukuran "
        "volume, rendering permukaan, registrasi ke seri lain, atau perencanaan biopsi - "
        "mensyaratkan irisan-irisan tersebut disusun kembali menjadi volume yang menempati ruang "
        "fisik yang benar."))
    add(("p",
        "Penyusunan kembali tersebut lazim diperlakukan sebagai detail implementasi. Padahal ia "
        "bukan satu operasi melainkan rangkaian operasi: setiap irisan harus ditempatkan pada "
        "indeks aksialnya yang benar, prapemrosesan spasial harus dibalik dalam urutan terbalik "
        "dari penerapannya, dan affine sumber harus dipulihkan agar volume berada di posisi yang "
        "ditentukan pemindai. Setiap tahap dapat gagal secara mandiri, dan kegagalannya tidak "
        "mengumumkan dirinya. Volume dengan header benar namun isi tercermin tetap merupakan "
        "berkas NIfTI yang sah; volume dengan voxel sempurna namun affine bawaan tetap "
        "menghasilkan angka ketika volumenya diukur. Publikasi jauh lebih sering melaporkan "
        "metrik yang mengikuti tahap ini daripada melaporkan bagaimana tahap tersebut "
        "diverifikasi."))
    add(("p",
        "Penelitian ini menangani kedua sisi keadaan tersebut pada dataset publik. Pertama, kami "
        "mengevaluasi 2D U-Net untuk segmentasi zonal di bawah protokol pemilihan yang "
        "ditetapkan sebelum data uji diperiksa, dengan membandingkan empat konfigurasi pelatihan "
        "faktor-tunggal. Kedua, kami mengimplementasikan rekonstruksi 2D ke 3D dengan verifikasi "
        "eksplisit pada setiap tahap, dan - bagian yang kami pandang paling dapat "
        "digeneralisasi - mengukur nilai setiap tahap verifikasi dengan menyusun ulang prediksi "
        "yang sama tanpa operasi tertentu, lalu melaporkan apa yang akan dilihat pembaca pada "
        "setiap kasus."))
    add(("p",
        "Kontribusi penelitian ini dirumuskan secara konservatif. Penelitian ini tidak "
        "mengusulkan arsitektur baru dan tidak mengklaim akurasi state-of-the-art; 2D U-Net "
        "adalah komponen, dan reprodusibilitas pipeline merupakan sifat, bukan temuan. "
        "Kontribusinya adalah perbandingan empiris konfigurasi pelatihan di bawah aturan yang "
        "ditetapkan di muka, termasuk hasil negatif bahwa tiga intervensi gagal menggeser "
        "performa melampaui variasi antar pasien, serta demonstrasi terukur bahwa validasi "
        "geometris semata tidak memadai untuk menegakkan bahwa suatu volume hasil rekonstruksi "
        "sudah benar."))

    # ------------------------------------------------------------------ metode
    add(("h2", "Metode"))

    add(("h3", "Desain Penelitian"))
    add(("p",
        "Penelitian ini bersifat eksperimental dan kuantitatif, seluruhnya dilakukan pada citra "
        "retrospektif yang tersedia publik dan telah dinyatakan anonim. Alur kerjanya mencakup "
        "prapemrosesan, ekstraksi irisan 2D, pelatihan empat konfigurasi faktor-tunggal, "
        "pemilihan model pada data validasi, satu kali evaluasi pada data uji, rekonstruksi "
        "prediksi model terpilih ke ruang voxel asli beserta verifikasinya, ablasi tahap "
        "rekonstruksi, analisis volumetrik, dan satu penerapan pada kohort eksternal independen."))
    add(("fig", (f"{FIG_DIR_RQ2}/F0_study_pipeline.png",
        "Gambar 1. Alur kerja penelitian secara keseluruhan. Empat konfigurasi pelatihan "
        "faktor-tunggal dibandingkan pada data validasi; konfigurasi terpilih dibekukan dan "
        "dievaluasi satu kali pada data uji, lalu digunakan untuk rekonstruksi, ablasi, analisis "
        "volumetrik, dan validasi eksternal.")))

    add(("h3", "Dataset"))
    add(("p",
        "Prostate158 digunakan sebagai landasan metodologis. Dataset ini terdiri atas 158 "
        "pemeriksaan MRI prostat biparametrik 3T dengan anotasi anatomi oleh pakar dalam format "
        "NIfTI. Masukan model adalah seri T2-weighted aksial dengan satu kanal. Target "
        "segmentasi adalah mask anatomi pembaca pertama dengan label bilangan bulat, di mana 0 "
        "adalah latar belakang, 1 adalah central gland, dan 2 adalah peripheral zone."))
    add(("p",
        "Makna label bilangan bulat tersebut diverifikasi, bukan diasumsikan, karena pemetaannya "
        "dilaporkan secara tidak konsisten dalam literatur dan pembacaan yang terbalik akan "
        "secara diam-diam menukar kedua zona pada seluruh metrik yang dilaporkan. Verifikasi "
        "menggabungkan inspeksi visual multiplanar terhadap kasus-kasus individual berdasarkan "
        "anatomi yang telah diketahui - peripheral zone membentuk sabit tipis pada aspek "
        "posterior kelenjar di antarmuka rektum, sedangkan central gland merupakan massa besar "
        "yang kerap bernodul dan mengisi bagian tengah serta anterior - dengan konsistensi "
        "morfometrik di seluruh kohort. Seluruh lini bukti saling bersesuaian."))

    add(("h3", "Pembagian Data"))
    add(("p",
        f"Pembagian resmi pada tingkat pasien digunakan tanpa perubahan: 119 pemeriksaan "
        f"pelatihan, {a['e1']['n']} validasi, dan {t['n_cases']} uji held-out. Pembagian pada "
        "tingkat pasien, bukan tingkat irisan, mencegah irisan dari satu pemeriksaan muncul di "
        "lebih dari satu subset. Kami memverifikasi bahwa himpunan pengenal pelatihan dan "
        "validasi saling lepas dan tidak satu pun beririsan dengan pengenal data uji."))

    add(("h3", "Prapemrosesan Citra"))
    add(("p",
        "Setiap operasi geometris diterapkan secara identik pada citra dan mask-nya, hanya "
        "berbeda pada orde interpolasi: linear untuk citra dan nearest-neighbour untuk mask, "
        "sehingga label bilangan bulat tidak pernah tercampur. Volume direorientasi ke kerangka "
        "RAS kanonik, intensitas dinormalisasi z-score per volume setelah pemotongan persentil, "
        "dan grid in-plane distandarkan menjadi 442x442 melalui zero-padding simetris atau "
        "pemotongan terpusat. Tidak dilakukan resampling pada sumbu through-plane, yang "
        "spasinya memang sudah seragam di seluruh kohort; menghindarinya menjaga grid irisan "
        "tetap identik dengan akuisisi dan menghapus satu sumber galat interpolasi dari jalur "
        "rekonstruksi."))
    add(("p",
        "Setiap parameter yang diperlukan untuk membalik operasi tersebut - transformasi "
        "orientasi, offset padding atau pemotongan per sumbu, serta kedalaman sebelum dan "
        "sesudah transformasi - disimpan per kasus, bukan dihitung ulang di kemudian waktu."))

    add(("h3", "Penyiapan Irisan 2D"))
    add(("p",
        "Setiap volume hasil prapemrosesan diuraikan menjadi irisan aksial, dan setiap irisan "
        "ditandai dengan indeks aksial asalnya. Indeks tersebut menyertai irisan melalui proses "
        "inferensi dan menjadi penentu posisi penulisannya kembali saat rekonstruksi, sehingga "
        "kebenaran hasil tidak bergantung pada urutan kebetulan yang dikeluarkan data loader. "
        "Seluruh irisan dipertahankan untuk evaluasi, termasuk irisan yang tidak memuat anatomi "
        "teranotasi."))

    add(("h3", "Arsitektur 2D U-Net"))
    add(("p",
        "Jaringan yang digunakan adalah U-Net encoder-decoder empat tingkat dengan 32 peta fitur "
        "awal, satu kanal masukan, dan tiga kanal keluaran, dengan total 7.762.531 parameter. "
        "Setiap tingkat menerapkan dua konvolusi 3x3 dengan batch normalization dan ReLU; "
        "downsampling menggunakan max pooling 2x2 dan upsampling menggunakan konvolusi transpose "
        "dengan skip connection dari tingkat encoder yang bersesuaian. Dropout sebesar 0,3 "
        "diterapkan pada bottleneck. Lapisan akhir menghasilkan logit per kelas, dan prediksi "
        "per irisan adalah arg-max atas ketiga kelas."))
    add(("fig", ("results/figures_manuscript/from_original/image5.png",
        "Gambar 2. Arsitektur 2D U-Net: encoder-decoder empat tingkat dengan 32 peta fitur awal, "
        "skip connection pada setiap tingkat, dan keluaran tiga kanal yang bersesuaian dengan "
        "latar belakang, central gland, dan peripheral zone.")))

    add(("h3", "Konfigurasi Pelatihan dan Pemilihan Model"))
    add(("p",
        "Empat konfigurasi dilatih. Masing-masing berbeda dari baseline tepat pada satu faktor, "
        "sehingga setiap perbedaan performa dapat diatribusikan pada faktor tersebut, bukan pada "
        "kombinasi perubahan. Baseline menggunakan loss cross-entropy. Konfigurasi kedua "
        "menambahkan suku soft Dice yang dihitung pada kedua kelas foreground dengan bobot setara "
        "terhadap cross-entropy. Konfigurasi ketiga menambahkan augmentasi geometrik dan "
        "intensitas pada baseline tanpa mengubah loss. Konfigurasi keempat menambahkan "
        "cross-entropy berbobot kelas pada konfigurasi kedua, dengan memperbesar bobot peripheral "
        "zone untuk menangani keterwakilannya yang rendah."))
    add(("p",
        "Seluruh pengaturan lain dibuat tetap antar konfigurasi: prapemrosesan identik, "
        "arsitektur identik, AdamW dengan learning rate 0,001 dan weight decay 0,0001, batch size "
        "8, 100 epoch, presisi penuh 32-bit, dan random seed tetap sebesar 42."))
    add(("p",
        "Aturan pemilihan ditetapkan sebelum data uji diperiksa: konfigurasi dengan macro Dice "
        "tertinggi pada 20 kasus validasi akan dipilih, dan tidak ada metrik uji yang turut "
        "menentukan keputusan tersebut. Setelah dipilih, checkpoint dibekukan dan selanjutnya "
        "diidentifikasi melalui nilai hash SHA-256 bobotnya, sehingga setiap analisis berikutnya "
        "dapat dibuktikan menggunakan model yang sama, bukan sekadar berkas dengan nama yang sama."))

    add(("h3", "Evaluasi Segmentasi"))
    add(("p",
        "Evaluasi dilakukan per kasus pada tingkat volume di ruang voxel asli, terhadap anotasi "
        "pakar yang dibaca dari disk tanpa modifikasi. Penilaian di ruang asli, bukan pada grid "
        "hasil prapemrosesan, berarti angka yang dilaporkan menggambarkan volume yang benar-benar "
        "akan diterima oleh pengguna hilir. Agregasi dilakukan per kasus dan tidak pernah "
        "digabung lintas irisan, karena penggabungan demikian akan memberi bobot lebih besar pada "
        "pemeriksaan dengan jumlah irisan banyak."))
    add(("p",
        "Metrik yang dilaporkan adalah Dice, intersection over union, Hausdorff distance "
        "persentil ke-95 (HD95), average surface distance (ASD), precision, dan recall. Jarak "
        "permukaan dihitung dalam milimeter menggunakan spasi voxel masing-masing kasus. Kelas "
        "yang tidak hadir baik pada prediksi maupun anotasi diperlakukan sebagai kesesuaian "
        "trivial; kelas yang hanya absen pada salah satunya membuat jarak permukaan tidak "
        "terdefinisi, dan kasus demikian dikeluarkan dari metrik tersebut alih-alih diberi nilai "
        "sembarang. Macro Dice adalah rerata kedua kelas foreground; Dice latar belakang "
        "dilaporkan terpisah karena memasukkannya akan menggelembungkan ringkasan dengan kelas "
        "yang secara trivial mudah."))
    add(("p",
        "Simpangan baku yang dilaporkan adalah simpangan baku populasi. Selang kepercayaan "
        "merupakan selang persentil bootstrap atas 2.000 resampel dengan seed tetap."))

    add(("h3", "Rekonstruksi 3D"))
    add(("p",
        "Irisan hasil prediksi disusun menjadi volume dengan menuliskan setiap irisan pada indeks "
        "aksial yang dibawanya, tidak pernah dengan menambahkan menurut urutan iterasi; indeks "
        "yang hilang atau ganda memicu galat alih-alih menghasilkan volume yang tampak masuk "
        "akal. Volume tersusun kemudian ditransformasi balik melalui rantai prapemrosesan dalam "
        "urutan terbalik, menggunakan interpolasi nearest-neighbour di seluruh tahap sehingga "
        "label bilangan bulat terjaga persis. Hasilnya ditulis sebagai citra NIfTI yang membawa "
        "affine dan spasi voxel sumber, tidak pernah affine identitas bawaan."))
    add(("fig", (f"{FIG_DIR_RQ2}/F5_reconstruction_workflow.png",
        "Gambar 3. Alur rekonstruksi 2D ke 3D beserta verifikasinya. Prediksi disusun berdasarkan "
        "indeks aksial eksplisit, ditransformasi balik dalam urutan terbalik dengan interpolasi "
        "nearest-neighbour, dan ditulis dengan affine sumber dipulihkan. Dua pemeriksaan yang "
        "saling bebas menyusul: daftar periksa geometris per kasus dan uji fidelitas round-trip "
        "tanpa model.")))

    add(("h3", "Validasi Geometris dan Fidelitas"))
    add(("p",
        "Kebenaran rekonstruksi dinilai melalui dua cara yang saling bebas, yang kami laporkan "
        "terpisah karena keduanya mendeteksi hal yang berbeda."))
    add(("p",
        "Yang pertama adalah daftar periksa geometris per kasus. Geometri dibaca kembali dari "
        "header berkas keluaran itu sendiri, bukan dari variabel yang digunakan untuk "
        "menyusunnya, lalu dibandingkan dengan sumber yang tidak dimodifikasi: bentuk larik, "
        "matriks affine, spasi voxel, kode orientasi, dan himpunan nilai label yang hadir. "
        "Membaca kembali dari header juga menangkap kehilangan presisi yang diperkenalkan oleh "
        "format berkas itu sendiri, yang menyimpan affine dalam presisi tunggal."))
    add(("p",
        "Yang kedua adalah uji fidelitas round-trip, yang sama sekali tidak melibatkan model. "
        "Anotasi pakar dilewatkan maju melalui prapemrosesan yang sama, disusun kembali dan "
        "ditransformasi balik oleh kode yang sama, lalu dibandingkan dengan anotasi aslinya. Pada "
        "konfigurasi ini transformasi mask berupa padding simetris diikuti permutasi sumbu "
        "bertanda, keduanya dapat dibalik secara eksak, sehingga hasil yang diharapkan adalah "
        "pemulihan sempurna. Kami karena itu melaporkannya sebagai hasil verifikasi lulus atau "
        "gagal dan secara sengaja menjauhkannya dari tabel hasil mana pun, sebab Dice bernilai "
        "1,0 yang disandingkan dengan akurasi model mengundang pembacaan bahwa model tersebut "
        "sempurna. Nilai itu merupakan sifat transformasi, bukan sifat model."))

    add(("h3", "Ablasi Tahap Rekonstruksi"))
    add(("p",
        "Untuk menetapkan nilai setiap tahap verifikasi, kami menyusun ulang prediksi 2D yang "
        "sama dengan operasi rekonstruksi tertentu dihilangkan. Karena seluruh varian berangkat "
        "dari prediksi yang identik, setiap perbedaan dapat diatribusikan pada tahap rekonstruksi "
        "semata, tanpa keterlibatan variasi model. Empat pintasan diuji: menghilangkan tahap "
        "inversi orientasi; memotong bagian tengah kembali ke ukuran in-plane asli alih-alih "
        "menggunakan offset yang tercatat; menuliskan volume dengan affine identitas alih-alih "
        "affine sumber; serta menumpuk irisan menurut urutan loader alih-alih indeks eksplisit. "
        "Varian kelima menggabungkan keempatnya."))
    add(("p",
        "Setiap varian diukur melalui tiga cara: kesesuaiannya dengan volume hasil rekonstruksi "
        "yang benar, yang mengisolasi besar kerusakan; Dice yang akan dilaporkannya terhadap "
        "anotasi pakar, yaitu angka yang akan dilihat pembaca; serta apakah daftar periksa "
        "geometris menandainya. Sebagai gerbang kebenaran, rekonstruksi rujukan disusun ulang "
        "hanya dari prediksi dan metadata yang tersimpan, dan disyaratkan identik bit per bit "
        "dengan volume yang dihasilkan proses utama, sehingga perbandingan dilakukan terhadap "
        "rujukan yang dapat direproduksi."))

    add(("h3", "Volume Zona"))
    add(("p",
        "Volume zona dihitung sebagai jumlah voxel tiap kelas dikalikan volume fisik satu voxel, "
        "yang diambil sebagai nilai absolut determinan bagian linear affine alih-alih hasil kali "
        "spasi pada header, sehingga akuisisi yang terotasi atau tergeser ditangani secara eksak. "
        "Kesesuaian dengan volume yang diturunkan dari anotasi pakar dinilai melalui analisis "
        "Bland-Altman, dengan melaporkan bias dan batas kesesuaian 95%."))
    add(("p",
        "Volume ditelaah karena merupakan besaran yang menghubungkan segmentasi ini dengan "
        "penggunaan klinis. Volume kelenjar total adalah penyebut pada PSA density, sehingga "
        "galat pada volume merambat langsung ke besaran yang digunakan dalam keputusan biopsi. "
        "Volume juga membuat kerja geometris menjadi terbaca: volume dalam mililiter adalah "
        "jumlah voxel dikalikan besaran yang dibaca dari affine, sehingga pipeline yang kehilangan "
        "affine melaporkan angka klinis yang salah dari voxel yang sepenuhnya benar."))

    add(("h3", "Validasi Eksternal"))
    add(("p",
        f"Model beku diterapkan satu kali pada kohort independen berisi {e['n_cases']} pemeriksaan "
        "PROSTATEx dengan anotasi zonal pakar yang dirilis publik. Tidak dilakukan pelatihan, "
        "penyetelan lanjut, pencarian ambang, maupun pemilihan model pada kohort ini; kohort "
        "digunakan murni untuk inferensi, melalui jalur kode inferensi-dan-rekonstruksi yang sama "
        "dengan evaluasi internal."))
    add(("p",
        "Anotasi sumber mendistribusikan peripheral zone dan sisa kelenjar sebagai dua mask biner "
        "terpisah. Keduanya digabungkan ke dalam konvensi tiga label milik model menggunakan uji "
        "lebih-besar-dari-nol alih-alih kesamaan, setelah ditemukan satu kasus yang memuat satu "
        "voxel bernilai 2 pada mask peripheral zone-nya yang akan terbuang oleh uji kesamaan. "
        "Sifat saling lepas kedua mask diukur ulang untuk setiap kasus, bukan diasumsikan. Kasus "
        "dirujuk melalui manifes kohort, karena 15 di antaranya menggunakan penamaan berkas yang "
        "tidak dapat diturunkan dari pengenal kasus."))
    add(("p",
        f"Kohort ini berbeda dari data pengembangan dalam hal-hal yang kami ukur, bukan sekadar "
        f"dideskripsikan secara kualitatif. Resolusi in-plane-nya terentang "
        f"{eg['inplane_min']}-{eg['inplane_max']} mm berbanding 0,46875 mm pada data internal, "
        f"spasi through-plane-nya terentang {eg['through_min']}-{eg['through_max']} mm, dan "
        f"medan pandang aksialnya rata-rata {eg['fov_mean']:.0f} mm berbanding "
        f"{eg['fov_internal']:.1f} mm, yakni sebesar {eg['fov_ratio']:.2f} kali. Karena pipeline "
        "tidak melakukan resampling in-plane, anatomi disajikan kepada model pada skala piksel "
        "yang berbeda dari saat pelatihan, dan medan pandang yang lebih luas berarti model "
        "melihat struktur sekitar yang tidak pernah ditemuinya selama pelatihan. Keduanya "
        "merupakan pergeseran kovariat yang nyata dan dilaporkan berdampingan dengan angka "
        "akurasi alih-alih diperlakukan sebagai gangguan."))

    add(("h3", "Analisis Statistik"))
    add(("p",
        "Perbedaan antar konfigurasi dinilai menggunakan uji Wilcoxon signed-rank pada nilai per "
        "kasus yang berpasangan, yang tidak mengasumsikan kenormalan dan sesuai untuk ukuran "
        "sampel yang terlibat. Perbandingan yang ditetapkan di muka adalah macro Dice konfigurasi "
        "terpilih terhadap baseline pada data uji."))
    add(("p",
        "Dua kelas uji dilaporkan di sini dengan peringatan eksplisit, karena melaporkannya tanpa "
        "peringatan tersebut akan melebih-lebihkan apa yang ditegakkannya. Uji yang dihitung pada "
        "data validasi bersifat deskriptif, bukan konfirmatori: data tersebut adalah data tempat "
        "model dipilih, sehingga klaim signifikansi yang diturunkan darinya bersifat sirkular. "
        "Uji per zona pada data uji bersifat post hoc; keduanya tidak ditetapkan di muka, dan "
        "dengan tiga perbandingan ambang terkoreksi Bonferroni adalah 0,0167. Kami melaporkan "
        "seluruh nilai tersebut karena informatif mengenai letak perbedaan antar konfigurasi, dan "
        "kami melabelinya agar tidak terbaca sebagai bukti konfirmatori."))

    add(("h3", "Visualisasi dan Lingkungan Komputasi"))
    add(("p",
        "Gambar kualitatif menampilkan, untuk kasus terpilih, citra T2-weighted, anotasi pakar, "
        "prediksi, tampilan bertumpuk keduanya, serta peta galat. Kasus representatif dipilih "
        "melalui aturan objektif - kasus dengan rerata Dice foreground tertinggi, median, dan "
        "terendah - bukan berdasarkan daya tarik visual. Proyeksi multiplanar dirender dengan "
        "rasio aspek tampilan ditetapkan dari spasi voxel, sehingga grid yang sangat anisotropik "
        "tidak ditampilkan seolah-olah isotropik."))
    add(("p",
        "Pelatihan dilakukan pada satu GPU NVIDIA Tesla T4 dengan PyTorch. Inferensi, "
        "rekonstruksi, ablasi, dan analisis volumetrik tidak memerlukan GPU dan dijalankan pada "
        "CPU. Seluruh kode analisis, berkas konfigurasi, dan skrip evaluasi tersedia pada "
        "repositori publik yang disitasi di bawah."))

    add(("h3", "Etika"))
    add(("p",
        "Penelitian ini menggunakan dataset publik yang telah dinyatakan anonim dan tidak "
        "melibatkan partisipan manusia, perekrutan, maupun intervensi. Persetujuan komite etik "
        "dan informed consent karena itu tidak diperlukan. Dataset digunakan sesuai lisensinya "
        "masing-masing, dan anotasi dikreditkan kepada penulis aslinya."))

    add(("tbl", ("Tabel 1. Karakteristik dataset dan konfigurasi eksperimen.",
        ["Butir", "Nilai"],
        [
            ["Dataset", "Prostate158 (158 pemeriksaan, MRI 3T, NIfTI)"],
            ["Modalitas masukan", "T2-weighted, satu kanal"],
            ["Target segmentasi", "Mask anatomi pembaca pertama, label {0,1,2}"],
            ["Kelas", "0 = latar belakang, 1 = central gland (CG), 2 = peripheral zone (PZ)"],
            ["Pembagian resmi", f"119 pelatihan / {a['e1']['n']} validasi / {t['n_cases']} uji held-out"],
            ["Satuan pembagian", "Tingkat pasien (kasus)"],
            ["Prapemrosesan", "Orientasi RAS; z-score per volume setelah pemotongan persentil; "
                              "crop/pad ke 442x442; tanpa resampling through-plane"],
            ["Arsitektur", "2D U-Net, 32 fitur awal, 1 kanal masuk / 3 kanal keluar, "
                           "7.762.531 parameter"],
            ["Optimizer", "AdamW, learning rate 0,001, weight decay 0,0001"],
            ["Batch size / epoch", "8 / 100"],
            ["Random seed", "42"],
            ["Presisi", "FP32"],
            ["Aturan pemilihan", "Macro Dice tertinggi pada data validasi, ditetapkan di muka"],
            ["Identitas model", "SHA-256 checkpoint beku"],
            ["Kohort eksternal", f"PROSTATEx, {e['n_cases']} pemeriksaan, inferensi saja"],
        ])))

    # ------------------------------------------------------------------- hasil
    add(("h2", "Hasil"))

    add(("h3", "Pemilihan Model pada Data Validasi"))
    add(("p",
        f"Tabel 2 melaporkan keempat konfigurasi pada {a['e1']['n']} kasus validasi. Konfigurasi "
        f"yang menambahkan suku soft Dice mencapai macro Dice tertinggi ({a['e1']['macro']:.4f}) "
        f"dan karena itu dipilih. Epoch terbaiknya adalah {a['e1']['epoch']}."))
    add(("p",
        f"Pengamatan yang lebih informatif adalah besarnya perbedaan tersebut. Keempat "
        f"konfigurasi hanya terentang {N['arm_spread']:.4f} macro Dice, sedangkan simpangan baku "
        f"antar kasus di dalam konfigurasi mana pun berkisar {N['arm_sd_range'][0]:.4f} hingga "
        f"{N['arm_sd_range'][1]:.4f}. Rentang antar intervensi dengan demikian beberapa kali "
        f"lipat lebih kecil daripada variasi antar pasien di dalam satu konfigurasi. Augmentasi "
        f"dan cross-entropy berbobot kelas tidak memperbaiki konfigurasi yang dimodifikasinya "
        f"(augmentasi {vt['e2_vs_baseline']['delta']:+.4f} terhadap baseline, "
        f"P={vt['e2_vs_baseline']['p']:.2f}; pembobotan kelas {vt['e3_vs_e1']['delta']:+.4f} "
        f"terhadap konfigurasi terpilih, P={vt['e3_vs_e1']['p']:.2f}). Kami melaporkan hal ini "
        f"sebagai hasil negatif alih-alih menghilangkannya: pada dataset ini, pada skala ini, dua "
        f"intervensi yang lazim direkomendasikan tidak memberi manfaat terukur."))
    add(("p",
        f"Uji berpasangan eksploratif pada data ini memihak konfigurasi terpilih dibandingkan "
        f"baseline ({vt['e1_vs_baseline']['delta']:+.4f}, P={vt['e1_vs_baseline']['p']:.4f}) "
        f"maupun arm augmentasi ({vt['e1_vs_e2']['delta']:+.4f}, "
        f"P={vt['e1_vs_e2']['p']:.4f}). Nilai-nilai tersebut bersifat deskriptif belaka. Data "
        f"validasi adalah data tempat pemilihan dilakukan, sehingga klaim signifikansi yang "
        f"dihitung padanya bersifat sirkular dan tidak diajukan sebagai bukti keunggulan."))

    add(("tbl", (f"Tabel 2. Empat konfigurasi pelatihan dan performanya pada {a['e1']['n']} kasus "
                 f"validasi. Setiap konfigurasi berbeda dari baseline tepat pada satu faktor. "
                 f"Macro Dice adalah rerata CG dan PZ.",
        ["Konfigurasi", "Intervensi", "Epoch terbaik", "Dice CG", "Dice PZ", "Macro Dice"],
        [[k.upper() if k != "baseline" else "Baseline",
          {"baseline": "Cross-entropy (baseline)",
           "e1": "+ soft Dice pada CG dan PZ",
           "e2": "+ augmentasi geometrik dan intensitas",
           "e3": "+ cross-entropy berbobot kelas"}[k],
          str(d["epoch"]), f"{d['cg']:.4f}", f"{d['pz']:.4f}", f"{d['macro']:.4f}"]
         for k, d in a.items()])))

    add(("h3", "Performa Segmentasi pada Data Uji"))
    add(("p",
        f"Model beku dievaluasi satu kali pada {t['n_cases']} kasus uji. Rerata (SB) Dice adalah "
        f"{cg_d['mean']:.4f} ({cg_d['sd']:.4f}) untuk central gland dan {pz_d['mean']:.4f} "
        f"({pz_d['sd']:.4f}) untuk peripheral zone, menghasilkan macro Dice {t['macro']['mean']:.4f}. "
        f"Dice latar belakang adalah {bg_d['mean']:.4f} ({bg_d['sd']:.4f}) dan dilaporkan "
        f"terpisah. Metrik lengkap disajikan pada Tabel 3."))
    add(("p",
        f"Kedua zona berperilaku berbeda, dan polanya konsisten. Peripheral zone memadukan "
        f"precision tinggi ({t['per_class']['pz']['precision']['mean']:.4f}) dengan recall yang "
        f"jauh lebih rendah ({t['per_class']['pz']['recall']['mean']:.4f}), sedangkan central "
        f"gland relatif seimbang ({t['per_class']['cg']['precision']['mean']:.4f} dan "
        f"{t['per_class']['cg']['recall']['mean']:.4f}). Apa yang dilabeli model sebagai "
        f"peripheral zone umumnya benar; model sekadar melabeli terlalu sedikit. Ini merupakan "
        f"profil kesulitan yang diharapkan bagi struktur perifer tipis yang batasnya hanya "
        f"menempati sebagian kecil citra."))
    add(("p",
        f"Macro Dice per kasus terentang {t['macro']['min']:.4f} hingga {t['macro']['max']:.4f}. "
        f"Dice central gland terentang {t['cg_range'][0]:.4f} hingga {t['cg_range'][1]:.4f} dan "
        f"Dice peripheral zone {t['pz_range'][0]:.4f} hingga {t['pz_range'][1]:.4f}. Variasi "
        f"antar pasien jauh lebih besar daripada variasi antar konfigurasi pelatihan yang "
        f"dibandingkan pada Tabel 2."))
    add(("p",
        f"Terhadap baseline pada kasus yang sama, konfigurasi terpilih memperbaiki macro Dice "
        f"sebesar {tb['macro']['delta']:+.4f}, lebih baik pada {tb['macro']['better_in']} dari "
        f"{tb['n']} kasus (W={tb['macro']['W']:.0f}, P={tb['macro']['p']:.3f}). Perbandingan yang "
        f"ditetapkan di muka ini tidak mencapai ambang konvensional, sehingga kami menyebut "
        f"perbaikan tersebut konsisten dalam arah alih-alih terbukti secara statistik. "
        f"Perbandingan post hoc yang dibatasi pada peripheral zone menghasilkan "
        f"{tb['pz']['delta']:+.4f}, lebih baik pada {tb['pz']['better_in']} dari {tb['n']} kasus "
        f"(W={tb['pz']['W']:.0f}, P={tb['pz']['p']:.3f}), yang mengindikasikan bahwa perbedaan "
        f"yang ada terpusat pada zona yang lebih sulit; dengan tiga perbandingan, ambang "
        f"terkoreksi adalah 0,0167, sehingga nilai ini dilaporkan sebagai eksploratif."))

    add(("tbl", (f"Tabel 3. Performa segmentasi model terpilih pada {t['n_cases']} kasus uji. "
                 f"Tingkat volume, ruang voxel asli, diagregasi per kasus. Nilai berupa rerata "
                 f"(SB), dengan selang kepercayaan bootstrap 95% untuk Dice.",
        ["Metrik", "Central gland", "Peripheral zone"],
        [
            ["Dice", f"{cg_d['mean']:.4f} ({cg_d['sd']:.4f})",
                     f"{pz_d['mean']:.4f} ({pz_d['sd']:.4f})"],
            ["Dice, SK 95%", f"{cg_d['ci_low']:.4f}-{cg_d['ci_high']:.4f}",
                             f"{pz_d['ci_low']:.4f}-{pz_d['ci_high']:.4f}"],
        ] + [
            [name,
             f"{t['per_class']['cg'][m]['mean']:.4f} ({t['per_class']['cg'][m]['sd']:.4f})",
             f"{t['per_class']['pz'][m]['mean']:.4f} ({t['per_class']['pz'][m]['sd']:.4f})"]
            for m, name in (("iou", "IoU"), ("hd95_mm", "HD95 (mm)"), ("asd_mm", "ASD (mm)"),
                            ("precision", "Precision"), ("recall", "Recall"))
        ] + [["Macro Dice", f"{t['macro']['mean']:.4f}", ""],
             ["Dice latar belakang", f"{bg_d['mean']:.4f} ({bg_d['sd']:.4f})", ""]])))

    add(("h3", "Rekonstruksi 3D dan Validasi Geometris"))
    add(("p",
        f"Seluruh {g['n_all_ok']} dari {g['n']} rekonstruksi lolos setiap pemeriksaan geometris. "
        f"Bentuk larik, spasi voxel, dan kode orientasi sesuai dengan sumber pada setiap kasus, "
        f"himpunan label terbatas pada tiga nilai yang diharapkan, dan selisih absolut maksimum "
        f"antara matriks affine hasil rekonstruksi dan sumber adalah {g['affine_max_abs_diff']:.1f} "
        f"di seluruh kohort - kesesuaian eksak, bukan sekadar kesesuaian dalam toleransi, yang "
        f"bermakna karena perbandingan dilakukan terhadap affine sebagaimana tersimpan dalam "
        f"header presisi tunggal berkas keluaran itu sendiri."))
    add(("p",
        "Uji fidelitas round-trip lolos: anotasi pakar, yang dilewatkan maju melalui prapemrosesan "
        "dan kembali melalui kode rekonstruksi yang sama, dipulihkan secara persis. Kami "
        "melaporkan hal ini sebagai hasil verifikasi, bukan sebagai metrik. Akurasi segmentasi "
        "itu sendiri adalah angka pada Tabel 3, dan keduanya tidak seharusnya muncul pada tabel "
        "yang sama."))

    add(("h3", "Nilai Setiap Tahap Rekonstruksi"))
    add(("p",
        "Ablasi menyusun ulang prediksi yang sama dengan operasi rekonstruksi tertentu "
        "dihilangkan. Rekonstruksi rujukan, yang disusun ulang secara independen dari prediksi "
        "dan metadata tersimpan, identik bit per bit dengan volume yang dihasilkan proses utama "
        "pada setiap kasus, sehingga perbandingan berikut dilakukan terhadap rujukan yang dapat "
        "direproduksi."))
    add(("p",
        f"Menghilangkan tahap inversi orientasi menghasilkan volume yang tercermin secara "
        f"anatomis. Dice yang dilaporkannya terhadap anotasi pakar turun dari {cg_d['mean']:.4f} "
        f"menjadi {v1['cg']['dice_vs_gt']:.4f} untuk central gland dan dari {pz_d['mean']:.4f} "
        f"menjadi {v1['pz']['dice_vs_gt']:.4f} untuk peripheral zone, dan peripheral zone "
        f"tergeser hingga {v1['pz']['centroid_shift_max_mm']:.0f} mm. Daftar periksa geometris "
        f"menandai {v1['cg']['n_caught']} dari {v1['cg']['n_cases']} kasus semacam itu. Inilah "
        f"temuan utama ablasi: daftar periksa memeriksa header berkas, header tersebut benar, dan "
        f"hanya data voxel yang salah, sehingga galat ini tidak terlihat olehnya."))
    add(("p",
        f"Menghilangkan pemulihan affine menghasilkan pola yang berkebalikan. Voxel tidak "
        f"tersentuh, sehingga Dice terhadap anotasi tetap pada {v3['cg']['dice_vs_gt']:.4f} dan "
        f"{v3['pz']['dice_vs_gt']:.4f} - Dice dihitung pada indeks voxel dan sama sekali tidak "
        f"dapat melihat affine. Namun kelenjar tergeser {v3['cg']['centroid_shift_mm']:.0f} mm "
        f"dalam koordinat pemindai dan setiap volume zona menggelembung "
        f"{v3['cg']['volume_pct_error']:.0f}%, karena voxel isotropik satuan bukanlah voxel yang "
        f"diakuisisi pemindai. Di sini daftar periksa menandai seluruh {v3['cg']['n_caught']} dari "
        f"{v3['cg']['n_cases']} kasus."))
    add(("p",
        "Kedua mode kegagalan dengan demikian saling melengkapi secara tepat. Metrik tumpang "
        "tindih mendeteksi volume yang tercermin dan buta terhadap affine yang hilang; daftar "
        "periksa geometris mendeteksi affine yang hilang dan buta terhadap volume yang tercermin. "
        "Pipeline yang hanya melaporkan metrik akurasi dan pipeline yang hanya memvalidasi header "
        "masing-masing buta terhadap salah satunya, dan inilah alasan untuk menjalankan kedua "
        "pemeriksaan sekaligus uji fidelitas round-trip yang membandingkan voxel alih-alih header."))
    add(("p",
        "Dua dari empat pintasan tidak menimbulkan kerusakan terukur pada kohort ini, dengan "
        "alasan yang berbeda dan layak dibedakan. Pemotongan terpusat aman secara struktural, "
        "karena pipeline ini selalu menempatkan padding di tengah, sehingga pemotongan terpusat "
        "tidak mungkin menyimpang dari offset tercatat pada kasus yang dipadding - namun ia tetap "
        "tidak dapat membalik pemotongan maju, yang akan muncul pada kohort dengan citra lebih "
        "besar. Penumpukan menurut urutan loader aman hanya secara kebetulan, karena loader ini "
        "kebetulan mengeluarkan irisan dalam urutan aksial; hal itu merupakan sifat loader, bukan "
        "sifat rekonstruksi, dan dapat berubah tanpa perubahan apa pun pada kode rekonstruksi. "
        "Keduanya tidak boleh dibaca sebagai bukti bahwa tahap tersebut tidak diperlukan."))
    add(("fig", (f"{FIG_DIR_RQ2}/F1_reconstruction_ablation.png",
        "Gambar 4. Biaya terukur dari setiap tahap rekonstruksi yang dihilangkan. Panel A "
        "menyajikan Dice yang akan dilaporkan setiap pintasan terhadap anotasi pakar, dari "
        "prediksi yang identik, berdampingan dengan pipeline yang benar. Panel B menunjukkan "
        "pemeriksaan mana yang mendeteksi kegagalan yang mana.")))
    add(("fig", (f"{FIG_DIR_RQ2}/F4_mirrored_volume_passes.png",
        "Gambar 5. Kasus yang sama direkonstruksi secara benar (baris atas) dan dengan tahap "
        "inversi orientasi dihilangkan (baris bawah). Kedua volume lolos setiap pemeriksaan "
        "geometris, karena header keduanya benar dan hanya data voxel yang berbeda. Volume pada "
        "baris bawah tercermin secara anatomis dan Dice peripheral zone-nya runtuh.")))

    add(("tbl", ("Tabel 4. Ablasi rekonstruksi. Setiap varian disusun ulang dari prediksi 2D yang "
                 "sama, sehingga perbedaan dapat diatribusikan pada tahap rekonstruksi semata. "
                 "Dice yang dilaporkan adalah angka yang akan dilihat pembaca terhadap anotasi "
                 "pakar.",
        ["Pintasan", "Dice CG dilaporkan", "Dice PZ dilaporkan", "Pergeseran sentroid (mm)",
         "Galat volume (%)", "Ditandai daftar periksa"],
        [[lbl,
          f"{ab[k]['cg']['dice_vs_gt']:.4f}", f"{ab[k]['pz']['dice_vs_gt']:.4f}",
          f"{ab[k]['cg']['centroid_shift_mm']:.1f}",
          f"{ab[k]['cg']['volume_pct_error']:.0f}",
          f"{ab[k]['cg']['n_caught']}/{ab[k]['cg']['n_cases']}"]
         for k, lbl in [("V1_no_reorientation", "Inversi orientasi dihilangkan"),
                        ("V2_naive_centre_crop", "Pemotongan terpusat, bukan offset tercatat"),
                        ("V3_identity_affine", "Affine tidak dipulihkan"),
                        ("V4_append_order", "Irisan ditumpuk menurut urutan loader"),
                        ("V5_all_naive", "Keempatnya digabung")]
         ] + [["Tidak ada (pipeline benar)",
               f"{v1['cg']['dice_vs_gt_correct']:.4f}", f"{v1['pz']['dice_vs_gt_correct']:.4f}",
               "0,0", "0", "0/19"]])))

    add(("h3", "Volume Zona Anatomi"))
    add(("p",
        f"Pada {v['n']} kasus uji, anotasi pakar memberikan rerata volume kelenjar total sebesar "
        f"{v['whole']['gt_mean']:.2f} mL, yang terdiri atas {v['cg']['gt_mean']:.2f} mL central "
        f"gland dan {v['pz']['gt_mean']:.2f} mL peripheral zone. Model meremehkan ketiganya. "
        f"Volume kelenjar total memiliki bias {v['whole']['bias']:.2f} mL "
        f"({v['whole']['pct_of_gt']:.1f}% dari rujukan), diremehkan pada "
        f"{v['whole']['under_in']} dari {v['n']} kasus, dengan batas kesesuaian 95% dari "
        f"{v['whole']['loa_low']:.2f} hingga {v['whole']['loa_high']:.2f} mL."))
    add(("p",
        f"Peripheral zone paling terdampak: bias {v['pz']['bias']:.2f} mL, yakni "
        f"{abs(v['pz']['pct_of_gt']):.1f}% dari volume sebenarnya, diremehkan pada "
        f"{v['pz']['under_in']} dari {v['n']} kasus. Inilah recall rendah pada Tabel 3 yang "
        f"dinyatakan dalam mililiter. Central gland relatif terestimasi baik, dengan bias "
        f"{v['cg']['bias']:.2f} mL ({abs(v['cg']['pct_of_gt']):.1f}%)."))
    add(("p",
        f"Konsekuensinya mengikuti aritmetika PSA density, yang membagi PSA serum dengan volume "
        f"kelenjar. Meremehkan penyebut akan menggelembungkan hasil baginya: pada kasus-kasus ini "
        f"PSA density akan dilebihkan sebesar {v['psad']['mean_pct']:.1f}% secara rerata dan "
        f"hingga {v['psad']['max_pct']:.0f}% pada kasus terburuk, dengan bias ke atas pada "
        f"{v['psad']['over_in']} dari {v['n']} kasus. Terhadap ambang yang lazim disitasi sebesar "
        f"0,15 ng/mL/cc, pasien dengan densitas sebenarnya "
        f"{v['psad']['threshold_equivalent']:.3f} akan terbaca tepat pada ambang tersebut. "
        f"Arahnya menuju lebih banyak biopsi alih-alih lebih sedikit, yang secara klinis "
        f"merupakan arah yang lebih aman, namun tetap merupakan mis-kalibrasi sistematis."))
    add(("p",
        "Kami menekankan batas kesesuaian di atas nilai bias. Bias sebesar beberapa mililiter "
        "akan saling meniadakan pada tingkat kohort; batas kesesuaian yang terentang lebih dari "
        "27 mL pada kelenjar berukuran sekitar 54 mL menggambarkan apa yang dapat terjadi pada "
        "pasien perorangan, dan pasien peroranganlah yang menjadi sasaran perhitungan PSA density."))
    add(("fig", (f"{FIG_DIR_RQ2}/F2_zone_volume_agreement.png",
        "Gambar 6. Kesesuaian volume zona. Plot Bland-Altman untuk central gland dan peripheral "
        "zone, serta galat PSA density per kasus yang dihasilkannya.")))

    add(("tbl", (f"Tabel 5. Volume zona pada {v['n']} kasus uji, dalam mililiter, dengan "
                 f"kesesuaian Bland-Altman terhadap anotasi pakar. Batas kesesuaian adalah "
                 f"bias +/- 1,96 SB.",
        ["Besaran", "Rujukan, rerata (SB)", "Prediksi, rerata (SB)", "Bias", "% dari rujukan",
         "Batas kesesuaian 95%"],
        [[name,
          f"{v[k]['gt_mean']:.2f} ({v[k]['gt_sd']:.2f})",
          f"{v[k]['pred_mean']:.2f} ({v[k]['pred_sd']:.2f})",
          f"{v[k]['bias']:+.2f}", f"{v[k]['pct_of_gt']:+.1f}",
          f"{v[k]['loa_low']:.2f} hingga {v[k]['loa_high']:.2f}"]
         for k, name in (("cg", "Central gland"), ("pz", "Peripheral zone"),
                         ("whole", "Kelenjar total"))])))

    add(("h3", "Validasi Eksternal"))
    add(("p",
        f"Diterapkan satu kali pada {e['n_cases']} pemeriksaan PROSTATEx tanpa adaptasi apa pun, "
        f"model beku mencapai rerata (SB) Dice sebesar {e['cg']['dice']['mean']:.4f} "
        f"({e['cg']['dice']['sd']:.4f}) untuk central gland dan {e['pz']['dice']['mean']:.4f} "
        f"({e['pz']['dice']['sd']:.4f}) untuk peripheral zone, dengan macro Dice "
        f"{e['macro']['mean']:.4f}. Dibandingkan hasil uji internal, ini merupakan penurunan "
        f"sebesar {t['macro']['mean'] - e['macro']['mean']:.4f} pada macro Dice."))
    add(("p",
        f"Pola zonal yang teramati secara internal bertahan dan menguat. Recall peripheral zone "
        f"turun menjadi {e['pz']['recall']['mean']:.4f} sementara precision-nya naik menjadi "
        f"{e['pz']['precision']['mean']:.4f}: pada data yang tidak dikenalnya, model menjadi lebih "
        f"konservatif, melabeli lebih sedikit dan lebih sering benar mengenai apa yang dilabelinya."))
    add(("p",
        f"Tumpang tindih tertransfer, namun metrik batas tidak tertransfer dengan cara yang sama, "
        f"dan pembedaan ini informatif. HD95 central gland memiliki median {e['hd95_cg']['median']:.1f} "
        f"mm - dekat dengan nilai internal - namun rerata {e['hd95_cg']['mean']:.1f} mm dan "
        f"maksimum {e['hd95_cg']['max']:.1f} mm. Sebanyak {e['hd95_cg']['n_over_30']} dari "
        f"{e['n_cases']} kasus melampaui 30 mm. Ini merupakan ekor distribusi yang berat alih-alih "
        f"penurunan merata, dan kasus-kasus yang terdampak tetap memiliki Dice tinggi, yang "
        f"mengindikasikan adanya volume prediksi palsu berukuran kecil yang jauh dari kelenjar "
        f"alih-alih batas yang salah secara menyeluruh. HD95 berkorelasi negatif dengan precision "
        f"di seluruh kohort (r={e['hd95_cg']['corr_with_precision']:+.2f}), konsisten dengan "
        f"pembacaan tersebut."))
    add(("p",
        f"Mekanisme yang paling mungkin adalah perbedaan medan pandang. Pemeriksaan eksternal "
        f"mencakup {eg['fov_mean']:.0f} mm secara in-plane berbanding {eg['fov_internal']:.1f} mm "
        f"pada data internal, yakni sebesar {eg['fov_ratio']:.2f} kali, sehingga memuat struktur "
        f"panggul yang tidak pernah ditemui model selama pelatihan, dan pada wilayah tambahan "
        f"itulah prediksi palsu dapat muncul. Kami menyatakan hal ini sebagai hipotesis yang "
        f"konsisten dengan pengukuran, bukan sebagai sebab yang telah ditegakkan; pembuktiannya "
        f"memerlukan analisis komponen terhubung pada volume yang terdampak, yang tidak kami "
        f"lakukan."))
    add(("p",
        f"Rekonstruksi berperilaku benar pada kohort ini meskipun geometrinya berbeda. Sebanyak "
        f"{eg['n_cropped']} dari {e['n_cases']} pemeriksaan berukuran cukup besar sehingga "
        f"standardisasi in-plane memotongnya alih-alih mem-padding-nya, yang menjalankan cabang "
        f"transformasi yang tidak pernah dipicu data pengembangan, dan tidak satu kasus pun "
        f"kehilangan anatomi teranotasi akibat pemotongan tersebut. Kohort ini juga diakuisisi "
        f"dengan konvensi orientasi yang berbeda dari data pengembangan, yang ditangani "
        f"rekonstruksi melalui transformasi balik yang sama."))
    add(("p",
        "Angka-angka ini bukan perbandingan setara dengan hasil internal. Anotator, pemindai, "
        "geometri akuisisi, dan definisi zonanya berbeda, dan anotasi eksternal dihasilkan oleh "
        "kelompok berbeda di bawah protokol berbeda. Besaran yang menjadi perhatian adalah besar "
        "selisihnya, yang ditafsirkan berdampingan dengan pergeseran kovariat yang sebagian "
        "menjelaskannya."))
    add(("fig", (f"{FIG_DIR_RQ2}/F3_external_validation.png",
        "Gambar 7. Validasi eksternal. Dice internal dan eksternal berdampingan; distribusi HD95 "
        "central gland pada kohort eksternal, yang menunjukkan ekor berat alih-alih pergeseran "
        "merata; serta hubungan antara HD95 dan precision.")))

    add(("tbl", (f"Tabel 6. Performa internal berbanding eksternal. Internal: {t['n_cases']} kasus "
                 f"uji Prostate158. Eksternal: {e['n_cases']} pemeriksaan PROSTATEx, inferensi "
                 f"saja. Nilai berupa rerata (SB).",
        ["Metrik", "Internal, CG", "Eksternal, CG", "Internal, PZ", "Eksternal, PZ"],
        [[name,
          f"{t['per_class']['cg'][m]['mean']:.4f} ({t['per_class']['cg'][m]['sd']:.4f})",
          f"{e['cg'][m]['mean']:.4f} ({e['cg'][m]['sd']:.4f})",
          f"{t['per_class']['pz'][m]['mean']:.4f} ({t['per_class']['pz'][m]['sd']:.4f})",
          f"{e['pz'][m]['mean']:.4f} ({e['pz'][m]['sd']:.4f})"]
         for m, name in (("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)"),
                         ("asd_mm", "ASD (mm)"), ("precision", "Precision"),
                         ("recall", "Recall"))])))

    add(("h3", "Hasil Kualitatif"))
    add(("p",
        f"Kasus representatif dipilih berdasarkan rerata Dice foreground alih-alih penampakan: "
        f"kasus {t['representatives']['good']} memperoleh skor tertinggi, kasus "
        f"{t['representatives']['median']} merupakan median, dan kasus "
        f"{t['representatives']['challenging']} terendah. Pada kasus berskor tertinggi kedua zona "
        f"terdelineasi baik dan ketidaksesuaian terbatas pada pita batas yang tipis. Pada kasus "
        f"berskor terendah central gland berhasil dipulihkan sementara peripheral zone "
        f"ter-undersegmentasi secara substansial, dengan central gland meluas ke wilayah yang oleh "
        f"anotasi ditetapkan sebagai peripheral zone - yakni kekeliruan kelas pada antarmuka "
        f"zonal, bukan kegagalan menemukan prostat."))
    add(("fig", (f"{VIZ}/patient_{t['representatives']['good']}/"
                 f"patient_{t['representatives']['good']}_slice_016.png",
        f"Gambar 8. Kasus representatif dengan performa tinggi (kasus "
        f"{t['representatives']['good']}). Dari kiri: citra T2-weighted, anotasi pakar, prediksi, "
        f"tampilan bertumpuk, dan peta galat.")))
    add(("fig", (f"{VIZ}/patient_{t['representatives']['challenging']}/"
                 f"patient_{t['representatives']['challenging']}_slice_015.png",
        f"Gambar 9. Kasus representatif yang menantang (kasus "
        f"{t['representatives']['challenging']}). Central gland berhasil dipulihkan sementara "
        f"peripheral zone ter-undersegmentasi pada antarmuka zonal.")))

    # -------------------------------------------------------------- pembahasan
    add(("h2", "Pembahasan"))

    add(("h3", "Hasil Utama"))
    add(("p",
        f"2D U-Net standar menyegmentasi central gland dengan Dice {cg_d['mean']:.4f} dan "
        f"peripheral zone dengan Dice {pz_d['mean']:.4f} pada data uji, dan tiga intervensi "
        f"pelatihan faktor-tunggal mengubah performa lebih kecil daripada variasi antar pasien. "
        f"Kami memandang paruh kedua kalimat tersebut sebagai temuan yang lebih berguna bagi RQ1. "
        f"Keempat konfigurasi hanya terentang {N['arm_spread']:.4f} macro Dice berbanding sebaran "
        f"per kasus di dalam satu konfigurasi yang beberapa kali lipat lebih besar, yang berarti "
        f"bahwa pada dataset ini, pada skala ini, fungsi loss dan kebijakan augmentasi bukanlah "
        f"tempat bersemayamnya galat yang tersisa. Faktor pembatasnya adalah delineasi batas pada "
        f"peripheral zone, dan pola konsisten berupa precision tinggi dengan recall rendah "
        f"melokalisasinya secara spesifik: model tidak keliru menganggap jaringan lain sebagai "
        f"peripheral zone, melainkan gagal mengklaim cukup banyak darinya."))
    add(("p",
        "Bagi RQ2, hasil utamanya adalah bahwa validasi geometris - verifikasi yang paling lazim "
        "dilaporkan, itu pun bila dilaporkan - tidak memadai. Rekonstruksi dengan header yang "
        "sepenuhnya benar dan data voxel yang tercermin lolos setiap butir daftar periksa - "
        "bentuk, affine, spasi, kode orientasi, dan himpunan label - sementara Dice yang "
        "dilaporkannya turun sekitar seperempat untuk central gland dan tiga perempat untuk "
        "peripheral zone. Kegagalan komplementernya, yakni larik voxel yang benar namun ditulis "
        "dengan affine bawaan, tidak terlihat oleh metrik tumpang tindih karena metrik tersebut "
        "dihitung pada indeks voxel, padahal kegagalan itu menggeser kelenjar hampir dua ratus "
        "milimeter dan menggelembungkan setiap volume hampir setengahnya."))
    add(("p",
        "Inilah yang memberi uji fidelitas round-trip alasan keberadaannya. Melewatkan anotasi "
        "pakar maju melalui prapemrosesan dan kembali melalui kode rekonstruksi yang sama, serta "
        "mensyaratkannya kembali tanpa perubahan, merupakan satu-satunya pemeriksaan dalam "
        "pipeline ini yang membandingkan voxel alih-alih header dan karena itu satu-satunya yang "
        "mendeteksi volume yang tercermin. Hasilnya berupa lulus atau gagal, bukan akurasi, dan "
        "karena alasan itulah kami sengaja menjauhkannya dari tabel hasil."))

    add(("h3", "Perbandingan dengan Penelitian Terdahulu"))
    add(("p",
        "Akurasi yang dilaporkan di sini berada di bawah implementasi rujukan Prostate158, yang "
        "menggunakan 3D residual U-Net dan melaporkan Dice central gland sekitar 0,877 serta Dice "
        "peripheral zone sekitar 0,754. Perbedaan tersebut konsisten dengan perbedaan arsitektur: "
        "jaringan 3D dapat memanfaatkan konteks through-plane yang tidak terlihat oleh jaringan "
        "2D. Kami karena itu tidak menyajikan hasil ini sebagai setara dengan rujukan tersebut, "
        "dan perbandingannya diajukan sebagai konteks, bukan sebagai klaim."))
    add(("p",
        "Hubungan antara kedua zona juga dilaporkan secara tidak konsisten lintas penelitian pada "
        "dataset ini. Beberapa melaporkan central gland sebagai zona yang lebih mudah; "
        "setidaknya satu melaporkan sebaliknya. Hasil kami menempatkan central gland sebagai yang "
        "jelas lebih mudah, dan urutan tersebut stabil pada kedua kohort yang ditelaah di sini. "
        "Sebagian ketidakkonsistenan antar penelitian tersebut secara masuk akal dapat "
        "diatribusikan pada protokol alih-alih pada model: agregasi per irisan alih-alih per "
        "kasus, evaluasi pada grid hasil resampling alih-alih ruang voxel asli, serta pembacaan "
        "atas bilangan label itu sendiri, masing-masing menggeser angka yang dilaporkan. Karena "
        "alasan itulah kami memverifikasi pemetaan label secara eksplisit."))
    add(("p",
        "Pada sisi rekonstruksi, literatur memperlakukan konsistensi spasial terutama sebagai "
        "sifat model, yang ditangani melalui cross-slice attention atau pascapemrosesan. Tahap "
        "penyusunan kembali itu sendiri, beserta verifikasinya, biasanya dideskripsikan dalam "
        "satu kalimat, itu pun bila ada. Sepanjang pengetahuan kami belum ada penelitian "
        "terdahulu yang mengukur nilai operasi-operasi individual dalam tahap tersebut dengan cara "
        "menghilangkannya. Pengukuran itulah yang kami pandang sebagai kontribusi penelitian ini "
        "yang dapat digeneralisasi, dan hal tersebut tidak spesifik pada pencitraan prostat: "
        "setiap pipeline yang menghasilkan prediksi per irisan lalu menyusunnya kembali terpapar "
        "pada dua mode kegagalan yang sama."))

    add(("h3", "Implikasi Klinis"))
    add(("p",
        "Dice merupakan metrik riset. Besaran yang menghubungkan segmentasi zonal dengan praktik "
        "klinis adalah volume, dan analisis di sini menunjukkan bahwa penerjemahannya tidak "
        "netral. Peremehan peripheral zone sebesar sekitar seperempat volumenya, dan kelenjar "
        "total sekitar seperdelapan, menggelembungkan PSA density sekitar sepertujuh secara "
        "rerata. Galat tersebut mengarah pada lebih banyak biopsi alih-alih lebih sedikit, yang "
        "merupakan arah lebih aman, namun bersifat sistematis alih-alih acak dan cukup besar "
        "untuk memindahkan pasien melintasi ambang keputusan."))
    add(("p",
        "Batas kesesuaian merupakan besaran operatif bagi penggunaan perorangan, dan nilainya "
        "lebar relatif terhadap ukuran kelenjar. Atas dasar itu kami tidak akan mengusulkan agar "
        "volume yang dihasilkan model ini digunakan untuk PSA density tanpa koreksi atau "
        "peninjauan manusia. Pembacaan konstruktifnya adalah bahwa melaporkan volume dalam "
        "mililiter beserta batas kesesuaian yang eksplisit membuat hal ini terlihat, sedangkan "
        "melaporkan Dice semata tidak."))

    add(("h3", "Keterbatasan"))
    add(("p",
        "Kohort pengembangan berupa satu dataset publik berisi 158 pemeriksaan, dan data uji "
        "memuat 19 kasus. Estimasi dari sampel sebesar itu tidak presisi, dan karena alasan "
        "tersebut kami melaporkan selang kepercayaan alih-alih estimasi titik semata. "
        "Arsitekturnya standar dan tidak dilakukan pencarian arsitektur."))
    add(("p",
        "Data uji tidak sepenuhnya murni. Data tersebut telah digunakan dua kali sebelumnya "
        "selama pengembangan pipeline, sebelum keempat konfigurasi pelatihan yang dibandingkan di "
        "sini dirancang, dan penggunaan terdahulu tersebut melibatkan model yang berbeda. Aturan "
        "pemilihan untuk perbandingan sekarang ditetapkan di muka dan hanya menggunakan data "
        "validasi, serta tidak ada metrik uji yang turut menentukannya. Kami mengungkapkan "
        "penggunaan terdahulu tersebut alih-alih mengklaim data uji yang belum pernah tersentuh."))
    add(("p",
        "Perbandingan statistik dibatasi oleh ukuran sampel dan oleh struktur penelitian. Uji yang "
        "ditetapkan di muka atas konfigurasi terpilih terhadap baseline tidak mencapai ambang "
        "konvensional. Uji yang dihitung pada data validasi bersifat sirkular dan dilaporkan "
        "sebagai deskriptif belaka; uji per zona pada data uji bersifat post hoc dan tidak akan "
        "bertahan terhadap koreksi untuk perbandingan ganda. Tidak satu pun di antaranya boleh "
        "dibaca sebagai penegakan keunggulan."))
    add(("p",
        "Kohort eksternal dianotasi oleh kelompok berbeda di bawah protokol berbeda, sehingga "
        "angka eksternal mencampurkan generalisasi model dengan perbedaan anotasi dan tidak dapat "
        "diuraikan menjadi keduanya. Pipeline tidak melakukan resampling in-plane, sehingga "
        "sebagian selisih eksternal dapat diatribusikan pada pergeseran skala piksel alih-alih "
        "kegagalan model; memisahkan keduanya memerlukan proses kedua dengan adapter resampling, "
        "yang tidak kami lakukan. Mekanisme yang diajukan bagi ekor berat pada galat batas "
        "eksternal merupakan hipotesis yang konsisten dengan pengukuran, bukan sebab yang telah "
        "dibuktikan."))
    add(("p",
        "Terakhir, tidak dilakukan reader study, dan tidak ada klaim yang dibuat mengenai "
        "kegunaan klinis, kesiapan penerapan, maupun dampak terhadap luaran pasien. Analisis "
        "volumetrik menetapkan besar potensi galat, bukan konsekuensi klinisnya dalam praktik."))

    add(("h3", "Simpulan"))
    add(("p",
        f"2D U-Net menyegmentasi central gland dengan Dice {cg_d['mean']:.4f} dan peripheral zone "
        f"dengan Dice {pz_d['mean']:.4f} pada data uji Prostate158, serta mempertahankan macro "
        f"Dice {e['macro']['mean']:.4f} pada kohort independen berisi {e['n_cases']} pemeriksaan "
        f"tanpa adaptasi. Tiga intervensi pelatihan faktor-tunggal tidak menghasilkan perbaikan "
        f"yang melampaui variasi antar pasien, yang melokalisasi galat tersisa pada delineasi "
        f"batas alih-alih pada fungsi objektif pelatihan."))
    add(("p",
        "Rekonstruksi prediksi per irisan menjadi volume yang valid secara spasial "
        "diimplementasikan dengan verifikasi pada setiap tahap dan divalidasi pada kedua kohort. "
        "Pengukuran atas nilai setiap tahap verifikasi menunjukkan bahwa validasi geometris dan "
        "metrik tumpang tindih masing-masing buta terhadap kelas kesalahan rekonstruksi yang "
        "berbeda, sehingga tidak satu pun memadai bila berdiri sendiri, dan bahwa uji fidelitas "
        "round-trip tanpa model adalah yang mendeteksi volume dengan header benar namun voxel "
        "salah. Karena volume zona masuk sebagai penyebut pada PSA density, kesalahan rekonstruksi "
        "yang tidak terlihat oleh Dice tetap dapat mengubah besaran klinis secara substansial. "
        "Kami karena itu merekomendasikan agar penelitian yang melaporkan hasil 3D turunan dari "
        "segmentasi 2D menyatakan bagaimana rekonstruksinya diverifikasi, serta melaporkan volume "
        "beserta batas kesesuaian berdampingan dengan metrik tumpang tindih."))

    # ------------------------------------------------------------ bagian akhir
    add(("h2", "Ucapan Terima Kasih"))
    add(("p",
        "Penulis mengucapkan terima kasih kepada para pembuat dataset Prostate158 dan anotasi "
        "zonal PROSTATEx yang telah menyediakan data serta anotasi pakar mereka secara publik. "
        "Penelitian ini tidak menerima pendanaan eksternal."))

    add(("h2", "Konflik Kepentingan"))
    add(("p", "Tidak ada."))

    add(("h2", "Ketersediaan Data dan Kode"))
    add(("p",
        "Kedua dataset yang digunakan dalam penelitian ini tersedia publik dari sumber aslinya di "
        "bawah lisensi masing-masing. Seluruh kode analisis, berkas konfigurasi, skrip evaluasi, "
        "dan kode pembangkit gambar tersedia pada "
        "https://github.com/amadeussandro/prostate-mri-segmentation-thesis. Setiap nilai "
        "kuantitatif yang dilaporkan dalam naskah ini diturunkan secara terprogram dari artefak "
        "hasil yang diproduksi kode tersebut. Checkpoint model beku diidentifikasi melalui nilai "
        "hash SHA-256 yang tercatat pada repositori, sehingga model yang digunakan pada setiap "
        "analisis yang dilaporkan di sini dapat diverifikasi, bukan diasumsikan."))

    add(("h2", "Singkatan"))
    add(("p",
        "ASD: average surface distance. CG: central gland. HD95: Hausdorff distance persentil "
        "ke-95. IoU: intersection over union. MRI: magnetic resonance imaging. PSA: "
        "prostate-specific antigen. PZ: peripheral zone. SB: simpangan baku. SK: selang "
        "kepercayaan."))

    add(("h2", "Daftar Pustaka"))
    from manuscript_content_en import build as _en_build
    refs = next(p for k, p in _en_build(N) if k == "refs")
    add(("refs", refs))

    return B
