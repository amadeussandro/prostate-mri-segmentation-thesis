"""Indonesian version of the JMIR revision (same structure and numbers as the English one).

Every number comes from `manuscript_numbers_jmir.jmir_numbers()`, exactly as in
`manuscript_content_en_jmir`, so the 2 versions cannot disagree. Decimal points
are turned into commas afterwards by `build_manuscript.localize`; to keep that
safe, this module writes thousands without a separator, P values with a leading
zero, and software versions with a "v" prefix (v3.13), which the converter leaves
alone. Technical terms without a settled Indonesian equivalent (Dice, round-trip
test, affine, header) are kept in English, as in the earlier Indonesian draft.
"""
from __future__ import annotations

from typing import List, Tuple

from manuscript_content_en_jmir import (FIG_J, REFERENCES, _r2, d3, mm, sg)  # noqa: F401

TITLE_CORE = ("Segmentasi Zona Anatomi pada MRI Prostat Menggunakan 2D U-Net dan "
              "Rekonstruksi 3D yang Tervalidasi Secara Spasial")
TITLE = f"{TITLE_CORE}: Studi Pengembangan dan Validasi Algoritma"

AUTHORS = ["Benedict Amadeus Sandro", "Eunike Endariahna Surbakti, S.Kom., M.T.I.",
           "David Agustriawan, S.Kom., M.Sc., Ph.D.", "Moeljono Widjaja, B.Sc., M.Sc., Ph.D."]
AFFIL = ["Program Studi Informatika, Fakultas Teknik dan Informatika, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Penulis Korespondensi: [[TO FILL: nama penulis korespondensi]], Program Studi "
        "Informatika, Fakultas Teknik dan Informatika, Universitas Multimedia Nusantara, "
        "Jl. Scientia Boulevard, Gading Serpong, Tangerang, Banten 15810, Indonesia. "
        "Telepon: [[TO FILL]]. Email: [[TO FILL]]. ORCID: Benedict Amadeus Sandro "
        "0009-0000-1459-9139; Eunike Endariahna Surbakti [[TO FILL]]; David Agustriawan "
        "[[TO FILL]]; Moeljono Widjaja [[TO FILL]]")


def fp(p: float) -> str:
    """P value with a leading zero (Indonesian text); the comma comes from localize()."""
    if p < 0.001:
        return "P<0.001"
    if p < 0.01:
        return f"P={p:.3f}"
    s = f"{p:.2f}"
    if s == "0.05":
        s = f"{p:.3f}"
    return "P=" + s


def th(n: int) -> str:
    """Integers without a thousands separator (a point would be read as a decimal)."""
    return str(int(n))


def msd(d: dict, f=d3) -> str:
    return f"{f(d['mean'])} ({f(d['sd'])})"


def build(N: dict) -> List[Tuple[str, object]]:
    a, ts, es, vs = N["arms"], N["t_s"], N["e_s"], N["vol_s"]
    st, tost, wil = N["stats"], N["stats"]["tost"], N["stats"]["wilcoxon_test_E1_vs_baseline"]
    r2, mf, nc, ov, psa = N["reader2"], N["mf"], N["nci"], N["ov"], N["psa"]
    g, ab, e = N["geometry"], N["ablation"], N["external"]
    eg, hd, gs, stn = e["geometry"], N["hd95_tail"], N["gland_size"], N["anterior_stroma"]
    margin = st["margin"]
    cg, pz = ts["cg"], ts["pz"]
    v1 = ab["V1_no_reorientation"]
    S = mf["summary"]
    p158, px, msd_, nci = S["prostate158"], S["prostatex"], S["msd_task05"], S["nci_isbi_2013"]
    r1 = nc["secondary"]["R1"]["cg"]
    r2pz = nc["secondary"]["R2"]["pz"]
    site15, site3 = "BMC 1.5T endorectal", "Radboud 3T"
    bs = nc["by_site"]
    hv = mf["nci_halfvoxel"]
    t_e1b = tost["test_E1_minus_baseline"]
    dec = nc["decision"]["rules"]
    nd, n1, n2 = mf["n_damaged"], mf["damaged_caught_L1"], mf["damaged_caught_L2"]
    cgm, pzm = r2["dice_CG_model_minus_interreader"], r2["dice_PZ_model_minus_interreader"]

    B: List[Tuple[str, object]] = []
    add = B.append

    # ------------------------------------------------------------------ front
    add(("h2", "Artikel Asli"))
    add(("title", TITLE))
    add(("authors", AUTHORS))
    add(("affil", AFFIL))
    add(("corr", CORR))

    # --------------------------------------------------------------- abstrak
    add(("h2", "Abstrak"))
    add(("abs", ("Latar Belakang",
        "Jaringan dua dimensi (2D) tetap menjadi pilihan praktis untuk magnetic resonance "
        "imaging (MRI) prostat, yang irisannya jauh lebih tebal daripada resolusi dalam "
        "bidangnya. Keluaran per irisan dari jaringan ini harus disusun kembali menjadi volume "
        "3D sebelum dapat dipakai untuk menghitung volume, menampilkan rendering, atau "
        "registrasi. Langkah ini rutin dilakukan, dan kesalahannya bisa tidak terlihat: sebuah "
        "volume dapat memiliki header yang sah tetapi tetap tercermin atau bergeser posisinya.")))
    add(("abs", ("Tujuan",
        "Mengembangkan 2D U-Net untuk segmentasi central gland (CG) dan peripheral zone (PZ), "
        "merekonstruksi keluarannya ke ruang voxel 3D asli dengan verifikasi eksplisit, dan "
        "mengukur pemeriksaan verifikasi mana yang mendeteksi kesalahan rekonstruksi yang mana.")))
    add(("abs", ("Metode",
        f"Empat konfigurasi pelatihan faktor-tunggal dibandingkan pada Prostate158 (119 data "
        f"latih, 20 validasi, dan 19 uji yang disisihkan), dengan uji ekuivalensi terhadap "
        f"margin ±{margin} Dice. Model terpilih dibekukan dan diidentifikasi dengan hash "
        f"checkpoint-nya. Rekonstruksi diverifikasi dengan pemeriksaan geometri berbasis header "
        f"dan round-trip test tanpa model. Enam jenis kesalahan rekonstruksi, sendiri-sendiri "
        f"maupun gabungan, kemudian disuntikkan ke {mf['v0_n']} segmentasi referensi dari 4 "
        f"kohort publik yang disimpan dalam 3 konvensi orientasi. Akurasi dibandingkan dengan "
        f"pembaca ahli kedua, dan model beku diterapkan pada PROSTATEx ({e['n_cases']} "
        f"pemeriksaan) dan NCI-ISBI 2013 ({nc['n']} pemeriksaan), yang terakhir dengan aturan "
        f"connected-component yang ditetapkan sebelum inferensi.")))
    add(("abs", ("Hasil",
        f"Pipeline yang benar mengembalikan seluruh {mf['v0_n']} segmentasi referensi secara "
        f"persis. Dari {th(nd)} rekonstruksi yang rusak, pemeriksaan geometri menandai {th(n1)} "
        f"dan round-trip test {th(n2)}; gabungan keduanya menandai seluruh {th(nd)}, tanpa "
        f"tanda palsu. Melewatkan reorientasi tidak menimbulkan kerusakan pada kohort yang "
        f"disimpan dalam orientasi jaringan itu sendiri, tetapi menurunkan Dice CG menjadi "
        f"{d3(nci['V1_no_reorientation']['dice_CG'])} pada kohort lain. Pada data uji, Dice "
        f"adalah {d3(cg['dice']['mean'])} (IK 95% {d3(cg['dice']['ci_low'])}–"
        f"{d3(cg['dice']['ci_high'])}) untuk CG dan {d3(pz['dice']['mean'])} "
        f"({d3(pz['dice']['ci_low'])}–{d3(pz['dice']['ci_high'])}) untuk PZ, dibandingkan Dice "
        f"antarpembaca {d3(r2['dice_CG_r1_vs_r2']['mean'])} dan "
        f"{d3(r2['dice_PZ_r1_vs_r2']['mean'])}; jarak batas tidak berbeda bermakna dari "
        f"variabilitas antarpembaca. Konfigurasi pelatihan ekuivalen atau tidak konklusif dalam "
        f"±{margin} Dice. Pada NCI-ISBI 2013, Dice CG adalah "
        f"{d3(nc['primary']['cg_dice_none']['mean'])}, dan aturan yang ditetapkan sebelumnya "
        f"menurunkan jarak Hausdorff persentil ke-95 CG dari "
        f"{mm(nc['primary']['cg_hd95_none']['mean'])} menjadi {mm(r1['hd95_after']['mean'])} mm "
        f"({fp(r1['hd95']['wilcoxon_p'])}).")))
    add(("abs", ("Kesimpulan",
        "Pemeriksaan header dan round-trip test tingkat voxel mendeteksi kesalahan rekonstruksi "
        "yang berbeda, dan hanya gabungan keduanya yang menangkap setiap kesalahan yang merusak. "
        "Terlihat atau tidaknya sebuah kesalahan bergantung pada cara data disimpan, sehingga "
        "pipeline yang diuji pada satu dataset publik dapat gagal tanpa terdeteksi pada dataset "
        "lain. Studi yang menurunkan hasil 3D dari segmentasi 2D sebaiknya melaporkan kedua "
        "pemeriksaan tersebut.")))
    add(("abs", ("Kata Kunci",
        "kanker prostat; magnetic resonance imaging; deep learning; segmentasi citra; "
        "pengolahan citra berbantuan komputer; kendali mutu; reprodusibilitas hasil; "
        "pengujian eksternal")))

    # ----------------------------------------------------------- pendahuluan
    add(("h2", "Pendahuluan"))
    add(("p",
        "Kanker prostat adalah kanker kedua yang paling sering didiagnosis pada pria di "
        "seluruh dunia [@sung]. MRI multiparametrik berperan sentral dalam deteksi dan "
        "penentuan stadiumnya, dan pelaporan terstruktur menyusun interpretasi berdasarkan "
        "anatomi zona kelenjar [@turkbey,weinreb]. PZ dan CG, yang mencakup zona transisi dan "
        "zona sentral [@mcneal], berbeda dalam tampilan dan dalam makna tumor yang tumbuh di "
        "dalamnya, dan volume zona menjadi masukan bagi besaran seperti densitas "
        "prostate-specific antigen (PSA) [@nordstrom]. Dengan demikian, segmentasi zona adalah "
        "masukan bagi pengukuran klinis, bukan tujuan akhir."))
    add(("p",
        "Deep learning merupakan pendekatan standar untuk tugas ini "
        "[@ronneberger,isensee,litjens_survey]. MRI prostat sangat anisotropik, dan jaringan 2D "
        "tetap menjadi pilihan yang wajar untuk geometri tersebut: jaringan ini tidak "
        "melakukan interpolasi sepanjang sumbu yang paling jarang, murah untuk dilatih, dan "
        "penelitian sebelumnya telah menyegmentasi kedua zona dengan cara ini [@bardis,astono], "
        "dan pada benchmark MRI prostat, jaringan berbasis konvolusi 2D menyamai atau melampaui "
        "3D U-Net [@zhang25d,hung_catnet]. "
        "Namun, keluarannya berupa tumpukan irisan. Apa pun yang memakai segmentasi sebagai "
        "objek 3D mengharuskan irisan dikembalikan ke posisi aksialnya, prapemrosesan dibalik, "
        "dan geometri sumber dipulihkan [@cheng]."))
    add(("p",
        "Kesalahan dalam rantai tersebut diketahui terjadi dan sering luput. Pembalikan "
        "kiri-kanan cukup umum pada data MRI publik sehingga pemeriksaan konsistensi khusus "
        "telah diusulkan [@glen], dan konversi antara DICOM dan NIfTI merupakan sumber kesalahan "
        "orientasi dan posisi yang terdokumentasi [@li_dcm2niix]. Kesalahan ini sulit terlihat "
        "karena tidak merusak berkas. Volume yang tercermin dengan header yang benar tetap "
        "merupakan citra NIfTI yang sah, dan array voxel yang benar yang ditulis dengan affine "
        "bawaan tetap menghasilkan volume dalam mililiter. Pedoman pelaporan meminta penulis "
        "menjelaskan prapemrosesan dan membagikan kode [@claim,tripod_ai], tetapi tidak menanyakan "
        "bagaimana volume hasil rekonstruksi diverifikasi."))
    add(("p",
        "Yang belum tersedia adalah bukti tentang pemeriksaan mana yang benar-benar mendeteksi "
        "kesalahan yang mana, dan apakah jawabannya berlaku lintas dataset. Penelitian ini "
        "memiliki 2 tujuan. Pertama, mengembangkan 2D U-Net untuk segmentasi CG dan PZ dengan "
        "protokol pemilihan yang ditetapkan sebelum data uji diperiksa, serta membandingkannya "
        "dengan pembaca ahli kedua dan pada 2 kohort eksternal. Kedua, yang kami pandang sebagai "
        "kontribusi yang lebih dapat dialihkan, merekonstruksi keluarannya ke ruang voxel asli "
        "dengan verifikasi eksplisit dan mengukur, dengan menyuntikkan kesalahan yang diketahui "
        "ke segmentasi referensi dari 4 kohort publik, pemeriksaan mana yang mendeteksi "
        "kesalahan yang mana."))

    # ---------------------------------------------------------------- metode
    add(("h2", "Metode"))
    add(("h3", "Pertimbangan Etis"))
    add(("p",
        "Penelitian ini tidak melibatkan partisipan manusia secara langsung. Seluruh citra dan "
        "anotasi berasal dari 4 koleksi publik yang telah dideidentifikasi (Prostate158, "
        "PROSTATEx beserta anotasi zonanya, NCI-ISBI 2013, dan tugas prostat Medical "
        "Segmentation Decathlon), yang dikumpulkan oleh institusi asalnya di bawah pengawasan "
        "etik masing-masing dan dirilis untuk penelitian sekunder, sebagaimana dijelaskan dalam "
        "publikasi sumbernya [@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli]. Koleksi "
        "tersebut didistribusikan dengan lisensi terbuka yang mengizinkan penggunaan ulang "
        "dengan atribusi (misalnya, NCI-ISBI 2013 dengan CC BY 3.0 dan Medical Segmentation "
        "Decathlon dengan CC BY-SA 4.0). Tidak ada informasi identitas yang diakses. Karena "
        "penelitian ini merupakan analisis sekunder atas data publik yang telah "
        "dideidentifikasi, persetujuan komite etik lokal tambahan dan informed consent tidak "
        "diperlukan."))
    add(("h3", "Desain Penelitian dan Pelaporan"))
    add(("p",
        "Penelitian ini merupakan studi retrospektif pengembangan dan validasi algoritma pada "
        "data publik yang telah dideidentifikasi. Pelaporannya mengikuti Checklist for "
        "Artificial Intelligence in Medical Imaging (CLAIM) 2024 [@claim] (Lampiran Multimedia "
        "1). Alur penelitian ditunjukkan pada Gambar 1. Penelitian ini memiliki 2 tingkat "
        "pemilihan model, keduanya ditetapkan sebelum data uji diperiksa. Satu analisis, yaitu "
        "aturan pascapemrosesan connected-component untuk kohort eksternal konfirmatori, "
        "dituliskan dan hash berkasnya dicatat sebelum dijalankan (Lampiran Multimedia 2)."))
    add(("fig", (f"{FIG_J}/F1_workflow.png",
        'Gambar 1. Alur penelitian dan jumlah kasus.')))

    add(("h3", "Sumber Data"))
    add(("p",
        "Empat kohort publik digunakan (Tabel 1). Prostate158 [@adams_cbm,adams_dib] menyediakan "
        "data pengembangan, dengan pembagian resmi tingkat pasien berupa 119 data latih, 20 "
        "validasi, dan 19 uji. Data validasi hanya dipakai untuk pemilihan model; data uji "
        "dipakai untuk pengujian internal. Masukannya adalah seri T2-weighted aksial; standar "
        "referensinya adalah mask anatomi pembaca 1, dengan 0 untuk latar, 1 untuk CG, dan 2 "
        "untuk PZ. Mask pembaca 2, yang tersedia untuk ke-19 kasus uji, hanya dipakai sebagai "
        "referensi kedua. Makna bilangan label diverifikasi terhadap anatomi, bukan diasumsikan, "
        "karena pembacaan yang terbalik akan menukar kedua zona secara diam-diam di setiap "
        "metrik."))
    add(("p",
        f"PROSTATEx [@litjens,clark] dengan anotasi zona yang dirilis publik "
        f"[@cuocolo_ejr,cuocolo_jmri] ({e['n_cases']} pemeriksaan) dan NCI-ISBI 2013 "
        f"[@bloch_nci] adalah kohort eksternal yang hanya dipakai untuk inferensi. Pada "
        f"PROSTATEx, kelas yang bersesuaian dengan CG juga mencakup anterior fibromuscular "
        f"stroma. NCI-ISBI 2013 menyediakan label CG dan PZ untuk {nc['n_total']} pemeriksaan "
        f"dari 2 lokasi, yaitu pemindai 1,5 T dengan koil endorektal dan pemindai 3 T dengan "
        f"koil permukaan. Labelnya dicocokkan dengan seri citra berdasarkan identitas pasien, "
        f"dan maknanya diverifikasi secara anatomi: label 1 terletak posterior terhadap label 2 "
        f"pada setiap kasus sehingga merupakan PZ. Dua kasus dikeluarkan, satu karena mask-nya "
        f"digambar pada seri lain (23 vs 34 irisan) dan satu karena hanya membawa satu label "
        f"kelenjar utuh, sehingga tersisa {nc['n']}. Tugas prostat Medical Segmentation "
        f"Decathlon (MSD) [@antonelli] ({mf['n_cases']['msd_task05']} dari 32 kasus berlabel "
        f"dengan kedua zona) hanya dipakai untuk eksperimen penyuntikan kesalahan, karena "
        f"disimpan dalam konvensi orientasi ketiga."))
    add(("p",
        f"Tumpang tindih antarkohort eksternal diperiksa dengan korelasi citra per voxel. Dari "
        f"pemeriksaan NCI-ISBI 3 T, {ov['n_msd']} merupakan citra MSD dengan anotasi "
        f"independen, dan 1 juga merupakan kasus PROSTATEx. Prostate158 diakuisisi di "
        f"institusi yang berbeda dari seluruh kohort eksternal, sehingga data pengembangan "
        f"tidak mungkin termasuk di dalamnya. Tumpang tindih ini tidak memengaruhi data latih; "
        f"hal ini dilaporkan karena kohort-kohort tersebut tidak sepenuhnya independen, dan "
        f"menyediakan set anotasi kedua untuk {ov['n_two']} citra eksternal."))
    add(("tbl", ("Tabel 1. Kohort yang digunakan dalam penelitian ini.",
        ["Kohort", "Penggunaan", "Pemeriksaan, n", "Pemindai dan lokasi", "Orientasi tersimpan",
         "Spasi dalam bidang, mm", "Anotasi referensi"],
        [["Prostate158", "Latih, validasi, uji", "158 (119/20/19)", "3 T, 1 institusi", "LPS",
          "0.27–0.50", "Zona (pembaca 1; pembaca 2 pada data uji)"],
         ["PROSTATEx", "Pengujian eksternal (sekunder)", str(e["n_cases"]), "3 T, 1 institusi",
          "LAS", f"{eg['inplane_min']:.2f}–{eg['inplane_max']:.2f}",
          "Zona; kelas CG mencakup stroma anterior"],
         ["NCI-ISBI 2013", "Pengujian eksternal (konfirmatori)",
          f"{nc['n']} dari {nc['n_total']}", "1,5 T koil endorektal dan 3 T, 2 institusi", "LPS",
          f"{_r2(nc['inplane_min'])}–{_r2(nc['inplane_max'])}", "Zona (CG, PZ)"],
         ["MSD Task05", "Hanya penyuntikan kesalahan", f"{mf['n_cases']['msd_task05']} dari 32",
          "3 T, 1 institusi", "RAS", "0.60–0.75", "Zona (zona transisi, PZ)"]])))

    add(("h3", "Prapemrosesan"))
    add(("p",
        "Volume direorientasi ke kerangka RAS kanonik. Intensitas dipotong pada persentil ke-0,5 "
        "dan ke-99,5 lalu dinormalisasi z-score per volume; mask tidak pernah dinormalisasi. "
        "Grid dalam bidang diseragamkan menjadi 442×442 dengan zero-padding simetris atau "
        "pemotongan di tengah, dan sumbu antar-irisan tidak di-resample. Dengan demikian, jalur "
        "mask tidak melibatkan interpolasi, dan setiap parameter yang dibutuhkan untuk "
        "membaliknya (transformasi orientasi, offset padding atau pemotongan per sumbu, dan "
        "kedalaman) disimpan untuk setiap kasus."))

    add(("h3", "Model dan Pelatihan"))
    add(("p",
        "Jaringan yang digunakan adalah 2D U-Net 4 tingkat [@ronneberger] dengan 32 feature map "
        "awal, 1 kanal masukan dan 3 kanal keluaran, serta 7762531 parameter (Gambar 2). Setiap "
        "tingkat menerapkan 2 konvolusi 3×3 dengan batch normalization dan rectified linear "
        "unit; downsampling memakai max pooling 2×2 dan upsampling memakai transposed "
        "convolution dengan skip connection. Dropout 0,2 diterapkan pada bottleneck. Prediksi "
        "adalah arg-max atas 3 kelas, sehingga keluarannya selalu berupa partisi. Arsitektur "
        "standar dipilih dengan sengaja: pipeline yang mengonfigurasi diri menunjukkan bahwa "
        "U-Net tanpa modifikasi tetap kompetitif [@isensee], dan arsitektur baru akan "
        "mencampuradukkan analisis rekonstruksi dengan desain model."))
    add(("fig", (f"{FIG_J}/F2_unet.png",
        'Gambar 2. Arsitektur 2D U-Net (garis putus-putus: skip connection).')))
    add(("p",
        "Empat konfigurasi dilatih, masing-masing berbeda dari pembandingnya pada 1 faktor: "
        "baseline cross-entropy; baseline ditambah suku soft Dice pada 2 kelas latar depan (E1); "
        "baseline ditambah augmentasi geometris dan intensitas (E2); serta E1 ditambah "
        "cross-entropy berbobot kelas yang memperbesar bobot PZ (E3). Dengan demikian, E3 "
        "merupakan perubahan faktor-tunggal terhadap E1, bukan terhadap baseline. Semua "
        "konfigurasi memakai AdamW (learning rate 0,001, weight decay 0,0001), batch size 8, "
        "100 epoch, presisi 32-bit, seed 42, dan sampler pelatihan yang mengambil irisan "
        "dengan dan tanpa latar depan dengan frekuensi harapan yang sama."))
    add(("p",
        "Dalam setiap run, epoch dengan Dice validasi tingkat irisan tertinggi dipertahankan. "
        "Antar-run, konfigurasi dengan macro Dice tingkat kasus tertinggi (rerata CG dan PZ) "
        "pada 20 kasus validasi dipilih. Tidak ada metrik data uji yang ikut dalam kedua "
        "keputusan tersebut. Checkpoint terpilih selanjutnya diidentifikasi dengan hash SHA-256 "
        "dari bobotnya."))

    add(("h3", "Rekonstruksi 3D dan Verifikasi"))
    add(("p",
        "Setiap irisan prediksi ditulis kembali ke indeks aksial yang dibawanya, tidak pernah "
        "menurut urutan loader; indeks yang hilang atau ganda memicu error. Volume kemudian "
        "ditransformasi balik melalui rantai prapemrosesan dalam urutan terbalik dengan "
        "interpolasi nearest-neighbor dan ditulis sebagai NIfTI dengan affine dan spasi sumber "
        "(Gambar 3). Empat pemeriksaan didefinisikan pada 3 tingkat."))
    add(("p",
        "Tingkat 1, pemeriksaan geometri, membaca header berkas yang ditulis dan membandingkan "
        "bentuk array, affine, spasi voxel, kode orientasi, dan himpunan label dengan sumbernya. "
        "Tingkat 2, round-trip test, tidak memakai model: mask referensi dilewatkan maju "
        "melalui prapemrosesan dan kembali melalui kode rekonstruksi yang sama, dan harus "
        "pulih voxel demi voxel. Karena transformasi mask berupa permutasi sumbu bertanda yang "
        "diikuti padding atau pemotongan, pemulihan persis memang diharapkan, dan hasilnya "
        "dilaporkan sebagai lulus atau gagal, tidak pernah sebagai akurasi. Tingkat 3 terdiri "
        "atas 2 pemeriksaan anatomi yang tidak memerlukan referensi sehingga dapat dijalankan "
        "saat inferensi: centroid PZ harus terletak posterior terhadap centroid CG dalam "
        "koordinat pemindai, dan kelenjar harus menempati rangkaian irisan yang bersambung "
        "dengan centroid dalam bidang yang tidak berpindah lebih dari 15 mm antara irisan yang "
        "berdekatan. Kedua ambang ditetapkan sebelum hasil penyuntikan kesalahan dibaca."))
    add(("fig", (f"{FIG_J}/F3_reconstruction_verification.png",
        'Gambar 3. Pipeline rekonstruksi dan 3 tingkat verifikasi.')))

    add(("h3", "Eksperimen Penyuntikan Kesalahan"))
    add(("p",
        f"Untuk mengukur apa yang dideteksi setiap pemeriksaan, segmentasi referensi dari "
        f"keempat kohort ({mf['v0_n']} kasus) dilewatkan melalui transformasi maju produksi "
        f"dan direkonstruksi dengan benar atau dengan 1 kesalahan: reorientasi tidak dibalik; "
        f"pemotongan di tengah alih-alih offset yang disimpan; affine identitas; affine yang "
        f"hanya membawa spasi voxel tanpa origin dan arah; irisan ditumpuk menurut urutan "
        f"loader; irisan ditumpuk dalam urutan acak (seed tetap); dan seluruh kesalahan "
        f"sekaligus. Karena tidak ada model yang terlibat, setiap perbedaan dari referensi "
        f"pasti disebabkan oleh kesalahan itu sendiri. Sebuah kasus dihitung rusak jika Dice "
        f"terhadap referensi turun di bawah 1, centroid berpindah, atau volume berubah. "
        f"Eksperimen ini memiliki gerbang: pipeline yang benar harus mengembalikan setiap "
        f"referensi secara persis sebelum kesalahan apa pun dianalisis. Kesalahan yang sama "
        f"juga diterapkan pada prediksi tersimpan dari 19 kasus uji, untuk menunjukkan apa yang "
        f"akan dilihat pembaca hasilnya."))

    add(("h3", "Evaluasi dan Pembanding Antarpembaca"))
    add(("p",
        "Seluruh metrik dihitung per kasus di ruang voxel asli terhadap mask yang dibaca dari "
        "disk: Dice, intersection over union, jarak Hausdorff persentil ke-95 (HD95), average "
        "surface distance, presisi, dan recall [@taha,maierhein], dengan jarak permukaan dalam "
        "milimeter dari spasi masing-masing kasus. Macro Dice adalah rerata 2 kelas latar "
        "depan. Pada 19 kasus uji, pembaca 2 dibandingkan dengan pembaca 1 menggunakan kode "
        "yang sama, sehingga kesalahan model dapat disandingkan dengan ketidaksepakatan para "
        "ahli, seperti pada studi segmentasi lain yang membandingkan dengan variasi "
        "antarpengamat [@nikolov]. Volume zona dihitung dari determinan absolut affine dan "
        "dibandingkan dengan analisis Bland-Altman [@bland]."))
    add(("p",
        "Karena nilai PSA tidak tersedia, konsekuensi klinis dari kesalahan volume diperiksa "
        "sebagai analisis sensitivitas. Untuk setiap pasien, dihitung rentang nilai PSA di mana "
        "volume referensi dan volume prediksi menempatkan densitas PSA pada sisi yang berbeda "
        "dari ambang 0,15 ng/mL/mL, dan proporsi pasangan pasien-PSA yang berpindah kelas "
        "dirangkum pada zona abu-abu diagnostik 4 hingga 10 ng/mL."))

    add(("h3", "Pengujian Eksternal"))
    add(("p",
        "Model beku diterapkan pada kedua kohort eksternal tanpa pelatihan, penyetelan, "
        "pencarian ambang, atau test-time augmentation, melalui kode inferensi dan "
        "rekonstruksi yang sama. PROSTATEx telah dianalisis sebelumnya, dan dekomposisi "
        "connected-component di sana menunjukkan bahwa ekor kesalahan batas CG-nya terdiri atas "
        "komponen positif palsu yang kecil dan jauh. Karena pengamatan itu bersifat post hoc, "
        "PROSTATEx diperlakukan sebagai kohort sekunder dan pascapemrosesan tidak diterapkan "
        "padanya."))
    add(("p",
        "Sebelum inferensi apa pun pada NCI-ISBI 2013, sebuah aturan dituliskan dan hash "
        "berkasnya dicatat di setiap keluaran (Lampiran Multimedia 2): pertahankan komponen CG "
        "6-connected terbesar (R1), dan, sebagai analisis sekunder, juga komponen PZ terbesar "
        "(R2). Aturan ini tidak memiliki parameter. Aturan terlebih dahulu diperiksa agar tidak "
        "merugikan pada 20 kasus validasi, dengan syarat rerata Dice tidak turun lebih dari "
        "0,005 dan rerata HD95 tidak naik. Luaran primer NCI-ISBI adalah pipeline tanpa "
        "modifikasi; efek aturan merupakan luaran sekunder yang ditetapkan sebelumnya. Satu "
        "koreksi dicatat sebelum inferensi: satu kasus yang hanya membawa satu label kelenjar "
        "utuh dikeluarkan."))

    add(("h3", "Analisis Statistik"))
    add(("p",
        f"Selisih berpasangan per kasus diuji dengan uji peringkat bertanda Wilcoxon. "
        f"Perbandingan yang ditetapkan sebelumnya adalah macro Dice konfigurasi terpilih "
        f"terhadap baseline pada data uji. Uji per zona bersifat post hoc dan dikoreksi dengan "
        f"metode Holm [@holm] untuk 2 zona. Karena selisih yang tidak bermakna tidak membuktikan "
        f"ketiadaan efek, ekuivalensi diuji dengan two one-sided tests (TOST) "
        f"[@schuirmann,lakens] pada selisih berpasangan dengan margin ±{margin} Dice; "
        f"ekuivalensi disimpulkan bila IK 90% berada di dalam margin. Margin tidak didaftarkan "
        f"sebelumnya. Margin dipilih lebih kecil daripada SB Dice antarpembaca untuk kedua zona "
        f"({d3(r2['dice_CG_r1_vs_r2']['sd'])} untuk CG dan {d3(r2['dice_PZ_r1_vs_r2']['sd'])} "
        f"untuk PZ). Uji pada data validasi bersifat deskriptif, karena data tersebut dipakai "
        f"untuk pemilihan. Sebaran dinyatakan sebagai SB sampel, dan IK merupakan interval "
        f"persentil bootstrap (10000 resampling, seed tetap). Pelatihan memakai 1 GPU NVIDIA "
        f"Tesla T4; seluruh analisis lain dijalankan pada CPU dengan Python v3.13 (Python "
        f"Software Foundation), PyTorch v2.14 (Linux Foundation), NiBabel v5.4, SimpleITK v2.4, "
        f"dan SciPy v1.18."))

    # ----------------------------------------------------------------- hasil
    add(("h2", "Hasil"))
    add(("h3", "Pemilihan Model dan Ekuivalensi Konfigurasi Pelatihan"))
    e2b, e3e1 = tost["val_E2_minus_baseline"], tost["val_E3_minus_E1"]
    add(("p",
        f"E1 memiliki macro Dice validasi tertinggi ({d3(a['e1']['macro'])}) dan dipilih "
        f"(Tabel 2). Keempat konfigurasi terentang {d3(N['arm_spread'])} macro Dice. "
        f"Augmentasi ekuivalen dengan baseline dalam ±{margin} untuk macro Dice dan Dice CG "
        f"({fp(e2b['macro']['tost_p'])} dan {fp(e2b['CG']['tost_p'])}), tetapi tidak terbukti "
        f"ekuivalen untuk PZ ({fp(e2b['PZ']['tost_p'])}). Pembobotan kelas ekuivalen dengan E1 "
        f"di setiap zona (semua {fp(max(e3e1[z]['tost_p'] for z in ('macro', 'CG', 'PZ')))} "
        f"atau lebih kecil). Kedua intervensi yang umum direkomendasikan ini tidak menghasilkan "
        f"selisih yang terbukti melampaui margin, meskipun ekuivalensi untuk PZ pada augmentasi "
        f"belum terbukti."))
    rowsT2 = []
    comp = {"baseline": None, "e1": ("val_E1_minus_baseline", "baseline"),
            "e2": ("val_E2_minus_baseline", "baseline"), "e3": ("val_E3_minus_E1", "E1")}
    for k, d in a.items():
        name = "Baseline" if k == "baseline" else k.upper()
        if comp[k] is None:
            diff, tp = "Referensi", "—"
        else:
            t = tost[comp[k][0]]["macro"]
            diff = (f"{sg(t['mean_diff'])} vs {comp[k][1]} "
                    f"({d3(t['ci90'][0])} s.d. {d3(t['ci90'][1])})")
            tp = fp(t["tost_p"])
        label = {"baseline": "Cross-entropy", "e1": "Baseline + soft Dice pada CG dan PZ",
                 "e2": "Baseline + augmentasi", "e3": "E1 + cross-entropy berbobot kelas"}[k]
        rowsT2.append([name, label, str(d["epoch"]), d3(d["cg"]), d3(d["pz"]), d3(d["macro"]),
                       diff, tp])
    add(("tbl", (f"Tabel 2. Konfigurasi pelatihan pada 20 kasus validasi. Masing-masing berbeda "
                 f"dari pembandingnya pada 1 faktor. Selisih: macro Dice, rerata (IK 90%). P TOST: "
                 f"ekuivalensi dalam ±{margin} Dice (deskriptif; data ini dipakai untuk "
                 f"pemilihan).",
        ["Konfigurasi", "Perubahan", "Epoch terbaik", "Dice CG", "Dice PZ", "Macro Dice",
         "Selisih (IK 90%)", "P TOST"], rowsT2)))

    add(("h3", "Performa pada Data Uji dan Kesepakatan Antarpembaca"))
    add(("p",
        f"Pada 19 kasus uji, rerata Dice adalah {d3(cg['dice']['mean'])} (SB "
        f"{d3(cg['dice']['sd'])}) untuk CG dan {d3(pz['dice']['mean'])} (SB "
        f"{d3(pz['dice']['sd'])}) untuk PZ (Tabel 3). PZ memadukan presisi tinggi "
        f"({d3(pz['precision']['mean'])}) dengan recall yang lebih rendah "
        f"({d3(pz['recall']['mean'])}): model jarang melabeli jaringan lain sebagai PZ, tetapi "
        f"melabeli PZ terlalu sedikit. Terhadap pembaca 1, Dice model lebih rendah daripada "
        f"pembaca 2 sebesar {d3(-cgm['mean_diff'])} untuk CG (IK 95% {d3(-cgm['ci95'][1])}–"
        f"{d3(-cgm['ci95'][0])}; {fp(cgm['wilcoxon_p'])}) dan {d3(-pzm['mean_diff'])} untuk PZ "
        f"({d3(-pzm['ci95'][1])}–{d3(-pzm['ci95'][0])}; {fp(pzm['wilcoxon_p'])}), yaitu "
        f"{cgm['ratio_of_means']*100:.1f}% dan {pzm['ratio_of_means']*100:.1f}% dari "
        f"kesepakatan antarpembaca. Jarak batas tidak berbeda bermakna dari variabilitas "
        f"antarpembaca, meskipun ekuivalensinya tidak diuji: HD95 adalah "
        f"{mm(r2['hd95_CG_model_vs_r1']['mean'])} mm dibanding "
        f"{mm(r2['hd95_CG_r1_vs_r2']['mean'])} mm untuk CG "
        f"({fp(r2['hd95_CG_model_minus_interreader']['wilcoxon_p'])}) dan "
        f"{mm(r2['hd95_PZ_model_vs_r1']['mean'])} mm dibanding "
        f"{mm(r2['hd95_PZ_r1_vs_r2']['mean'])} mm untuk PZ "
        f"({fp(r2['hd95_PZ_model_minus_interreader']['wilcoxon_p'])})."))
    m = wil["macro"]
    add(("p",
        f"Terhadap baseline, perbandingan macro yang ditetapkan sebelumnya tidak bermakna "
        f"({sg(m['mean_diff'])}, lebih baik pada {m['n_better']} dari {m['n']} kasus; "
        f"{fp(m['p_raw'])}) dan juga tidak ekuivalen (IK 90% {d3(t_e1b['macro']['ci90'][0])} "
        f"s.d. {d3(t_e1b['macro']['ci90'][1])}), sehingga efek keseluruhannya tidak konklusif. "
        f"Secara post hoc, Dice CG ekuivalen ({fp(t_e1b['CG']['tost_p'])}) dan Dice PZ lebih "
        f"tinggi {d3(wil['PZ']['mean_diff'])} (lebih baik pada {wil['PZ']['n_better']} dari "
        f"{wil['PZ']['n']} kasus; P terkoreksi Holm {fp(wil['PZ']['p_holm_zones'])[2:]})."))

    def rr(name, key, f=d3):
        return [name, msd(cg[key], f), "—", msd(pz[key], f), "—"]
    add(("tbl", ("Tabel 3. Performa model terpilih pada data uji (19 kasus, ruang voxel asli, "
                 "rerata (SB) per kasus) dan kesepakatan antara 2 pembaca ahli pada kasus yang "
                 "sama.",
        ["Metrik", "CG, model vs pembaca 1", "CG, pembaca 2 vs pembaca 1",
         "PZ, model vs pembaca 1", "PZ, pembaca 2 vs pembaca 1"],
        [["Dice", msd(cg["dice"]), msd(r2["dice_CG_r1_vs_r2"]), msd(pz["dice"]),
          msd(r2["dice_PZ_r1_vs_r2"])],
         ["Dice, IK 95%", f"{d3(cg['dice']['ci_low'])}–{d3(cg['dice']['ci_high'])}", "—",
          f"{d3(pz['dice']['ci_low'])}–{d3(pz['dice']['ci_high'])}", "—"],
         ["HD95, mm", msd(cg["hd95_mm"], mm), msd(r2["hd95_CG_r1_vs_r2"], mm),
          msd(pz["hd95_mm"], mm), msd(r2["hd95_PZ_r1_vs_r2"], mm)],
         ["Average surface distance, mm", msd(cg["asd_mm"], mm),
          msd(r2["asd_CG_r1_vs_r2"], mm), msd(pz["asd_mm"], mm), msd(r2["asd_PZ_r1_vs_r2"], mm)],
         rr("Intersection over union", "iou"), rr("Presisi", "precision"),
         rr("Recall", "recall")])))

    add(("h3", "Verifikasi Rekonstruksi dan Penyuntikan Kesalahan"))
    add(("p",
        f"Seluruh {g['n']} rekonstruksi data uji lulus pemeriksaan geometri, dengan affine yang "
        f"identik dengan sumbernya dalam batas pembulatan presisi tunggal, dan round-trip test "
        f"memulihkan setiap mask referensi secara persis. Dalam eksperimen penyuntikan "
        f"kesalahan, pipeline yang benar mengembalikan seluruh {mf['v0_n']} referensi secara "
        f"persis (Tabel 4). Dari {th(nd)} rekonstruksi yang rusak, pemeriksaan geometri menandai "
        f"{th(n1)} ({n1/nd*100:.1f}%), round-trip test {th(n2)} ({n2/nd*100:.1f}%), dan gabungan "
        f"keduanya seluruh {th(nd)}, sementara tidak satu pun dari keduanya menandai "
        f"{mf['n_undamaged']} rekonstruksi berkesalahan yang tidak menimbulkan kerusakan "
        f"(Gambar 4)."))
    v6 = [c["V6_spacing_only"]["shift_CG_mm"] for c in S.values()]
    add(("p",
        f"Kedua pemeriksaan saling melengkapi. Volume yang tidak direorientasi membawa header "
        f"yang benar tetapi voxel yang salah, sehingga pemeriksaan geometri meloloskannya di "
        f"setiap kohort dan hanya round-trip test yang menangkapnya. Pada prediksi data uji, "
        f"volume ini akan dilaporkan dengan Dice CG {d3(v1['cg']['dice_vs_gt'])} dan Dice PZ "
        f"{d3(v1['pz']['dice_vs_gt'])}, dengan centroid PZ bergeser rerata "
        f"{mm(v1['pz']['centroid_shift_mm'])} mm (maksimum "
        f"{mm(v1['pz']['centroid_shift_max_mm'])} mm), sambil tetap lulus setiap pemeriksaan "
        f"header (Gambar 5). Sebaliknya, affine identitas atau affine yang hanya membawa spasi "
        f"tidak mengubah voxel, sehingga Dice tidak berubah dan hanya pemeriksaan geometri yang "
        f"mendeteksinya. Affine yang hanya membawa spasi adalah yang lebih berbahaya: volume "
        f"tetap persis sama (kesalahan 0%) sementara kelenjar berpindah {mm(min(v6))} hingga "
        f"{mm(max(v6))} mm, sehingga pemeriksaan mutu berbasis volume pun akan meloloskannya."))
    add(("p",
        f"Terlihat atau tidaknya sebuah kesalahan bergantung pada cara data disimpan. "
        f"Melewatkan reorientasi tidak menimbulkan kerusakan pada MSD, yang disimpan dalam "
        f"orientasi RAS yang dipakai jaringan; pada PROSTATEx (LAS) hanya menghasilkan "
        f"pembalikan kiri-kanan (Dice CG {d3(px['V1_no_reorientation']['dice_CG'])}, PZ "
        f"{d3(px['V1_no_reorientation']['dice_PZ'])}); dan pada 2 kohort LPS menghancurkan "
        f"tumpang tindih (Dice CG {d3(nci['V1_no_reorientation']['dice_CG'])} hingga "
        f"{d3(p158['V1_no_reorientation']['dice_CG'])}). Pipeline yang dikembangkan dan diuji "
        f"hanya pada MSD karenanya akan lulus, lalu gagal tanpa terdeteksi pada data LPS. "
        f"Pemeriksaan anatomi, yang tidak memerlukan referensi, menandai "
        f"{th(mf['damaged_caught_L3'])} dari {th(nd)} kasus rusak "
        f"({mf['damaged_caught_L3']/nd*100:.0f}%). Pemeriksaan ini melewatkan pembalikan "
        f"kiri-kanan, yang tidak mengubah urutan anterior-posterior, dan menghasilkan "
        f"{mf['v0_L3b_false']} alarm palsu pada pipeline yang benar "
        f"({mf['v0_L3b_false']/mf['v0_n']*100:.1f}%), sehingga hanya berfungsi sebagai "
        f"penyaring saat inferensi, bukan pengganti round-trip test. Pemotongan di tengah dan "
        f"penumpukan menurut urutan loader tidak menimbulkan kerusakan di kohort mana pun, "
        f"karena pipeline ini melakukan padding secara simetris dan loader-nya mengeluarkan "
        f"irisan dalam urutan aksial. Ini adalah sifat implementasi ini, bukan bukti bahwa "
        f"langkah tersebut tidak diperlukan."))
    add(("p",
        f"Eksperimen ini juga mengungkap inkonsistensi geometri yang nyata. Seluruh {hv['n']} "
        f"mask NCI-ISBI dari lokasi 3 T memiliki ukuran, spasi, dan arah yang sama dengan seri "
        f"citranya, tetapi origin yang tersimpan bergeser tepat setengah voxel pada setiap "
        f"sumbu. Anotasinya sejajar dengan voxel citra, sehingga anotasi disalin ke grid citra. "
        f"Resampling di ruang fisik, respons yang lazim ketika 2 origin berbeda, akan "
        f"memindahkan label dan menurunkan kesesuaiannya dengan anotasi asli menjadi Dice "
        f"{d3(hv['pz_dice_mean'])} untuk PZ (minimum {d3(hv['pz_dice_min'])}) dan "
        f"{d3(hv['cg_dice_mean'])} untuk CG (minimum {d3(hv['cg_dice_min'])}) sebelum model apa "
        f"pun diterapkan. Citra yang sama yang didistribusikan dalam MSD juga bergeser setengah "
        f"voxel dari geometri turunan DICOM."))

    def cell(c, v):
        x = c[v]
        if (x["dice_CG"] == 1 and x["dice_PZ"] == 1 and abs(x["vol_err_CG_pct"]) < .01
                and x["shift_CG_mm"] < .01):
            return "Tidak rusak"
        if x["dice_CG"] == 1 and x["dice_PZ"] == 1:
            ve = x["vol_err_CG_pct"]
            vtxt = "0%" if abs(ve) < 0.5 else f"{ve:+.0f}%".replace("-", "−")
            return f"Dice 1.000; geser {x['shift_CG_mm']:.0f} mm; volume {vtxt}"
        return f"Dice {d3(x['dice_CG'])}/{d3(x['dice_PZ'])}"
    faults = [("V1_no_reorientation", "Reorientasi tidak dibalik", "Round-trip"),
              ("V3_identity_affine", "Affine identitas", "Geometri"),
              ("V6_spacing_only", "Affine hanya spasi", "Geometri"),
              ("V4b_shuffled_order", "Urutan irisan acak", "Round-trip"),
              ("V2_centre_crop", "Pemotongan tengah alih-alih offset tersimpan", "—"),
              ("V4_append_order", "Penumpukan menurut urutan loader", "—")]
    add(("tbl", (f"Tabel 4. Penyuntikan kesalahan pada {mf['v0_n']} segmentasi referensi. Sel "
                 f"berisi Dice CG/PZ terhadap referensi atau, bila Dice tidak berubah, "
                 f"pergeseran centroid CG dan kesalahan volume (rerata kohort). \"Tidak rusak\": "
                 f"hasil rekonstruksi identik dengan referensi. Kolom terakhir menyebut "
                 f"pemeriksaan yang menandai 100% kasus rusak di setiap kohort.",
        ["Kesalahan", f"Prostate158, LPS (n={mf['n_cases']['prostate158']})",
         f"PROSTATEx, LAS (n={mf['n_cases']['prostatex']})",
         f"MSD, RAS (n={mf['n_cases']['msd_task05']})",
         f"NCI-ISBI, LPS (n={mf['n_cases']['nci_isbi_2013']})", "Terdeteksi oleh"],
        [[lbl, cell(p158, k), cell(px, k), cell(msd_, k), cell(nci, k), det]
         for k, lbl, det in faults])))
    add(("fig", (f"{FIG_J}/F5_detection_matrix.png",
        'Gambar 4. Proporsi kasus rusak yang ditandai setiap pemeriksaan, per kesalahan dan kohort. Arsiran: kesalahan tidak mengubah hasil rekonstruksi (no damage).')))
    add(("fig", (f"{FIG_J}/jmir_F4_mirrored_volume_passes.png",
        'Gambar 5. Rekonstruksi yang benar (atas) dan rekonstruksi tercermin yang tetap lulus setiap pemeriksaan header (bawah).')))

    add(("h3", "Volume Zona dan Densitas PSA"))
    w = vs["whole"]
    add(("p",
        f"Model meremehkan volume kelenjar utuh sebesar {mm(-w['bias'])} mL "
        f"({abs(w['pct_of_gt']):.1f}% dari referensi; batas kesepakatan 95% {mm(w['loa_low'])} "
        f"hingga {mm(w['loa_high'])} mL), terutama melalui PZ ({mm(vs['pz']['bias'])} mL; "
        f"{sg(vs['pz']['pct_of_gt'], 1)}%) (Gambar 6). Kekurangan PZ ini lebih besar daripada, "
        f"dan berlawanan arah dengan, selisih antarpembaca "
        f"({sg(r2['vol_PZ_r2_minus_r1']['bias_ml'], 1)} mL), sehingga derau label saja tidak "
        f"mungkin menjelaskannya. Dalam analisis sensitivitas, {psa['n_overestimated']} dari "
        f"{psa['n']} pasien akan memiliki densitas PSA yang terlalu tinggi, dan "
        f"{psa['n_ever_reclassified']} dari {psa['n']} akan melewati ambang 0,15 ng/mL/mL pada "
        f"suatu nilai PSA di zona abu-abu 4 hingga 10 ng/mL. Pada zona tersebut, "
        f"{psa['share_pairs_reclassified_pct']:.1f}% pasangan pasien-PSA berpindah kelas; "
        f"rentang PSA tempat seorang pasien salah diklasifikasikan memiliki median lebar "
        f"{psa['width_median']:.1f} ng/mL (maksimum {psa['width_max']:.1f})."))
    add(("fig", (f"{FIG_J}/jmir_F2_zone_volume_agreement.png",
        'Gambar 6. Kesepakatan volume zona dan kesalahan densitas PSA pada data uji.')))

    add(("h3", "Pengujian Eksternal"))
    pr = nc["primary"]
    add(("p",
        f"Pada PROSTATEx, Dice adalah {d3(es['cg']['dice']['mean'])} (SB "
        f"{d3(es['cg']['dice']['sd'])}) untuk CG dan {d3(es['pz']['dice']['mean'])} (SB "
        f"{d3(es['pz']['dice']['sd'])}) untuk PZ (Tabel 5). HD95 CG memiliki median "
        f"{mm(es['cg']['hd95_mm']['median'])} mm tetapi rerata {mm(es['cg']['hd95_mm']['mean'])} "
        f"mm, ekor berat yang ditimbulkan oleh komponen positif palsu yang jauh. Seluruh "
        f"{e['n_cases']} rekonstruksi lulus pemeriksaan geometri, termasuk {eg['n_cropped']} "
        f"pemeriksaan yang cukup besar sehingga dipotong alih-alih di-padding, dan tidak satu "
        f"pun kehilangan anatomi yang dianotasi. Penghitungan ulang Dice CG terpisah untuk "
        f"belahan anterior dan posterior menunjukkan bahwa selisih terhadap hasil internal "
        f"terletak di posterior ({sg(stn['gap']['cg_dice_posterior'])}), bukan di anterior "
        f"({sg(stn['gap']['cg_dice_anterior'])}), sehingga stroma anterior yang termasuk dalam "
        f"kelas CG kohort ini tidak menjelaskannya."))
    add(("p",
        f"Pada NCI-ISBI 2013, seluruh {nc['geometry_ok']} rekonstruksi lulus pemeriksaan "
        f"geometri. Dice CG adalah {d3(pr['cg_dice_none']['mean'])} (SB "
        f"{d3(pr['cg_dice_none']['sd'])}; IK 95% {d3(pr['cg_dice_none']['ci95'][0])}–"
        f"{d3(pr['cg_dice_none']['ci95'][1])}) dan Dice PZ {d3(pr['pz_dice_none']['mean'])} (SB "
        f"{d3(pr['pz_dice_none']['sd'])}), lebih rendah di lokasi 1,5 T dengan koil endorektal "
        f"(CG {d3(bs[site15]['cg_dice_none']['mean'])}) daripada di lokasi 3 T "
        f"({d3(bs[site3]['cg_dice_none']['mean'])}). Aturan yang ditetapkan sebelumnya lulus "
        f"pemeriksaan kerugian internal (Dice CG {sg(dec['R1']['cg']['dice']['mean_diff'])}, "
        f"HD95 {sg(dec['R1']['cg']['hd95']['mean_diff'], 1)} mm). Pada NCI-ISBI 2013, aturan "
        f"ini menurunkan HD95 CG dari {mm(pr['cg_hd95_none']['mean'])} menjadi "
        f"{mm(r1['hd95_after']['mean'])} mm (median {mm(pr['cg_hd95_none']['median'])} menjadi "
        f"{mm(r1['hd95_after']['median'])}; {fp(r1['hd95']['wilcoxon_p'])}) dan menaikkan Dice "
        f"CG sebesar {d3(r1['dice']['mean_diff'])} ({fp(r1['dice']['wilcoxon_p'])}), memenuhi "
        f"kriteria yang ditetapkan sebelumnya. Kasus dengan HD95 CG di atas 30 mm turun dari "
        f"{nc['cg_tail_before']} menjadi {nc['cg_tail_after']} dari {nc['n']}. Untuk PZ, aturan "
        f"sekunder menurunkan HD95 dari {mm(pr['pz_hd95_none']['mean'])} menjadi "
        f"{mm(r2pz['hd95_after']['mean'])} mm ({fp(r2pz['hd95']['wilcoxon_p'])}) dengan Dice "
        f"tidak berubah ({sg(r2pz['dice']['mean_diff'])})."))
    add(("p",
        f"Pada {ov['n_two']} citra eksternal yang dianotasi secara independen untuk NCI-ISBI dan "
        f"MSD, kedua set anotasi bersesuaian dengan Dice {d3(ov['msd_vs_nci_cg'])} untuk CG dan "
        f"{d3(ov['msd_vs_nci_pz'])} untuk PZ, mendekati nilai antarpembaca Prostate158. Model "
        f"mencapai {d3(ov['model_vs_nci_cg'])} dan {d3(ov['model_vs_nci_pz'])} terhadap anotasi "
        f"NCI-ISBI. Selisihnya terhadap kesepakatan antaranotator karenanya melebar di luar "
        f"domain, menjadi {d3(-ov['gap_cg'])} untuk CG dan {d3(-ov['gap_pz'])} untuk PZ "
        f"(keduanya {fp(max(ov['gap_cg_p'], ov['gap_pz_p']))})."))

    def nrow(name, key, f=d3):
        dd = lambda x: f"{f(x['mean'])} ({f(x['sd'])})"  # noqa: E731
        return [name, dd(es[key[0]][key[1]]), dd(pr[key[2]]), dd(bs[site15][key[2]]),
                dd(bs[site3][key[2]])]
    add(("tbl", (f"Tabel 5. Pengujian eksternal model beku, rerata (SB). Aturan (komponen CG "
                 f"terhubung terbesar) ditetapkan sebelumnya untuk NCI-ISBI 2013; untuk "
                 f"PROSTATEx, perhitungan yang sama bersifat post hoc dan hanya ditampilkan "
                 f"sebagai pembanding.",
        ["Metrik", f"PROSTATEx (n={e['n_cases']})", f"NCI-ISBI, semua (n={nc['n']})",
         f"NCI-ISBI, 1,5 T (n={nc['n_sites'][site15]})",
         f"NCI-ISBI, 3 T (n={nc['n_sites'][site3]})"],
        [nrow("Dice CG", ("cg", "dice", "cg_dice_none")),
         nrow("Dice PZ", ("pz", "dice", "pz_dice_none")),
         nrow("HD95 CG, mm", ("cg", "hd95_mm", "cg_hd95_none"), mm),
         nrow("HD95 PZ, mm", ("pz", "hd95_mm", "pz_hd95_none"), mm),
         ["HD95 CG setelah aturan, mm", f"{mm(hd['overall']['hd95_after_mean'])} (post hoc)",
          f"{mm(r1['hd95_after']['mean'])} ({mm(r1['hd95_after']['sd'])})",
          mm(bs[site15]["cg_hd95_R1_mean"]), mm(bs[site3]["cg_hd95_R1_mean"])],
         ["Dice CG setelah aturan", f"{d3(hd['overall']['dice_after_mean'])} (post hoc)",
          f"{d3(r1['dice_after']['mean'])} ({d3(r1['dice_after']['sd'])})", "—", "—"]])))
    add(("fig", (f"{FIG_J}/jmir_F3_external_validation.png",
        'Gambar 7. Pengujian eksternal pada PROSTATEx. Batang galat: SB.')))

    add(("h3", "Ukuran Kelenjar"))
    ge, gi = gs["external"], gs["internal"]
    add(("p",
        f"Pada PROSTATEx, Dice CG berkorelasi dengan volume CG referensi (ρ Spearman="
        f"{ge['cg']['corr']['spearman_r']:.2f}; {fp(ge['cg']['corr']['spearman_p'])}), dari "
        f"{d3(ge['cg']['tertiles'][0]['dice_mean'])} pada tertil volume terkecil hingga "
        f"{d3(ge['cg']['tertiles'][2]['dice_mean'])} pada tertil terbesar. PZ menunjukkan arah "
        f"yang sama dengan lebih lemah (ρ={ge['pz']['corr']['spearman_r']:.2f}; "
        f"{fp(ge['pz']['corr']['spearman_p'])}). Kohort internal menunjukkan arah yang sama "
        f"untuk CG (ρ={gi['cg']['corr']['spearman_r']:.2f}; "
        f"{fp(gi['cg']['corr']['spearman_p'])}) tetapi kurang bertenaga dengan 19 kasus. "
        f"Kesalahan volume relatif tidak membesar seiring ukuran kelenjar pada kohort internal "
        f"(ρ={gs['volume_error_vs_size']['spearman_r']:.2f}; "
        f"{fp(gs['volume_error_vs_size']['spearman_p'])})."))

    # ------------------------------------------------------------- pembahasan
    add(("h2", "Pembahasan"))
    add(("h3", "Hasil Utama"))
    add(("p",
        f"Temuan utama menyangkut verifikasi, bukan akurasi. Pada {mf['v0_n']} segmentasi "
        f"referensi dari 4 kohort dan 3 konvensi orientasi, pemeriksaan geometri berbasis "
        f"header dan round-trip test tingkat voxel masing-masing melewatkan kelas kesalahan "
        f"rekonstruksi yang berbeda, dan hanya gabungan keduanya yang menandai setiap kesalahan "
        f"yang merusak, tanpa alarm palsu. Terlihat atau tidaknya suatu kesalahan bergantung "
        f"pada data: reorientasi yang terlewat yang sama bisa tidak berbahaya, berupa "
        f"pembalikan kiri-kanan, atau menghilangkan hampir seluruh tumpang tindih pada kohort "
        f"yang berbeda. Affine yang hanya membawa spasi mempertahankan setiap volume secara "
        f"persis sambil memindahkan kelenjar lebih dari 10 cm, sehingga baik metrik tumpang "
        f"tindih maupun kewajaran volume tidak akan mengungkapnya. Saat menyiapkan sebuah kohort "
        f"publik, penalaran yang sama mengungkap konvensi origin setengah voxel yang oleh "
        f"resampling standar akan diubah menjadi penurunan Dice hingga "
        f"{d3(1 - hv['pz_dice_min'])} untuk PZ."))
    add(("p",
        f"2D U-Net itu sendiri berkinerja mendekati, tetapi di bawah, kesepakatan ahli: "
        f"{cgm['ratio_of_means']*100:.0f}% dan {pzm['ratio_of_means']*100:.0f}% dari Dice "
        f"antarpembaca untuk CG dan PZ, dengan jarak batas yang tidak berbeda bermakna dari "
        f"jarak antarpembaca. Intervensi pelatihan berpengaruh lebih kecil daripada variasi "
        f"antarpasien: keempat konfigurasi terentang {d3(N['arm_spread'])} macro Dice, "
        f"sedangkan SB Dice CG antarpasien pada data uji saja {d3(cg['dice']['sd'])}; "
        f"augmentasi dan pembobotan kelas ekuivalen dengan pembandingnya dalam ±{margin} Dice "
        f"pada sebagian besar perbandingan, dan efek keseluruhan suku soft Dice tidak "
        f"konklusif, dengan peningkatan post hoc yang terbatas pada PZ. Pada 2 kohort "
        f"eksternal, akurasi menurun, paling besar di lokasi 1,5 T dengan koil endorektal dan "
        f"paling besar untuk PZ. Aturan connected-component, yang ditetapkan dan diperiksa "
        f"secara internal sebelum dipakai, menghilangkan hampir seluruh ekor kesalahan batas "
        f"eksternal."))

    add(("h3", "Perbandingan dengan Penelitian Terdahulu"))
    add(("p",
        f"Akurasinya berada di bawah implementasi rujukan Prostate158, sebuah 3D residual U-Net "
        f"yang melaporkan Dice sekitar 0,877 untuk CG dan 0,754 untuk PZ pada pembagian yang "
        f"sama [@adams_cbm], yaitu {cg['dice']['mean']/0.877*100:.1f}% dan "
        f"{pz['dice']['mean']/0.754*100:.1f}% dari nilai tersebut. Kekurangan yang lebih besar "
        f"pada PZ sesuai dengan dugaan dari arsitekturnya, karena batas PZ membentang miring "
        f"terhadap bidang aksial, tempat jaringan 2D tidak memiliki konteks. Karena itu, kami "
        f"tidak menyajikan model ini sebagai pesaing metode 3D; perannya di sini adalah sumber "
        f"keluaran per irisan yang realistis."))
    add(("p",
        "Kesalahan orientasi pada data neuroimaging telah terdokumentasi dengan baik: Glen dkk "
        "[@glen] mendapati pembalikan kiri-kanan cukup umum pada koleksi MRI publik sehingga "
        "layak diperiksa secara otomatis, dan konversi dari DICOM ke NIfTI merupakan sumber "
        "kesalahan semacam itu yang telah dikenal [@li_dcm2niix]. Studi-studi tersebut "
        "berkaitan dengan data masukan. Penelitian ini memperluas pertanyaannya ke keluaran "
        "pipeline segmentasi, tempat kesalahan ditimbulkan oleh pipeline itu sendiri dan tidak "
        "dapat ditemukan dengan memeriksa citra sumber, serta mengukur pemeriksaan mana yang "
        "mendeteksi kesalahan yang mana lintas dataset alih-alih mengusulkan satu pemeriksaan. "
        "Temuan bahwa pemeriksaan anatomi melewatkan pembalikan kiri-kanan konsisten dengan "
        "Glen dkk, yang memakai perbandingan berbasis citra untuk mendeteksi pembalikan "
        "tersebut. Hasil lintas dataset kami menambahkan bahwa konvensi orientasi sebuah "
        "benchmark menentukan apakah kesalahan reorientasi dapat diamati sama sekali di "
        "dalamnya. Tugas prostat MSD, benchmark yang banyak dipakai [@antonelli], disimpan "
        "dalam orientasi yang dipakai sebagian besar jaringan secara internal, sehingga "
        "menjadikannya kohort yang paling tidak mampu mengungkap kesalahan ini."))
    add(("p",
        "Pedoman pelaporan meminta rincian prapemrosesan dan kode [@claim,tripod_ai], dan "
        "sebuah studi dapat memenuhinya sambil menghasilkan volume yang tercermin, karena volume "
        "semacam itu memiliki header, bentuk, spasi, dan himpunan label yang benar. Penelitian "
        "yang mengadaptasi model 2D dan video ke data medis 3D [@yamagishi] berakhir pada "
        "langkah penyusunan ulang yang sama. Kami menyarankan agar studi yang menurunkan hasil "
        "3D dari keluaran per irisan melaporkan pemeriksaan header dan round-trip test tingkat "
        "voxel; yang terakhir hanya memerlukan satu lintasan maju dan balik dari anotasi yang "
        "sudah ada."))

    add(("h3", "Implikasi Klinis"))
    add(("p",
        "Dice adalah metrik penelitian; volume adalah jalan segmentasi zona menuju penggunaan "
        "klinis. Kekurangan volume PZ dari model merambat ke densitas PSA. Dalam analisis "
        "sensitivitas, sebagian besar pasien akan melewati ambang 0,15 ng/mL/mL di suatu titik "
        "pada zona abu-abu diagnostik, meskipun rentang PSA tempat hal itu terjadi sempit bagi "
        "sebagian besar dari mereka. Kesalahannya mengarah ke densitas yang terlalu tinggi. "
        "Bersama batas kesepakatan yang lebar relatif terhadap ukuran kelenjar, ini berarti "
        "volume dari model ini tidak boleh dipakai untuk densitas PSA tanpa peninjauan. "
        "Melaporkan volume beserta batas kesepakatannya membuat hal ini terlihat; melaporkan "
        "Dice saja tidak."))

    add(("h3", "Keterbatasan"))
    add(("p",
        "Penelitian ini memiliki beberapa kekuatan: protokol pemilihan yang ditetapkan sebelum "
        "data uji diperiksa, model yang dibekukan dan diidentifikasi dengan hash bobotnya, "
        "pembaca ahli kedua sebagai pembanding, aturan pascapemrosesan yang ditetapkan sebelum "
        "uji konfirmatorinya, dan desain penyuntikan kesalahan yang membuat setiap perbedaan "
        "pasti berasal dari kesalahan itu sendiri. Penelitian ini juga memiliki keterbatasan. "
        "Kohort pengembangan terdiri atas 158 pemeriksaan dari 1 institusi, dan data uji "
        "berjumlah 19 kasus, sehingga estimasi internal kurang presisi; IK dilaporkan karena "
        "alasan tersebut. Setiap konfigurasi dilatih sekali dengan 1 seed, sehingga variasi "
        "antar-seed, yang dengan sendirinya dapat mengubah performa segmentasi [@akesson], "
        "tidak diestimasi, dan hasil ekuivalensi menggambarkan run-run ini, bukan "
        "konfigurasinya secara umum. Data uji telah dipakai dua kali selama pengembangan "
        "pipeline sebelumnya dengan model yang berbeda; aturan pemilihan hanya memakai data "
        "validasi, tetapi data uji tidak sepenuhnya belum tersentuh. Karena itu, analisis "
        "tambahan dalam penelitian ini yang memerlukan kohort konfirmatori dijalankan pada "
        "NCI-ISBI 2013, yang belum pernah dipakai sebelumnya."))
    add(("p",
        f"Kohort eksternal dianotasi dengan protokol yang berbeda, sehingga perbedaan eksternal "
        f"menggabungkan generalisasi model dengan perbedaan anotasi, dan kesalahan pada standar "
        f"referensi dapat menggeser performa yang terukur dengan sendirinya [@wang_labels]; "
        f"perbandingan 2 anotasi membatasi, tetapi tidak menghilangkan, hal ini. Kohort-kohort "
        f"tersebut juga sebagian tumpang tindih ({ov['n_msd']} citra yang sama antara NCI-ISBI "
        f"dan MSD, dan 1 antara NCI-ISBI dan PROSTATEx); hal ini tidak memengaruhi data latih, "
        f"dan mengeluarkan kasus yang sama dengan PROSTATEx tidak mengubah hasil NCI-ISBI pada "
        f"presisi yang dilaporkan (Dice CG 0,776 vs 0,777; HD95 CG setelah aturan 9,1 mm pada "
        f"keduanya). Penyuntikan kesalahan memakai himpunan kesalahan tetap yang "
        f"diimplementasikan pada 1 pipeline; pipeline lain dapat gagal dengan cara lain, dan 2 "
        f"kesalahan yang tidak berbahaya di sini belum tentu tidak berbahaya di tempat lain. "
        f"Analisis PSA memakai rentang nilai PSA, bukan nilai terukur, dan tidak dilakukan "
        f"studi pembaca, sehingga tidak ada klaim tentang kegunaan klinis atau luaran pasien. "
        f"Akurasi juga bergantung pada ukuran kelenjar, yang membatasi penggunaan pada kelenjar "
        f"kecil."))

    add(("h3", "Kesimpulan"))
    add(("p",
        "Merekonstruksi segmentasi per irisan menjadi volume 3D dapat gagal dengan cara yang "
        "tetap menghasilkan berkas yang sah, dan dalam penelitian ini pemeriksaan header dan "
        "round-trip test tingkat voxel masing-masing melewatkan kelas kegagalan yang berbeda. "
        "Hanya gabungan keduanya yang mendeteksi setiap kesalahan yang merusak pada 4 kohort "
        "publik, dan terlihat atau tidaknya suatu kesalahan bergantung pada orientasi "
        "penyimpanan data. 2D U-Net menyegmentasi central gland mendekati kesepakatan ahli dan "
        "peripheral zone kurang baik, dan aturan connected-component yang ditetapkan "
        "sebelumnya menghilangkan sebagian besar kesalahan batas eksternalnya. Studi yang "
        "menurunkan hasil 3D dari segmentasi 2D sebaiknya melaporkan kedua pemeriksaan "
        "verifikasi dan melaporkan volume beserta batas kesepakatannya di samping Dice."))

    # --------------------------------------------------------- bagian akhir
    add(("h2", "Ucapan Terima Kasih"))
    add(("p",
        "Penulis berterima kasih kepada pembuat Prostate158, PROSTATEx beserta anotasi zonanya, "
        "NCI-ISBI 2013, dan Medical Segmentation Decathlon yang telah membuka datanya untuk "
        "publik. Kecerdasan artifisial generatif (Claude, Anthropic) dipakai untuk membantu "
        "pengembangan kode, penulisan skrip analisis, pembuatan gambar, serta penyusunan draf dan "
        "penyuntingan bahasa naskah [[TO FILL: konfirmasi bahwa daftar ini sesuai pemakaiannya]]. Seluruh konten yang dihasilkan telah "
        "ditinjau dan diverifikasi oleh penulis, yang bertanggung jawab penuh atas karya ini. "
        "Penulis juga berterima kasih kepada Universitas Multimedia Nusantara atas dukungannya "
        "terhadap penelitian ini."))
    add(("h2", "Pendanaan"))
    add(("p", "Penelitian ini didukung oleh Universitas Multimedia Nusantara [[TO FILL: nama "
              "dan nomor hibah, bila ada]]. Pemberi dana tidak berperan dalam desain penelitian, "
              "analisis data, interpretasi, maupun keputusan publikasi."))
    add(("h2", "Ketersediaan Data"))
    add(("p",
        "Seluruh dataset yang dianalisis tersedia secara publik dari sumber aslinya "
        "[@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli]. Dataset dipakai sebagaimana "
        "adanya; seluruh prapemrosesan yang diterapkan dalam penelitian ini dijelaskan di "
        "bagian Metode."))
    add(("h2", "Ketersediaan Kode"))
    add(("p",
        "Kode analisis, berkas konfigurasi, aturan yang ditetapkan sebelumnya, dan skrip yang "
        "menghasilkan setiap angka dalam artikel ini tersedia secara publik di repositori "
        "proyek [@repo]. Model beku diidentifikasi di sana dengan hash SHA-256 dari bobotnya. "
        "Ketiga tingkat verifikasi juga disediakan sebagai skrip mandiri yang hanya bergantung "
        "pada NumPy dan NiBabel (tools/verify_reconstruction.py), sehingga dapat diterapkan pada "
        "pipeline lain."))
    add(("h2", "Kontribusi Penulis"))
    add(("p",
        "Konseptualisasi: BAS, EES. Metodologi: BAS. Perangkat lunak: BAS. Analisis formal: "
        "BAS. Validasi: BAS. Visualisasi: BAS. Penulisan draf awal: BAS. Penulisan, tinjauan, "
        "dan penyuntingan: EES, DA, MW, BAS. Supervisi: EES. [[TO FILL: konfirmasi peran, "
        "termasuk peran DA dan MW]]"))
    add(("h2", "Konflik Kepentingan"))
    add(("p", "Tidak ada."))
    add(("h2", "Lampiran Multimedia 1"))
    add(("p", "Daftar periksa CLAIM 2024."))
    add(("h2", "Lampiran Multimedia 2"))
    add(("p", "Penetapan awal aturan connected-component dan analisis NCI-ISBI 2013, beserta "
              "erratum dan hash berkas yang dicatat."))
    add(("h2", "Lampiran Multimedia 3"))
    add(("p", "Hasil tambahan: riwayat pelatihan, hasil per kasus, geometri kohort, dan gambar "
              "tambahan."))
    add(("h2", "Daftar Pustaka"))
    add(("refs", REFERENCES))
    add(("h2", "Singkatan"))
    for k, v in [("2D", "dua dimensi"), ("3D", "tiga dimensi"), ("CG", "central gland"),
                 ("CLAIM", "Checklist for Artificial Intelligence in Medical Imaging"),
                 ("HD95", "jarak Hausdorff persentil ke-95"),
                 ("IK", "interval kepercayaan"),
                 ("LAS/LPS/RAS", "konvensi orientasi yang menamai arah 3 sumbu array pertama"),
                 ("MRI", "magnetic resonance imaging"), ("MSD", "Medical Segmentation Decathlon"),
                 ("PSA", "prostate-specific antigen"), ("PZ", "peripheral zone"),
                 ("SB", "simpangan baku"), ("TOST", "two one-sided tests")]:
        add(("p", f"{k}: {v}"))

    # Singkatan di dalam gambar didefinisikan di akhir keterangannya, seperti versi Inggris.
    tails = {
        1: "CG: central gland; LAS/LPS/RAS: konvensi orientasi; MSD: Medical Segmentation "
           "Decathlon; PSA: prostate-specific antigen; PZ: peripheral zone; TOST: two one-sided "
           "tests.",
        2: "bg: latar; CG: central gland; PZ: peripheral zone; T2W: T2-weighted.",
        3: "CG: central gland; NIfTI: Neuroimaging Informatics Technology Initiative; PZ: "
           "peripheral zone; RAS: orientasi right-anterior-superior.",
        4: "CG: central gland; LAS/LPS/RAS: konvensi orientasi; MSD: Medical Segmentation "
           "Decathlon; PZ: peripheral zone.",
        5: "CG: central gland; PZ: peripheral zone.",
        6: "PSA: prostate-specific antigen.",
        7: "CG: central gland; HD95: jarak Hausdorff persentil ke-95; PZ: peripheral zone.",
    }
    for i, (k, v) in enumerate(B):
        if k == "fig":
            n = int(v[1].split(".")[0].split()[-1])
            B[i] = (k, (v[0], f"{v[1]} {tails[n]}"))
    return B
