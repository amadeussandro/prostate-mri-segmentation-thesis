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

from manuscript_content_en_jmir import (FIG_J, FIG_J10, REFERENCES, _r2, d3, mm, pend, sg)  # noqa: F401

TITLE_CORE = ("Segmentasi Zona Anatomi pada MRI Prostat Menggunakan 2D U-Net dan "
              "Rekonstruksi 3D yang Tervalidasi Secara Spasial")
TITLE = f"{TITLE_CORE}: Studi Pengembangan dan Validasi Algoritma"

AUTHORS = ["Benedict Amadeus Sandro", "Eunike Endariahna Surbakti, S.Kom., M.T.I.",
           "David Agustriawan, S.Kom., M.Sc., Ph.D.", "Moeljono Widjaja, B.Sc., M.Sc., Ph.D."]
AFFIL = ["Program Studi Informatika, Fakultas Teknik dan Informatika, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Penulis Korespondensi: Eunike Endariahna Surbakti, S.Kom., M.T.I., Program Studi "
        "Informatika, Fakultas Teknik dan Informatika, Universitas Multimedia Nusantara, "
        "Jl. Scientia Boulevard, Gading Serpong, Tangerang, Banten 15810, Indonesia. "
        "Telepon: +62 21 5422 0808. Email: eunike.endariahna@umn.ac.id. ORCID: Benedict Amadeus Sandro "
        "0009-0000-1459-9139; Eunike Endariahna Surbakti 0000-0002-8442-8615; David Agustriawan "
        "0000-0003-1185-1145; Moeljono Widjaja 0000-0003-3002-7426")


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
    # N["picai"] menambahkan analisis densitas PSA pada PI-CAI (v10); bila None, teks v9
    # dibangun tanpa perubahan. pcr berisi hasilnya setelah summary.json tersedia.
    pc = N.get("picai")
    pcr = pc["results"] if pc else None
    # Kunci tingkat atas menggambarkan model FINAL (ensemble 3 seed + perataan cermin).
    # N["single"] adalah model seed 42; perbandingan terhadap baseline yang ditetapkan
    # sebelumnya dan uji ekuivalensi validasi dijalankan dengannya (N["stats_single"]).
    a, ts, es, vs = N["arms"], N["t_s"], N["e_s"], N["vol_s"]
    st = N["stats_single"]
    tost, wil = st["tost"], st["wilcoxon_test_E1_vs_baseline"]
    wilF = N["stats"]["wilcoxon_test_E1_vs_baseline"]
    one, sd_, fvs = N["single"], N["seeds"], N["final_vs_single"]
    ts1, es1, nc1, r21 = one["t_s"], one["e_s"], one["nci"], one["reader2"]
    arm2, arm25 = sd_["arm"]["2d"], sd_["arm"]["2.5d"]
    n_seed = len(sd_["seed_list"])
    r2, mf, nc, ov, psa = N["reader2"], N["mf"], N["nci"], N["ov"], N["psa"]
    g, ab, e = N["geometry"], N["ablation"], N["external"]
    eg, hd, gs, stn = e["geometry"], N["hd95_tail"], N["gland_size"], N["anterior_stroma"]
    margin = st["margin"]
    cg, pz = ts["cg"], ts["pz"]
    v1 = ab["V1_no_reorientation"]
    S = mf["summary"]
    p158, px, msd_, nci = S["prostate158"], S["prostatex"], S["msd_task05"], S["nci_isbi_2013"]
    r1 = nc["secondary"]["R1"]["cg"]
    r2pz = nc1["secondary"]["R2"]["pz"]     # R2 lulus pemeriksaan internal hanya untuk model seed-tunggal
    site15, site3 = "BMC 1.5T endorectal", "Radboud 3T"
    bs = nc["by_site"]
    hv = mf["nci_halfvoxel"]
    t_e1b = tost["test_E1_minus_baseline"]
    dec = nc["decision"]["rules"]
    lit = N["lit"]
    lit_space = {k: v for k, v in lit["space"].items() if k != "not stated"}
    if not (lit["reasm_yes"] == 0 and lit["inv_yes"] == 0 and lit["geo_k"] == 0
            and list(lit_space) == ["resampled or preprocessed grid"] and lit_space[
                "resampled or preprocessed grid"] == 2 and lit["n_depend"] > lit["n"] / 2):
        raise RuntimeError("kalimat audit literatur tidak lagi sesuai dengan hasil audit")
    nd, n1, n2 = mf["n_damaged"], mf["damaged_caught_L1"], mf["damaged_caught_L2"]
    cgm, pzm = r2["dice_CG_model_minus_interreader"], r2["dice_PZ_model_minus_interreader"]

    B: List[Tuple[str, object]] = []
    add = B.append

    if pcr:
        pp, pv, pb = pcr["primary"], pcr["volume_model_vs_report"], pcr["volume_model_vs_bosma"]
        rc, pdup = pcr["reclassification"], pc["dup"]
        if not (pp["noninferior"] and not pp["superior"] and pv["bias"] < 0 and pb["bias"] < 0
                and rc["model"]["sensitivity"] > rc["report"]["sensitivity"]
                and rc["model"]["specificity"] < rc["report"]["specificity"]):
            raise RuntimeError("kalimat PI-CAI tidak lagi sesuai dengan summary.json; tulis ulang")

        def ci(x, f=d3):
            return f"{f(x[0])} sampai {f(x[1])}"

        def dci(r):   # selisih beserta IK-nya
            return f"{d3(r['diff'])} (IK 95% {ci(r['diff_ci95'])})"
        mgn = f"{pc['margin']:.2f}".replace("-", "−")

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
        f"margin ±{margin} Dice. Dengan aturan yang ditetapkan sebelumnya, "
        f"konfigurasi terpilih direplikasi dengan {n_seed} seed, dengan dan tanpa masukan "
        f"2.5D; model final, yaitu ensemble softmax dari seed yang diadopsi dengan perataan "
        f"cermin kiri-kanan, dibekukan. Rekonstruksi "
        f"diverifikasi dengan pemeriksaan geometri berbasis header "
        f"dan round-trip test tanpa model. Enam jenis kesalahan rekonstruksi, sendiri-sendiri "
        f"maupun gabungan, kemudian disuntikkan ke {mf['v0_n']} segmentasi referensi dari 4 "
        f"kohort publik yang disimpan dalam 3 konvensi orientasi. Akurasi dibandingkan dengan "
        f"pembaca ahli kedua, dan model final diterapkan pada PROSTATEx (n={e['n_cases']}) "
        f"dan NCI-ISBI 2013 (n={nc['n']}), yang terakhir dengan aturan "
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
        f"{d3(r2['dice_PZ_r1_vs_r2']['mean'])}; baik Dice maupun jarak batas tidak berbeda "
        f"bermakna dari variabilitas antarpembaca, meskipun ekuivalensinya tidak terbukti. "
        f"Konfigurasi pelatihan ekuivalen atau tidak konklusif dalam ±{margin} Dice, rentangnya "
        f"({d3(N['arm_spread'])}) serupa dengan rentang antar-seed ({d3(sd_['range_2d'])}), dan "
        f"masukan 2.5D tidak menaikkan Dice validasi ({sg(sd_['diff'])}). Pada NCI-ISBI 2013, "
        f"Dice CG adalah {d3(nc['primary']['cg_dice_none']['mean'])}; aturan yang ditetapkan "
        f"sebelumnya menurunkan jarak "
        f"Hausdorff persentil ke-95 CG dari {mm(nc1['primary']['cg_hd95_none']['mean'])} menjadi "
        f"{mm(nc1['secondary']['R1']['cg']['hd95_after']['mean'])} mm "
        f"({fp(nc1['secondary']['R1']['cg']['hd95']['wilcoxon_p'])}) pada uji konfirmatorinya "
        f"dengan model seed-tunggal, dan dari "
        f"{mm(nc['primary']['cg_hd95_none']['mean'])} menjadi {mm(r1['hd95_after']['mean'])} "
        f"mm dengan model final.")))
    add(("abs", ("Kesimpulan",
        "Pemeriksaan header dan round-trip test tingkat voxel mendeteksi kesalahan rekonstruksi "
        "yang berbeda, dan hanya gabungan keduanya yang menangkap setiap kesalahan yang merusak. "
        "Terlihat atau tidaknya sebuah kesalahan bergantung pada cara data disimpan, sehingga "
        "pipeline yang diuji pada satu dataset dapat gagal tanpa terdeteksi pada dataset "
        "lain. Studi yang menurunkan hasil 3D dari segmentasi 2D sebaiknya melaporkan kedua "
        f"pemeriksaan tersebut; tidak satu pun dari {lit['n']} studi prostat yang diaudit "
        "melakukannya.")))
    add(("abs", ("Kata Kunci",
        "kanker prostat; magnetic resonance imaging; deep learning; segmentasi citra; "
        "pengolahan citra berbantuan komputer; kendali mutu; reprodusibilitas hasil; "
        "pengujian eksternal")))

    if pcr:
        # Dengan PI-CAI, abstrak di atas melebihi batas 450 kata: teks yang lebih ringkas
        # menggantikannya. Tanpa PI-CAI, abstrak v9 tetap seperti semula.
        short = {
            "Latar Belakang":
                "Jaringan dua dimensi (2D) tetap praktis untuk magnetic resonance imaging (MRI) "
                "prostat, yang irisannya tebal. Keluaran per irisannya harus disusun kembali "
                "menjadi volume 3D, langkah rutin yang kesalahannya bisa tidak terlihat: sebuah "
                "volume dapat memiliki header yang sah tetapi tetap tercermin atau bergeser.",
            "Tujuan":
                "Mengembangkan 2D U-Net untuk central gland (CG) dan peripheral zone (PZ), "
                "merekonstruksi dan memverifikasi keluarannya di ruang voxel 3D asli, mengukur "
                "pemeriksaan mana yang mendeteksi kesalahan rekonstruksi yang mana, dan menguji "
                "volumenya untuk densitas prostate-specific antigen (PSA).",
            "Metode":
                f"Empat konfigurasi pelatihan dibandingkan pada Prostate158 (119 data latih, 20 "
                f"validasi, 19 uji); dengan aturan yang ditetapkan sebelumnya, konfigurasi "
                f"terpilih direplikasi dengan {n_seed} seed, dengan dan tanpa masukan 2.5D, dan "
                f"ensemble dengan perataan cermin dibekukan sebagai model final. Rekonstruksi "
                f"diverifikasi dengan pemeriksaan geometri berbasis header dan round-trip test "
                f"tanpa model. Enam jenis kesalahan rekonstruksi disuntikkan ke {mf['v0_n']} "
                f"segmentasi referensi dari 4 kohort publik. Model dibandingkan dengan pembaca "
                f"ahli kedua dan diterapkan pada PROSTATEx (n={e['n_cases']}) dan NCI-ISBI 2013 "
                f"(n={nc['n']}). Dalam analisis lanjutan yang juga ditetapkan sebelumnya pada "
                f"PI-CAI (n={th(pp['n'])}), densitas PSA dari volume seluruh kelenjar hasil model "
                f"dibandingkan dengan densitas PSA dari volume pada laporan radiologi untuk "
                f"mendeteksi kanker yang bermakna klinis (margin noninferioritas {mgn} pada area "
                f"under the curve [AUC]).",
            "Hasil":
                f"Pipeline yang benar mengembalikan seluruh {mf['v0_n']} referensi secara persis. "
                f"Dari {th(nd)} rekonstruksi yang rusak, pemeriksaan geometri menandai {th(n1)} "
                f"dan round-trip test {th(n2)}; gabungan keduanya menandai semuanya, tanpa tanda "
                f"palsu. Melewatkan reorientasi tidak merusak pada satu kohort, tetapi menurunkan "
                f"Dice CG menjadi {d3(nci['V1_no_reorientation']['dice_CG'])} pada kohort lain. "
                f"Pada data uji, Dice adalah {d3(cg['dice']['mean'])} (IK 95% "
                f"{d3(cg['dice']['ci_low'])}–{d3(cg['dice']['ci_high'])}) untuk CG dan "
                f"{d3(pz['dice']['mean'])} ({d3(pz['dice']['ci_low'])}–"
                f"{d3(pz['dice']['ci_high'])}) untuk PZ, dibandingkan Dice antarpembaca "
                f"{d3(r2['dice_CG_r1_vs_r2']['mean'])} dan {d3(r2['dice_PZ_r1_vs_r2']['mean'])}. "
                f"Rentang antarkonfigurasi pelatihan ({d3(N['arm_spread'])}) serupa dengan "
                f"rentang antar-seed ({d3(sd_['range_2d'])}), dan masukan 2.5D tidak membantu. "
                f"Pada NCI-ISBI 2013, aturan connected-component yang ditetapkan sebelumnya "
                f"menurunkan jarak Hausdorff persentil ke-95 CG dari "
                f"{mm(nc1['primary']['cg_hd95_none']['mean'])} menjadi "
                f"{mm(nc1['secondary']['R1']['cg']['hd95_after']['mean'])} mm "
                f"({fp(nc1['secondary']['R1']['cg']['hd95']['wilcoxon_p'])}) pada uji "
                f"konfirmatorinya. Pada PI-CAI, AUC adalah {d3(pp['auc_a'])} dengan volume model "
                f"dan {d3(pp['auc_b'])} dengan volume laporan (selisih {d3(pp['diff'])}; IK 95% "
                f"{ci(pp['diff_ci95'])}): noninferior, tidak superior. Volume model lebih kecil "
                f"{mm(-pv['bias'])} mL, sehingga pada {pc['threshold']:.2f} ng/mL/mL "
                f"sensitivitas naik ({rc['model']['sensitivity']:.2f} vs "
                f"{rc['report']['sensitivity']:.2f}) dan spesifisitas turun "
                f"({rc['model']['specificity']:.2f} vs {rc['report']['specificity']:.2f}).",
            "Kesimpulan":
                "Pemeriksaan header dan round-trip test tingkat voxel mendeteksi kesalahan "
                "rekonstruksi yang berbeda, dan hanya gabungan keduanya yang menangkap setiap "
                "kesalahan yang merusak. Terlihat atau tidaknya sebuah kesalahan bergantung pada "
                "cara data disimpan. Studi yang menurunkan hasil 3D dari segmentasi 2D sebaiknya "
                f"melaporkan kedua pemeriksaan tersebut; tidak satu pun dari {lit['n']} studi "
                "prostat yang diaudit melakukannya. Densitas PSA dari volume model mengurutkan "
                "pasien sama baiknya dengan densitas PSA dari volume laporan, tetapi ambangnya "
                "tidak dapat dipindahkan begitu saja.",
        }
        B[:] = [(k, (v[0], short[v[0]])) if k == "abs" and v[0] in short else (k, v) for k, v in B]
        n_abs = sum(len(t.split()) for t in short.values())
        if n_abs > 450:
            raise RuntimeError(f"abstrak berisi {n_abs} kata; batas JMIR 450")

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
        "bawaan tetap menghasilkan volume dalam mililiter. Alat untuk pelaporan dan penilaian "
        "kritis studi kecerdasan artifisial sudah banyak; sebuah scoping review terbaru "
        "mengidentifikasi 46 di antaranya [@cabello]. Pedoman yang paling relevan di sini meminta "
        "penulis menjelaskan prapemrosesan dan membagikan kode [@claim,tripod_ai], tetapi tidak "
        "menanyakan bagaimana volume hasil rekonstruksi diverifikasi."))
    add(("p",
        "Yang belum tersedia adalah bukti tentang pemeriksaan mana yang benar-benar mendeteksi "
        "kesalahan yang mana, dan apakah jawabannya berlaku lintas dataset. Penelitian ini "
        "memiliki 2 tujuan. Pertama, mengembangkan 2D U-Net untuk segmentasi CG dan PZ dengan "
        "aturan pemilihan yang hanya memakai data validasi, termasuk replikasi seed dan varian "
        "2.5D, serta membandingkannya "
        "dengan pembaca ahli kedua dan pada 2 kohort eksternal, karena model pencitraan cenderung "
        "berkinerja lebih rendah pada data eksternal yang independen daripada yang ditunjukkan "
        "validasi internal [@feng_mvi]. Kedua, yang kami pandang sebagai "
        "kontribusi yang lebih dapat dialihkan, merekonstruksi keluarannya ke ruang voxel asli "
        "dengan verifikasi eksplisit dan mengukur, dengan menyuntikkan kesalahan yang diketahui "
        "ke segmentasi referensi dari 4 kohort publik, pemeriksaan mana yang mendeteksi "
        "kesalahan yang mana."
        + (" Satu analisis lanjutan, yang ditetapkan sebelum datanya diunduh, menanyakan apakah "
           "volume yang dihasilkan berguna di titik ia menyentuh klinik: apakah densitas PSA "
           "yang dihitung dari volume seluruh kelenjar hasil model mendeteksi kanker prostat "
           "yang bermakna klinis sama baiknya dengan densitas PSA yang dihitung dari volume pada "
           "laporan radiologi." if pc else "")))

    # ---------------------------------------------------------------- metode
    add(("h2", "Metode"))
    add(("h3", "Pertimbangan Etis"))
    add(("p",
        "Penelitian ini tidak melibatkan partisipan manusia secara langsung. Seluruh citra"
        f"{', anotasi, dan nilai klinis' if pc else ' dan anotasi'} berasal dari "
        f"{5 if pc else 4} koleksi publik yang telah dideidentifikasi (Prostate158, "
        "PROSTATEx beserta anotasi zonanya, NCI-ISBI 2013, "
        + ("tugas prostat Medical Segmentation Decathlon, dan set pelatihan dan pengembangan "
           "publik tantangan PI-CAI" if pc else
           "dan tugas prostat Medical Segmentation Decathlon") +
        "), yang dikumpulkan oleh institusi asalnya di bawah pengawasan "
        "etik masing-masing dan dirilis untuk penelitian sekunder, sebagaimana dijelaskan dalam "
        f"publikasi sumbernya [@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli"
        f"{',picai_lancet,picai_data' if pc else ''}]. Koleksi "
        "tersebut didistribusikan dengan lisensi terbuka yang mengizinkan penggunaan ulang "
        "dengan atribusi (misalnya, NCI-ISBI 2013 dengan CC BY 3.0"
        + (", Medical Segmentation Decathlon dengan CC BY-SA 4.0, dan PI-CAI dengan CC BY-NC "
           "4.0, yang juga membatasi penggunaan ulang untuk keperluan nonkomersial" if pc else
           " dan Medical Segmentation Decathlon dengan CC BY-SA 4.0") +
        "). Tidak ada informasi identitas yang diakses. Karena "
        "penelitian ini merupakan analisis sekunder atas data publik yang telah "
        "dideidentifikasi, persetujuan komite etik lokal tambahan dan informed consent tidak "
        "diperlukan."))
    add(("h3", "Desain Penelitian dan Pelaporan"))
    add(("p",
        "Penelitian ini merupakan studi retrospektif pengembangan dan validasi algoritma pada "
        "data publik yang telah dideidentifikasi. Pelaporannya mengikuti Checklist for "
        "Artificial Intelligence in Medical Imaging (CLAIM) 2024 [@claim] (Lampiran Multimedia "
        "1). Alur penelitian ditunjukkan pada Gambar 1. Pemilihan model hanya memakai data "
        "validasi di setiap tingkat. "
        + ("Tiga analisis dituliskan, dengan hash berkasnya dicatat sebelum dijalankan: aturan "
           "pascapemrosesan connected-component untuk kohort eksternal konfirmatori dan "
           "perluasan berupa replikasi seed dan masukan 2.5D yang menghasilkan model final "
           "(Lampiran Multimedia 2), serta analisis densitas PSA pada PI-CAI (Lampiran "
           "Multimedia 5). " if pc else
           "Dua analisis dituliskan, dengan hash berkasnya dicatat "
           "sebelum dijalankan (Lampiran Multimedia 2): aturan pascapemrosesan "
           "connected-component untuk kohort eksternal konfirmatori, serta perluasan berupa "
           "replikasi seed dan masukan 2.5D yang menghasilkan model final. ") +
        "Perluasan tersebut "
        "ditulis setelah model seed-tunggal dievaluasi pada data uji, sehingga merupakan revisi "
        "yang ditetapkan sebelumnya dan bukan bagian dari protokol awal. Bila suatu analisis "
        "pertama kali dijalankan dengan model seed-tunggal, hasilnya dilaporkan berdampingan "
        "dengan hasil model final."))
    add(("fig", (f"{FIG_J10 if pc else FIG_J}/F1_workflow.png",
        'Gambar 1. Alur penelitian dan jumlah kasus.')))

    add(("h3", "Sumber Data"))
    add(("p",
        f"{'Lima' if pc else 'Empat'} kohort publik digunakan (Tabel 1). Prostate158 [@adams_cbm,adams_dib] menyediakan "
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
        f"disimpan dalam konvensi orientasi ketiga."
        + (" Set pelatihan dan pengembangan publik tantangan PI-CAI [@picai_lancet,picai_data] "
           "hanya dipakai untuk memvalidasi densitas PSA, sebagaimana dijelaskan di bawah."
           if pc else "")))
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
          "3 T, 1 institusi", "RAS", "0.60–0.75", "Zona (zona transisi, PZ)"]]
        + ([["PI-CAI", "Hanya validasi densitas PSA",
             f"{th(pc['n_eligible'])} dari {th(pc['n_patients'])} pasien",
             f"Beberapa pemindai, {len(pc['centers'])} pusat", "/".join(pc["axcodes"]),
             f"{_r2(pc['inplane_min'])}–{_r2(pc['inplane_max'])}",
             "Tidak ada (label csPCa tingkat kasus, PSA terukur, volume laporan)"]] if pc else []))))

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
        "keputusan tersebut."))

    add(("h3", "Replikasi Seed, Masukan 2.5D, dan Model Final"))
    add(("p",
        f"Setiap konfigurasi hanya dilatih sekali, padahal seed acak saja dapat mengubah "
        f"performa segmentasi [@akesson]. Karena itu, konfigurasi terpilih dilatih ulang dengan "
        f"2 seed tambahan (43 dan 44). Secara paralel, varian 2.5D dilatih dengan {n_seed} seed "
        f"yang sama: irisan yang disegmentasi beserta 2 irisan aksial tetangganya ditumpuk "
        f"sebagai 3 kanal masukan (7763107 parameter), dengan irisan tepi diulang di kedua "
        f"ujung volume. Tiga "
        f"irisan adalah konteks yang paling didukung bukti untuk MRI prostat "
        f"[@zhang25d,hung_catnet]. Kedua lengan hanya berbeda pada seed dan jumlah kanal "
        f"masukan; arsitektur, loss, dan rekonstruksi per irisan tidak berubah."))
    add(("p",
        f"Aturannya ditetapkan sebelum run tersebut dijalankan (Lampiran Multimedia 2). Masukan "
        f"2.5D hanya diadopsi bila rerata macro Dice validasinya lintas seed melampaui masukan "
        f"2D sedikitnya sebesar {sd_['margin']} Dice. Model final adalah rata-rata keluaran "
        f"softmax dari "
        f"{n_seed} seed pada lengan yang diadopsi, suatu bentuk ensemble yang lazim [@isensee]. "
        f"Perataan dengan prediksi untuk irisan yang dicerminkan kiri-kanan, yang kemudian "
        f"dicerminkan kembali, hanya ditambahkan bila menaikkan macro Dice validasi ensemble. "
        f"Tidak ada langkah yang membaca data uji. Model final kemudian dibekukan, "
        f"diidentifikasi dengan hash SHA-256 dari bobot setiap anggotanya, dan dievaluasi satu "
        f"kali pada data uji."))

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
        ("Karena Prostate158 tidak menyediakan nilai PSA, konsekuensi klinis dari kesalahan "
         "volume terlebih dahulu diperiksa padanya sebagai analisis sensitivitas. " if pc else
         "Karena nilai PSA tidak tersedia, konsekuensi klinis dari kesalahan volume diperiksa "
         "sebagai analisis sensitivitas. ") +
        "Untuk setiap pasien, dihitung rentang nilai PSA di mana "
        "volume referensi dan volume prediksi menempatkan densitas PSA pada sisi yang berbeda "
        "dari ambang 0,15 ng/mL/mL, dan proporsi pasangan pasien-PSA yang berpindah kelas "
        "dirangkum pada zona abu-abu diagnostik 4 hingga 10 ng/mL."
        + (" Nilai PSA terukur kemudian dianalisis pada PI-CAI." if pc else "")))

    add(("h3", "Pengujian Eksternal"))
    add(("p",
        "Model final yang dibekukan diterapkan pada kedua kohort eksternal tanpa pelatihan, "
        "penyetelan, atau "
        "pencarian ambang, melalui kode inferensi dan rekonstruksi yang sama; perataan cermin "
        "adalah bagian dari model beku dan tidak diubah. Kedua kohort pertama kali dianalisis "
        "dengan model seed-tunggal, mengikuti protokol di bawah ini, dan analisisnya diulang "
        "tanpa perubahan dengan model final. Pada PROSTATEx, dekomposisi connected-component "
        "atas prediksi model seed-tunggal menunjukkan bahwa ekor kesalahan batas CG terdiri atas "
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
        "utuh dikeluarkan. Dengan demikian, uji konfirmatori aturan ini adalah uji yang "
        "dijalankan dengan model seed-tunggal. Untuk model final, pemeriksaan kerugian dan "
        "analisis NCI-ISBI diulang tanpa perubahan, sebagaimana disyaratkan dalam penetapan "
        "awal replikasi seed."))

    if pc:
        mg = f"{pc['margin']:.2f}".replace("-", "−")
        add(("h3", "Densitas PSA pada PI-CAI"))
        add(("p",
            f"Kelayakan volume model untuk menghitung densitas PSA diuji pada set pelatihan dan "
            f"pengembangan publik (v2.0) tantangan PI-CAI [@picai_lancet,picai_data]. Set ini terdiri atas "
            f"{th(pc['n_studies'])} studi MRI biparametrik dari {th(pc['n_patients'])} pasien di "
            f"{len(pc['centers'])} pusat di Belanda, dan untuk setiap studi menyediakan kadar PSA "
            f"terukur, volume prostat dari laporan radiologi, serta label kanker prostat yang "
            f"signifikan secara klinis (csPCa), yaitu grade group International Society of "
            f"Urological Pathology (ISUP) 2 atau lebih. Studi positif dikonfirmasi secara "
            f"histologis; studi negatif dikonfirmasi secara histologis (ISUP grade group 1 atau "
            f"kurang) atau, tanpa biopsi maupun tindak lanjut, oleh pembacaan MRI dengan kategori "
            f"Prostate Imaging-Reporting and Data System (PI-RADS) 2 atau kurang. Nilai-nilai "
            f"tersebut, serta segmentasi AI kelenjar utuh yang dirilis publik (Bosma22b) dan hanya "
            f"dipakai sebagai pembanding, diambil dari repositori label tantangan tersebut pada "
            f"commit yang tetap [@picai_labels]."))
        add(("p",
            f"Analisis ini dituliskan, dan hash berkasnya dicatat, sebelum berkas PI-CAI apa pun "
            f"diunduh (Lampiran Multimedia 5). Sebuah erratum yang ditulis sebelum citra apa pun "
            f"dibuka mengoreksi deskripsi standar referensi untuk studi negatif; erratum tersebut "
            f"tidak mengubah titik akhir, ambang, maupun populasi analisis. Studi paling awal dari "
            f"setiap pasien dianalisis, dan pasien tanpa nilai PSA dikeluarkan. Hanya seri "
            f"T2-weighted aksial yang dipakai. Setiap citra dikonversi dari format MetaImage ke "
            f"NIfTI dengan SimpleITK, dan konversi hanya diterima bila koordinat dunia dari 8 "
            f"voxel sudut berselisih tidak lebih dari {pc['corner_tol_mm']:g} mm. Model final yang "
            f"dibekukan diterapkan melalui kode inferensi dan rekonstruksi analisis utama, tanpa "
            f"perubahan dan tanpa pascapemrosesan. Volume kelenjar utuh adalah jumlah volume CG "
            f"dan PZ, dan densitas PSA adalah PSA dibagi volume tersebut; pasien dengan segmentasi "
            f"kosong tetap dianalisis, dengan densitas PSA diberi peringkat tertinggi."))
        add(("p",
            f"Titik akhir primer adalah selisih area di bawah kurva receiver operating "
            f"characteristic (AUC) untuk csPCa antara densitas PSA dari volume model dan densitas "
            f"PSA dari volume laporan, model dikurangi laporan, pada pasien yang memiliki volume "
            f"laporan. Setiap AUC merupakan estimasi empiris (Mann-Whitney), dan IK persentil 95% "
            f"untuk selisih berpasangan diperoleh dari bootstrap yang distratifikasi menurut "
            f"luaran ({th(pc['n_boot'])} resampling, seed tetap). Densitas PSA dari model "
            f"dinyatakan noninferior bila batas bawah IK ini melampaui {mg}, dan superior hanya "
            f"bila batas tersebut juga melampaui 0."))
        add(("p",
            f"Titik akhir sekunder bersifat deskriptif. Kesesuaian volume model dengan volume "
            f"laporan dan dengan volume Bosma22b dinilai dengan analisis Bland-Altman, dan Dice "
            f"kelenjar utuh terhadap Bosma22b dihitung bila mask tersebut berada pada grid "
            f"T2-weighted. Volume laporan umumnya merupakan estimasi elipsoid dan Bosma22b sendiri "
            f"merupakan keluaran model, sehingga tidak satu pun perbandingan tersebut mengukur "
            f"akurasi. Pada densitas PSA {pc['threshold']:.2f} ng/mL/mL, pasien yang "
            f"diklasifikasikan berbeda oleh kedua volume dihitung, dan sensitivitas serta "
            f"spesifisitas setiap densitas PSA dihitung. AUC densitas PSA dari volume Bosma22b "
            f"dilaporkan sebagai acuan, dan titik akhir primer dilaporkan per pusat. Titik akhir "
            f"primer juga diulang dengan aturan R1, tanpa studi yang gagal pada pemeriksaan "
            f"geometri, dan tanpa studi yang citranya merupakan duplikat citra PROSTATEx, NCI-ISBI "
            f"2013, atau MSD. Duplikat diidentifikasi seperti pada kohort eksternal: bentuk array "
            f"yang sama setelah reorientasi dan koefisien Pearson r per voxel sedikitnya "
            f"{pc['dup_r']:.2f}. Pemeriksaan geometri diterapkan pada setiap rekonstruksi, dan "
            f"round-trip test pada setiap mask Bosma22b yang tidak kosong di grid aslinya."))

    add(("h3", "Audit Pelaporan pada Studi Terpublikasi"))
    y, mo, dd = (int(x) for x in lit["run_date"].split("-"))
    bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
             "September", "Oktober", "November", "Desember"][mo - 1]
    add(("p",
        "Untuk mengetahui seberapa sering penelitian terpublikasi melaporkan langkah "
        "rekonstruksi, kami mengaudit studi deep learning yang menyegmentasi kelenjar prostat "
        "atau zonanya pada MRI dengan setidaknya 1 jaringan yang menerima irisan 2D, atau "
        "tumpukan irisan bertetangga (2.5D), sebagai masukan. Pencarian (kueri tetap pada "
        f"subset akses terbuka PubMed Central, dijalankan sekali pada {dd} {bulan} {y}), "
        "kriteria kelayakan, dan 10 butir ekstraksi ditetapkan dalam protokol yang ditulis "
        "sebelum penyaringan, dan hash berkasnya dicatat dalam keluaran (Lampiran Multimedia 4). "
        "Satu penulis menyaring teks lengkap dan mengode setiap butir, dengan kutipan sebagai "
        "dasar setiap kode positif; butir yang tidak dinyatakan dalam teks dikode sebagai tidak "
        "dilaporkan."))

    add(("h3", "Analisis Statistik"))
    add(("p",
        f"Selisih berpasangan per kasus diuji dengan uji peringkat bertanda Wilcoxon. "
        f"Perbandingan yang ditetapkan sebelumnya adalah macro Dice konfigurasi terpilih, "
        f"sebagai model seed-tunggal, terhadap baseline pada data uji, yang merupakan kontras "
        f"faktor-tunggal. Uji per zona bersifat post hoc dan dikoreksi dengan "
        f"metode Holm [@holm] untuk 2 zona. Model final juga dibandingkan dengan baseline dan "
        f"dengan model seed-tunggal; perbandingan ini tidak ditetapkan sebelumnya, dan yang "
        f"pertama menggabungkan efek loss, ensemble, dan perataan cermin. Karena selisih yang "
        f"tidak bermakna tidak membuktikan "
        f"ketiadaan efek, ekuivalensi diuji dengan two one-sided tests (TOST) "
        f"[@schuirmann,lakens] pada selisih berpasangan dengan margin ±{margin} Dice; "
        f"ekuivalensi disimpulkan bila IK 90% berada di dalam margin. Margin tidak didaftarkan "
        f"sebelumnya. Margin dipilih lebih kecil daripada SB Dice antarpembaca untuk kedua zona "
        f"({d3(r2['dice_CG_r1_vs_r2']['sd'])} untuk CG dan {d3(r2['dice_PZ_r1_vs_r2']['sd'])} "
        f"untuk PZ). Uji pada data validasi bersifat deskriptif, karena data tersebut dipakai "
        f"untuk pemilihan. Sebaran dinyatakan sebagai SB sampel, antarkasus atau, bila "
        f"disebutkan, antar-seed, dan IK merupakan interval "
        f"persentil bootstrap (10000 resampling, seed tetap). Pelatihan, serta inferensi pada "
        f"PROSTATEx dengan model final, memakai 1 GPU NVIDIA "
        + (f"Tesla T4, dan inferensi pada PI-CAI memakai "
           f"{pend('GPU model(s) of the PI-CAI runs') if pcr is None else 'model GPU yang sama' if pcr['model_identity']['gpus'] == ['Tesla T4'] else ' dan '.join(pcr['model_identity']['gpus'])}; "
           f"pada {pc['cpu_gpu_n']} studi PI-CAI yang juga dijalankan pada CPU, prediksinya "
           f"{'identik voxel demi voxel' if pc['cpu_gpu_identical'] == pc['cpu_gpu_n'] else pend('CPU and GPU predictions differ')}. "
           f"Seluruh analisis lain dijalankan pada CPU dengan Python v3.13 (Python " if pc else
           f"Tesla T4; seluruh analisis lain dijalankan pada CPU dengan Python v3.13 (Python ") +
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

    add(("h3", "Replikasi Seed, Masukan 2.5D, dan Model Final"))
    ens, enst = sd_["ensemble"], sd_["ensemble_tta"]
    if sd_["adopted"] != "2d" or not sd_["tta_kept"]:
        raise RuntimeError("teks di bawah menggambarkan ensemble 2D dengan perataan cermin")
    add(("p",
        f"Pada {n_seed} seed, E1 memiliki rerata macro Dice validasi {d3(arm2['mean_macro'])} "
        f"(SB {d3(arm2['sd_macro'])}) dengan masukan 2D dan {d3(arm25['mean_macro'])} (SB "
        f"{d3(arm25['sd_macro'])}) dengan masukan 2.5D (Tabel 3). Selisihnya "
        f"({sg(sd_['diff'])}) berada di bawah margin yang ditetapkan sebelumnya, yaitu "
        f"{sd_['margin']}, sehingga masukan 2D dipertahankan. Masukan 2.5D tidak memperbaiki "
        f"PZ, zona yang ingin dibantunya ({d3(arm25['mean_pz'])} vs {d3(arm2['mean_pz'])}), dan "
        f"lebih bervariasi antar-seed. Rentang antara 3 seed ({d3(sd_['range_2d'])} untuk "
        f"masukan 2D dan {d3(sd_['range_25d'])} untuk masukan 2.5D) serupa dengan atau lebih "
        f"besar daripada rentang antara 4 konfigurasi pelatihan ({d3(N['arm_spread'])}), "
        f"sehingga selisih pada Tabel 2 seukuran dengan selisih yang dapat timbul hanya karena "
        f"pergantian seed. Ensemble seed 2D mencapai {d3(ens['macro'])}, di atas setiap "
        f"anggotanya. "
        f"Perataan cermin menaikkannya menjadi {d3(enst['macro'])} sehingga dipertahankan, dan "
        f"model ini dibekukan sebagai model final."))
    rowsT3 = []
    for arm, lab, st_ in (("2d", "2D", arm2), ("2.5d", "2.5D", arm25)):
        for s_ in sd_["seed_list"]:
            v = sd_["per_seed"][arm][s_]
            rowsT3.append([f"Masukan {lab}, seed {s_}", str(v["epoch"]), d3(v["cg"]),
                           d3(v["pz"]), d3(v["macro"])])
        rowsT3.append([f"Masukan {lab}, rerata (SB) antar-seed", "—",
                       f"{d3(st_['mean_cg'])} ({d3(st_['sd_cg'])})",
                       f"{d3(st_['mean_pz'])} ({d3(st_['sd_pz'])})",
                       f"{d3(st_['mean_macro'])} ({d3(st_['sd_macro'])})"])
    rowsT3.append(["Ensemble seed 2D", "—", d3(ens["cg"]), d3(ens["pz"]), d3(ens["macro"])])
    rowsT3.append(["Ensemble dengan perataan cermin kiri-kanan (model final)", "—",
                   d3(enst["cg"]), d3(enst["pz"]), d3(enst["macro"])])
    add(("tbl", ("Tabel 3. Replikasi seed konfigurasi terpilih (E1) dengan masukan 2D dan 2.5D, "
                 "serta ensemble, pada 20 kasus validasi (Dice tingkat kasus).",
        ["Model", "Epoch terbaik", "Dice CG", "Dice PZ", "Macro Dice"], rowsT3)))

    add(("h3", "Performa pada Data Uji dan Kesepakatan Antarpembaca"))
    cg1, pz1, ft = ts1["cg"], ts1["pz"], fvs["test"]
    cgm1, pzm1 = r21["dice_CG_model_minus_interreader"], r21["dice_PZ_model_minus_interreader"]
    hcg, hpz = r2["hd95_CG_model_minus_interreader"], r2["hd95_PZ_model_minus_interreader"]
    if min(x["wilcoxon_p"] for x in (cgm, pzm, hcg, hpz)) < 0.05:
        raise RuntimeError("teks antarpembaca mengandaikan tidak ada selisih yang bermakna")
    add(("p",
        f"Pada 19 kasus uji, rerata Dice model final adalah {d3(cg['dice']['mean'])} (SB "
        f"{d3(cg['dice']['sd'])}) untuk CG dan {d3(pz['dice']['mean'])} (SB "
        f"{d3(pz['dice']['sd'])}) untuk PZ (Tabel 4). Untuk model seed-tunggal, nilainya "
        f"{d3(cg1['dice']['mean'])} (SB {d3(cg1['dice']['sd'])}) dan {d3(pz1['dice']['mean'])} "
        f"(SB {d3(pz1['dice']['sd'])}); macro Dice model final lebih tinggi "
        f"{d3(ft['macro']['mean_diff'])} (IK 95% {d3(ft['macro']['ci_low'])} s.d. "
        f"{d3(ft['macro']['ci_high'])}; lebih baik pada {ft['macro']['n_better']} dari "
        f"{ft['macro']['n']} kasus; {fp(ft['macro']['p'])}). PZ memadukan presisi tinggi "
        f"({d3(pz['precision']['mean'])}) dengan recall yang lebih rendah "
        f"({d3(pz['recall']['mean'])}): model jarang melabeli jaringan lain sebagai PZ, tetapi "
        f"melabeli PZ terlalu sedikit."))
    add(("p",
        f"Terhadap pembaca 1, Dice model final lebih rendah daripada "
        f"pembaca 2 sebesar {d3(-cgm['mean_diff'])} untuk CG (IK 95% {d3(-cgm['ci95'][1])} s.d. "
        f"{d3(-cgm['ci95'][0])}; {fp(cgm['wilcoxon_p'])}) dan {d3(-pzm['mean_diff'])} untuk PZ "
        f"(IK 95% {d3(-pzm['ci95'][1])} s.d. {d3(-pzm['ci95'][0])}; {fp(pzm['wilcoxon_p'])}), "
        f"yaitu {cgm['ratio_of_means']*100:.1f}% dan {pzm['ratio_of_means']*100:.1f}% dari "
        f"kesepakatan antarpembaca. Kedua selisih tidak bermakna. Namun, ekuivalensinya tidak "
        f"diuji, dan dengan 19 kasus IK tersebut tidak menyingkirkan kekurangan sebesar "
        f"{d3(-cgm['ci95'][0])} untuk CG atau {d3(-pzm['ci95'][0])} untuk PZ. Untuk model "
        f"seed-tunggal, kedua kekurangan tersebut bermakna ({d3(-cgm1['mean_diff'])}, "
        f"{fp(cgm1['wilcoxon_p'])}; {d3(-pzm1['mean_diff'])}, {fp(pzm1['wilcoxon_p'])}). Jarak "
        f"batas juga tidak berbeda bermakna dari variabilitas antarpembaca: HD95 adalah "
        f"{mm(r2['hd95_CG_model_vs_r1']['mean'])} mm dibanding "
        f"{mm(r2['hd95_CG_r1_vs_r2']['mean'])} mm untuk CG "
        f"({fp(r2['hd95_CG_model_minus_interreader']['wilcoxon_p'])}) dan "
        f"{mm(r2['hd95_PZ_model_vs_r1']['mean'])} mm dibanding "
        f"{mm(r2['hd95_PZ_r1_vs_r2']['mean'])} mm untuk PZ "
        f"({fp(r2['hd95_PZ_model_minus_interreader']['wilcoxon_p'])})."))
    m, mF = wil["macro"], wilF["macro"]
    add(("p",
        f"Terhadap baseline, perbandingan macro yang ditetapkan sebelumnya, yang dijalankan "
        f"dengan model seed-tunggal, tidak bermakna "
        f"({sg(m['mean_diff'])}, lebih baik pada {m['n_better']} dari {m['n']} kasus; "
        f"{fp(m['p_raw'])}) dan juga tidak ekuivalen (IK 90% {d3(t_e1b['macro']['ci90'][0])} "
        f"s.d. {d3(t_e1b['macro']['ci90'][1])}), sehingga efek suku soft Dice saja tidak "
        f"konklusif. "
        f"Secara post hoc, Dice CG ekuivalen ({fp(t_e1b['CG']['tost_p'])}) dan Dice PZ lebih "
        f"tinggi {d3(wil['PZ']['mean_diff'])} (lebih baik pada {wil['PZ']['n_better']} dari "
        f"{wil['PZ']['n']} kasus; P terkoreksi Holm {fp(wil['PZ']['p_holm_zones'])[2:]}). Model "
        f"final melampaui baseline sebesar {d3(mF['mean_diff'])} macro Dice (lebih baik pada "
        f"{mF['n_better']} dari {mF['n']} kasus; {fp(mF['p_raw'])}). Perbandingan ini tidak "
        f"ditetapkan sebelumnya, dan mencerminkan ensemble serta perataan cermin di samping "
        f"loss."))

    def rr(name, key, f=d3):
        return [name, msd(cg[key], f), "—", msd(pz[key], f), "—"]
    add(("tbl", ("Tabel 4. Performa model final pada data uji (19 kasus, ruang voxel asli, "
                 "rerata (SB) per kasus) dan kesepakatan antara 2 pembaca ahli pada kasus yang "
                 "sama.",
        ["Metrik", "CG, model vs pembaca 1", "CG, pembaca 2 vs pembaca 1",
         "PZ, model vs pembaca 1", "PZ, pembaca 2 vs pembaca 1"],
        [["Dice", msd(cg["dice"]), msd(r2["dice_CG_r1_vs_r2"]), msd(pz["dice"]),
          msd(r2["dice_PZ_r1_vs_r2"])],
         ["Dice, IK 95%", f"{d3(cg['dice']['ci_low'])}–{d3(cg['dice']['ci_high'])}", "—",
          f"{d3(pz['dice']['ci_low'])}–{d3(pz['dice']['ci_high'])}", "—"],
         ["Dice, model seed-tunggal", msd(cg1["dice"]), "—", msd(pz1["dice"]), "—"],
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
        f"persis (Tabel 5). Dari {th(nd)} rekonstruksi yang rusak, pemeriksaan geometri menandai "
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
        return f"Dice {d3(x['dice_CG'])} / {d3(x['dice_PZ'])}"
    faults = [("V1_no_reorientation", "Reorientasi tidak dibalik", "Round-trip"),
              ("V3_identity_affine", "Affine identitas", "Geometri"),
              ("V6_spacing_only", "Affine hanya spasi", "Geometri"),
              ("V4b_shuffled_order", "Urutan irisan acak", "Round-trip"),
              ("V2_centre_crop", "Pemotongan tengah alih-alih offset tersimpan", "—"),
              ("V4_append_order", "Penumpukan menurut urutan loader", "—")]
    add(("tbl", (f"Tabel 5. Penyuntikan kesalahan pada {mf['v0_n']} segmentasi referensi. Sel "
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
        f"mungkin menjelaskannya. Kekurangan ini juga tidak berubah dari model seed-tunggal "
        f"({mm(one['vol_s']['pz']['bias'])} mL): ensemble memperbaiki tumpang tindih tanpa "
        f"mengoreksi kecenderungan meremehkan volume. Dalam analisis sensitivitas, "
        f"{psa['n_overestimated']} dari "
        f"{psa['n']} pasien akan memiliki densitas PSA yang terlalu tinggi, dan "
        f"{psa['n_ever_reclassified']} dari {psa['n']} akan melewati ambang 0,15 ng/mL/mL pada "
        f"suatu nilai PSA di zona abu-abu 4 hingga 10 ng/mL. Pada zona tersebut, "
        f"{psa['share_pairs_reclassified_pct']:.1f}% pasangan pasien-PSA berpindah kelas; "
        f"rentang PSA tempat seorang pasien salah diklasifikasikan memiliki median lebar "
        f"{psa['width_median']:.1f} ng/mL (maksimum {psa['width_max']:.1f})."))
    add(("fig", (f"{FIG_J}/jmir_F2_zone_volume_agreement.png",
        'Gambar 6. Kesepakatan volume zona dan kesalahan densitas PSA pada data uji.')))

    add(("h3", "Pengujian Eksternal"))
    pr, pr1 = nc["primary"], nc1["primary"]
    r1s = nc1["secondary"]["R1"]["cg"]                    # uji konfirmatori aturan
    fx, fn_ = fvs["prostatex"], fvs["nci"]
    e1x = one["external"]
    if not (e["hd95_cg"]["n_over_30"] < e1x["hd95_cg"]["n_over_30"]
            and hd["overall"]["hd95_after_mean"] < hd["overall"]["hd95_mean"]
            and hd["tail"]["multi_component_frac"] > 0.9):
        raise RuntimeError("teks ekor PROSTATEx tidak lagi sesuai dengan data")
    add(("p",
        f"Pada PROSTATEx, Dice adalah {d3(es['cg']['dice']['mean'])} (SB "
        f"{d3(es['cg']['dice']['sd'])}) untuk CG dan {d3(es['pz']['dice']['mean'])} (SB "
        f"{d3(es['pz']['dice']['sd'])}) untuk PZ (Tabel 6), dibandingkan "
        f"{d3(es1['cg']['dice']['mean'])} dan {d3(es1['pz']['dice']['mean'])} untuk model "
        f"seed-tunggal (macro Dice {sg(fx['macro']['mean_diff'])}; lebih baik pada "
        f"{fx['macro']['n_better']} dari {fx['macro']['n']} kasus; {fp(fx['macro']['p'])}). "
        f"HD95 CG memiliki median "
        f"{mm(es['cg']['hd95_mm']['median'])} mm tetapi rerata {mm(es['cg']['hd95_mm']['mean'])} "
        f"mm, ekor yang ditimbulkan oleh komponen positif palsu yang jauh. Ekor ini lebih "
        f"pendek daripada pada model seed-tunggal, yang reratanya "
        f"{mm(es1['cg']['hd95_mm']['mean'])} mm: kasus di atas 30 mm turun dari "
        f"{e1x['hd95_cg']['n_over_30']} menjadi {e['hd95_cg']['n_over_30']} dari "
        f"{e['n_cases']}. Seluruh "
        f"{e['n_cases']} rekonstruksi lulus pemeriksaan geometri, termasuk {eg['n_cropped']} "
        f"pemeriksaan yang cukup besar sehingga dipotong alih-alih di-padding, dan tidak satu "
        f"pun kehilangan anatomi yang dianotasi. Penghitungan ulang Dice CG terpisah untuk "
        f"belahan anterior dan posterior menunjukkan bahwa selisih terhadap hasil internal "
        f"terletak di posterior ({sg(stn['gap']['cg_dice_posterior'])}), bukan di anterior "
        f"({sg(stn['gap']['cg_dice_anterior'])}), sehingga stroma anterior yang termasuk dalam "
        f"kelas CG kohort ini tidak menjelaskannya."))
    def met(x):
        return (x["hd95"]["mean_diff"] < 0 and x["hd95"]["wilcoxon_p"] < 0.05
                and x["dice"]["mean_diff"] >= -0.005)
    dec1 = nc1["decision"]["rules"]
    dR2 = dec["R2"]["pz"]["hd95"]["mean_diff"]
    if not (dec["R1"]["passed"] and dec1["R1"]["passed"] and dec1["R2"]["passed"]
            and not dec["R2"]["passed"] and isinstance(nc["secondary"]["R2"], str) and dR2 > 0
            and met(r1) and met(r1s) and met(r2pz)
            and bs[site15]["cg_dice_none"]["mean"] < bs[site3]["cg_dice_none"]["mean"]):
        raise RuntimeError("teks NCI-ISBI tidak lagi sesuai dengan data; tulis ulang")
    add(("p",
        f"Pada NCI-ISBI 2013, seluruh {nc['geometry_ok']} rekonstruksi lulus pemeriksaan "
        f"geometri. Dice CG adalah {d3(pr['cg_dice_none']['mean'])} (SB "
        f"{d3(pr['cg_dice_none']['sd'])}; IK 95% {d3(pr['cg_dice_none']['ci95'][0])}–"
        f"{d3(pr['cg_dice_none']['ci95'][1])}) dan Dice PZ {d3(pr['pz_dice_none']['mean'])} (SB "
        f"{d3(pr['pz_dice_none']['sd'])}), lebih rendah di lokasi 1,5 T dengan koil endorektal "
        f"(CG {d3(bs[site15]['cg_dice_none']['mean'])}) daripada di lokasi 3 T "
        f"({d3(bs[site3]['cg_dice_none']['mean'])}). Untuk model seed-tunggal, nilainya "
        f"{d3(pr1['cg_dice_none']['mean'])} dan {d3(pr1['pz_dice_none']['mean'])} (perubahan "
        f"dengan model final: CG {sg(fn_['cg_dice_none']['mean_diff'])}, "
        f"{fp(fn_['cg_dice_none']['p'])}; PZ {sg(fn_['pz_dice_none']['mean_diff'])}, "
        f"{fp(fn_['pz_dice_none']['p'])})."))
    add(("p",
        f"Pada uji konfirmatorinya, yang dijalankan dengan model seed-tunggal, aturan yang "
        f"ditetapkan sebelumnya menurunkan HD95 CG dari {mm(pr1['cg_hd95_none']['mean'])} "
        f"menjadi {mm(r1s['hd95_after']['mean'])} mm ({fp(r1s['hd95']['wilcoxon_p'])}) dan "
        f"menaikkan Dice CG sebesar {d3(r1s['dice']['mean_diff'])} "
        f"({fp(r1s['dice']['wilcoxon_p'])}), memenuhi kriteria yang ditetapkan sebelumnya. "
        f"Ketika diulang dengan model final, aturan ini kembali lulus pemeriksaan kerugian "
        f"internal (Dice CG {sg(dec['R1']['cg']['dice']['mean_diff'])}, HD95 "
        f"{sg(dec['R1']['cg']['hd95']['mean_diff'], 1)} mm) dan menurunkan HD95 CG dari "
        f"{mm(pr['cg_hd95_none']['mean'])} menjadi "
        f"{mm(r1['hd95_after']['mean'])} mm (median {mm(pr['cg_hd95_none']['median'])} menjadi "
        f"{mm(r1['hd95_after']['median'])}; {fp(r1['hd95']['wilcoxon_p'])}), dengan perubahan "
        f"Dice CG {sg(r1['dice']['mean_diff'])} ({fp(r1['dice']['wilcoxon_p'])}). Kasus dengan "
        f"HD95 CG di atas 30 mm turun dari "
        f"{nc['cg_tail_before']} menjadi {nc['cg_tail_after']} dari {nc['n']}. Untuk PZ, aturan "
        f"sekunder pada uji konfirmatori menurunkan HD95 dari {mm(pr1['pz_hd95_none']['mean'])} "
        f"menjadi {mm(r2pz['hd95_after']['mean'])} mm ({fp(r2pz['hd95']['wilcoxon_p'])}; "
        f"perubahan Dice {sg(r2pz['dice']['mean_diff'])}). Dengan model final, aturan ini gagal "
        f"dalam pemeriksaan kerugian internal karena HD95 PZ validasi naik {mm(dR2)} mm, "
        f"sehingga tidak diterapkan, sesuai ketentuan yang ditetapkan sebelumnya."))
    add(("p",
        f"Pada {ov['n_two']} citra eksternal yang dianotasi secara independen untuk NCI-ISBI dan "
        f"MSD, kedua set anotasi bersesuaian dengan Dice {d3(ov['msd_vs_nci_cg'])} untuk CG dan "
        f"{d3(ov['msd_vs_nci_pz'])} untuk PZ, mendekati nilai antarpembaca Prostate158. Model "
        f"final mencapai {d3(ov['model_vs_nci_cg'])} dan {d3(ov['model_vs_nci_pz'])} terhadap "
        f"anotasi NCI-ISBI. Berbeda dengan pada data uji, selisihnya terhadap kesepakatan "
        f"antaranotator bermakna di luar domain, yaitu {d3(-ov['gap_cg'])} untuk CG dan "
        f"{d3(-ov['gap_pz'])} untuk PZ (keduanya {fp(max(ov['gap_cg_p'], ov['gap_pz_p']))}"
        f"{'' if max(ov['gap_cg_p'], ov['gap_pz_p']) < 0.001 else ' atau lebih kecil'})."))
    if max(ov["gap_cg_p"], ov["gap_pz_p"]) >= 0.05 or ov["gap_cg"] >= 0 or ov["gap_pz"] >= 0:
        raise RuntimeError("teks 2 anotasi mengandaikan selisih bermakna di luar domain")

    def nrow(name, key, f=d3, src=None, bsrc=None):
        src, bsrc = src or (es, pr), bsrc or bs
        dd = lambda x: f"{f(x['mean'])} ({f(x['sd'])})"  # noqa: E731
        return [name, dd(src[0][key[0]][key[1]]), dd(src[1][key[2]]), dd(bsrc[site15][key[2]]),
                dd(bsrc[site3][key[2]])]
    one_src = dict(src=(es1, pr1), bsrc=nc1["by_site"])
    add(("tbl", (f"Tabel 6. Pengujian eksternal model final, rerata (SB), dengan Dice model "
                 f"seed-tunggal sebagai pembanding. Aturan (komponen CG "
                 f"terhubung terbesar) ditetapkan sebelumnya untuk NCI-ISBI 2013; untuk "
                 f"PROSTATEx, perhitungan yang sama bersifat post hoc dan hanya ditampilkan "
                 f"sebagai pembanding.",
        ["Metrik", f"PROSTATEx (n={e['n_cases']})", f"NCI-ISBI, semua (n={nc['n']})",
         f"NCI-ISBI, 1,5 T (n={nc['n_sites'][site15]})",
         f"NCI-ISBI, 3 T (n={nc['n_sites'][site3]})"],
        [nrow("Dice CG", ("cg", "dice", "cg_dice_none")),
         nrow("Dice PZ", ("pz", "dice", "pz_dice_none")),
         nrow("Dice CG, model seed-tunggal", ("cg", "dice", "cg_dice_none"), **one_src),
         nrow("Dice PZ, model seed-tunggal", ("pz", "dice", "pz_dice_none"), **one_src),
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

    if pcr:
        cn = {"RUMC": "Radboud University Medical Center (RUMC)",
              "ZGT": "Ziekenhuisgroep Twente (ZGT)",
              "PCNN": "Prostaat Centrum Noord-Nederland (PCNN)"}
        ex, bc, sv = pcr["excluded"], pcr["by_center"], pcr["sensitivity"]
        rt, mg_, bp = pcr["roundtrip_bosma"], pc["merge"], pcr["bosma_psad_vs_report"]
        dq = pcr["dice_vs_bosma"]
        if not (pcr["geometry_ok"] == pcr["n_inferred"] == pc["n_eligible"]
                and rt["n_tested"] - rt["passed"] == 1 and pcr["n_empty_model"] == 1
                and sv["rule_R1"]["noninferior"] and sv["without_duplicates"]["noninferior"]
                and sv["without_geometry_failures"] == "no geometry failures"
                and list(pdup["centers"]) == ["RUMC"] and pdup["by_source"]["MSD"] == 0
                and mg_["n_also_inferred_elsewhere"] == mg_["n_also_inferred_identical"]
                and bc["RUMC"]["noninferior"] and bc["ZGT"]["noninferior"]
                and not bc["PCNN"]["noninferior"] and bp["superior"]):
            raise RuntimeError("teks Hasil PI-CAI tidak lagi sesuai dengan summary.json")
        add(("h3", "Densitas PSA pada PI-CAI"))
        add(("p",
            f"Dari {th(pc['n_studies'])} studi milik {th(pc['n_patients'])} pasien, "
            f"{ex['later study of the same patient']} studi lanjutan dari pasien yang sama dan "
            f"{ex['no PSA']} pasien tanpa nilai PSA dikeluarkan, sehingga tersisa "
            f"{th(pc['n_eligible'])} pasien ({pc['centers']['RUMC']} dari {cn['RUMC']}, "
            f"{pc['centers']['ZGT']} dari {cn['ZGT']}, dan {pc['centers']['PCNN']} dari "
            f"{cn['PCNN']}); {pc['n_cspca']} di antaranya "
            f"({pc['n_cspca'] / pc['n_eligible'] * 100:.1f}%) memiliki csPCa. Seluruh "
            f"{th(pc['n_converted'])} citra berhasil dikonversi, dengan selisih sudut terbesar di "
            f"bawah 0,001 mm. Setiap rekonstruksi lulus pemeriksaan geometri, dan round-trip "
            f"test mengembalikan {th(rt['passed'])} dari {th(rt['n_tested'])} masker Bosma22b "
            f"secara persis. Satu masker sisanya melampaui batas pemotongan foreground dan "
            f"disebut dalam repositori label sebagai segmentasi yang keliru [@picai_labels]; "
            f"studi tersebut tetap berada dalam semua analisis. Model menghasilkan segmentasi "
            f"kosong pada {pcr['n_empty_model']} pasien. Inferensi berjalan dalam "
            f"{len(mg_['runs'])} sesi dari 1 arsip kode dan bobot yang sama, dan "
            f"{mg_['n_also_inferred_elsewhere']} studi yang diinferensi dalam 2 sesi "
            f"menghasilkan volume yang identik."))
        add(("p",
            f"Perbandingan primer mencakup {th(pp['n'])} pasien yang memiliki volume laporan "
            f"({pp['n_pos']} dengan csPCa); {ex['no reported prostate volume']} pasien tidak "
            f"memilikinya. AUC untuk csPCa adalah {d3(pp['auc_a'])} (IK 95% "
            f"{d3(pp['auc_a_ci95'][0])}–{d3(pp['auc_a_ci95'][1])}) untuk densitas PSA dari "
            f"volume model dan {d3(pp['auc_b'])} ({d3(pp['auc_b_ci95'][0])}–"
            f"{d3(pp['auc_b_ci95'][1])}) untuk densitas PSA dari volume laporan, dengan selisih "
            f"{dci(pp)} (Tabel 7). Batas bawahnya melampaui margin {mgn}, sehingga "
            f"noninferioritas terbukti; batas itu tidak melampaui 0, sehingga superioritas tidak "
            f"terbukti. Dengan aturan R1, selisihnya {dci(sv['rule_R1'])}. Tidak ada studi yang "
            f"gagal pemeriksaan geometri, sehingga analisis sensitivitas tersebut tidak berlaku."))
        add(("p",
            f"Dari {th(pc['n_eligible'])} studi, {pdup['n_studies']} merupakan duplikat citra "
            f"yang sudah dipakai dalam penelitian ini ({pdup['by_source']['PROSTATEx']} di "
            f"PROSTATEx dan {pdup['by_source']['NCI-ISBI']} di NCI-ISBI 2013, dengan "
            f"{pdup['by_source']['PROSTATEx'] + pdup['by_source']['NCI-ISBI'] - pdup['n_studies']} "
            f"di keduanya); semuanya berasal dari RUMC. Tidak satu pun dari citra ini dipakai "
            f"untuk pelatihan. Tanpa {sv['without_duplicates']['n_removed']} di antaranya yang "
            f"berada dalam perbandingan primer, selisihnya {dci(sv['without_duplicates'])} pada "
            f"{th(sv['without_duplicates']['n'])} pasien. Per pusat, selisihnya "
            f"{dci(bc['RUMC'])} di RUMC, {dci(bc['ZGT'])} di ZGT, dan {dci(bc['PCNN'])} di PCNN. "
            f"Di PCNN kedua AUC rendah ({d3(bc['PCNN']['auc_a'])} dan "
            f"{d3(bc['PCNN']['auc_b'])}) dan batas bawahnya jatuh di bawah margin; analisis per "
            f"pusat bersifat deskriptif dan ukuran sampelnya tidak dirancang untuk menguji "
            f"noninferioritas."))
        add(("p",
            f"Volume model lebih kecil daripada volume laporan dengan rerata "
            f"{mm(-pv['bias'])} mL (batas kesepakatan 95% {ci(pv['loa95'], mm)} mL; median "
            f"selisih relatif {mm(pv['median_rel_diff'] * 100)}%; ρ Spearman="
            f"{pv['spearman']:.2f}) dan lebih kecil daripada volume Bosma22b sebesar "
            f"{mm(-pb['bias'])} mL ({ci(pb['loa95'], mm)} mL; "
            f"{mm(pb['median_rel_diff'] * 100)}%; ρ={pb['spearman']:.2f}). Dice seluruh kelenjar "
            f"terhadap Bosma22b memiliki median {d3(dq['median'])} (IQR "
            f"{d3(dq['q25_q75'][0])}–{d3(dq['q25_q75'][1])}; rerata {d3(dq['mean'])}). Volume "
            f"yang lebih kecil menghasilkan densitas PSA yang lebih tinggi, sehingga pada "
            f"{pc['threshold']:.2f} ng/mL/mL volume model mengklasifikasikan "
            f"{rc['model']['n_positive']} dari {th(pp['n'])} pasien sebagai positif, sedangkan "
            f"volume laporan {rc['report']['n_positive']}. Sensitivitasnya "
            f"{d3(rc['model']['sensitivity'])} (volume laporan: "
            f"{d3(rc['report']['sensitivity'])}), dan spesifisitasnya "
            f"{d3(rc['model']['specificity'])} (volume laporan: "
            f"{d3(rc['report']['specificity'])}). Kedua volume mengklasifikasikan "
            f"{rc['discordant']} pasien secara berbeda: {rc['model_pos_report_neg']} positif "
            f"hanya dengan volume model dan {rc['model_neg_report_pos']} hanya dengan volume "
            f"laporan. Densitas PSA dari volume Bosma22b memiliki AUC {d3(bp['auc_a'])}, lebih "
            f"tinggi {dci(bp)} daripada densitas PSA dari volume laporan."))

        def prow(name, auc, lo_hi, diff, r):
            return [name, f"{d3(auc)} ({d3(lo_hi[0])}–{d3(lo_hi[1])})", diff,
                    d3(r["sensitivity"]), d3(r["specificity"])]

        def dcell(r):
            return f"{d3(r['diff'])} ({ci(r['diff_ci95'])})"
        add(("tbl", (f"Tabel 7. Densitas PSA untuk mendeteksi kanker prostat yang bermakna klinis "
                     f"pada PI-CAI ({th(pp['n'])} pasien, {pp['n_pos']} dengan kanker), menurut "
                     f"sumber volume seluruh kelenjar. Sensitivitas dan spesifisitas dihitung "
                     f"pada {pc['threshold']:.2f} ng/mL/mL.",
            ["Sumber volume", "AUC (IK 95%)", "Selisih terhadap volume laporan (IK 95%)",
             "Sensitivitas", "Spesifisitas"],
            [prow("Model final", pp["auc_a"], pp["auc_a_ci95"], dcell(pp), rc["model"]),
             prow("Laporan radiologi", pp["auc_b"], pp["auc_b_ci95"], "Pembanding", rc["report"]),
             prow("Bosma22b (segmentasi AI publik)", bp["auc_a"], bp["auc_a_ci95"], dcell(bp),
                  rc["bosma"])])))

    add(("h3", "Pelaporan Rekonstruksi pada Studi Terpublikasi"))
    add(("p",
        f"Dari {lit['n_records']} rekaman, {lit['n']} studi memenuhi kriteria "
        f"({lit['dim'].get('2D', 0)} dengan masukan 2D, {lit['dim'].get('2D+3D', 0)} "
        f"membandingkan masukan 2D dengan 3D, dan {lit['dim'].get('2.5D', 0)} dengan masukan "
        f"2.5D); alasan setiap eksklusi tercantum dalam Lampiran Multimedia 4. Tidak satu pun "
        f"menjelaskan cara keluaran per irisan disusun ulang menjadi volume, dan tidak satu pun "
        f"dari {lit['inv_n']} studi yang melaporkan resampling, pengubahan ukuran, atau "
        f"pemotongan citra menjelaskan cara langkah itu dibalik. Tidak satu pun melaporkan "
        f"pemeriksaan terhadap volume hasil rekonstruksi (0 dari {lit['n']}; IK 95% "
        f"0%–{lit['geo_hi']*100:.1f}%). Hanya 2 studi yang menyatakan grid tempat metrik "
        f"dihitung, dan keduanya memakai grid hasil prapemrosesan, bukan grid citra sumber. "
        f"Meski demikian, {lit['n_depend']} studi melaporkan tumpang tindih per volume atau "
        f"jarak dalam milimeter, yang keduanya mensyaratkan volume yang tersusun dengan benar."))

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
        f"Model 2D final berkinerja mendekati kesepakatan ahli pada data uji: "
        f"{cgm['ratio_of_means']*100:.1f}% dan {pzm['ratio_of_means']*100:.1f}% dari Dice "
        f"antarpembaca untuk CG dan PZ, dengan Dice maupun jarak batas yang tidak berbeda "
        f"bermakna dari nilai antarpembaca. Dengan 19 kasus, ini berarti tidak ada selisih yang "
        f"terdeteksi, bukan bukti ekuivalensi. Intervensi pelatihan berpengaruh lebih kecil "
        f"daripada variasi antarpasien dan kira-kira sebesar variasi antar-seed: "
        f"keempat konfigurasi terentang {d3(N['arm_spread'])} macro Dice dan 3 seed dari 1 "
        f"konfigurasi terentang {d3(sd_['range_2d'])} hingga {d3(sd_['range_25d'])}, "
        f"sedangkan SB Dice CG antarpasien pada data uji saja {d3(cg['dice']['sd'])}; "
        f"augmentasi dan pembobotan kelas ekuivalen dengan pembandingnya dalam ±{margin} Dice "
        f"pada sebagian besar perbandingan, efek keseluruhan suku soft Dice tidak "
        f"konklusif, dengan peningkatan post hoc yang terbatas pada PZ, dan masukan 2.5D tidak "
        f"memberi peningkatan. Langkah yang jelas menaikkan akurasi pada data uji adalah "
        f"meng-ensemble 3 seed dengan perataan cermin ({sg(ft['macro']['mean_diff'])} macro "
        f"Dice dibanding satu seed), dan peningkatan ini terbawa ke kedua kohort eksternal. "
        f"Akurasi tetap menurun di luar domain, paling besar di lokasi 1,5 T dengan koil "
        f"endorektal dan "
        f"paling besar untuk PZ. Aturan connected-component, yang ditetapkan dan diperiksa "
        f"secara internal sebelum dipakai, menghilangkan sebagian besar ekor kesalahan batas "
        f"eksternal."))
    if pcr:
        add(("p",
            f"Pada {th(pp['n'])} pasien dari 3 pusat, densitas PSA yang dihitung dari volume "
            f"seluruh kelenjar hasil model membedakan kanker yang bermakna klinis tidak lebih "
            f"buruk daripada densitas PSA yang dihitung dari volume pada laporan radiologi, "
            f"dalam margin yang ditetapkan sebelum data diunduh. Hasilnya tidak lebih baik, dan "
            f"volume model secara sistematis lebih kecil daripada volume laporan maupun volume "
            f"dari sebuah segmentasi AI publik."))
    if not (fx["macro"]["mean_diff"] > 0 and fx["macro"]["p"] < 0.05
            and fn_["cg_dice_none"]["mean_diff"] > 0 and fn_["cg_dice_none"]["p"] < 0.05
            and fn_["pz_dice_none"]["mean_diff"] > 0):
        raise RuntimeError("'peningkatan terbawa ke kedua kohort eksternal' tidak lagi berlaku")

    add(("h3", "Perbandingan dengan Penelitian Terdahulu"))
    add(("p",
        f"Akurasinya tetap berada di bawah implementasi rujukan Prostate158, sebuah 3D residual "
        f"U-Net yang melaporkan Dice sekitar 0,877 untuk CG dan 0,754 untuk PZ pada pembagian "
        f"yang sama [@adams_cbm]; model final mencapai {cg['dice']['mean']/0.877*100:.1f}% dan "
        f"{pz['dice']['mean']/0.754*100:.1f}% dari nilai tersebut. Kekurangan yang lebih besar "
        f"pada PZ sesuai dengan dugaan dari arsitekturnya, karena batas PZ membentang miring "
        f"terhadap bidang aksial, tempat jaringan 2D tidak memiliki konteks. Memberikan konteks "
        f"tersebut melalui masukan 2.5D tidak membantu di sini, sedangkan penelitian terdahulu "
        f"pada MRI prostat melaporkan peningkatan kecil dari 3 irisan masukan "
        f"[@zhang25d,hung_catnet]. Peningkatan tersebut sekitar 0,002 hingga 0,025 Dice, ukuran "
        f"yang tidak dapat dipisahkan dari variasi seed oleh 3 seed per lengan (SB "
        f"{d3(arm25['sd_macro'])} untuk lengan 2.5D), sehingga hasil kami paling tepat dibaca "
        f"sebagai tidak ada manfaat yang terdeteksi pada ukuran sampel ini, bukan bukti yang "
        f"menentang masukan 2.5D. Hasil ini konsisten dengan laporan bahwa efek acak selama "
        f"pelatihan saja menimbulkan selisih performa segmentasi yang sering mencapai "
        f"kebermaknaan statistik [@akesson]. Kami "
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
        "semacam itu memiliki header, bentuk, spasi, dan himpunan label yang benar. Dalam audit "
        f"kami, tidak satu pun dari {lit['n']} studi prostat 2D atau 2.5D terpublikasi "
        f"melaporkan pemeriksaan rekonstruksinya, padahal {lit['n_depend']} studi melaporkan "
        "hasil yang bergantung padanya. Penelitian "
        "yang mengadaptasi model 2D dan video ke data medis 3D [@yamagishi] berakhir pada "
        "langkah penyusunan ulang yang sama. Kami menyarankan agar studi yang menurunkan hasil "
        "3D dari keluaran per irisan melaporkan pemeriksaan header dan round-trip test tingkat "
        "voxel; yang terakhir hanya memerlukan satu lintasan maju dan balik dari anotasi yang "
        "sudah ada."))

    add(("h3", "Implikasi Klinis"))
    pzb, pzb1 = vs["pz"]["bias"], one["vol_s"]["pz"]["bias"]
    if not (pzb < 0 and pzb1 < 0 and abs(pzb - pzb1) < 0.5 and fvs["test"]["pz_dice"]["mean_diff"] > 0
            and psa["n_ever_reclassified"] > psa["n"] / 2):
        raise RuntimeError("paragraf Implikasi Klinis tidak lagi sesuai dengan data")
    add(("p",
        "Dice adalah metrik penelitian; volume adalah jalan segmentasi zona menuju penggunaan "
        "klinis. Kekurangan volume PZ dari model merambat ke densitas PSA. Ensembling, yang "
        f"menaikkan Dice PZ, tidak mengubah bias ini ({mm(pzb)} vs {mm(pzb1)} mL), sehingga bias "
        "tersebut tampaknya sistematis, bukan kebetulan dari 1 kali pelatihan, dan tumpang "
        "tindih yang lebih baik tidak dapat diasumsikan memperbaiki volume. Dalam analisis "
        "sensitivitas, sebagian besar pasien akan melewati ambang 0,15 ng/mL/mL di suatu titik "
        "pada zona abu-abu diagnostik, meskipun rentang PSA tempat hal itu terjadi sempit bagi "
        "sebagian besar dari mereka. Kesalahannya mengarah ke densitas yang terlalu tinggi. "
        "Bersama batas kesepakatan yang lebar relatif terhadap ukuran kelenjar, ini berarti "
        "volume dari model ini tidak boleh dipakai untuk densitas PSA tanpa peninjauan. "
        "Melaporkan volume beserta batas kesepakatannya membuat hal ini terlihat; melaporkan "
        "Dice saja tidak."))
    if pcr:
        add(("p",
            f"Analisis PI-CAI, dengan nilai PSA terukur, mempertajam kesimpulan ini tanpa "
            f"membalikkannya. AUC hanya bergantung pada urutan pasien, dan kekurangan volume "
            f"yang searah pada sebagian besar pasien nyaris tidak mengubah urutan itu (ρ "
            f"Spearman={pv['spearman']:.2f} antara kedua volume); itulah sebabnya kemampuan "
            f"membedakan tetap terjaga. Ambang adalah perkara lain. Volume yang mediannya "
            f"{mm(-pv['median_rel_diff'] * 100)}% lebih kecil memindahkan banyak pasien ke atas "
            f"{pc['threshold']:.2f} ng/mL/mL, sehingga sensitivitas naik "
            f"{(rc['model']['sensitivity'] - rc['report']['sensitivity']) * 100:.0f} poin "
            f"persentase dan spesifisitas turun "
            f"{(rc['report']['specificity'] - rc['model']['specificity']) * 100:.0f} poin. "
            f"Dengan demikian, volume model dapat menggantikan volume laporan ketika pasien "
            f"diurutkan menurut risikonya, tetapi ambang yang ditetapkan dengan volume laporan "
            f"tidak boleh diterapkan padanya tanpa penyesuaian. Kami tidak menurunkan ambang "
            f"terkoreksi ataupun koreksi volume, karena keduanya tidak ditetapkan sebelumnya dan "
            f"koreksi yang dicocokkan pada data ini memerlukan pengujian tersendiri. Volume "
            f"laporan pun bukan standar referensi: densitas PSA dari segmentasi otomatis kedua "
            f"sedikit lebih baik daripada densitas PSA dari laporan, yang mengisyaratkan bahwa "
            f"pengukuran pada laporan itu sendiri merupakan sumber kesalahan."))

    add(("h3", "Keterbatasan"))
    add(("p",
        "Penelitian ini memiliki beberapa kekuatan: aturan pemilihan yang hanya memakai data "
        "validasi, replikasi seed dengan aturan yang ditetapkan sebelum dijalankan, model yang "
        "dibekukan dan diidentifikasi dengan hash bobotnya, "
        "pembaca ahli kedua sebagai pembanding, aturan pascapemrosesan yang ditetapkan sebelum "
        "uji konfirmatorinya, dan desain penyuntikan kesalahan yang membuat setiap perbedaan "
        "pasti berasal dari kesalahan itu sendiri. Penelitian ini juga memiliki keterbatasan. "
        "Kohort pengembangan terdiri atas 158 pemeriksaan dari 1 institusi, dan data uji "
        "berjumlah 19 kasus, sehingga estimasi internal kurang presisi; IK dilaporkan karena "
        "alasan tersebut. Hanya konfigurasi terpilih yang direplikasi lintas seed, dengan 3 "
        "seed per lengan; baseline, E2, dan E3 masing-masing dilatih sekali, sehingga hasil "
        "ekuivalensi menggambarkan run-run tersebut, bukan konfigurasinya secara umum, dan 3 "
        "seed hanya memberi estimasi kasar atas variasi antar-seed [@akesson]. Data uji telah "
        "dipakai dua kali selama pengembangan pipeline sebelumnya dengan model yang berbeda, "
        "dan sekali untuk model seed-tunggal sebelum perluasan replikasi seed ditulis. Setiap "
        "aturan pemilihan hanya memakai data validasi, tetapi perluasan tersebut dirancang "
        "ketika hasil uji sebelumnya sudah diketahui, dan data uji tidak sepenuhnya belum "
        "tersentuh. Karena itu, aturan connected-component diuji pada NCI-ISBI 2013, yang belum "
        "pernah dipakai sebelumnya; uji konfirmatori tersebut milik model seed-tunggal, dan "
        "analisis model final mengulanginya pada kohort yang saat itu sudah pernah "
        "dilihat. Model final juga memerlukan 6 lintasan maju per irisan (3 anggota, "
        "masing-masing pada irisan dan cerminannya), dibandingkan 1 untuk satu jaringan."))
    if not (abs(nc["excl_px"]["cg_dice"] - pr["cg_dice_none"]["mean"]) <= 0.001
            and abs(nc["excl_px"]["cg_hd95_R1"] - r1["hd95_after"]["mean"]) <= 0.1):
        raise RuntimeError("'tidak lebih dari 0,001 Dice dan 0,1 mm' tidak lagi berlaku")
    add(("p",
        f"Kohort eksternal dianotasi dengan protokol yang berbeda, sehingga perbedaan eksternal "
        f"menggabungkan generalisasi model dengan perbedaan anotasi, dan kesalahan pada standar "
        f"referensi dapat menggeser performa yang terukur dengan sendirinya [@wang_labels]; "
        f"perbandingan 2 anotasi membatasi, tetapi tidak menghilangkan, hal ini. Kohort-kohort "
        f"tersebut juga sebagian tumpang tindih ({ov['n_msd']} citra yang sama antara NCI-ISBI "
        f"dan MSD, dan 1 antara NCI-ISBI dan PROSTATEx); hal ini tidak memengaruhi data latih, "
        f"dan mengeluarkan kasus yang sama dengan PROSTATEx mengubah hasil NCI-ISBI tidak lebih "
        f"dari 0,001 Dice dan 0,1 mm (Dice CG {d3(nc['excl_px']['cg_dice'])} vs "
        f"{d3(pr['cg_dice_none']['mean'])}; HD95 CG setelah aturan "
        f"{mm(nc['excl_px']['cg_hd95_R1'])} vs {mm(r1['hd95_after']['mean'])} mm). Penyuntikan "
        f"kesalahan memakai himpunan kesalahan tetap yang "
        f"diimplementasikan pada 1 pipeline; pipeline lain dapat gagal dengan cara lain, dan 2 "
        f"kesalahan yang tidak berbahaya di sini belum tentu tidak berbahaya di tempat lain. "
        + ((f"Analisis densitas PSA pada Prostate158 memakai rentang nilai PSA; nilai terukur "
            f"hanya tersedia pada PI-CAI. Kohort tersebut berasal dari 3 pusat di 1 negara. "
            f"Sebagian studi negatifnya dikonfirmasi dengan pembacaan MRI tanpa biopsi, sehingga "
            f"sebagian kanker mungkin terlewat; bias verifikasi ini berlaku sama bagi kedua "
            f"densitas PSA, tetapi pengaruhnya terhadap nilai AUC itu sendiri tidak diketahui. "
            f"Volume laporan diukur dalam layanan rutin dengan metode yang tidak tercatat. "
            f"Sebanyak {pdup['n_studies']} studi PI-CAI merupakan duplikat citra kohort "
            f"eksternal kami, sehingga kohort-kohort tersebut tidak independen dari PI-CAI, "
            f"meskipun mengeluarkan duplikat itu tidak mengubah hasil. Margin noninferioritas "
            f"merupakan pilihan kami sendiri, noninferioritas AUC tidak mengatakan apa pun "
            f"tentang kalibrasi, dan pada 1 dari 3 pusat batas bawah interval kepercayaannya "
            f"jatuh di bawah margin. Tidak dilakukan studi pembaca, sehingga tidak ada klaim "
            f"tentang luaran pasien. ") if pcr else
           (f"Analisis PSA memakai rentang nilai PSA, bukan nilai terukur, dan tidak dilakukan "
            f"studi pembaca, sehingga tidak ada klaim tentang kegunaan klinis atau luaran pasien. "))
        + f"Akurasi juga bergantung pada ukuran kelenjar, yang membatasi penggunaan pada kelenjar "
        f"kecil. Terakhir, audit literatur hanya mencakup artikel akses terbuka di PubMed "
        f"Central, memakai istilah pencarian pada judul, dan dilakukan oleh 1 penelaah, serta "
        f"mengukur apa yang dilaporkan studi, bukan apa yang sebenarnya mereka lakukan."))

    add(("h3", "Kesimpulan"))
    add(("p",
        "Merekonstruksi segmentasi per irisan menjadi volume 3D dapat gagal dengan cara yang "
        "tetap menghasilkan berkas yang sah, dan dalam penelitian ini pemeriksaan header dan "
        "round-trip test tingkat voxel masing-masing melewatkan kelas kegagalan yang berbeda. "
        "Hanya gabungan keduanya yang mendeteksi setiap kesalahan yang merusak pada 4 kohort "
        "publik, dan terlihat atau tidaknya suatu kesalahan bergantung pada orientasi "
        "penyimpanan data. Ensemble 3 seed dari 2D U-Net menyegmentasi kedua zona tanpa selisih "
        "yang terdeteksi dari kesepakatan ahli pada data uji yang kecil, sementara selisih "
        "antarkonfigurasi pelatihan kira-kira sebesar selisih antar-seed, dan masukan "
        "2.5D tidak memberi manfaat. Aturan connected-component yang ditetapkan "
        "sebelumnya menghilangkan sebagian besar kesalahan batas eksternal. Studi yang "
        "menurunkan hasil 3D dari segmentasi 2D sebaiknya melaporkan kedua pemeriksaan "
        "verifikasi dan melaporkan volume beserta batas kesepakatannya di samping Dice."
        + (" Pada kohort multipusat yang tidak dipakai dalam pengembangan, densitas PSA dari "
           "volume model noninferior terhadap densitas PSA dari volume laporan untuk mendeteksi "
           "kanker yang bermakna klinis, tetapi volume model yang lebih kecil menggeser titik "
           "operasi pada ambang yang lazim, sehingga keduanya dapat dipertukarkan untuk "
           "pengurutan risiko dan tidak untuk keputusan berbasis ambang." if pcr else "")))

    # --------------------------------------------------------- bagian akhir
    add(("h2", "Ucapan Terima Kasih"))
    add(("p",
        "Penulis berterima kasih kepada pembuat Prostate158, PROSTATEx beserta anotasi zonanya, "
        + ("NCI-ISBI 2013, Medical Segmentation Decathlon, dan tantangan PI-CAI yang telah "
           "membuka datanya untuk " if pc else
           "NCI-ISBI 2013, dan Medical Segmentation Decathlon yang telah membuka datanya untuk ") +
        "publik. Kecerdasan artifisial generatif (Claude, Anthropic) dipakai untuk membantu "
        "pengembangan kode, penulisan skrip analisis, pembuatan gambar, serta penyusunan draf dan "
        "penyuntingan bahasa naskah. Seluruh konten yang dihasilkan telah "
        "ditinjau dan diverifikasi oleh penulis, yang bertanggung jawab penuh atas karya ini. "
        "Penulis juga berterima kasih kepada Universitas Multimedia Nusantara atas dukungannya "
        "terhadap penelitian ini."))
    add(("h2", "Pendanaan"))
    add(("p", "Penelitian ini didukung oleh Universitas Multimedia Nusantara. Pemberi dana tidak berperan dalam desain penelitian, "
              "analisis data, interpretasi, maupun keputusan publikasi."))
    add(("h2", "Ketersediaan Data"))
    add(("p",
        "Seluruh dataset yang dianalisis tersedia secara publik dari sumber aslinya "
        f"[@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli{',picai_data,picai_labels' if pc else ''}]. "
        "Dataset dipakai sebagaimana "
        "adanya; seluruh prapemrosesan yang diterapkan dalam penelitian ini dijelaskan di "
        "bagian Metode."
        + (" Label dan nilai klinis PI-CAI dipakai pada commit repositori yang dicatat dalam "
           "Lampiran Multimedia 5." if pc else "")))
    add(("h2", "Ketersediaan Kode"))
    add(("p",
        f"Kode analisis, berkas konfigurasi, {'ketiga' if pc else 'kedua'} penetapan awal, dan skrip yang "
        "menghasilkan setiap angka dalam artikel ini tersedia secara publik di repositori "
        "proyek [@repo]. Model final yang dibekukan diidentifikasi di sana dengan hash SHA-256 "
        "dari bobot ketiga anggotanya, beserta catatan pemilihannya. "
        "Ketiga tingkat verifikasi juga disediakan sebagai skrip mandiri yang hanya bergantung "
        "pada NumPy dan NiBabel (tools/verify_reconstruction.py), sehingga dapat diterapkan pada "
        "pipeline lain."))
    add(("h2", "Kontribusi Penulis"))
    add(("p",
        "Konseptualisasi: BAS, EES. Metodologi: BAS. Perangkat lunak: BAS. Analisis formal: "
        "BAS. Validasi: BAS. Visualisasi: BAS. Penulisan draf awal: BAS. Penulisan, tinjauan, "
        "dan penyuntingan: EES, DA, MW, BAS. Supervisi: EES."))
    add(("h2", "Konflik Kepentingan"))
    add(("p", "Tidak ada."))
    add(("h2", "Lampiran Multimedia 1"))
    add(("p", "Daftar periksa CLAIM 2024."))
    add(("h2", "Lampiran Multimedia 2"))
    add(("p", "Penetapan awal aturan connected-component dan analisis NCI-ISBI 2013, beserta "
              "erratum dan hash berkas yang dicatat; serta penetapan awal perluasan replikasi "
              "seed dan masukan 2.5D, beserta hash berkas yang dicatat dan catatan pemilihannya."))
    add(("h2", "Lampiran Multimedia 3"))
    add(("p", "Hasil tambahan: riwayat pelatihan, hasil per kasus, hasil model seed-tunggal dan "
              "model final secara berdampingan, geometri kohort, dan gambar tambahan."))
    add(("h2", "Lampiran Multimedia 4"))
    add(("p", "Audit pelaporan studi terpublikasi: protokol beserta hash berkasnya, catatan "
              "pencarian, keputusan penyaringan beserta alasannya, dan pengodean per studi beserta "
              "kutipan pendukung."))
    if pc:
        add(("h2", "Lampiran Multimedia 5"))
        add(("p", "Densitas PSA pada PI-CAI: penetapan awal beserta hash berkas yang dicatat, "
                  "erratum, commit repositori label, alur kohort, catatan konversi dan verifikasi, "
                  "serta hasil per studi."))
    add(("h2", "Daftar Pustaka"))
    add(("refs", REFERENCES))
    add(("h2", "Singkatan"))
    abbr = [("2D", "dua dimensi"), ("3D", "tiga dimensi"), ("CG", "central gland"),
            ("CLAIM", "Checklist for Artificial Intelligence in Medical Imaging"),
            ("HD95", "jarak Hausdorff persentil ke-95"),
            ("IK", "interval kepercayaan"),
            ("LAS/LPS/RAS", "konvensi orientasi yang menamai arah 3 sumbu array pertama"),
            ("MRI", "magnetic resonance imaging"), ("MSD", "Medical Segmentation Decathlon"),
            ("PSA", "prostate-specific antigen"), ("PZ", "peripheral zone"),
            ("SB", "simpangan baku"), ("TOST", "two one-sided tests")]
    if pc:
        abbr += [("AUC", "area di bawah kurva receiver operating characteristic"),
                 ("csPCa", "kanker prostat yang signifikan secara klinis"),
                 ("ISUP", "International Society of Urological Pathology"),
                 ("PI-CAI", "Prostate Imaging: Cancer AI"),
                 ("PI-RADS", "Prostate Imaging-Reporting and Data System")]
        abbr.sort(key=lambda kv: kv[0].lower())
    for k, v in abbr:
        add(("p", f"{k}: {v}"))

    # Singkatan di dalam gambar didefinisikan di akhir keterangannya, seperti versi Inggris.
    tails = {
        1: ("AUC: area under the receiver operating characteristic curve; " if pc else "")
           + "CG: central gland; LAS/LPS/RAS: konvensi orientasi; MSD: Medical Segmentation "
           "Decathlon; " + ("PI-CAI: Prostate Imaging: Cancer AI; " if pc else "")
           + "PSA: prostate-specific antigen; PZ: peripheral zone; TOST: two one-sided "
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
