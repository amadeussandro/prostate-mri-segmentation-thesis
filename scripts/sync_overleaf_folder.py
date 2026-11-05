"""Bring the local Overleaf upload folder up to date, safely.

    python scripts/sync_overleaf_folder.py

SKRIPSI-Overleaf mirrors the Overleaf project's layout and is what the student
uploads by hand. This copies in every file that changed in the thesis source,
adds what is new, and builds a small zip of just the changed files.

Four pages are deliberately kept OUT of the upload folder: 02b-HalamanAI,
03a-Persetujuan, 04-Publikasi, 05-Persembahan. The student filled them in
directly on Overleaf (verified 2026-11-06 by hashing the Overleaf copies), and
the local copies are still the template. Uploading them would erase that work,
so they are removed here - only after checking that the source folder still
holds a byte-identical copy - and the README says why.
"""
import hashlib
import io
import os
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = r"D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840"
STAGE = r"D:\1.AmadeusKuliah\Skripsi\SKRIPSI-Overleaf"
ZIP = r"D:\1.AmadeusKuliah\Skripsi\SKRIPSI-Overleaf-update.zip"

CHANGED_TEX = [
    r"src\1.Awal\01-HalamanJudul.tex",
    r"src\2.Isi\Bab2.tex",
    r"src\2.Isi\Bab3.tex",
    r"src\2.Isi\Bab4.tex",
    r"src\3.Akhir\lampiran.tex",
]
GENERATED = [r"src\_generated\%s" % f for f in (
    "tabel_konfigurasi_eksperimen.tex", "tabel_geometri_kohort.tex",
    "tabel_perkasus_internal.tex", "tabel_eksternal_lengkap.tex")]
PICS_NEW = [r"assets\pics\%s" % f for f in (
    "fig_kurva_pelatihan.png", "fig_pemilihan_epoch.png", "fig_rq1_ikhtisar.png",
    "fig_rq1_sebaran.png", "fig_rq1_semua_metrik.png", "fig_e3_mekanisme.png",
    "fig_validasi_vs_uji.png", "fig_uji_perkasus.png", "fig_rq1_perjalanan.png",
    "fig_proyeksi_3d_baik.png", "fig_proyeksi_3d_median.png", "fig_proyeksi_3d_sulit.png")]
PERSONAL = [r"src\1.Awal\%s" % f for f in (
    "02b-HalamanAI.tex", "03a-Persetujuan.tex", "04-Publikasi.tex", "05-Persembahan.tex")]

README = """\
PEMBARUAN SKRIPSI UNTUK OVERLEAF
================================

Folder ini sudah diperbarui dan strukturnya sama dengan proyek Overleaf.
Unggah per folder, ke folder dengan nama yang sama di Overleaf.

YANG BARU / BERUBAH
  assets/pics/      12 gambar BARU  -> WAJIB diunggah, tanpa ini Overleaf error:
                    {pics}
  src/_generated/   4 tabel (folder ini sudah ada di Overleaf)
  src/1.Awal/       01-HalamanJudul.tex   (typo "PROGRAM STUDI1" diperbaiki)
  src/2.Isi/        Bab3.tex, Bab4.tex    (Bab2.tex sudah sama dengan Overleaf)
  src/3.Akhir/      lampiran.tex

  Semua .tex di atas SUDAH terunggah ke Overleaf. Kalau ragu, unggah ulang
  saja: isinya sama, aman ditimpa.

SENGAJA TIDAK ADA DI FOLDER INI - JANGAN DIUNGGAH DARI MANA PUN
  src/1.Awal/02b-HalamanAI.tex
  src/1.Awal/03a-Persetujuan.tex
  src/1.Awal/04-Publikasi.tex
  src/1.Awal/05-Persembahan.tex
  Keempatnya sudah Mas isi langsung di Overleaf. Salinan lokal masih
  template kosong; mengunggahnya akan menghapus isian Mas.

JANGAN SENTUH
  BACKUP_ORI_TEMPLATE_UMN (hanya ada di Overleaf)

MASIH HARUS MAS GANTI SENDIRI (lampiran wajib)
  assets/pics/fig_individual_source_sample_turnitin.png -> hasil Turnitin Mas
  assets/pdf/Form_Bimbingan_atau_Konsultasi_Skripsi.pdf -> formulir bertanda tangan
"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def copy(rel):
    src, dst = os.path.join(SRC, rel), os.path.join(STAGE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    existed = os.path.exists(dst)
    same = existed and sha(src) == sha(dst)
    if not same:
        shutil.copyfile(src, dst)
    return "sama" if same else ("diperbarui" if existed else "baru")


def main():
    for rel in CHANGED_TEX + GENERATED + PICS_NEW:
        print("  %-11s %s" % (copy(rel), rel))

    for rel in PERSONAL:
        st, src = os.path.join(STAGE, rel), os.path.join(SRC, rel)
        if not os.path.exists(st):
            print("  %-11s %s" % ("sudah tiada", rel))
            continue
        if not os.path.exists(src) or sha(src) != sha(st):
            raise SystemExit("REFUSING to remove %s: the source copy is missing or differs, "
                             "so it may hold edits that exist nowhere else" % rel)
        os.remove(st)
        print("  %-11s %s  (salinan identik tetap ada di folder sumber)" % ("dikeluarkan", rel))

    io.open(os.path.join(STAGE, "README_UNGGAH.txt"), "w", encoding="utf-8").write(
        README.format(pics="\n                    ".join(os.path.basename(p) for p in PICS_NEW)))

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in [r for r in CHANGED_TEX if not r.endswith("Bab2.tex")] + GENERATED + PICS_NEW:
            z.write(os.path.join(STAGE, rel), rel.replace("\\", "/"))
        z.write(os.path.join(STAGE, "README_UNGGAH.txt"), "README_UNGGAH.txt")
    print("\nzip pembaruan: %s (%.1f MB, %d berkas)"
          % (ZIP, os.path.getsize(ZIP) / 1e6, len(zipfile.ZipFile(ZIP).namelist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
