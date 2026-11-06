"""Mirror the complete thesis into the manual-upload folder, and zip it.

    python scripts/sync_overleaf_folder.py

SKRIPSI-Overleaf is what the student uploads to Overleaf by hand. It is a full
mirror of everything the thesis needs to compile - every chapter, front-matter
page, generated table, figure, style file - and SKRIPSI-Overleaf.zip holds the
same files at the zip root, so Overleaf's "New Project > Upload Project" builds
a working project from it in one step.

Left out on purpose: editor backups (*.bak*), the backup folder
_cadangan_*, artikel_ilmiah.tex (Overleaf would pick it as the main document),
the template's unused references.bib, and src/3.Akhir/pustaka.bib (its basename
collides with the root pustaka.bib and once emptied the bibliography).

The zip is refused if any \\input or \\includegraphics would not resolve, or if
more than one file declares \\documentclass.
"""
import hashlib
import io
import os
import re
import shutil
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = r"D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840"
STAGE = r"D:\1.AmadeusKuliah\Skripsi\SKRIPSI-Overleaf"
ZIP = r"D:\1.AmadeusKuliah\Skripsi\SKRIPSI-Overleaf.zip"

EXCLUDE_FILES = {"artikel_ilmiah.tex", "references.bib"}
EXCLUDE_PATHS = {os.path.join("src", "3.Akhir", "pustaka.bib")}

# Pages whose Overleaf versions held the student's own entries. 05 has been
# restored exactly (verified against the Overleaf hash); the other three could
# not be read back and are still the template.
TEMPLATE_PERSONAL = ["src/1.Awal/02b-HalamanAI.tex", "src/1.Awal/03a-Persetujuan.tex",
                     "src/1.Awal/04-Publikasi.tex"]

README = """\
SKRIPSI - PAKET LENGKAP UNTUK OVERLEAF
======================================

Isi folder ini (dan SKRIPSI-Overleaf.zip) adalah proyek skripsi LENGKAP:
semua bab, halaman awal, tabel, gambar, dan berkas gaya. {n} berkas.

CARA PALING MUDAH
  Overleaf > New Project > Upload Project > pilih SKRIPSI-Overleaf.zip
  Semua berkas langsung berada di root proyek. Dokumen utama: SKRIPSI.tex
  (kalau Overleaf memilih yang lain: Menu > Main document > SKRIPSI.tex).

KALAU MEMAKAI PROYEK YANG SEKARANG
  Unggah ISI zip ke root proyek, bukan sebagai satu folder. Kalau semua
  berkas masuk ke satu subfolder, \\input dan gambar tidak akan ditemukan.
  Setel juga Menu > Main document > SKRIPSI.tex (saat ini masih main.tex).

PERLU MAS PERIKSA SETELAH DIUNGGAH
  src/1.Awal/05-Persembahan.tex  -> SUDAH berisi motto Mas (Kolose 3:23).
  Tiga halaman berikut masih VERSI TEMPLATE. Isian Mas di Overleaf ada di
  riwayatnya (History) proyek lama; pulihkan dari sana atau isi ulang:
{personal}
  Lampiran wajib, ganti dengan dokumen Mas sendiri:
    assets/pics/fig_individual_source_sample_turnitin.png  -> hasil Turnitin
    assets/pdf/Form_Bimbingan_atau_Konsultasi_Skripsi.pdf  -> formulir bimbingan
  settings.tex masih memuat isian kosong (tanggal, NIDN pembimbing).

YANG BARU DIBANDING VERSI SEBELUMNYA
  22 gambar hasil (sebelumnya 10), 17 tabel, folder src/_generated/,
  koreksi Bab 3 dan Bab 4, lampiran berisi hasil penelitian.
"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wanted(rel):
    parts = rel.split(os.sep)
    name = parts[-1]
    if any(p.startswith("_cadangan") for p in parts):
        return False
    if ".bak" in name or name in EXCLUDE_FILES or rel in EXCLUDE_PATHS:
        return False
    return True


def collect():
    out = []
    for dirpath, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if not d.startswith("_cadangan")]
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), SRC)
            if wanted(rel):
                out.append(rel)
    return sorted(out)


def check(root, files):
    """Every reference resolves and exactly one \\documentclass exists."""
    problems = []
    classes = []
    present = set(f.replace(os.sep, "/") for f in files)
    for rel in files:
        if not rel.endswith(".tex"):
            continue
        for line in io.open(os.path.join(root, rel), encoding="utf-8", errors="replace"):
            if line.lstrip().startswith("%"):
                continue
            if "\\documentclass" in line:
                classes.append(rel)
            for m in re.finditer(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", line):
                kind, p = m.groups()
                cand = [p, p + ".tex"] if kind == "input" else [p]
                if not any(c in present for c in cand):
                    problems.append("%s -> %s" % (rel, p))
    if len(classes) != 1:
        problems.append("documentclass in %s" % classes)
    return problems


def main():
    files = collect()

    # Mirror: copy what changed, then drop anything the mirror should no
    # longer hold.
    counts = {"baru": 0, "diperbarui": 0, "sama": 0}
    for rel in files:
        src, dst = os.path.join(SRC, rel), os.path.join(STAGE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and sha(src) == sha(dst):
            counts["sama"] += 1
            continue
        counts["diperbarui" if os.path.exists(dst) else "baru"] += 1
        shutil.copyfile(src, dst)
    keep = set(files) | {"README_UNGGAH.txt"}
    for dirpath, _, fs in os.walk(STAGE):
        for f in fs:
            rel = os.path.relpath(os.path.join(dirpath, f), STAGE)
            if rel not in keep:
                os.remove(os.path.join(STAGE, rel))
                print("  dihapus dari cermin: %s" % rel)

    problems = check(STAGE, files)
    if problems:
        print("DITOLAK - rujukan tidak lengkap:")
        for p in problems:
            print("  " + p)
        return 1

    io.open(os.path.join(STAGE, "README_UNGGAH.txt"), "w", encoding="utf-8").write(
        README.format(n=len(files),
                      personal="\n".join("    " + p for p in TEMPLATE_PERSONAL)))

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files + ["README_UNGGAH.txt"]:
            z.write(os.path.join(STAGE, rel), rel.replace(os.sep, "/"))

    print("cermin: %d berkas (baru %d, diperbarui %d, sama %d)"
          % (len(files), counts["baru"], counts["diperbarui"], counts["sama"]))
    print("rujukan: semua \\input dan \\includegraphics ditemukan; satu \\documentclass")
    print("zip: %s (%.1f MB, %d berkas)"
          % (ZIP, os.path.getsize(ZIP) / 1e6, len(zipfile.ZipFile(ZIP).namelist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
