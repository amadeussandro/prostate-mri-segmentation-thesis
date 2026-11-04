"""Content audit of the Indonesian manuscript, plus an EN-vs-ID number cross-check.

content_audit.py only ever read the English file, so the Indonesian version had
been corrected but never verified. The two language versions are generated from
separate content modules, so a number can drift in one without the other
noticing: the cross-check here compares every reported figure across the pair.

No pattern in this file uses \\b. An earlier revision lost its word boundaries to
backslash mangling, which turned them into literal backspace characters and made
the cross-check report a mismatch that did not exist. Explicit lookarounds cannot
fail that way, and every mismatch is now printed with its context so an artefact
can be told from a real divergence.
"""
import io
import os
import re
import sys

import docx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Overridable, so the check is not pinned to one machine's layout.
J = (sys.argv[1] if len(sys.argv) > 1
     else os.environ.get("JOURNAL_DIR", r"D:\1.AmadeusKuliah\Skripsi\Journal-JMIR"))
EN = os.path.join(J, "JurnalJMIR-BenedictAmadeusSandro-EN.docx")
ID = os.path.join(J, "JurnalJMIR-BenedictAmadeusSandro-ID.docx")

HEADLINE = {
    "0.8442": "Dice CG internal",
    "0.6852": "Dice PZ internal",
    "0.7647": "Dice makro internal",
    "0.8137": "Dice CG eksternal",
    "0.6247": "Dice PZ eksternal",
    "0.7192": "Dice makro eksternal",
    "0.5960": "ablasi, tanpa reorientasi CG",
    "182.8": "ablasi, pergeseran sentroid affine identitas",
    "0.7287": "Dice CG tertil terkecil",
    "0.8754": "Dice CG tertil terbesar",
    "0.766": "Spearman r ukuran kelenjar",
}

OVERREACH = [
    (r"secara signifikan lebih (baik|tinggi)", "klaim signifikansi"),
    (r"membuktikan", "'membuktikan' (lebih kuat dari 'demonstrate')"),
    (r"terbukti (unggul|lebih unggul|superior)", "klaim keunggulan"),
    (r"siap (secara )?klinis|tervalidasi secara klinis", "klaim kesiapan klinis"),
    (r"mengalahkan|melampaui semua", "klaim mengalahkan"),
]

MUST_APPEAR_IN_LIMITATIONS = {
    "ukuran kelenjar": r"ukuran kelenjar|tertil ukuran|kelenjar kecil",
    "pemakaian ulang data uji": (r"tidak lagi murni|dipakai dua kali"
                                 r"|pemakaian sebelumnya|tidak sepenuhnya murni"),
    "tanpa reader study": r"reader study",
    "dataset tunggal": r"satu dataset publik|dataset publik tunggal",
    "resampling in-plane": r"resampling in-plane|resampling bidang",
}

ABSTRACT_CHECKS = [
    ("temuan ukuran kelenjar", r"ukuran kelenjar|tertil ukuran|tertil terkecil"),
    ("hasil negatif", r"lebih kecil dari|tidak mencapai|di bawah"),
    ("validasi eksternal", r"eksternal|PROSTATEx|204"),
    ("ablasi rekonstruksi", r"cermin|mirror|ablasi|geometri"),
    ("terjemahan klinis", r"densitas PSA|PSA density"),
]

# A reported metric: digits, a decimal point, then three or four digits, not
# part of a longer number and not part of a dotted string such as a DOI.
METRIC = re.compile(r"(?<![\d.])(\d+\.\d{3,4})(?![\d.])")

# A comma between digits cannot be classified by shape: Indonesian "0,766" is a
# decimal and English "2,000" is a thousands separator, and the two are
# identical patterns. Treating them with one rule destroyed every three-decimal
# Indonesian number and reported thirteen false mismatches. So normalise per
# language instead, which is unambiguous: English uses the comma only as a
# thousands separator, Indonesian only as a decimal mark. Verified against both
# documents - every comma-plus-three-digits token in the Indonesian file begins
# "0,", and the English file has only 2,000 resamples and 7,762,531 parameters.
COMMA_BETWEEN_DIGITS = re.compile(r"(?<=\d),(?=\d)")


def read(path):
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs]
    tbl = "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    return paras, "\n".join(paras) + "\n" + tbl


def section(paras, start, *stops):
    out, on = [], False
    for t in paras:
        s = t.strip()
        if re.match(start, s, re.I):
            on = True
            continue
        if on and any(re.match(x, s, re.I) for x in stops):
            break
        if on:
            out.append(t)
    return "\n".join(out)


def norm(s, lang):
    """Put both languages' numbers into one comparable form."""
    if lang == "en":
        return COMMA_BETWEEN_DIGITS.sub("", s)      # 2,000 -> 2000
    return COMMA_BETWEEN_DIGITS.sub(".", s)         # 0,766 -> 0.766


def context(text, needle, pad=80):
    i = text.find(needle)
    if i < 0:
        return "(tidak ditemukan dalam teks ternormalisasi)"
    return "..." + text[max(0, i - pad):i + len(needle) + 50].replace("\n", " ") + "..."


def main():
    paras, whole = read(ID)
    txt = "\n".join(paras)
    abstract = section(paras, r"^Abstrak$", r"^Pendahuluan$")
    limits = section(paras, r"^Keterbatasan$", r"^Simpulan$")
    concl = section(paras, r"^Simpulan$", r"^Ucapan Terima Kasih")

    issues = []

    print("=" * 74)
    print("1. ANGKA HEADLINE (versi ID)")
    print("=" * 74)
    for num, what in HEADLINE.items():
        n = whole.count(num) + whole.count(num.replace(".", ","))
        if not n:
            issues.append("%s (%s) tidak muncul sama sekali" % (num, what))
        print("  %-8s %-44s x%-2d %s" % (num, what, n, "" if n else "<-- HILANG"))

    print()
    print("=" * 74)
    print("2. KLAIM BERLEBIHAN")
    print("=" * 74)
    hit = False
    for pat, label in OVERREACH:
        for m in re.finditer(pat, txt, re.I):
            ctx = txt[max(0, m.start() - 70):m.start() + 60].replace("\n", " ")
            print("  %-42s ...%s..." % (label, ctx))
            issues.append("kemungkinan klaim berlebihan: %s" % label)
            hit = True
    if not hit:
        print("  tidak ada")

    print()
    print("=" * 74)
    print("3. CAKUPAN KETERBATASAN")
    print("=" * 74)
    for label, pat in MUST_APPEAR_IN_LIMITATIONS.items():
        ok = bool(re.search(pat, limits, re.I))
        if not ok:
            issues.append("keterbatasan tidak menyebut: %s" % label)
        print("  %-26s %s" % (label, "ada" if ok else "TIDAK ADA"))

    print()
    print("=" * 74)
    print("4. ABSTRAK vs HASIL")
    print("=" * 74)
    for label, pat in ABSTRACT_CHECKS:
        ok = bool(re.search(pat, abstract, re.I))
        if not ok:
            issues.append("abstrak tidak memuat: %s" % label)
        print("  %-26s %s" % (label, "ada" if ok else "TIDAK ADA"))

    print()
    print("=" * 74)
    print("5. SIMPULAN vs hasil p = 0,055")
    print("=" * 74)
    hedged = bool(re.search(r"tidak mencapai|tidak signifikan|arah.{0,24}konsisten"
                            r"|konsisten secara arah", concl + limits, re.I))
    print("  simpulan/keterbatasan berhati-hati:", "ya" if hedged else "TIDAK")
    if not hedged:
        issues.append("hasil p=0,055 mungkin dilebihkan di Simpulan")

    print()
    print("=" * 74)
    print("6. SILANG EN vs ID - apakah angka yang dilaporkan sama?")
    print("=" * 74)
    _, en_whole = read(EN)
    en_norm, id_norm = norm(en_whole, "en"), norm(whole, "id")
    a = set(METRIC.findall(en_norm))
    b = set(METRIC.findall(id_norm))
    only_en, only_id = sorted(a - b), sorted(b - a)
    print("  angka 3-4 desimal: EN %d, ID %d, sama %d" % (len(a), len(b), len(a & b)))
    for n in only_en:
        print("    hanya di EN: %s" % n)
        print("      %s" % context(en_norm, n))
    for n in only_id:
        print("    hanya di ID: %s" % n)
        print("      %s" % context(id_norm, n))
    if only_en or only_id:
        issues.append("%d angka tidak cocok antara EN dan ID"
                      % (len(only_en) + len(only_id)))

    print()
    print("=" * 74)
    print("ISSUES: %d" % len(issues))
    for i in issues:
        print("  -", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
