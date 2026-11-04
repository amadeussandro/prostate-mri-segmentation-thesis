"""Generate the skripsi DAFTAR PUSTAKA as plain LaTeX from pustaka.bib.

Why not BibTeX: an earlier compile printed the DAFTAR PUSTAKA heading and its
TOC entry while listing no references and showing every citation as [?]. Both
only appear when a thebibliography environment is processed, so BibTeX had run
and emitted an EMPTY list - which happens when none of the cited keys are found
in the database it read. The template ships a second file also named
pustaka.bib under src/3.Akhir/, and \\bibliography{pustaka} resolves by
basename, so which one wins depends on search order.

Writing the bibliography as ordinary LaTeX removes that dependency entirely:
\\cite resolves straight from \\bibitem on the second pdflatex run, with no
BibTeX pass, no .bst and no basename lookup. pustaka.bib stays as the
machine-readable record.

IEEE numbers by order of first citation, so that order is computed from the
sources in the order SKRIPSI.tex inputs them, never hand-maintained.

Usage:
    python scripts/gen_daftarpustaka.py
    python scripts/gen_daftarpustaka.py --check
"""

from __future__ import annotations

import io
import os
import re
import sys

ROOT = r"D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840"
BIB = os.path.join(ROOT, "pustaka.bib")
MAIN = os.path.join(ROOT, "SKRIPSI.tex")
OUT = os.path.join(ROOT, "src", "3.Akhir", "daftarpustaka.tex")

BS = chr(92)
OQ = "``"
CQ = chr(39) * 2
MAX_NAMES = 6          # IEEE: first six authors, then et al.


def strip_braces(v: str) -> str:
    v = v.strip()
    while v.startswith("{") and v.endswith("}"):
        v = v[1:-1].strip()
    return v


def parse_bib(path: str) -> dict:
    text = io.open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\s*$)",
                              m.group(3), re.S):
            fields[fm.group(1).lower()] = strip_braces(fm.group(2))
        out[m.group(2)] = (m.group(1).lower(), fields)
    return out


def fmt_author(raw: str) -> str:
    """'Sung, Hyuna and Ferlay, Jacques and others' -> 'H. Sung, J. Ferlay, et al.'"""
    parts = [p.strip() for p in re.split(r"\s+and\s+", raw.strip()) if p.strip()]
    etal = False
    if parts and parts[-1].lower() == "others":
        parts, etal = parts[:-1], True
    if len(parts) > MAX_NAMES:
        parts, etal = parts[:MAX_NAMES], True

    names = []
    for p in parts:
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
        else:
            bits = p.split()
            last, first = bits[-1], " ".join(bits[:-1])
        initials = " ".join(w[0] + "." for w in first.replace(".", " ").split() if w)
        names.append((initials + " " + last).strip() if initials else last)

    s = ", ".join(names)
    return s + ", " + BS + "emph{et al.}" if etal else s


def fmt_entry(typ: str, f: dict) -> str:
    bits = []
    if f.get("author"):
        bits.append(fmt_author(f["author"]) + ",")
    if f.get("title"):
        bits.append(OQ + f["title"].rstrip(".") + "," + CQ)
    venue = f.get("journal") or f.get("booktitle")
    if venue:
        bits.append(BS + "emph{" + venue + "},")
    if f.get("series"):
        bits.append(f["series"] + ",")
    if f.get("volume"):
        bits.append("vol.~" + f["volume"] + ",")
    if f.get("number"):
        bits.append("no.~" + f["number"] + ",")
    if f.get("pages"):
        bits.append(("pp.~" if "--" in f["pages"] else "p.~") + f["pages"] + ",")
    if f.get("publisher") and typ == "inproceedings":
        bits.append(f["publisher"] + ",")
    if f.get("year"):
        bits.append(f["year"] + ".")
    if f.get("doi"):
        bits.append("doi:" + f["doi"] + ".")
    return " ".join(bits)


def input_order(main_path: str) -> list:
    files = []
    for line in io.open(main_path, encoding="utf-8").read().split("\n"):
        if line.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", line):
            files.append(m.group(1).strip())
    return files


def first_citation_order(main_path: str) -> list:
    order, seen = [], set()
    for rel in input_order(main_path):
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        if not path.endswith(".tex"):
            path += ".tex"
        if not os.path.exists(path):
            continue
        text = re.sub(r"(?m)(?<!\\)%.*$", "",
                      io.open(path, encoding="utf-8", errors="replace").read())
        for m in re.finditer(r"\\cite\{([^}]*)\}", text):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key and key not in seen:
                    seen.add(key)
                    order.append(key)
    return order


def main() -> int:
    check = "--check" in sys.argv
    entries = parse_bib(BIB)
    order = first_citation_order(MAIN)

    missing = [k for k in order if k not in entries]
    if missing:
        print("ERROR: cited but absent from pustaka.bib: %s" % missing)
        return 1

    print("cited references: %d (numbered by order of first citation)" % len(order))
    for i, k in enumerate(order, 1):
        print("  [%2d] %s" % (i, k))

    lines = [
        "%" + "-" * 77 + "%",
        "% DAFTAR PUSTAKA",
        "%",
        "% Dibangkitkan oleh scripts/gen_daftarpustaka.py dari pustaka.bib.",
        "% Ditulis sebagai thebibliography biasa, bukan melalui BibTeX, agar daftar",
        "% pustaka dan nomor sitasi tidak bergantung pada lintasan BibTeX maupun pada",
        "% nama berkas basis data. pustaka.bib tetap disimpan sebagai catatan.",
        "%",
        "% Urutan nomor mengikuti gaya IEEE: urut kemunculan sitasi pertama.",
        "% JANGAN diedit tangan - ubah pustaka.bib lalu bangkitkan ulang.",
        "%" + "-" * 77 + "%",
        "",
        BS + "begin{thebibliography}{" + str(len(order)) + "}",
        BS + "small",
        "",
    ]
    for key in order:
        typ, f = entries[key]
        lines += ["\\bibitem{%s}" % key, fmt_entry(typ, f), ""]
    lines += [BS + "end{thebibliography}", ""]

    text = "\n".join(lines)
    if check:
        print("\n" + text[:900])
        return 0

    io.open(OUT, "w", encoding="utf-8", newline="").write(text)
    print("\nwrote %s (%d bibitem)" % (OUT, len(order)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
