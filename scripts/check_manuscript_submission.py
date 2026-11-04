"""Pre-submission check on the built journal .docx files.

Checks the things a desk editor or a reviewer sees first, against JMIR's own
instructions and the accepted paper supplied as the model. Reports pass/fail per
item rather than an overall verdict, so what is still open stays visible.
"""
import io
import os
import re
import sys
import zipfile

import docx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# The folder holding the built .docx files. Overridable, so the check is not
# pinned to one machine's layout: pass it as the first argument or set
# JOURNAL_DIR.
J = (sys.argv[1] if len(sys.argv) > 1
     else os.environ.get("JOURNAL_DIR", r"D:\1.AmadeusKuliah\Skripsi\Journal-JMIR"))
FILES = [("EN", os.path.join(J, "JurnalJMIR-BenedictAmadeusSandro-EN.docx")),
         ("ID", os.path.join(J, "JurnalJMIR-BenedictAmadeusSandro-ID.docx"))]

REF_WORDS = 13255          # the accepted JMIR paper supplied as the model
ABSTRACT_MAX = 450


def check(lab, path):
    out, bad = [], 0
    d = docx.Document(path)
    txt = "\n".join(p.text for p in d.paragraphs)
    words = sum(len(p.text.split()) for p in d.paragraphs) + sum(
        len(c.text.split()) for t in d.tables for r in t.rows for c in r.cells)

    def ok(cond, label, detail=""):
        nonlocal bad
        if not cond:
            bad += 1
        out.append("  %s %-44s %s" % ("PASS" if cond else "FAIL", label, detail))

    # --- metadata -----------------------------------------------------------
    cp = d.core_properties
    ok(bool(cp.author and cp.author.strip()), "author set in document metadata",
       cp.author or "(empty)")
    tool = (cp.last_modified_by or "").lower()
    ok("python" not in tool and "claude" not in tool and "ai" != tool,
       "no tool attribution in metadata", cp.last_modified_by or "(empty)")
    ok(bool(cp.title and "word document" not in (cp.title or "").lower()),
       "document title set", (cp.title or "")[:46])

    # --- no AI mention anywhere ---------------------------------------------
    hits = re.findall(r"\b(claude|anthropic|chatgpt|openai|copilot)\b", txt, re.I)
    ok(not hits, "no AI mention in body text", ", ".join(set(hits)) or "none")

    # --- abstract ------------------------------------------------------------
    keys = ("Background|Latar", "Objective|Tujuan", "Methods|Metode",
            "Results|Hasil", "Conclusions|Simpulan|Kesimpulan")
    aw = 0
    for p in d.paragraphs:
        if re.match(r"^(%s):" % "|".join(k for k in keys), p.text.strip()):
            aw += len(p.text.split())
    ok(0 < aw <= ABSTRACT_MAX, "structured abstract <= %d words" % ABSTRACT_MAX,
       "%d words" % aw)

    kw = [p.text for p in d.paragraphs
          if re.match(r"^(Keywords|Kata\s+Kunci)\s*:", p.text.strip(), re.I)]
    nk = len(re.split(r"[;,]", kw[0].split(":", 1)[1])) if kw else 0
    ok(3 <= nk <= 10, "3-10 keywords", "%d" % nk)

    # --- references ----------------------------------------------------------
    refs = re.findall(r"(?m)^\s*(\d+)\.\s+\w", txt)
    nref = len(set(refs))
    ok(nref >= 25, "reference list present", "%d entries" % nref)
    dois = len(re.findall(r"doi:", txt, re.I))
    ok(dois >= nref * 0.9, "references carry DOIs", "%d doi: markers" % dois)
    jmir = len(re.findall(r"JMIR|J Med Internet Res", txt))
    ok(jmir >= 5, "at least 5 JMIR-family citations", "%d mentions" % jmir)
    ok("[?]" not in txt, "no unresolved citation markers",
       "%d found" % txt.count("[?]"))

    # --- figures and tables ---------------------------------------------------
    figword = "Figure" if lab == "EN" else "Gambar"
    tabword = "Table" if lab == "EN" else "Tabel"
    fnums = sorted({int(m) for m in re.findall(figword + r" (\d+)\.", txt)})
    tnums = sorted({int(m) for m in re.findall(tabword + r" (\d+)\.", txt)})
    ok(fnums == list(range(1, len(fnums) + 1)), "figures numbered consecutively",
       str(fnums))
    ok(tnums == list(range(1, len(tnums) + 1)), "tables numbered consecutively",
       str(tnums))
    ok(len(d.inline_shapes) == len(fnums), "every figure number has an image",
       "%d images / %d numbers" % (len(d.inline_shapes), len(fnums)))
    ok(len(d.tables) == len(tnums), "every table number has a table",
       "%d tables / %d numbers" % (len(d.tables), len(tnums)))

    # --- formatting defects ----------------------------------------------------
    # A DOI suffix such as 10.1016/j.compbiomed.2022.105817 contains a long
    # decimal that is not float noise, so match with the preceding context.
    noise = [m.group(1) for m in re.finditer(r"(?<!doi:)(?<![\w./])(\d+\.\d{6,})", txt)
             if "doi" not in txt[max(0, m.start() - 40):m.start()].lower()]
    ok(not noise, "no float32 noise in numbers", ", ".join(noise[:3]) or "none")

    # --- size -----------------------------------------------------------------
    mb = os.path.getsize(path) / 1e6
    ok(mb < 20, "file size reasonable for submission", "%.1f MB" % mb)
    ok(words < REF_WORDS, "shorter than the accepted model paper",
       "%d vs %d words" % (words, REF_WORDS))

    with zipfile.ZipFile(path) as z:
        media = sum(i.file_size for i in z.infolist()
                    if i.filename.startswith("word/media/"))
    out.append("  ---- %s: %d words, %.1f MB (%.1f MB images), %d figures, %d tables"
               % (lab, words, mb, media / 1e6, len(d.inline_shapes), len(d.tables)))
    return out, bad


def main():
    total = 0
    for lab, path in FILES:
        print("=" * 74)
        print(lab, os.path.basename(path))
        print("=" * 74)
        lines, bad = check(lab, path)
        print("\n".join(lines))
        total += bad
        print()
    print("=" * 74)
    print("FAILURES: %d" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
