"""Content-level audit of the journal: consistency, overreach, and omissions.

Format is already clean, so this looks at what a reviewer reads rather than what
a desk editor checks: whether the abstract matches the results, whether every
headline number is stated the same way everywhere, whether any claim is stronger
than the statistics support, and whether the limitations cover what the results
actually revealed.
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
DOC = os.path.join(J, "JurnalJMIR-BenedictAmadeusSandro-EN.docx")

HEADLINE = {
    "0.8442": "internal CG Dice",
    "0.6852": "internal PZ Dice",
    "0.7647": "internal macro Dice",
    "0.8137": "external CG Dice",
    "0.6247": "external PZ Dice",
    "0.7192": "external macro Dice",
    "0.5960": "ablation, no reorientation CG",
    "182.8": "ablation, identity affine centroid shift",
}

OVERREACH = [
    (r"significantly (better|higher|improv)", "claims significance"),
    (r"\bproves?\b", "'proves'"),
    (r"\bdemonstrates? that .{0,30}\bsuperior", "claims superiority"),
    (r"state[- ]of[- ]the[- ]art performance", "SOTA claim"),
    (r"\bclinically (ready|validated|proven)\b", "clinical readiness claim"),
    (r"\boutperform", "'outperform'"),
]

MUST_APPEAR_IN_LIMITATIONS = {
    "gland size": r"gland size|smallest|size tertile",
    "test-set reuse": r"not pristine|used twice|earlier use",
    "no reader study": r"reader study",
    "single dataset": r"single public dataset|one public dataset",
    "in-plane resampling": r"in-plane resampling",
}


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


def main():
    d = docx.Document(DOC)
    paras = [p.text for p in d.paragraphs]
    txt = "\n".join(paras)
    tbl = "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    whole = txt + "\n" + tbl

    abstract = section(paras, r"^Abstract$", r"^Introduction$")
    limits = section(paras, r"^Limitations$", r"^Conclusions$")
    concl = section(paras, r"^Conclusions$", r"^Acknowledg")

    issues = []

    print("=" * 74)
    print("1. HEADLINE NUMBERS - stated consistently?")
    print("=" * 74)
    for num, what in HEADLINE.items():
        n_body = whole.count(num)
        in_abs = num in abstract
        mark = "" if n_body else "   <-- MISSING ENTIRELY"
        if not n_body:
            issues.append("%s (%s) appears nowhere" % (num, what))
        print("  %-8s %-38s body x%-2d abstract:%s%s"
              % (num, what, n_body, "yes" if in_abs else "no ", mark))

    print()
    print("=" * 74)
    print("2. OVERREACHING CLAIMS")
    print("=" * 74)
    found = False
    for pat, label in OVERREACH:
        for m in re.finditer(pat, txt, re.I):
            ctx = txt[max(0, m.start() - 70):m.start() + 60].replace("\n", " ")
            print("  %-26s ...%s..." % (label, ctx))
            issues.append("possible overreach: %s" % label)
            found = True
    if not found:
        print("  none found")

    print()
    print("=" * 74)
    print("3. LIMITATIONS COVERAGE")
    print("=" * 74)
    for label, pat in MUST_APPEAR_IN_LIMITATIONS.items():
        hit = bool(re.search(pat, limits, re.I))
        if not hit:
            issues.append("limitations do not mention: %s" % label)
        print("  %-22s %s" % (label, "covered" if hit else "NOT COVERED"))

    print()
    print("=" * 74)
    print("4. ABSTRACT vs RESULTS")
    print("=" * 74)
    checks = [
        ("gland-size finding", r"gland size|size tertile|smallest third"),
        ("the negative result", r"less than|smaller than .{0,24}variation|did not reach"),
        ("external validation", r"external|PROSTATEx|204"),
        ("reconstruction ablation", r"mirror|ablation|geometric"),
        ("clinical translation", r"PSA density"),
    ]
    for label, pat in checks:
        hit = bool(re.search(pat, abstract, re.I))
        if not hit:
            issues.append("abstract omits: %s" % label)
        print("  %-24s %s" % (label, "present" if hit else "ABSENT"))

    print()
    print("=" * 74)
    print("5. CONCLUSIONS vs the p = .055 result")
    print("=" * 74)
    hedged = bool(re.search(r"did not reach|not statistically|consistent in direction"
                            r"|direction.{0,20}consistent", concl + limits, re.I))
    print("  conclusions/limits hedge the pre-specified test:",
          "yes" if hedged else "NO - check this")
    if not hedged:
        issues.append("the p=.055 result may be overstated in Conclusions")

    print()
    print("=" * 74)
    print("ISSUES: %d" % len(issues))
    for i in issues:
        print("  -", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
