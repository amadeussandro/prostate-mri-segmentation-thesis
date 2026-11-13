"""Structured audit: how 2D prostate MRI segmentation studies report the 2D-to-3D step.

The protocol is results/literature_audit/protocol.md (written before the search). Steps:

    python scripts/literature_audit.py search      # PMC query, full texts -> fulltext/
    python scripts/literature_audit.py screen      # title, type and 2D/3D cues per article
    python scripts/literature_audit.py snippets --ids PMC1,PMC2   # evidence windows for coding
    python scripts/literature_audit.py summarize   # extraction.csv -> summary.json, report.md
    python scripts/literature_audit.py sample      # draws the second-reviewer sample (once)
    python scripts/literature_audit.py agreement --reviewer-file <completed form>

Screening decisions and extracted items are written by hand to screening.csv and extraction.csv,
each `yes` with a quotation from the article. Only `search` touches the network.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "literature_audit")
FULL = os.path.join(OUT, "fulltext")
PROTOCOL = os.path.join(OUT, "protocol.md")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERY = ('prostate[ti] AND segmentation[ti] AND (MRI[tiab] OR "magnetic resonance"[tiab]) AND '
         '("deep learning"[tiab] OR convolutional[tiab] OR "U-Net"[tiab] OR "neural network"[tiab]) '
         'AND 2017:2026[pdat] AND "open access"[filter]')

ITEMS = ["dimensionality", "target", "metric_level", "metric_space", "reassembly_described",
         "inverse_preprocessing_described", "geometry_check_reported", "orientation_mentioned",
         "surface_metric_mm", "code_available"]

CUES = {
    "dim2d": r"\b2[.-]?5\s?-?D\b|\b2-?D\b|two[- ]dimensional|slice[- ]wise|slice[- ]by[- ]slice|per[- ]slice",
    "dim3d": r"\b3-?D\b|three[- ]dimensional|volumetric",
    "reassembly": r"stack(?:ed|ing)?\b|re-?assembl|re-?construct\w*\s+(?:the\s+)?(?:3D|volume)|"
                  r"concatenat\w+\s+(?:the\s+)?(?:slice|prediction)|back\s+into\s+(?:a\s+|the\s+)?(?:3D|volume)",
    "inverse": r"original\s+(?:image\s+)?(?:resolution|space|size|spacing|dimension|geometry|grid)|"
               r"resampl\w*\s+back|back\s+to\s+(?:the\s+)?original|native\s+(?:resolution|space)|invers\w+",
    "geometry": r"\baffine\b|\bheader\b|re-?orient\w*|\borientation\b|\bRAS\b|\bLPS\b|\bflip\w*|"
                r"\bmisalign\w*|spatial(?:ly)?\s+(?:correct|consisten|align)\w*",
    "space": r"(?:computed|calculated|evaluated|measured)\s+(?:on|in|at|per)\s+[^.]{0,60}|"
             r"per[- ](?:patient|case|volume|subject)|volume[- ](?:level|wise|based)|slice[- ]level|3D\s+Dice",
    "surface": r"Hausdorff|HD\s?95|surface\s+distance|\bASD\b|\bASSD\b",
    "code": r"github\.com|gitlab|code\s+(?:is|are|will\s+be)\s+(?:publicly\s+)?available|zenodo",
}


def _sha_lf(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _get(url: str, params: dict) -> bytes:
    params = dict(params, tool="literature-audit")
    q = urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - NCBI rate limits; retry with back-off
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed: {url}")


def search() -> None:
    os.makedirs(FULL, exist_ok=True)
    res = json.loads(_get(EUTILS + "esearch.fcgi", {"db": "pmc", "term": QUERY, "retmode": "json",
                                                     "retmax": 500}))["esearchresult"]
    ids = res["idlist"]
    for i in range(0, len(ids), 10):
        batch = ids[i:i + 10]
        xml = _get(EUTILS + "efetch.fcgi", {"db": "pmc", "id": ",".join(batch), "retmode": "xml"})
        root = ET.fromstring(xml)
        for art in root.findall("article"):
            pmc = next((e.text for e in art.iter("article-id") if e.get("pub-id-type") in ("pmc", "pmcid")), None)
            pmc = pmc if pmc and pmc.startswith("PMC") else f"PMC{pmc}"
            open(os.path.join(FULL, f"{pmc}.xml"), "wb").write(ET.tostring(art))
        time.sleep(0.5)
    rec = {"query": QUERY, "database": "pmc", "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "count": int(res["count"]), "ids": ids, "n_fulltext": len(os.listdir(FULL)),
           "protocol_sha256_lf": _sha_lf(PROTOCOL)}
    json.dump(rec, open(os.path.join(OUT, "search.json"), "w"), indent=1)
    print(f"{rec['count']} records, {rec['n_fulltext']} full texts")


def _text(e) -> str:
    return re.sub(r"\s+", " ", "".join(e.itertext())).strip() if e is not None else ""


def load(pmc: str) -> dict:
    art = ET.parse(os.path.join(FULL, f"{pmc}.xml")).getroot()
    meta = art.find(".//article-meta")
    ids = {e.get("pub-id-type"): e.text for e in meta.findall("article-id")}
    year = next((y.text for y in meta.iter("year")), "")
    body = art.find("body")
    secs = []
    if body is not None:
        for s in body.iter("sec"):
            t = _text(s.find("title"))
            own = " ".join(_text(p) for p in s.findall("p"))
            if own:
                secs.append((t, own))
    return {"pmc": pmc, "pmid": ids.get("pmid", ""), "doi": ids.get("doi", ""), "year": year,
            "type": art.get("article-type", ""), "title": _text(meta.find(".//article-title")),
            "journal": _text(art.find(".//journal-title")), "abstract": _text(meta.find("abstract")),
            "secs": secs, "body_chars": sum(len(x[1]) for x in secs)}


def all_ids() -> list:
    return sorted(f[:-4] for f in os.listdir(FULL) if f.endswith(".xml"))


def screen() -> None:
    for pmc in all_ids():
        a = load(pmc)
        body = " ".join(x[1] for x in a["secs"])
        n2 = len(re.findall(CUES["dim2d"], body, re.I))
        n3 = len(re.findall(CUES["dim3d"], body, re.I))
        print(f"{pmc} | {a['year']} | {a['type']} | 2D-cues {n2} | 3D-cues {n3} | chars {a['body_chars']} | {a['title'][:120]}")


def snippets(ids: list, cats: list, width: int, per: int, abstract: int = 0) -> None:
    for pmc in ids:
        a = load(pmc)
        print(f"\n##### {pmc} ({a['year']}) {a['title'][:110]}")
        if abstract:
            print("  ABSTRACT: " + a["abstract"][:abstract])
        for cat in cats:
            hits = []
            for title, txt in a["secs"]:
                for m in re.finditer(CUES[cat], txt, re.I):
                    s = txt[max(0, m.start() - width): m.end() + width]
                    hits.append(f"[{title[:25]}] ...{s}...")
            seen, out = set(), []
            for h in hits:
                k = h[-60:]
                if k not in seen:
                    seen.add(k)
                    out.append(h)
            print(f"  <{cat}> {len(hits)} hits")
            for h in out[:per]:
                print("    " + h)


def _cp(k: int, n: int):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def summarize() -> None:
    scr = list(csv.DictReader(open(os.path.join(OUT, "screening.csv"), encoding="utf-8")))
    ext = list(csv.DictReader(open(os.path.join(OUT, "extraction.csv"), encoding="utf-8")))
    inc = {r["pmc"] for r in scr if r["decision"] == "include"}
    if inc != {r["pmc"] for r in ext}:
        raise RuntimeError("extraction.csv must hold exactly the included studies")
    srch = json.load(open(os.path.join(OUT, "search.json")))
    if srch["protocol_sha256_lf"] != _sha_lf(PROTOCOL):
        raise RuntimeError("protocol.md changed after the search")
    n = len(ext)
    counts = {it: {} for it in ITEMS}
    for r in ext:
        for it in ITEMS:
            counts[it][r[it]] = counts[it].get(r[it], 0) + 1
    k = counts["geometry_check_reported"].get("yes", 0)
    lo, hi = _cp(k, n)
    reasons = {}
    for r in scr:
        if r["decision"] != "include":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    s = {"n_records": srch["count"], "n_fulltext": srch["n_fulltext"], "n_screened": len(scr),
         "n_included": n, "exclusion_reasons": reasons, "counts": counts,
         "geometry_check": {"k": k, "n": n, "ci95": [lo, hi]},
         "search_run_utc": srch["run_utc"], "protocol_sha256_lf": srch["protocol_sha256_lf"]}
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    lines = ["# Literature audit: reporting of the 2D-to-3D step", "",
             f"Search run {srch['run_utc']}; {srch['count']} PMC open-access records; "
             f"{len(scr)} screened on full text; {n} included.", "",
             "Exclusions: " + "; ".join(f"{v} {k_}" for k_, v in sorted(reasons.items(), key=lambda x: -x[1])), "",
             "| Item | Counts (of included studies) |", "|---|---|"]
    for it in ITEMS:
        lines.append(f"| {it} | " + ", ".join(f"{v_} {k_}" for k_, v_ in sorted(counts[it].items(), key=lambda x: -x[1])) + " |")
    changed = n - counts["inverse_preprocessing_described"].get("n/a", 0)
    lines += ["", f"Geometry check reported: {k}/{n} (exact 95% CI {lo*100:.1f}%-{hi*100:.1f}%).",
              f"Inverse preprocessing described: {counts['inverse_preprocessing_described'].get('yes', 0)}/{changed} "
              "studies that reported resampling, resizing or cropping (the rest reported none).", "",
              "## Included studies", "",
              "| PMCID | First author | Year | Input | Target | Reassembly | Inverse | Geometry check | Metric space |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ext, key=lambda r: (r["year"], r["pmc"])):
        lines.append(f"| {r['pmc']} | {r['first_author']} | {r['year']} | {r['dimensionality']} | {r['target']} | "
                     f"{r['reassembly_described']} | {r['inverse_preprocessing_described']} | "
                     f"{r['geometry_check_reported']} | {r['metric_space']} |")
    lines += ["", "## Limitations (stated in the protocol before the search)", "",
              "One database (PMC open-access subset), title-level search terms, one reviewer. The audit describes "
              "reporting, not practice: a study may have verified its reconstruction without saying so. Uncertain "
              "codings are flagged in extraction.csv and were coded `no` unless the text stated the item.", "",
              "Per-study coding with supporting quotations: extraction.csv. Screening with reasons: screening.csv. "
              f"Protocol SHA-256 (LF): {srch['protocol_sha256_lf']}."]
    s["inverse"] = {"k": counts["inverse_preprocessing_described"].get("yes", 0), "n": changed}
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------- second reviewer
SR = os.path.join(OUT, "second_review")
SEED = 20261112
REASONS = ["all models 3D", "lesion-only", "target not prostate", "not MRI", "review or non-original",
           "not a segmentation model", "input dimensionality not reported"]
ALLOWED = {
    "decision": ["include", "exclude"],
    "dimensionality": ["2D", "2.5D", "2D+3D", "2.5D+3D"],
    "target": ["gland", "zones", "both"],
    "metric_level": ["volume", "slice", "both", "not stated"],
    "metric_space": ["original voxel space", "resampled or preprocessed grid", "not stated"],
    "reassembly_described": ["yes", "no"],
    "inverse_preprocessing_described": ["yes", "no", "n/a"],
    "geometry_check_reported": ["yes", "no"],
    "orientation_mentioned": ["yes", "no"],
    "surface_metric_mm": ["yes", "no"],
    "code_available": ["yes", "no"],
}
KEY = ["reassembly_described", "inverse_preprocessing_described", "geometry_check_reported"]
NOTES = [
    "Eligibility is judged on the model, not on the acquisition: a 2D MRI sequence with a 3D network is 3D.",
    "If an article does not say whether its network takes 2D slices or 3D volumes (for example, nnU-Net "
    "with no configuration named), exclude it as 'input dimensionality not reported'.",
    "A network that applies 2D operations to a stack of slices with attention across slices counts as 2.5D.",
    "A radiotherapy target described as the prostate (clinical target volume of the prostate) counts as gland.",
    "inverse_preprocessing_described is n/a only if the article reports no resampling, resizing or cropping "
    "of the images used at inference; resizing used only as training augmentation, or crops used only to "
    "form training patches, do not count.",
    "geometry_check_reported: visual inspection of segmentation quality does not count; only a check stated "
    "to concern the reconstructed volume's position, orientation, header or alignment counts.",
    "orientation_mentioned: any statement about image orientation, including the acquisition plane named "
    "as an orientation, counts; 'orientation' used for shape priors or augmentation does not.",
    "surface_metric_mm: the unit must be stated in the text or a table header; a distance with no unit is no.",
    "code_available: a public link to the study's own code; links only to third-party libraries, or code "
    "'available on request', are no.",
    "metric_level and metric_space: code what the text states; if it does not state it, code 'not stated'.",
]


def sample() -> None:
    import random
    spec = os.path.join(SR, "sample_spec.md")
    if os.path.exists(os.path.join(SR, "sample.json")):
        raise RuntimeError("the sample has already been drawn; it is drawn once")
    scr = list(csv.DictReader(open(os.path.join(OUT, "screening.csv"), encoding="utf-8")))
    inc = sorted(r["pmc"] for r in scr if r["decision"] == "include")
    exc = sorted(r["pmc"] for r in scr if r["decision"] != "include")
    rng = random.Random(SEED)
    s_inc, s_exc = rng.sample(inc, 8), rng.sample(exc, 10)
    order = s_inc + s_exc
    rng.shuffle(order)
    json.dump({"spec_sha256_lf": _sha_lf(spec), "seed": SEED, "included_sample": s_inc,
               "excluded_sample": s_exc, "order": order,
               "drawn_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
              open(os.path.join(SR, "sample.json"), "w"), indent=1)
    title = {r["pmc"]: r["title"] for r in scr}
    cols = ["order", "pmc", "link", "title", "decision", "reason"] + ITEMS + ["evidence", "notes"]
    with open(os.path.join(SR, "coding_form_reviewer2.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, pmc in enumerate(order, 1):
            w.writerow([i, pmc, f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/", title[pmc]]
                       + [""] * (len(cols) - 4))
    _instructions(order, title)
    print(f"sample drawn: {len(order)} records; form and instructions in {SR}")


def _instructions(order: list, title: dict) -> None:
    from docx import Document
    d = Document()
    d.core_properties.author = "Benedict Amadeus Sandro"
    d.add_heading("Second-reviewer coding: instructions", level=1)
    for p in [
        "Thank you for checking this audit. Please work alone and do not look at the first reviewer's "
        "files (screening.csv, extraction.csv, report.md) or at the manuscript before you finish.",
        "For each of the 18 articles below, open the PubMed Central link and read the full text "
        "(methods, results, tables, figure captions, and data or code availability statements).",
        "Fill one row per article in coding_form_reviewer2.csv (Excel or Google Sheets). First decide "
        "eligibility (decision, and reason if excluded). If you include the article, code all 10 items, "
        "using only the allowed values below, and paste a short quotation (a few words) into 'evidence' "
        "for every item you code yes. Use 'notes' for anything you were unsure about.",
        "When in doubt, code what the text states: an item the article does not state is 'no' or "
        "'not stated'. The full protocol follows the tables.",
    ]:
        d.add_paragraph(p)
    d.add_heading("Articles", level=2)
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, ["Order", "PMCID and link", "Title"]):
        c.text = h
    for i, pmc in enumerate(order, 1):
        cells = t.add_row().cells
        cells[0].text, cells[1].text = str(i), f"{pmc}\nhttps://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"
        cells[2].text = title[pmc]
    d.add_heading("Allowed values", level=2)
    t = d.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Column", "Values"
    for k, v in [("decision", ALLOWED["decision"]), ("reason (if excluded)", REASONS + ["other: <text>"])] + \
            [(k, ALLOWED[k]) for k in ITEMS]:
        cells = t.add_row().cells
        cells[0].text, cells[1].text = k, " | ".join(v)
    d.add_heading("Coding notes", level=2)
    for n in NOTES:
        d.add_paragraph(n, style="List Bullet")
    d.add_heading("Protocol (verbatim)", level=2)
    for line in open(PROTOCOL, encoding="utf-8").read().splitlines():
        if line.startswith("#"):
            d.add_heading(line.lstrip("# ").strip(), level=3)
        elif line.strip() and not set(line.strip()) <= set("|-: "):
            d.add_paragraph(line.strip().replace("`", ""))
    d.save(os.path.join(SR, "Instructions_reviewer2.docx"))


def _kappa(a: list, b: list):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if len(cats) < 2 or pe >= 1 else (po - pe) / (1 - pe)


def agreement(path: str) -> None:
    smp = json.load(open(os.path.join(SR, "sample.json")))
    if smp["spec_sha256_lf"] != _sha_lf(os.path.join(SR, "sample_spec.md")):
        raise RuntimeError("sample_spec.md changed after the sample was drawn")
    raw = open(path, encoding="utf-8-sig").read()
    dialect = csv.Sniffer().sniff(raw.splitlines()[0], delimiters=",;\t")
    r2 = {r["pmc"].strip(): {k: (v or "").strip() for k, v in r.items()}
          for r in csv.DictReader(raw.splitlines(), dialect=dialect)}
    if set(r2) != set(smp["order"]):
        raise RuntimeError(f"reviewer file must hold exactly the 18 sampled records; differs by {set(r2) ^ set(smp['order'])}")
    for pmc, r in r2.items():
        bad = [k for k in ["decision"] + (ITEMS if r["decision"] == "include" else [])
               if r[k] not in ALLOWED[k]]
        if bad:
            raise RuntimeError(f"{pmc}: values outside the allowed list in {bad}")
    scr = {r["pmc"]: r for r in csv.DictReader(open(os.path.join(OUT, "screening.csv"), encoding="utf-8"))}
    ext = {r["pmc"]: r for r in csv.DictReader(open(os.path.join(OUT, "extraction.csv"), encoding="utf-8"))}
    ids = smp["order"]
    d1, d2 = [scr[p]["decision"] for p in ids], [r2[p]["decision"] for p in ids]
    dis = [{"pmc": p, "field": "decision", "reviewer1": x, "reviewer2": y,
            "reason1": scr[p]["reason"], "reason2": r2[p]["reason"]}
           for p, x, y in zip(ids, d1, d2) if x != y]
    both = [p for p in ids if scr[p]["decision"] == "include" and r2[p]["decision"] == "include"]
    items = {}
    for it in ITEMS:
        a, b = [ext[p][it] for p in both], [r2[p][it] for p in both]
        if not both:
            continue
        items[it] = {"n": len(both), "agree": sum(x == y for x, y in zip(a, b)),
                     "pct": 100 * sum(x == y for x, y in zip(a, b)) / len(both), "kappa": _kappa(a, b)}
        dis += [{"pmc": p, "field": it, "reviewer1": x, "reviewer2": y}
                for p, x, y in zip(both, a, b) if x != y]
    res = {"n_records": len(ids), "eligibility": {"agree": len(ids) - sum(1 for x in dis if x["field"] == "decision"),
                                                 "pct": 100 * sum(x == y for x, y in zip(d1, d2)) / len(ids),
                                                 "kappa": _kappa(d1, d2)},
           "n_both_included": len(both), "items": items,
           "key_items_agree": all(items[k]["agree"] == items[k]["n"] for k in KEY if k in items),
           "disagreements": dis, "spec_sha256_lf": smp["spec_sha256_lf"], "reviewer_file": os.path.basename(path)}
    json.dump(res, open(os.path.join(SR, "agreement.json"), "w"), indent=1)
    fk = lambda k: "undefined" if k is None else f"{k:.2f}"  # noqa: E731
    lines = ["# Second-reviewer agreement", "",
             f"Eligibility: {res['eligibility']['agree']}/{len(ids)} records agree "
             f"({res['eligibility']['pct']:.0f}%; kappa {fk(res['eligibility']['kappa'])}).",
             f"Items, over the {len(both)} studies both reviewers included:", "",
             "| Item | Agree | % | Kappa |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['agree']}/{v['n']} | {v['pct']:.0f} | {fk(v['kappa'])} |" for k, v in items.items()]
    lines += ["", "## Disagreements (to be resolved against the article text)", ""]
    lines += [f"- {x['pmc']} {x['field']}: reviewer 1 `{x['reviewer1']}`, reviewer 2 `{x['reviewer2']}`" for x in dis] or ["- none"]
    open(os.path.join(SR, "agreement.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["search", "screen", "snippets", "summarize", "sample", "agreement"])
    ap.add_argument("--reviewer-file", default=os.path.join(SR, "coding_form_reviewer2_completed.csv"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--cats", default="dim2d,reassembly,inverse,geometry,space")
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--abstract", type=int, default=0, help="print this many abstract characters")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.step == "search":
        search()
    elif a.step == "screen":
        screen()
    elif a.step == "sample":
        sample()
    elif a.step == "agreement":
        agreement(a.reviewer_file)
    elif a.step == "snippets":
        ids = all_ids() if a.ids == "all" else [x.strip() for x in a.ids.split(",") if x.strip()]
        snippets(ids, a.cats.split(","), a.width, a.per, a.abstract)
    else:
        summarize()


if __name__ == "__main__":
    main()
