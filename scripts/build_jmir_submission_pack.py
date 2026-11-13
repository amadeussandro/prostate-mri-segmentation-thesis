"""Build the JMIR submission package next to the revised manuscript.

Writes to <journal-dir>/JMIR_submission_<tag>/:
  Multimedia_Appendix_1_CLAIM2024_mapping.docx  where each CLAIM topic is addressed
  Multimedia_Appendix_2_prespecification.docx   both prespecifications, erratum, recorded hashes
  Multimedia_Appendix_3_supplementary.docx      supplementary results (old extended material,
                                                plus the single-seed vs final-model section)
  Multimedia_Appendix_4_literature_audit.docx   audit of how published 2D studies report the
                                                2D-to-3D step: protocol, search, screening, coding
  Cover_letter.docx                            draft, with [[TO FILL]] markers
  figures/Figure_N.png                          <=1200x1200 px, <=5 MB, as JMIR asks

Usage:
    python scripts/build_jmir_submission_pack.py --journal-dir "D:/.../Journal-JMIR" --tag 071126
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from docx import Document
from docx.shared import Pt
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))


def _doc(title: str) -> Document:
    d = Document()
    d.core_properties.author = "Benedict Amadeus Sandro"
    d.core_properties.title = title
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    d.add_heading(title, level=1)
    return d


def _table(d, header, rows):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, header):
        c.text = h
    for r in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, r):
            c.text = str(v)
    return t


def appendix_claim(out: str, picai: bool = False) -> None:
    d = _doc("Multimedia Appendix 1. Where the CLAIM 2024 topics are addressed")
    d.add_paragraph(
        "This table maps the topics of the Checklist for Artificial Intelligence in Medical "
        "Imaging (CLAIM) 2024 to the sections of the manuscript. Item numbers and line numbers "
        "must be transferred to the official CLAIM 2024 form (Radiology: Artificial Intelligence, "
        "doi:10.1148/ryai.240300, Appendix) once the final page layout is fixed. "
        "[[TO FILL: transfer to the official form, with page and line numbers]]")
    rows = [
        ("Title and abstract", "Identification as an AI study; structured summary",
         "Title; Abstract (Background, Objective, Methods, Results, Conclusions)", "Yes"),
        ("Introduction", "Scientific and clinical background; objectives and hypotheses",
         "Introduction, paragraphs 1-4 (2 aims stated)", "Yes"),
        ("Methods: study design", "Prospective or retrospective; study goal",
         "Study Design and Reporting", "Yes"),
        ("Methods: data", "Data sources; eligibility; preprocessing; acquisition protocol; "
         "missing data; de-identification",
         "Data Sources; Table 1; Figure 1 (case counts); Preprocessing; exclusions with reasons (NCI-ISBI 2013)",
         "Yes (acquisition protocols only as reported by the source datasets)"),
        ("Methods: reference standard", "Definition; rationale; annotators; inter-reader "
         "variability", "Data Sources (reader 1 as reference standard; reader 2 as second "
         "reference); Evaluation and Inter-Reader Benchmark", "Yes"),
        ("Methods: data partitions", "Allocation to training, tuning (validation), and "
         "internal testing; level of partition", "Data Sources (official patient-level split "
         "119/20/19); Model and Training", "Yes"),
        ("Methods: model", "Architecture; software; initialization", "Model and Training; "
         "Figure 2; Statistical Analysis (software and versions)", "Yes"),
        ("Methods: training", "Training details; hyperparameters; number of models trained; "
         "model selection; ensembling",
         "Model and Training (optimizer, learning rate, batch, epochs, seed, sampler; selection "
         "on validation data only); Seed Replicates, 2.5D Input, and the Final Model (9 runs in "
         "all: 4 configurations, 2 further seeds, 3 seeds of the 2.5D variant; prespecified "
         "adoption rule; softmax ensemble of 3 seeds with left-right flip averaging; weight "
         "hashes); Table 3", "Yes"),
        ("Methods: evaluation", "Metrics; statistical measures of significance and "
         "uncertainty; robustness; explainability; external testing",
         "Figure 3; Evaluation and Inter-Reader Benchmark; Statistical Analysis (Wilcoxon, Holm, TOST, "
         "bootstrap CI); External Testing; Fault-Injection Experiment", "Yes (no explainability "
         "method: the output is a segmentation shown directly)"),
        ("Results: data", "Flow of cases; demographic and clinical characteristics",
         "Table 1; Data Sources (exclusions); overlap between cohorts", "Partly: patient "
         "demographics are not distributed with the public datasets"),
        ("Results: model performance", "Performance with uncertainty; comparison with "
         "reference; failure analysis", "Tables 2-6; Figures 4-7; inter-reader comparison; "
         "single-seed and final models side by side; gland-size analysis", "Yes"),
        ("Discussion", "Limitations; implications for practice", "Limitations; Clinical "
         "Implications", "Yes"),
        ("Other information", "Registration; protocol; data and code availability; funding",
         "Multimedia Appendix 2 (2 prespecifications with file hashes); Data Availability; "
         "Code Availability; Funding",
         "No trial registration (not applicable to a retrospective algorithm study)"),
    ]
    if picai:   # v10: the PI-CAI PSA density analysis
        add = {
            "Introduction": ("2 aims stated", "2 aims and a further prespecified analysis stated"),
            "Methods: data": ("exclusions with reasons (NCI-ISBI 2013)",
                              "exclusions with reasons (NCI-ISBI 2013; PI-CAI); PSA Density on PI-CAI"),
            "Methods: reference standard": (
                "Evaluation and Inter-Reader Benchmark",
                "Evaluation and Inter-Reader Benchmark; PSA Density on PI-CAI (csPCa label and "
                "how negative studies were confirmed)"),
            "Methods: evaluation": (
                "External Testing; Fault-Injection Experiment",
                "External Testing; Fault-Injection Experiment; PSA Density on PI-CAI (paired AUC "
                "difference, stratified bootstrap, noninferiority margin fixed in advance)"),
            "Results: data": ("overlap between cohorts",
                              "overlap between cohorts; PSA Density on PI-CAI (flow of patients, "
                              "centers, duplicates)"),
            "Results: model performance": ("Tables 2-6", "Tables 2-7"),
            "Other information": ("Multimedia Appendix 2 (2 prespecifications with file hashes)",
                                  "Multimedia Appendices 2 and 5 (3 prespecifications with file "
                                  "hashes)"),
        }
        new = []
        for r in rows:
            if r[0] in add:
                old, repl = add.pop(r[0])
                if old not in r[2]:
                    raise RuntimeError(f"CLAIM row {r[0]!r} changed; update the PI-CAI wording")
                r = (r[0], r[1], r[2].replace(old, repl), r[3])
            new.append(r)
        if add:
            raise RuntimeError(f"CLAIM rows not found: {list(add)}")
        rows = new
    _table(d, ["CLAIM 2024 section", "Topic", "Where addressed in the manuscript", "Reported"], rows)
    d.save(out)


def _verbatim(d, path: str, head: str) -> None:
    d.add_heading(head, level=2)
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("#"):
            d.add_heading(line.lstrip("# ").strip(), level=3)
        elif line.strip():
            d.add_paragraph(line.lstrip("- ").strip(),
                            style="List Bullet" if line.lstrip().startswith("- ") else None)


def _sha_lf(path: str) -> str:
    """SHA-256 of the file as committed (LF line endings), whatever the checkout wrote."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _repeat_outcome(dec: dict, decf: dict) -> str:
    """One sentence on which rules passed the internal check in each run."""
    r, rf = dec["rules"], decf["rules"]
    if not (r["R1"]["passed"] and r["R2"]["passed"] and rf["R1"]["passed"] and not rf["R2"]["passed"]):
        raise RuntimeError("the internal-check outcomes changed; rewrite the Appendix 2 sentence")
    up = rf["R2"]["pz"]["hd95"]["mean_diff"]
    return (f"In the confirmatory run both R1 and R2 passed the internal check. In the "
            f"repetition, R1 passed and R2 failed, because validation PZ HD95 increased by "
            f"{up:.1f} mm, so only R1 was applied to NCI-ISBI 2013 with the final model.")


def appendix_prespec(out: str) -> None:
    from manuscript_numbers_jmir import MODELS, SEED_PRESPEC, SELECTION
    d = _doc("Multimedia Appendix 2. Prespecifications: (A) the connected-component rule and "
             "the NCI-ISBI 2013 analysis; (B) seed replicates and 2.5D input")
    j_ = lambda rel: json.load(open(os.path.join(PROJECT, rel), encoding="utf-8"))

    # ---- A. connected-component rule
    pre = os.path.join(PROJECT, "results/lcc_prespecification.md")
    err = os.path.join(PROJECT, "results/lcc_prespecification_erratum.md")
    sha = hashlib.sha256(open(pre, "rb").read()).hexdigest()
    one, fin = MODELS["single"]["nci"], MODELS["final"]["nci"]
    dec, nci = j_(f"{one}/valid_prostate158/decision.json"), j_(f"{one}/nci/summary.json")
    decf, ncif = j_(f"{fin}/valid_prostate158/decision.json"), j_(f"{fin}/nci/summary.json")
    for x in (dec, nci, decf, ncif):
        if x["prespec_sha256"] != sha:
            raise RuntimeError("a run recorded a different hash of the rule's prespecification")
    d.add_heading("A. Connected-component rule and the NCI-ISBI 2013 analysis", level=2)
    d.add_paragraph(
        f"SHA-256 of the prespecification file: {sha}. The same hash is recorded in the "
        f"internal check output (decision.json: {dec['prespec_sha256']}) and in the NCI-ISBI "
        f"2013 output (summary.json: {nci['prespec_sha256']}, generated {nci['generated_utc']}) "
        f"of the confirmatory run, made with the single-seed model (checkpoint SHA-256: "
        f"{nci['checkpoint_sha256']}).")
    d.add_paragraph(
        f"The analysis was repeated without change with the final model, as part B required. "
        f"Those outputs record the same hash (decision.json: {decf['prespec_sha256']}; "
        f"summary.json: {ncif['prespec_sha256']}, generated {ncif['generated_utc']}). The "
        f"statement in the prespecification that there would be no re-selection of the "
        f"checkpoint and no test-time augmentation describes the confirmatory run; the final "
        f"model was defined later, by the prespecification in part B, and its NCI-ISBI 2013 "
        f"analysis is reported as a repetition, not as the confirmatory test. "
        + _repeat_outcome(dec, decf))
    _verbatim(d, pre, "A1. Prespecification (verbatim)")
    _verbatim(d, err, "A2. Erratum (verbatim)")

    # ---- B. seed replicates and 2.5D input
    sp = os.path.join(PROJECT, SEED_PRESPEC)
    sel = j_(SELECTION)
    sha_b = _sha_lf(sp)
    if sha_b != sel["prespecification_sha256"]:
        raise RuntimeError("seed prespecification no longer matches the hash in the selection record")
    d.add_heading("B. Seed replicates and 2.5D input", level=2)
    d.add_paragraph(
        f"SHA-256 of the prespecification file: {sha_b}. The same hash is recorded in the "
        f"selection record (selection.json: {sel['prespecification_sha256']}, generated "
        f"{sel['generated_utc']}), which also records that the test split was not read "
        f"(test_split_read: {str(sel['test_split_read']).lower()}). This prespecification was "
        f"written after the single-seed model had been evaluated on the held-out test set and "
        f"on both external cohorts.")
    _verbatim(d, sp, "B1. Prespecification (verbatim)")
    _verbatim(d, os.path.join(PROJECT, "results/e1_seed_selection/report.md"),
              "B2. Selection record (verbatim)")
    d.save(out)


def final_model_section() -> list:
    """Tables S4 to S6: the single-seed and final models side by side."""
    import csv
    from manuscript_numbers_jmir import MODELS, jmir_numbers
    N = jmir_numbers()
    one, fvs, sd = N["single"], N["final_vs_single"], N["seeds"]
    d3 = lambda x: f"{x:.3f}".replace("-", "\u2212")
    d1 = lambda x: f"{x:.1f}".replace("-", "\u2212")
    sg = lambda x, nd=3: f"{x:+.{nd}f}".replace("-", "\u2212")

    def fp(p):
        if p < 0.001:
            return "P<.001"
        if p < 0.01:
            return f"P={p:.3f}".replace("0.", ".")
        t = f"{p:.2f}"
        if t == "0.05":
            t = f"{p:.3f}"
        return "P=" + t.replace("0.", ".", 1)

    ms = lambda x, f=d3: f"{f(x['mean'])} ({f(x['sd'])})"
    B = [("h3", "Seed Replicates and the Final Model")]
    B.append(("p",
        "The final model is the softmax ensemble of 3 seeds of the selected configuration with "
        "left-right flip averaging (Table 3 of the main text). Table S4 sets it beside the "
        "single-seed model on every cohort, Table S5 gives its held-out results case by case, "
        "and Table S6 identifies its members. The paired comparisons in Table S4 were not "
        "prespecified."))
    rows = []
    cohorts = (("Prostate158 held-out test", N["t_s"], one["t_s"], fvs["test"], None),
               ("PROSTATEx", N["e_s"], one["e_s"], fvs["prostatex"], None),
               ("NCI-ISBI 2013", N["nci"]["primary"], one["nci"]["primary"], fvs["nci"], "nci"))
    for name, f, s_, dv, kind in cohorts:
        for zone, lab in (("cg", "CG"), ("pz", "PZ")):
            for met, mlab, fmt in (("dice", "Dice", d3), ("hd95", "HD95, mm", d1)):
                if kind == "nci":
                    a, b = f[f"{zone}_{met}_none"], s_[f"{zone}_{met}_none"]
                    x = dv[f"{zone}_{met}_none"]
                else:
                    key = "dice" if met == "dice" else "hd95_mm"
                    a, b, x = f[zone][key], s_[zone][key], dv[f"{zone}_{key}"]
                rows.append([name, f"{lab} {mlab}", str(x["n"]), ms(b, fmt), ms(a, fmt),
                             sg(x["mean_diff"], 3 if met == "dice" else 1),
                             f"{x['n_better']} of {x['n']}", fp(x["p"])])
    B.append(("tbl", ("Table S4. Single-seed model (seed 42) and final model on every cohort, "
                      "mean (SD) across cases. Difference: final minus single-seed, mean of the "
                      "paired differences; P: Wilcoxon signed-rank test. Cases improved: higher "
                      "Dice or lower HD95 with the final model.",
                      ["Cohort", "Metric", "Cases, n", "Single-seed model", "Final model",
                       "Difference", "Cases improved", "P"], rows)))
    per = {}
    with open(os.path.join(PROJECT, MODELS["final"]["test"], "metrics_per_case.csv"),
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per.setdefault(r["patient_id"], {})[int(r["class_id"])] = r
    rows = [[c, d3(float(per[c][1]["dice"])), d3(float(per[c][2]["dice"])),
             d3((float(per[c][1]["dice"]) + float(per[c][2]["dice"])) / 2),
             d1(float(per[c][1]["hd95_mm"])), d1(float(per[c][2]["hd95_mm"]))]
            for c in sorted(per)]
    if len(rows) != N["t_s"]["macro"]["n"]:
        raise RuntimeError("per-case table does not cover the held-out set")
    B.append(("tbl", (f"Table S5. Per-case results of the final model on the {len(rows)} held-out "
                      f"test cases (original voxel space, against reader 1).",
                      ["Case", "CG Dice", "PZ Dice", "Macro Dice", "CG HD95, mm", "PZ HD95, mm"],
                      rows)))
    rows = [[str(m["seed"]), str(m["epoch"]), d3(sd["per_seed"]["2d"][m["seed"]]["macro"]),
             m["sha256"]] for m in sd["members"]]
    B.append(("tbl", ("Table S6. Members of the final model. Validation macro Dice is for the "
                      "member alone (20 validation cases).",
                      ["Seed", "Best epoch", "Validation macro Dice", "SHA-256 of the weights"],
                      rows)))
    return B


def _verbatim_md(d, path: str, head: str) -> None:
    """Like _verbatim, but renders Markdown tables as Word tables."""
    d.add_heading(head, level=2)
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip().strip("`") for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            _table(d, rows[0], rows[1:])
            continue
        if line.startswith("#"):
            d.add_heading(line.lstrip("# ").strip(), level=3)
        elif line.strip():
            d.add_paragraph(line.lstrip("- ").strip().replace("`", ""),
                            style="List Bullet" if line.lstrip().startswith("- ") else None)
        i += 1


def appendix_picai(out: str) -> str:
    """Multimedia Appendix 5: the PI-CAI prespecification, its lock, and the records of the run."""
    import shutil
    R = os.path.join(PROJECT, "results/picai_psad")
    j_ = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
    pre = os.path.join(PROJECT, "results/picai_psad_prespecification.md")
    lock, elig, summ, merge = (j_("prespec_lock.json"), j_("eligibility.json"),
                               j_("summary.json"), j_("colab_merge.json"))
    if not (_sha_lf(pre) == lock["sha256_lf"] == elig["prespec_sha256_lf"]
            == summ["prespec_sha256_lf"]):
        raise RuntimeError("the PI-CAI prespecification no longer matches its lock")
    d = _doc("Multimedia Appendix 5. PSA density on PI-CAI: prespecification and records")
    d.add_paragraph(
        f"The prespecification below was written, and the SHA-256 of its file recorded "
        f"({lock['sha256_lf']}; line endings normalized to LF), before any PI-CAI file was "
        f"downloaded. The erratum was written before any image was opened. Labels and clinical "
        f"values were read from the picai_labels repository at commit {summ['labels_commit']} "
        f"(marksheet SHA-256 {summ['marksheet_sha256']}).")
    _verbatim_md(d, pre, "A. Prespecification (verbatim)")
    _verbatim_md(d, os.path.join(R, "erratum_2026-11-12.md"), "B. Erratum (verbatim)")
    d.add_heading("C. Model and inference record", level=2)
    mi = summ["model_identity"]
    d.add_paragraph(
        f"Frozen final model: softmax ensemble of the 2D seeds {mi['seeds']} with left-right "
        f"flip averaging. SHA-256 of the member weights: {'; '.join(mi['member_sha256'])}. "
        f"Inference ran on {' and '.join(mi['gpus'])} (PyTorch {' and '.join(mi['torch'])}) in "
        f"{len(mi['runs'])} sessions from one archive of code and weights (SHA-256 "
        f"{mi['bundle_sha256']}). {merge['n_also_inferred_elsewhere']} studies were inferred in 2 "
        f"sessions; {merge['n_also_inferred_identical']} of them gave identical volumes.")
    _table(d, ["Session", "Image archives (folds)", "Studies used"],
           [[r["name"], ", ".join(map(str, r["folds_assigned"])), merge["by_run"][r["name"]]]
            for r in mi["runs"]])
    _verbatim_md(d, os.path.join(R, "report.md"), "D. Results as generated by the analysis script")
    _verbatim_md(d, os.path.join(R, "roundtrip_failure_note.md"),
                 "E. The Bosma22b mask that the round-trip test did not restore")
    d.add_paragraph("Per-study results (volumes, geometry check, Dice against Bosma22b) are in "
                    "the accompanying file Multimedia_Appendix_5_per_study_results.csv.")
    d.save(out)
    csv_out = os.path.join(os.path.dirname(out), "Multimedia_Appendix_5_per_study_results.csv")
    shutil.copy2(os.path.join(R, "per_case.csv"), csv_out)
    return csv_out


def appendix_literature(out: str) -> None:
    """Protocol, search record, screening and per-study coding of the literature audit."""
    import csv
    from manuscript_numbers_jmir import LIT, literature_audit
    lit = literature_audit()           # also checks the protocol hash against the search record
    base = os.path.join(PROJECT, LIT)
    srch = json.load(open(os.path.join(base, "search.json"), encoding="utf-8"))
    summ = json.load(open(os.path.join(base, "summary.json"), encoding="utf-8"))
    scr = list(csv.DictReader(open(os.path.join(base, "screening.csv"), encoding="utf-8")))
    ext = list(csv.DictReader(open(os.path.join(base, "extraction.csv"), encoding="utf-8")))
    d = _doc("Multimedia Appendix 4. Audit of published reporting of the 2D-to-3D step")
    d.add_paragraph(
        "This appendix documents the audit summarized in the Methods and Results (Reporting of "
        "Reconstruction in Published Studies). The protocol was written before any search result "
        "was screened; its SHA-256 (computed with LF line endings) was recorded by the search "
        f"script: {srch['protocol_sha256_lf']}. All files are in the project repository under "
        f"{LIT}/ and are produced by scripts/literature_audit.py; full texts are re-downloaded by "
        "its search step.")
    _verbatim_md(d, os.path.join(base, "protocol.md"), "A. Protocol (verbatim)")

    d.add_heading("B. Search record", level=2)
    _table(d, ["Field", "Value"], [
        ["Database", "PubMed Central, open-access subset (NCBI E-utilities)"],
        ["Query", srch["query"]],
        ["Run (UTC)", srch["run_utc"]],
        ["Records returned", str(srch["count"])],
        ["Full texts retrieved", str(srch["n_fulltext"])],
        ["Screened on full text", str(summ["n_screened"])],
        ["Included", str(summ["n_included"])],
    ])

    d.add_heading("C. Summary of the extracted items", level=2)
    _table(d, ["Item", f"Counts among the {summ['n_included']} included studies"],
           [[it, ", ".join(f"{v} {k}" for k, v in sorted(c.items(), key=lambda x: -x[1]))]
            for it, c in summ["counts"].items()])
    g = summ["geometry_check"]
    d.add_paragraph(
        f"Geometry check reported: {g['k']}/{g['n']} (exact Clopper-Pearson 95% CI "
        f"{g['ci95'][0]*100:.1f}%-{g['ci95'][1]*100:.1f}%). Inverse preprocessing described: "
        f"{lit['inv_yes']}/{lit['inv_n']} studies that reported resampling, resizing, or "
        "cropping; the others reported no preprocessing that changed the image geometry (coded "
        "n/a). Studies reporting per-volume overlap or a surface distance in millimeters: "
        f"{lit['n_depend']}/{summ['n_included']}.")

    d.add_heading("D. Screening decisions", level=2)
    _table(d, ["PMCID", "Year", "Title", "Decision", "Reason", "Note"],
           [[r["pmc"], r["year"], r["title"], r["decision"], r["reason"] or "-", r["note"]] for r in
            sorted(scr, key=lambda r: (r["decision"] != "include", r["year"], r["pmc"]))])

    d.add_heading("E. Coding of the included studies", level=2)
    d.add_paragraph("Input: 2D, 2.5D, or 2D+3D (compared both). Target: gland, zones, or both. "
                    "Level and space: where the metrics were computed. n/a: no preprocessing that "
                    "changed the geometry was reported.")
    _table(d, ["PMCID", "First author, year", "DOI", "Input", "Target", "Level", "Space",
               "Reassembly", "Inverse", "Geometry check", "Orientation", "Surface distance in mm",
               "Code link"],
           [[r["pmc"], f"{r['first_author']}, {r['year']}", r["doi"], r["dimensionality"],
             r["target"], r["metric_level"], r["metric_space"], r["reassembly_described"],
             r["inverse_preprocessing_described"], r["geometry_check_reported"],
             r["orientation_mentioned"], r["surface_metric_mm"], r["code_available"]]
            for r in sorted(ext, key=lambda r: (r["year"], r["pmc"]))])

    d.add_heading("F. Supporting evidence and flags", level=2)
    d.add_paragraph("Short quotations or descriptions that support each positive code, and notes on "
                    "uncertain codings, which were coded conservatively.")
    _table(d, ["PMCID", "Evidence", "Flags"],
           [[r["pmc"], r["evidence"] or "-", r["flags"] or "-"]
            for r in sorted(ext, key=lambda r: (r["year"], r["pmc"]))])
    d.save(out)


def appendix_supplementary(out: str, template: str) -> None:
    """The supplementary material of the earlier extended manuscript, unchanged in content."""
    import manuscript_content_en as old
    from manuscript_numbers import all_numbers, extended
    from build_manuscript import render, resolve_citations, strip_orphan_media
    blocks = old.supplementary(all_numbers(), extended())
    note = ("p", "Multimedia Appendix 3. Supplementary results. Figures and tables are numbered S1, "
                 "S2, and so on. Figures S1 to S8 and Tables S1 to S3, and the text around them, "
                 "describe the single-seed model (E1, seed 42), with which those analyses were "
                 "run; where that text says \"the selected model,\" it means this single-seed "
                 "model. The last section, Seed Replicates and the Final Model, gives the "
                 "results of the final model. In Figures S1 to S8 and Tables S1 to S3, SDs are "
                 "population SDs (as in the analysis artifacts); the main text and Tables S4 to "
                 "S6 report sample SDs.")
    blocks = [("title", "Multimedia Appendix 3. Supplementary Results"), note] + blocks
    blocks += final_model_section()
    blocks, _ = resolve_citations(blocks, old.REFERENCES)
    render(blocks, out, template)
    strip_orphan_media(out)


def cover_letter(out: str, title: str, picai: bool = False) -> None:
    from manuscript_numbers_jmir import literature_audit
    lit = literature_audit()
    if lit["geo_k"] != 0 or lit["reasm_yes"] != 0:
        raise RuntimeError("the cover-letter sentence on the literature audit no longer holds")
    d = _doc("Cover letter")
    for para in [
        "[[TO FILL: date]]",
        # v10 targets JMIR Formative Research (the supervisor's choice, confirmed 2026-11-13).
        "To the Editor, JMIR Formative Research" if picai else "To the Editor, JMIR AI",
        f"We submit the manuscript \"{title}\" for consideration as an original paper.",
        "Two-dimensional segmentation networks remain common in prostate MRI, and their output "
        "has to be reassembled into a 3D volume before any volume, rendering, or registration "
        "can use it. This study measures, across 470 reference segmentations from 4 public "
        "cohorts stored in 3 orientation conventions, which verification check detects which "
        "reconstruction error. A header-based check and a voxel-level round-trip test each "
        "missed a different class of error, and only their combination detected every "
        "damaging fault. Whether a fault was visible at all depended on how the dataset was "
        f"stored. In a protocol-driven audit of {lit['n']} published 2D and 2.5D prostate MRI "
        "segmentation studies, none described how slice outputs were reassembled or reported a "
        "check of the reconstructed volume. "
        "We also report a half-voxel origin inconsistency in a public challenge dataset, "
        "an inter-reader benchmark, seed replicates and a 2.5D variant under a rule fixed "
        "before they were run, and a confirmatory external test on NCI-ISBI 2013 with a "
        "post-processing rule specified before inference."
        + (" Finally, in an analysis specified before its data were downloaded, PSA density "
           "computed from the model's whole-gland volume was noninferior, and not superior, to "
           "PSA density computed from the reported volume for detecting clinically significant "
           "prostate cancer in the public PI-CAI cohort; the model's volumes were systematically "
           "smaller, and we report the consequence for the usual threshold." if picai else ""),
        ("We believe the work fits the journal as an algorithm development and validation "
         "study at a formative stage: it develops a model on public data, tests it externally "
         "under rules fixed in advance, and reports where it is not yet ready for clinical use. "
         "It also addresses reproducibility and verification of AI pipelines in medical "
         "imaging, and its recommendation applies to any study that derives 3D results from "
         "slice-wise output." if picai else
         "We believe the work fits the journal because it addresses reproducibility and "
         "verification of AI pipelines in medical imaging, and its recommendation applies to "
         "any study that derives 3D results from slice-wise output."),
        ("This work is derived from the first author's undergraduate thesis research at "
         "Universitas Multimedia Nusantara. " if picai else
         "This work is derived from the first author's undergraduate thesis at Universitas "
         "Multimedia Nusantara, \"Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat "
         "menggunakan 2D U-Net dan Rekonstruksi 3D\" (2026) [[TO FILL: confirm year and status; "
         "repository link if public]]. ") +
        "The manuscript has not been published or submitted elsewhere, and no preprint has been "
        "posted [[TO FILL: confirm]]. All data are public and de-identified. Generative AI "
        "assistance is disclosed in the Acknowledgments [[TO FILL: confirm]].",
        "Suggested reviewers: [[TO FILL: optional]].",
        "Sincerely,",
        "Eunike Endariahna Surbakti, MTI (corresponding author), on behalf of all authors",
        "Department of Informatics, Faculty of Engineering and Informatics, Universitas "
        "Multimedia Nusantara, Tangerang, Indonesia",
        "eunike.endariahna@umn.ac.id",
    ]:
        d.add_paragraph(para)
    d.save(out)


def figures(manuscript: str, out_dir: str) -> list:
    """Export each figure as PNG within 1200x1200 px and 5 MB, in manuscript order."""
    import manuscript_content_en_jmir as content
    from manuscript_numbers_jmir import jmir_numbers
    blocks = content.build(jmir_numbers(picai=bool(manuscript)))
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for i, (path, _cap) in enumerate((p for k, p in blocks if k == "fig"), 1):
        im = Image.open(os.path.join(PROJECT, path)).convert("RGB")
        im.thumbnail((1200, 1200), Image.LANCZOS)
        dst = os.path.join(out_dir, f"Figure_{i}.png")
        im.save(dst, optimize=True)
        assert os.path.getsize(dst) <= 5 * 1024 * 1024, dst
        done.append((dst, im.size, os.path.getsize(dst)))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--picai", action="store_true",
                    help="v10: add the PI-CAI analysis (Appendix 5, CLAIM rows, cover letter, Figure 1)")
    a = ap.parse_args()
    out = os.path.join(a.journal_dir, f"JMIR_submission_{a.tag}")
    if a.picai and os.path.exists(out):
        raise SystemExit(f"{out} exists: choose a new --tag; older packs are never overwritten")
    os.makedirs(out, exist_ok=True)
    import manuscript_content_en_jmir as content
    appendix_claim(os.path.join(out, "Multimedia_Appendix_1_CLAIM2024_mapping.docx"), a.picai)
    appendix_prespec(os.path.join(out, "Multimedia_Appendix_2_prespecification.docx"))
    appendix_supplementary(os.path.join(out, "Multimedia_Appendix_3_supplementary.docx"), a.template)
    appendix_literature(os.path.join(out, "Multimedia_Appendix_4_literature_audit.docx"))
    if a.picai:
        print("  " + os.path.basename(appendix_picai(
            os.path.join(out, "Multimedia_Appendix_5_PICAI_prespecification.docx"))))
    cover_letter(os.path.join(out, "Cover_letter.docx"), content.TITLE, a.picai)
    for dst, size, b in figures("picai" if a.picai else None, os.path.join(out, "figures")):
        print(f"  {os.path.basename(dst)} {size[0]}x{size[1]} {b/1e6:.2f} MB")
    print("wrote", out)


if __name__ == "__main__":
    main()
