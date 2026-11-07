"""Build the JMIR submission package next to the revised manuscript.

Writes to <journal-dir>/JMIR_submission_<tag>/:
  Multimedia_Appendix_1_CLAIM2024_mapping.docx  where each CLAIM topic is addressed
  Multimedia_Appendix_2_prespecification.docx   rule, erratum, recorded hashes
  Multimedia_Appendix_3_supplementary.docx      supplementary results (old extended material)
  Cover_letter.docx                             draft, with [[TO FILL]] markers
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


def appendix_claim(out: str) -> None:
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
         "Data Sources; Table 1; Preprocessing; Figure 2; exclusions with reasons (NCI-ISBI 2013)",
         "Yes (acquisition protocols only as reported by the source datasets)"),
        ("Methods: reference standard", "Definition; rationale; annotators; inter-reader "
         "variability", "Data Sources (reader 1 as reference standard; reader 2 as second "
         "reference); Evaluation and Inter-Reader Benchmark", "Yes"),
        ("Methods: data partitions", "Allocation to training, tuning (validation), and "
         "internal testing; level of partition", "Data Sources (official patient-level split "
         "119/20/19); Model and Training", "Yes"),
        ("Methods: model", "Architecture; software; initialization", "Model and Training; "
         "Figure 3; Statistical Analysis (software and versions)", "Yes"),
        ("Methods: training", "Training details; hyperparameters; model selection",
         "Model and Training (optimizer, learning rate, batch, epochs, seed, sampler; 2-level "
         "selection rule fixed in advance; checkpoint hash)", "Yes"),
        ("Methods: evaluation", "Metrics; statistical measures of significance and "
         "uncertainty; robustness; explainability; external testing",
         "Evaluation and Inter-Reader Benchmark; Statistical Analysis (Wilcoxon, Holm, TOST, "
         "bootstrap CI); External Testing; Fault-Injection Experiment", "Yes (no explainability "
         "method: the output is a segmentation shown directly)"),
        ("Results: data", "Flow of cases; demographic and clinical characteristics",
         "Table 1; Data Sources (exclusions); overlap between cohorts", "Partly: patient "
         "demographics are not distributed with the public datasets"),
        ("Results: model performance", "Performance with uncertainty; comparison with "
         "reference; failure analysis", "Tables 2-5; Figures 5-8; inter-reader comparison; "
         "gland-size analysis", "Yes"),
        ("Discussion", "Limitations; implications for practice", "Limitations; Clinical "
         "Implications", "Yes"),
        ("Other information", "Registration; protocol; data and code availability; funding",
         "Multimedia Appendix 2 (prespecified rule with file hash); Data Availability; Funding",
         "No trial registration (not applicable to a retrospective algorithm study)"),
    ]
    _table(d, ["CLAIM 2024 section", "Topic", "Where addressed in the manuscript", "Reported"], rows)
    d.save(out)


def appendix_prespec(out: str) -> None:
    d = _doc("Multimedia Appendix 2. Prespecification of the connected-component rule "
             "and the NCI-ISBI 2013 analysis")
    pre = os.path.join(PROJECT, "results/lcc_prespecification.md")
    err = os.path.join(PROJECT, "results/lcc_prespecification_erratum.md")
    sha = hashlib.sha256(open(pre, "rb").read()).hexdigest()
    dec = json.load(open(os.path.join(PROJECT, "results/external_nci_isbi_e1/valid_prostate158/decision.json")))
    nci = json.load(open(os.path.join(PROJECT, "results/external_nci_isbi_e1/nci/summary.json")))
    d.add_paragraph(
        f"SHA-256 of the prespecification file: {sha}. The same hash is recorded in the "
        f"internal check output (decision.json: {dec['prespec_sha256']}) and in the NCI-ISBI "
        f"2013 output (summary.json: {nci['prespec_sha256']}, generated {nci['generated_utc']}). "
        f"Model checkpoint SHA-256: {nci['checkpoint_sha256']}.")
    for path, head in ((pre, "Prespecification (verbatim)"), (err, "Erratum (verbatim)")):
        d.add_heading(head, level=2)
        for line in open(path, encoding="utf-8").read().splitlines():
            if line.startswith("#"):
                d.add_heading(line.lstrip("# ").strip(), level=3)
            elif line.strip():
                d.add_paragraph(line.lstrip("- ").strip(), style="List Bullet" if line.lstrip().startswith("- ") else None)
    d.save(out)


def appendix_supplementary(out: str, template: str) -> None:
    """The supplementary material of the earlier extended manuscript, unchanged in content."""
    import manuscript_content_en as old
    from manuscript_numbers import all_numbers, extended
    from build_manuscript import render, resolve_citations, strip_orphan_media
    blocks = old.supplementary(all_numbers(), extended())
    note = ("p", "Multimedia Appendix 3. Supplementary results. Figures and tables are numbered S1, "
                 "S2, and so on. In this appendix, SDs are population SDs (as in the analysis "
                 "artifacts); the main text reports sample SDs.")
    blocks = [("title", "Multimedia Appendix 3. Supplementary Results"), note] + blocks
    blocks, _ = resolve_citations(blocks, old.REFERENCES)
    render(blocks, out, template)
    strip_orphan_media(out)


def cover_letter(out: str, title: str) -> None:
    d = _doc("Cover letter")
    for para in [
        "[[TO FILL: date]]",
        "To the Editor, JMIR AI [[TO FILL: or JMIR Biomedical Engineering]]",
        f"We submit the manuscript \"{title}\" for consideration as an original paper.",
        "Two-dimensional segmentation networks remain common in prostate MRI, and their output "
        "has to be reassembled into a 3D volume before any volume, rendering, or registration "
        "can use it. This study measures, across 470 reference segmentations from 4 public "
        "cohorts stored in 3 orientation conventions, which verification check detects which "
        "reconstruction error. A header-based check and a voxel-level round-trip test each "
        "missed a different class of error, and only their combination detected every "
        "damaging fault. Whether a fault was visible at all depended on how the dataset was "
        "stored. We also report a half-voxel origin inconsistency in a public challenge dataset, "
        "an inter-reader benchmark, and a confirmatory external test on NCI-ISBI 2013 with a "
        "post-processing rule specified before inference.",
        "We believe the work fits the journal because it addresses reproducibility and "
        "verification of AI pipelines in medical imaging, and its recommendation applies to "
        "any study that derives 3D results from slice-wise output.",
        "This work is derived from the first author's undergraduate thesis at Universitas "
        "Multimedia Nusantara [[TO FILL: thesis title, year, and repository link if public]]. "
        "The manuscript has not been published or submitted elsewhere, and no preprint has been "
        "posted [[TO FILL: confirm]]. All data are public and de-identified. Generative AI "
        "assistance is disclosed in the Acknowledgments [[TO FILL: confirm]].",
        "Suggested reviewers: [[TO FILL: optional]].",
        "Sincerely,",
        "[[TO FILL: corresponding author name, affiliation, email]]",
    ]:
        d.add_paragraph(para)
    d.save(out)


def figures(manuscript: str, out_dir: str) -> list:
    """Export each figure as PNG within 1200x1200 px and 5 MB, in manuscript order."""
    import manuscript_content_en_jmir as content
    from manuscript_numbers_jmir import jmir_numbers
    blocks = content.build(jmir_numbers())
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
    a = ap.parse_args()
    out = os.path.join(a.journal_dir, f"JMIR_submission_{a.tag}")
    os.makedirs(out, exist_ok=True)
    import manuscript_content_en_jmir as content
    appendix_claim(os.path.join(out, "Multimedia_Appendix_1_CLAIM2024_mapping.docx"))
    appendix_prespec(os.path.join(out, "Multimedia_Appendix_2_prespecification.docx"))
    appendix_supplementary(os.path.join(out, "Multimedia_Appendix_3_supplementary.docx"), a.template)
    cover_letter(os.path.join(out, "Cover_letter.docx"), content.TITLE)
    for dst, size, b in figures(None, os.path.join(out, "figures")):
        print(f"  {os.path.basename(dst)} {size[0]}x{size[1]} {b/1e6:.2f} MB")
    print("wrote", out)


if __name__ == "__main__":
    main()
