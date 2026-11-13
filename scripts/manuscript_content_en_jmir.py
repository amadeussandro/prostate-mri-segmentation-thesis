"""JMIR revision of the English manuscript (title unchanged, study-type subtitle added).

Same block grammar as `manuscript_content_en`. Numbers come from
`manuscript_numbers_jmir.jmir_numbers()`. House style applied here:
  * sample SD; text "mean x (SD y)"; tables "mean (SD)"
  * Dice to 3 decimals, distances to 1 decimal, percentages to 1 decimal
  * P without a leading zero: P<.001, P=.006 (3 decimals below .01), else 2
  * Spearman rho as the Greek letter; x for dimensions; US spelling
Anything the authors must still supply is written as [[TO FILL: ...]] so the
pre-submission check can find it.
"""
from __future__ import annotations

from typing import List, Tuple

TITLE_CORE = ("Segmentation of Anatomical Zones in Prostate MRI Using a 2D U-Net "
              "and Spatially Validated 3D Reconstruction")
STUDY_TYPE = "Algorithm Development and Validation Study"
TITLE = f"{TITLE_CORE}: {STUDY_TYPE}"

AUTHORS = ["Benedict Amadeus Sandro",
           "Eunike Endariahna Surbakti, MTI", "David Agustriawan, PhD",
           "Moeljono Widjaja, PhD"]
AFFIL = ["Department of Informatics, Faculty of Engineering and Informatics, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Corresponding Author: Eunike Endariahna Surbakti, MTI, Department of "
        "Informatics, Faculty of Engineering and Informatics, Universitas Multimedia Nusantara, "
        "Jl. Scientia Boulevard, Gading Serpong, Tangerang, Banten 15810, Indonesia. "
        "Phone: +62 21 5422 0808. Email: eunike.endariahna@umn.ac.id. ORCID: Benedict Amadeus Sandro "
        "0009-0000-1459-9139; Eunike Endariahna Surbakti 0000-0002-8442-8615; David Agustriawan "
        "0000-0003-1185-1145; Moeljono Widjaja 0000-0003-3002-7426")

FIG_RQ2 = "results/figures_rq2"
FIG_SUP = "results/figures_supervisor"
FIG_J = "results/figures_jmir_final"   # figures regenerated for the final model
FIG_J10 = "results/figures_jmir_v10"   # v10: the workflow figure with the PI-CAI box


def fp(p: float) -> str:
    """JMIR P value: no leading zero; 3 decimals below .01; P<.001 floor."""
    if p < 0.001:
        return "P<.001"
    if p < 0.01:
        return f"P={p:.3f}".replace("0.", ".")
    s = f"{p:.2f}"
    if s == "0.05":  # .05 is ambiguous either side of the threshold: give 3 decimals
        s = f"{p:.3f}"
    return "P=" + s.replace("0.", ".", 1)


def d3(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def mm(x: float) -> str:
    return f"{x:.1f}".replace("-", "−")


def _date(iso: str) -> str:
    """2026-11-11 -> November 11, 2026 (JMIR house style)."""
    import datetime as _dt
    d = _dt.date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def sg(x: float, nd: int = 3) -> str:
    """Signed value with a true minus sign."""
    return f"{x:+.{nd}f}".replace("-", "−")


def _r2(x: float) -> str:
    """2 decimals, half up (0.625 -> 0.63), to match the other rows of Table 1."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def msd(d: dict, f=d3) -> str:
    return f"{f(d['mean'])} ({f(d['sd'])})"


def pend(what: str) -> str:
    """A visible marker for a PI-CAI number that does not exist yet (draft builds only)."""
    return f"[[PENDING: {what}]]"


def build(N: dict) -> List[Tuple[str, object]]:
    # N["picai"] adds the PI-CAI PSA density analysis (v10); when it is None the v9
    # text is built unchanged. pcr holds its results once summary.json exists.
    pc = N.get("picai")
    pcr = pc["results"] if pc else None
    # Top-level keys describe the FINAL model (3-seed ensemble with flip averaging).
    # N["single"] is the seed-42 model; the prespecified baseline comparison and the
    # validation equivalence tests were run with it, so they read N["stats_single"].
    a, ts, es, vs = N["arms"], N["t_s"], N["e_s"], N["vol_s"]
    st = N["stats_single"]
    tost, wil = st["tost"], st["wilcoxon_test_E1_vs_baseline"]
    wilF = N["stats"]["wilcoxon_test_E1_vs_baseline"]      # final model vs baseline
    one, sd_, fvs = N["single"], N["seeds"], N["final_vs_single"]
    ts1, es1, nc1, r21 = one["t_s"], one["e_s"], one["nci"], one["reader2"]
    arm2, arm25 = sd_["arm"]["2d"], sd_["arm"]["2.5d"]
    r2, mf, nc, ov, psa = N["reader2"], N["mf"], N["nci"], N["ov"], N["psa"]
    g, ab, e = N["geometry"], N["ablation"], N["external"]
    eg, hd, gs, stn = e["geometry"], N["hd95_tail"], N["gland_size"], N["anterior_stroma"]
    margin = st["margin"]
    cg, pz = ts["cg"], ts["pz"]
    v1, v3 = ab["V1_no_reorientation"], ab["V3_identity_affine"]
    S = mf["summary"]
    p158, px, msd_, nci = S["prostate158"], S["prostatex"], S["msd_task05"], S["nci_isbi_2013"]
    r1 = nc["secondary"]["R1"]["cg"]
    r2pz = nc1["secondary"]["R2"]["pz"]     # R2 passed its internal check for the single-seed model only
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
        raise RuntimeError("the literature-audit sentences no longer match the audit; rewrite them")

    B: List[Tuple[str, object]] = []
    add = B.append

    abs_m = abs_r = abs_c = ""
    if pcr:
        pp, pv, pb = pcr["primary"], pcr["volume_model_vs_report"], pcr["volume_model_vs_bosma"]
        rc, pdup = pcr["reclassification"], pc["dup"]
        if not (pp["noninferior"] and not pp["superior"] and pv["bias"] < 0 and pb["bias"] < 0
                and rc["model"]["sensitivity"] > rc["report"]["sensitivity"]
                and rc["model"]["specificity"] < rc["report"]["specificity"]):
            raise RuntimeError("the PI-CAI sentences no longer match summary.json; rewrite them")

        def ci(x, f=d3):
            return f"{f(x[0])} to {f(x[1])}"

        def dci(r):   # difference with its CI
            return f"{d3(r['diff'])} (95% CI {ci(r['diff_ci95'])})"
        mgn = f"{pc['margin']:.2f}".replace("-", "−")
        abs_m = (f" In a further prespecified analysis, PSA density from the model's whole-gland "
                 f"volume was compared with PSA density from the reported volume for detecting "
                 f"clinically significant cancer in {pp['n']:,} patients of the PI-CAI cohort "
                 f"(noninferiority margin {mgn} in the area under the curve [AUC]).")
        abs_r = (f" On PI-CAI, the AUC was {d3(pp['auc_a'])} with the model's volume and "
                 f"{d3(pp['auc_b'])} with the reported volume (difference {d3(pp['diff'])}; 95% CI "
                 f"{ci(pp['diff_ci95'])}): noninferior, not superior. The model's volumes were "
                 f"{mm(-pv['bias'])} mL smaller, so at {pc['threshold']:.2f} ng/mL/mL sensitivity "
                 f"rose ({rc['model']['sensitivity']:.2f} vs {rc['report']['sensitivity']:.2f}) and "
                 f"specificity fell ({rc['model']['specificity']:.2f} vs "
                 f"{rc['report']['specificity']:.2f}).")
        abs_c = (" PSA density from the model's volume ranked patients as well as that from "
                 "reported volumes, but its thresholds cannot be transferred unchanged.")

    # ------------------------------------------------------------------ front
    add(("h2", "Original Paper"))
    add(("title", TITLE))
    add(("authors", AUTHORS))
    add(("affil", AFFIL))
    add(("corr", CORR))

    # --------------------------------------------------------------- abstract
    add(("h2", "Abstract"))
    add(("abs", ("Background",
        "Two-dimensional (2D) networks remain a practical choice for prostate magnetic resonance "
        "imaging (MRI), whose slices are much thicker than their in-plane resolution. Their "
        "slice-wise output must be reassembled into a 3D volume before any volume, rendering, or "
        "registration can use it. That step is routine, and its errors can be silent: "
        "a volume can carry a valid header and still be mirrored or misplaced.")))
    add(("abs", ("Objective",
        "To develop a 2D U-Net for segmenting the central gland (CG) and peripheral zone (PZ), to "
        "reconstruct its output in the original 3D voxel space with explicit verification, and "
        "to measure which verification check detects which reconstruction error.")))
    add(("abs", ("Methods",
        f"Four single-factor training configurations were compared on Prostate158 (119 training, "
        f"20 validation, 19 test examinations), with equivalence tested against a "
        f"margin of ±{margin} Dice. Under a prespecified rule, the selected "
        f"configuration was replicated with {len(sd_['seed_list'])} seeds, with and without 2.5D "
        f"input; the final model, a flip-averaged ensemble of the adopted seeds, was frozen. "
        f"Reconstruction was verified by a header-based geometry check and a "
        f"model-free round-trip test. Six reconstruction faults, alone and combined, were injected into "
        f"{mf['v0_n']} reference segmentations from 4 public cohorts stored in 3 orientation "
        f"conventions. Accuracy was compared with a second expert reader, and the final model "
        f"was applied to PROSTATEx (n={e['n_cases']}) and NCI-ISBI 2013 "
        f"(n={nc['n']}), the latter with a connected-component rule specified "
        f"before inference." + abs_m)))
    add(("abs", ("Results",
        f"The correct pipeline restored all {mf['v0_n']} reference segmentations exactly. Of "
        f"{mf['n_damaged']:,} damaged reconstructions, the geometry check flagged "
        f"{mf['damaged_caught_L1']:,} and the round-trip test {mf['damaged_caught_L2']:,}; "
        f"together they flagged all {mf['n_damaged']:,}, with no false flags. Omitting "
        f"reorientation caused no damage in the cohort stored in the network's own orientation "
        f"but reduced CG Dice to {d3(nci['V1_no_reorientation']['dice_CG'])} in another. On the "
        f"held-out set, Dice was {d3(cg['dice']['mean'])} (95% CI {d3(cg['dice']['ci_low'])}"
        f"-{d3(cg['dice']['ci_high'])}) for the CG and {d3(pz['dice']['mean'])} "
        f"({d3(pz['dice']['ci_low'])}-{d3(pz['dice']['ci_high'])}) for the PZ, against "
        f"inter-reader Dice of {d3(r2['dice_CG_r1_vs_r2']['mean'])} and "
        f"{d3(r2['dice_PZ_r1_vs_r2']['mean'])}; neither Dice nor boundary distance differed "
        f"significantly from inter-reader variability, although equivalence was not shown. "
        f"Training configurations were equivalent or inconclusive within ±{margin} Dice, their "
        f"range ({d3(N['arm_spread'])}) was similar to the range across seeds "
        f"({d3(sd_['range_2d'])}), and 2.5D input did not improve validation Dice "
        f"({sg(sd_['diff'])}). On NCI-ISBI 2013, CG Dice was "
        f"{d3(nc['primary']['cg_dice_none']['mean'])}; the prespecified rule reduced CG "
        f"95th-percentile Hausdorff distance "
        f"from {mm(nc1['primary']['cg_hd95_none']['mean'])} to "
        f"{mm(nc1['secondary']['R1']['cg']['hd95_after']['mean'])} mm "
        f"({fp(nc1['secondary']['R1']['cg']['hd95']['wilcoxon_p'])}) in its confirmatory test "
        f"with the single-seed model, and from {mm(nc['primary']['cg_hd95_none']['mean'])} to "
        f"{mm(r1['hd95_after']['mean'])} mm with the final model." + abs_r)))
    add(("abs", ("Conclusions",
        "A header check and a voxel-level round-trip test detect different reconstruction "
        "errors, and only their combination caught every damaging fault. Whether a fault is "
        "visible depends on how the data were stored, so a pipeline tested on one "
        "dataset can fail silently on another. Studies deriving 3D results from 2D "
        f"segmentation should report both checks; none of {lit['n']} audited prostate studies "
        "did." + abs_c)))
    add(("abs", ("Keywords",
        "prostatic neoplasms; magnetic resonance imaging; deep learning; image segmentation; "
        "image processing, computer-assisted; quality control; reproducibility of results; "
        "external testing")))

    if pcr:
        # With PI-CAI the abstract above exceeds the 450-word limit: a shorter text
        # replaces it. Without PI-CAI the v9 abstract stays as it was.
        short = {
            "Background":
                "Two-dimensional (2D) networks remain practical for prostate magnetic resonance "
                "imaging (MRI), whose slices are thick. Their slice-wise output must be "
                "reassembled into a 3D volume, a routine step whose errors can be silent: a "
                "volume can carry a valid header and still be mirrored or misplaced.",
            "Objective":
                "To develop a 2D U-Net for the central gland (CG) and peripheral zone (PZ), to "
                "reconstruct and verify its output in the original 3D voxel space, to measure "
                "which check detects which reconstruction error, and to test its volumes for "
                "prostate-specific antigen (PSA) density.",
            "Methods":
                f"Four training configurations were compared on Prostate158 (119 training, 20 "
                f"validation, 19 test examinations); under a prespecified rule, the selected one "
                f"was replicated with {len(sd_['seed_list'])} seeds, with and without 2.5D input, "
                f"and a flip-averaged ensemble was frozen as the final model. Reconstruction was "
                f"verified by a header-based geometry check and a model-free round-trip test. Six "
                f"reconstruction faults were injected into {mf['v0_n']} reference segmentations "
                f"from 4 public cohorts. The model was compared with a second expert reader and "
                f"applied to PROSTATEx (n={e['n_cases']}) and NCI-ISBI 2013 (n={nc['n']}). In a "
                f"further prespecified analysis on PI-CAI (n={pp['n']:,}), PSA density from the "
                f"model's whole-gland volume was compared with PSA density from the reported "
                f"volume for detecting clinically significant cancer (noninferiority margin "
                f"{mgn} in area under the curve [AUC]).",
            "Results":
                f"The correct pipeline restored all {mf['v0_n']} references exactly. Of "
                f"{mf['n_damaged']:,} damaged reconstructions, the geometry check flagged "
                f"{mf['damaged_caught_L1']:,} and the round-trip test "
                f"{mf['damaged_caught_L2']:,}; together they flagged all, with no false flags. "
                f"Omitting reorientation caused no damage in one cohort but reduced CG Dice to "
                f"{d3(nci['V1_no_reorientation']['dice_CG'])} in another. On the held-out set, "
                f"Dice was {d3(cg['dice']['mean'])} (95% CI {d3(cg['dice']['ci_low'])}-"
                f"{d3(cg['dice']['ci_high'])}) for the CG and {d3(pz['dice']['mean'])} "
                f"({d3(pz['dice']['ci_low'])}-{d3(pz['dice']['ci_high'])}) for the PZ, against "
                f"inter-reader Dice of {d3(r2['dice_CG_r1_vs_r2']['mean'])} and "
                f"{d3(r2['dice_PZ_r1_vs_r2']['mean'])}. The range across training configurations "
                f"({d3(N['arm_spread'])}) was similar to that across seeds "
                f"({d3(sd_['range_2d'])}), and 2.5D input did not help. On NCI-ISBI 2013, a "
                f"prespecified connected-component rule reduced CG 95th-percentile Hausdorff "
                f"distance from {mm(nc1['primary']['cg_hd95_none']['mean'])} to "
                f"{mm(nc1['secondary']['R1']['cg']['hd95_after']['mean'])} mm "
                f"({fp(nc1['secondary']['R1']['cg']['hd95']['wilcoxon_p'])}) in its confirmatory "
                f"test." + abs_r,
            "Conclusions":
                "A header check and a voxel-level round-trip test detect different "
                "reconstruction errors, and only their combination caught every damaging fault. "
                "Whether a fault is visible depends on how the data were stored. Studies "
                "deriving 3D results from 2D segmentation should report both checks; none of "
                f"{lit['n']} audited prostate studies did." + abs_c,
        }
        B[:] = [(k, (v[0], short[v[0]])) if k == "abs" and v[0] in short else (k, v) for k, v in B]
        n_abs = sum(len(t.split()) for t in short.values())
        if n_abs > 450:
            raise RuntimeError(f"abstract has {n_abs} words; JMIR allows 450")

    # ----------------------------------------------------------- introduction
    add(("h2", "Introduction"))
    add(("p",
        "Prostate cancer is the second most frequently diagnosed cancer in men worldwide [@sung]. "
        "Multiparametric MRI is central to its detection and staging, and structured reporting "
        "organizes interpretation around the zonal anatomy of the gland [@turkbey,weinreb]. The "
        "PZ and the CG, which comprises the transition and central zones [@mcneal], differ in "
        "appearance and in the significance of tumors arising in them, and zone volumes feed "
        "quantities such as prostate-specific antigen (PSA) density [@nordstrom]. Zonal "
        "segmentation is therefore an input to clinical measurement, not an end in itself."))
    add(("p",
        "Deep learning is the standard approach to this task [@ronneberger,isensee,litjens_survey]. "
        "Prostate MRI is strongly anisotropic, and 2D networks remain a reasonable choice under "
        "that geometry: they do not interpolate along the sparsest axis, they are inexpensive to "
        "train, prior work has segmented both zones this way [@bardis,astono], and in benchmarks "
        "on prostate MRI, networks built from 2D convolutions have matched or outperformed a 3D "
        "U-Net [@zhang25d,hung_catnet]. Their output, "
        "however, is a stack of slices. Anything that uses the segmentation as a 3D object "
        "requires those slices to be placed back at their axial positions, the preprocessing to "
        "be inverted, and the source geometry to be restored [@cheng]."))
    add(("p",
        "Errors in that chain are known to occur and to go unnoticed. Left-right flips are "
        "common enough in public MRI data that a dedicated consistency check has been proposed "
        "[@glen], and conversion between DICOM and NIfTI is a documented source of orientation "
        "and position errors [@li_dcm2niix]. Such errors are difficult to see because they do "
        "not corrupt the file. A mirrored volume with a correct header is a valid NIfTI image, "
        "and a correct voxel array written with a default affine still returns a volume in "
        "milliliters. Tools for reporting and appraising studies of artificial "
        "intelligence are numerous, with 46 identified by a recent scoping review [@cabello]. The guidelines "
        "most relevant here ask authors to describe preprocessing and to share code "
        "[@claim,tripod_ai], but they do not ask how a reconstructed volume was verified."))
    add(("p",
        "What is missing is evidence about which verification actually detects which error, and "
        "whether that answer holds across datasets. This study had 2 aims. The first was to "
        "develop a 2D U-Net for CG and PZ segmentation under selection rules that used "
        "validation data only, including seed replicates and a 2.5D variant, and to benchmark it "
        "against a second expert reader and "
        "on 2 external cohorts, since imaging models tend to perform worse on independent "
        "external data than internal validation suggests [@feng_mvi]. The second, which we "
        "regard as the more transferable "
        "contribution, was to reconstruct its output in the original voxel space with explicit "
        "verification and to measure, by injecting known faults into reference segmentations "
        "from 4 public cohorts, which check detects which fault."
        + (" A further analysis, specified before its data were downloaded, asked whether the "
           "resulting volumes are useful where they reach the clinic: whether PSA density "
           "computed from the model's whole-gland volume detects clinically significant "
           "prostate cancer as well as PSA density computed from the volume in the radiology "
           "report." if pc else "")))

    # ---------------------------------------------------------------- methods
    add(("h2", "Methods"))
    add(("h3", "Ethical Considerations"))
    add(("p",
        "No human participants were directly involved in this study. All imaging data"
        f"{', annotations, and clinical values' if pc else ' and annotations'} came from "
        f"{5 if pc else 4} publicly available, deidentified collections (Prostate158, "
        "PROSTATEx with its released zonal annotations, NCI-ISBI 2013, "
        + ("the Medical Segmentation Decathlon prostate task, and the public training and "
           "development set of the PI-CAI challenge" if pc else
           "and the Medical Segmentation Decathlon prostate task") +
        "), which were collected by the contributing "
        "institutions under their own ethical oversight and released for secondary research "
        "use, as described in the source publications [@adams_dib,cuocolo_ejr,clark,bloch_nci,"
        f"antonelli{',picai_lancet,picai_data' if pc else ''}]. The collections are distributed "
        "under open licenses that permit reuse with "
        "attribution (for example, NCI-ISBI 2013 under CC BY 3.0"
        + (", the Medical Segmentation Decathlon under CC BY-SA 4.0, and PI-CAI under CC BY-NC "
           "4.0, which also limits reuse to noncommercial purposes" if pc else
           " and the Medical Segmentation Decathlon under CC BY-SA 4.0") +
        "). No identifiable information was accessed. As this work "
        "was a secondary analysis of publicly available, deidentified data, it did not require "
        "additional local institutional review board approval or informed consent."))
    add(("h3", "Study Design and Reporting"))
    add(("p",
        "This was a retrospective algorithm development and validation study on public, "
        "de-identified data. It is reported following the Checklist for Artificial Intelligence "
        "in Medical Imaging (CLAIM) 2024 [@claim] (Multimedia Appendix 1). The workflow is shown "
        "in Figure 1. Model selection used validation data only at every level. "
        + ("Three analyses were specified in writing, with the file hash recorded before they "
           "were run: a connected-component post-processing rule for the confirmatory external "
           "cohort and the seed-replicate and 2.5D extension that produced the final model "
           "(Multimedia Appendix 2), and the analysis of PSA density on PI-CAI (Multimedia "
           "Appendix 5). " if pc else
           "Two analyses "
           "were specified in writing, with the file hash recorded before they were run "
           "(Multimedia Appendix 2): a connected-component post-processing rule for the "
           "confirmatory external cohort, and the seed-replicate and 2.5D extension that produced "
           "the final model. ") +
        "The extension was written after the single-seed model had been "
        "evaluated on the held-out set, so it is a prespecified revision and not part of the "
        "original protocol. Where an analysis was first run with the single-seed model, its "
        "result is reported alongside that of the final model."))
    add(("fig", (f"{FIG_J10 if pc else FIG_J}/F1_workflow.png",
        'Figure 1. Study workflow and case counts.')))

    add(("h3", "Data Sources"))
    add(("p",
        f"{'Five' if pc else 'Four'} public cohorts were used (Table 1). Prostate158 [@adams_cbm,adams_dib] provided "
        "development data, with its official patient-level split of 119 training, 20 "
        "validation, and 19 held-out test examinations. The validation split served only for "
        "model selection; the test split was used for internal testing. The input was the "
        "axial T2-weighted series; the reference standard was the reader-1 anatomical mask, with 0 for background, 1 for the "
        "CG, and 2 for the PZ. The reader-2 masks, available for all 19 test cases, were used "
        "only as a second reference. The meaning of the label integers was verified against "
        "anatomy rather than assumed, because an inverted reading would silently exchange the 2 "
        "zones in every metric."))
    add(("p",
        f"PROSTATEx [@litjens,clark] with publicly released zonal annotations "
        f"[@cuocolo_ejr,cuocolo_jmri] ({e['n_cases']} examinations) and NCI-ISBI 2013 "
        f"[@bloch_nci] were external cohorts used for inference only. In PROSTATEx, the class "
        f"corresponding to the CG also includes the anterior fibromuscular stroma. NCI-ISBI 2013 "
        f"provides CG and PZ labels for {nc['n_total']} examinations from 2 sites, a 1.5 T "
        f"scanner with an endorectal coil and a 3 T scanner with a surface coil. Its labels were "
        f"matched to the image series by patient identifier and their meaning was verified "
        f"anatomically: label 1 lies posterior to label 2 in every case and is therefore the PZ. "
        f"Two cases were excluded, one whose mask was drawn on a different series (23 vs 34 "
        f"slices) and one carrying a single whole-gland label, leaving {nc['n']}. The Medical "
        f"Segmentation Decathlon (MSD) prostate task [@antonelli] ({mf['n_cases']['msd_task05']} "
        f"of 32 labeled cases with both zones) was used only for the fault-injection experiment, "
        f"because it is stored in a third orientation convention."
        + (" The public training and development set of the PI-CAI challenge "
           "[@picai_lancet,picai_data] was used only to validate PSA density, as described "
           "below." if pc else "")))
    add(("p",
        f"Overlap between the external cohorts was checked by voxel-wise image correlation. Of "
        f"the NCI-ISBI 3 T examinations, {ov['n_msd']} are MSD images with independent annotations, "
        f"and 1 is also a PROSTATEx case. Prostate158 was acquired at a different institution "
        f"from every external cohort, so the development data cannot be among them. The overlap does not affect any training data; it is "
        f"reported because the cohorts are not fully independent of each other, and it supplied "
        f"a second annotation set for {ov['n_two']} external images."))
    add(("tbl", ("Table 1. Cohorts used in this study.",
        ["Cohort", "Use", "Examinations, n", "Scanner and site", "Stored orientation",
         "In-plane spacing, mm", "Reference annotation"],
        [["Prostate158", "Training, validation, held-out test", "158 (119/20/19)",
          "3 T, 1 institution", "LPS", "0.27-0.50", "Zonal (reader 1; reader 2 on test)"],
         ["PROSTATEx", "External testing (secondary)", str(e["n_cases"]),
          "3 T, 1 institution", "LAS", f"{eg['inplane_min']:.2f}-{eg['inplane_max']:.2f}",
          "Zonal; CG class includes anterior stroma"],
         ["NCI-ISBI 2013", "External testing (confirmatory)",
          f"{nc['n']} of {nc['n_total']}", "1.5 T endorectal coil and 3 T, 2 institutions", "LPS",
          f"{_r2(nc['inplane_min'])}-{_r2(nc['inplane_max'])}", "Zonal (CG, PZ)"],
         ["MSD Task05", "Fault injection only", f"{mf['n_cases']['msd_task05']} of 32",
          "3 T, 1 institution", "RAS", "0.60-0.75", "Zonal (transition zone, PZ)"]]
        + ([["PI-CAI", "PSA density validation only",
             f"{pc['n_eligible']:,} of {pc['n_patients']:,} patients",
             f"Several scanners, {len(pc['centers'])} centers", "/".join(pc["axcodes"]),
             f"{_r2(pc['inplane_min'])}-{_r2(pc['inplane_max'])}",
             "None (case-level csPCa label, measured PSA, reported volume)"]] if pc else []))))

    add(("h3", "Preprocessing"))
    add(("p",
        "Volumes were reoriented to a canonical RAS "
        "frame. Intensities were clipped at the 0.5th and 99.5th percentiles and z-score "
        "normalized per volume; the mask was never normalized. The in-plane grid was "
        "standardized to 442×442 by symmetric zero-padding or center-cropping, and the "
        "through-plane axis was not resampled. The mask path therefore involves no "
        "interpolation, and every parameter needed to invert it (orientation transform, per-axis "
        "pad or crop offsets, and depths) was stored per case."))

    add(("h3", "Model and Training"))
    add(("p",
        "The network was a 4-level 2D U-Net [@ronneberger] with 32 initial feature maps, 1 input "
        "and 3 output channels, and 7,762,531 parameters (Figure 2). Each level applied 2 "
        "3×3 convolutions with batch normalization and rectified linear units; "
        "downsampling used 2×2 max pooling and upsampling used transposed convolutions with "
        "skip connections. Dropout of 0.2 was applied at the bottleneck. The prediction was the "
        "arg-max over 3 classes, so the output is a partition by construction. A standard "
        "architecture was chosen deliberately: self-configuring pipelines show that an "
        "unmodified U-Net remains competitive [@isensee], and a novel architecture would "
        "confound the reconstruction analysis with model design."))
    add(("fig", (f"{FIG_J}/F2_unet.png",
        'Figure 2. The 2D U-Net architecture (dashed lines: skip connections).')))
    add(("p",
        "Four configurations were trained, each differing from its comparator in 1 factor: a "
        "cross-entropy baseline; the baseline plus a soft Dice term on the 2 foreground classes "
        "(E1); the baseline plus geometric and intensity augmentation (E2); and E1 plus "
        "class-weighted cross-entropy that up-weights the PZ (E3). E3 is therefore a "
        "single-factor change relative to E1, not to the baseline. All configurations used "
        "AdamW (learning rate 0.001, weight decay 0.0001), batch size 8, 100 epochs, 32-bit "
        "precision, seed 42, and a training sampler that drew slices with and without "
        "foreground at equal expected frequency."))
    add(("p",
        "Within each run, the epoch with the highest slice-level validation Dice was retained. "
        "Between runs, the configuration with the highest case-level macro Dice (mean of CG and "
        "PZ) on the 20 validation cases was selected. No test metric contributed to either "
        "decision."))

    add(("h3", "Seed Replicates, 2.5D Input, and the Final Model"))
    add(("p",
        f"Every configuration had been trained once, and the random seed alone can change "
        f"segmentation performance [@akesson]. The selected configuration was therefore "
        f"retrained with 2 further seeds (43 and 44). In parallel, a 2.5D variant was trained "
        f"with the same {len(sd_['seed_list'])} seeds: the slice to be segmented and its 2 axial "
        f"neighbors were stacked as 3 input channels (7,763,107 parameters), with the edge "
        f"slice repeated at either "
        f"end of the volume. Three slices is the context best supported for prostate MRI "
        f"[@zhang25d,hung_catnet]. The 2 arms differed only in the seed and the number of input "
        f"channels; the architecture, the loss, and the slice-wise reconstruction were unchanged."))
    add(("p",
        f"The rule was fixed before these runs (Multimedia Appendix 2). The 2.5D input would be "
        f"adopted only if its mean validation macro Dice across seeds exceeded that of the 2D "
        f"input by at least {sd_['margin']}. The final model would be the average of the softmax "
        f"outputs of the adopted arm's {len(sd_['seed_list'])} seeds, a standard form of "
        f"ensembling [@isensee]. Averaging with the prediction for the left-right mirrored "
        f"slice, mirrored back, would be added only if it raised the ensemble's validation "
        f"macro Dice. No step read the test split. The final model was then frozen, identified "
        f"by the SHA-256 hashes of its members' weights, and evaluated once on the held-out "
        f"set."))

    add(("h3", "3D Reconstruction and Verification"))
    add(("p",
        "Each predicted slice was written back to the axial index it carried, never in "
        "loader order; a missing or duplicated index raised an error. The volume was then "
        "inverse-transformed through the preprocessing chain in reverse order with "
        "nearest-neighbor interpolation and written as NIfTI with the source affine and spacing "
        "(Figure 3). Four checks were defined, at 3 levels."))
    add(("p",
        "Level 1, the geometry check, reads the written file's own header and compares array "
        "shape, affine, voxel spacing, orientation codes, and label set with the source. Level "
        "2, the round-trip test, uses no model: the reference mask is passed forward through "
        "preprocessing and back through the same reconstruction code and must be recovered "
        "voxel for voxel. Because the mask transform is a signed axis permutation followed by a "
        "pad or crop, exact recovery is expected, and the outcome is reported as pass or fail, "
        "never as an accuracy. Level 3 comprises 2 anatomical checks that need no reference and "
        "can therefore run at inference time: the PZ centroid must lie posterior to the CG "
        "centroid in scanner coordinates, and the gland must occupy a contiguous run of slices "
        "with its in-plane centroid moving no more than 15 mm between adjacent slices. Both "
        "thresholds were fixed before any fault-injection result was read."))
    add(("fig", (f"{FIG_J}/F3_reconstruction_verification.png",
        'Figure 3. Reconstruction pipeline and the 3 verification levels.')))

    add(("h3", "Fault-Injection Experiment"))
    add(("p",
        f"To measure what each check detects, the reference segmentations of all 4 cohorts "
        f"({mf['v0_n']} cases) were passed through the production forward transform and "
        f"reconstructed either correctly or with 1 fault: reorientation not inverted; center "
        f"crop instead of the stored offsets; an identity affine; an affine carrying the voxel "
        f"spacing but no origin or direction; slices stacked in loader order; slices stacked in "
        f"shuffled order (fixed seed); and all faults combined. Because no model is involved, any "
        f"difference from the reference is caused by the fault alone. A case was counted as "
        f"damaged if Dice against the reference fell below 1, a centroid moved, or a volume "
        f"changed. The experiment was gated: the correct pipeline had to return every reference "
        f"exactly before any fault was analyzed. The same faults were also applied to the saved "
        f"predictions of the 19 held-out cases, to show what a reader of the results would have "
        f"seen."))

    add(("h3", "Evaluation and Inter-Reader Benchmark"))
    add(("p",
        "All metrics were computed per case in the original voxel space against the mask read "
        "from disk: Dice, intersection over union, 95th-percentile Hausdorff distance (HD95), "
        "average surface distance, precision, and recall [@taha,maierhein], with surface "
        "distances in millimeters from each case's own spacing. Macro Dice is the mean of the 2 "
        "foreground classes. On the 19 test cases, reader 2 was compared with reader 1 using the "
        "same code, so that the model's error could be set against expert disagreement, as in "
        "other segmentation studies that benchmark against interobserver variation [@nikolov]. Zone "
        "volumes were computed from the absolute determinant of the affine and compared by "
        "Bland-Altman analysis [@bland]."))
    add(("p",
        ("Because Prostate158 provides no PSA values, the clinical consequence of volume "
           "error was first examined on it as a sensitivity analysis. " if pc else
           "Because no PSA values are available, the clinical consequence of volume error was "
           "examined as a sensitivity analysis. ") +
        "For each patient, the range of PSA values at which "
        "the reference and predicted volumes place PSA density on opposite sides of 0.15 "
        "ng/mL/mL was computed, and the share of patient-PSA pairs reclassified was summarized "
        "over the 4 to 10 ng/mL diagnostic gray zone."
        + (" Measured PSA values were then analyzed on PI-CAI." if pc else "")))

    add(("h3", "External Testing"))
    add(("p",
        "The frozen final model was applied to both external cohorts with no training, "
        "tuning, or "
        "threshold search, through the same inference and reconstruction code; the flip "
        "averaging is part of the frozen model and was not adjusted. Both cohorts were first "
        "analyzed with the single-seed model, under the protocol below, and the analysis was "
        "repeated unchanged with the final model. In PROSTATEx, a connected-component "
        "decomposition of the single-seed predictions showed that the CG boundary-error tail "
        "consisted of small, distant "
        "false-positive components. Because that observation was post hoc, PROSTATEx is "
        "treated as a secondary cohort and the post-processing was not applied to it."))
    add(("p",
        "Before any inference on NCI-ISBI 2013, a rule was written down and its file hash "
        "recorded in every output (Multimedia Appendix 2): keep the largest 6-connected CG "
        "component (R1), and, secondarily, also the largest PZ component (R2). The rule has no "
        "parameters. It was first checked for harm on the 20 validation cases, where it "
        "had to change mean Dice by no less than −0.005 and not increase mean HD95. The "
        "primary NCI-ISBI outcome was the unmodified pipeline; the rule's effect was the "
        "prespecified secondary outcome. One correction was recorded before inference: a case "
        "carrying a single whole-gland label was excluded. The confirmatory test of the rule is "
        "therefore the one run with the single-seed model. For the final model, the harm check "
        "and the NCI-ISBI analysis were repeated without change, as the seed prespecification "
        "required."))

    if pc:
        mg = f"{pc['margin']:.2f}".replace("-", "−")
        add(("h3", "PSA Density on PI-CAI"))
        add(("p",
            f"Whether the model's volumes are fit for computing PSA density was tested on the public training "
            f"and development set (version 2.0) of the PI-CAI challenge [@picai_lancet,picai_data]. "
            f"It comprises {pc['n_studies']:,} biparametric MRI studies of {pc['n_patients']:,} "
            f"patients from {len(pc['centers'])} centers in the Netherlands and provides, for each "
            f"study, the measured PSA level, the prostate volume from the radiology report, and a "
            f"label for clinically significant prostate cancer (csPCa), defined as International "
            f"Society of Urological Pathology (ISUP) grade group 2 or higher. Positive studies were "
            f"confirmed histologically; negative studies were confirmed either histologically "
            f"(ISUP grade group 1 or lower) or, without biopsy or follow-up, by an MRI reading of "
            f"Prostate Imaging-Reporting and Data System (PI-RADS) category 2 or lower. These "
            f"values, and a publicly released AI segmentation of the whole gland (Bosma22b) that "
            f"was used only for comparison, were taken from the challenge's label repository at a "
            f"fixed commit [@picai_labels]."))
        add(("p",
            f"The analysis was specified in writing, and its file hash recorded, before any PI-CAI "
            f"file was downloaded (Multimedia Appendix 5). An erratum written before any image was "
            f"opened corrected the description of the reference standard for negative studies; it "
            f"changed no end point, threshold, or analysis population. The earliest study of each "
            f"patient was analyzed, and patients without a PSA value were excluded. Only the axial "
            f"T2-weighted series was used. Each image was converted from the MetaImage format to "
            f"NIfTI with SimpleITK, and a conversion was accepted only if the world coordinates of "
            f"the 8 corner voxels agreed within {pc['corner_tol_mm']:g} mm. The frozen final model "
            f"was applied through the inference and reconstruction code of the main analysis, "
            f"unchanged and without post-processing. Whole-gland volume was the sum of the CG and "
            f"PZ volumes, and PSA density was PSA divided by that volume; a patient with an empty "
            f"segmentation stayed in the analysis, with PSA density ranked highest."))
        add(("p",
            f"The primary end point was the difference in the area under the receiver operating "
            f"characteristic curve (AUC) for csPCa between PSA density from the model's volume and "
            f"PSA density from the reported volume, model minus report, in patients with a "
            f"reported volume. Each AUC was the empirical (Mann-Whitney) estimate, and the paired "
            f"difference received a 95% percentile CI from a bootstrap stratified by outcome "
            f"({pc['n_boot']:,} resamples, fixed seed). PSA density from the model was to be called "
            f"noninferior if the lower bound of this CI exceeded {mg}, and superior only if it "
            f"also exceeded 0."))
        add(("p",
            f"Secondary end points were descriptive. Agreement of the model's volume with the "
            f"reported volume and with the Bosma22b volume was assessed by Bland-Altman analysis, "
            f"and whole-gland Dice against Bosma22b was computed where that mask lies on the "
            f"T2-weighted grid. The reported volume is usually an ellipsoid estimate and Bosma22b "
            f"is itself a model output, so neither comparison measures accuracy. At a PSA density "
            f"of {pc['threshold']:.2f} ng/mL/mL, the patients classified differently by the 2 "
            f"volumes were counted, and the sensitivity and specificity of each PSA density were "
            f"computed. The AUC of PSA density from the Bosma22b volume was reported for "
            f"reference, and the primary end point was reported by center. The primary end point "
            f"was also repeated with rule R1, without studies that failed the geometry check, and "
            f"without studies whose image duplicates one in PROSTATEx, NCI-ISBI 2013, or MSD. "
            f"Duplicates were identified as for the external cohorts: the same array shape after "
            f"reorientation and a voxel-wise Pearson r of at least {pc['dup_r']:.2f}. The geometry "
            f"check was applied to every reconstruction, and the round-trip test to every "
            f"nonempty Bosma22b mask on its own grid."))

    add(("h3", "Audit of Published Reporting"))
    add(("p",
        "To learn how often published work reports the reconstruction step, we audited deep "
        "learning studies that segment the prostate gland or its zones on MRI with at least 1 "
        "network taking 2D slices, or stacks of neighboring slices (2.5D), as input. The search "
        "(a fixed query of the open-access subset of PubMed Central, run once on "
        f"{_date(lit['run_date'])}), the eligibility criteria, and the 10 extracted items were "
        "set in a protocol written before screening, and its file hash is recorded in the "
        "outputs (Multimedia Appendix 4). One author screened the full texts and coded each "
        "item, supporting each positive code with a quotation; an item not stated in the text "
        "was coded as not reported."))

    add(("h3", "Statistical Analysis"))
    add(("p",
        f"Paired per-case differences were tested with the Wilcoxon signed-rank test. The "
        f"prespecified comparison was macro Dice of the selected configuration, as a single-seed "
        f"model, against the baseline on the held-out set, which is a single-factor contrast. "
        f"Per-zone tests were post hoc and adjusted with the Holm "
        f"method [@holm] across the 2 zones. The final model was also compared with the "
        f"baseline and with the single-seed model; these comparisons were not prespecified, and "
        f"the first combines the effects of the loss, ensembling, and flip averaging. Because a "
        f"nonsignificant difference does not show "
        f"absence of an effect, equivalence was tested with 2 one-sided tests (TOST) "
        f"[@schuirmann,lakens] on paired differences with a margin of ±{margin} Dice; "
        f"equivalence was concluded when the 90% CI lay within the margin. The margin was not "
        f"registered in advance. It was chosen to be smaller than the SD of inter-reader Dice for either zone "
        f"({d3(r2['dice_CG_r1_vs_r2']['sd'])} for the CG and "
        f"{d3(r2['dice_PZ_r1_vs_r2']['sd'])} for the PZ). Tests on the validation split are "
        f"descriptive, because that split was used for selection. Dispersion is the sample SD, "
        f"across cases or, where stated, across seeds, "
        f"and CIs are bootstrap percentile intervals (10,000 resamples, fixed seed). Training, "
        f"and inference on PROSTATEx with the final model, "
        + (f"used a single NVIDIA Tesla T4 GPU, and inference on PI-CAI used "
           f"{pend('GPU model(s) of the PI-CAI runs') if pcr is None else 'the same GPU model' if pcr['model_identity']['gpus'] == ['Tesla T4'] else ' and '.join(pcr['model_identity']['gpus'])}; "
           f"on {pc['cpu_gpu_n']} PI-CAI studies also run on a CPU, the predictions were "
           f"{'identical voxel for voxel' if pc['cpu_gpu_identical'] == pc['cpu_gpu_n'] else pend('CPU and GPU predictions differ')}. "
           f"All other analyses ran on a CPU with Python " if pc else
           f"used a single NVIDIA Tesla T4 GPU; all other analyses ran on a CPU with Python ") +
        f"(version 3.13; Python Software Foundation), PyTorch (version 2.14; Linux Foundation), "
        f"NiBabel (version 5.4), SimpleITK (version 2.4), and SciPy (version 1.18)."))

    # ---------------------------------------------------------------- results
    add(("h2", "Results"))
    add(("h3", "Model Selection and Equivalence of Training Configurations"))
    e1b, e2b, e3e1 = (tost["val_E1_minus_baseline"], tost["val_E2_minus_baseline"],
                      tost["val_E3_minus_E1"])
    add(("p",
        f"E1 had the highest validation macro Dice ({d3(a['e1']['macro'])}) and was selected "
        f"(Table 2). The 4 configurations spanned {d3(N['arm_spread'])} macro Dice. "
        f"Augmentation was equivalent to the baseline within ±{margin} for macro and CG "
        f"Dice ({fp(e2b['macro']['tost_p'])} and {fp(e2b['CG']['tost_p'])}) but not shown "
        f"equivalent for the PZ ({fp(e2b['PZ']['tost_p'])}). Class weighting was equivalent to "
        f"E1 in every zone (all {fp(max(e3e1[z]['tost_p'] for z in ('macro', 'CG', 'PZ')))} or "
        f"lower). Neither of these commonly recommended interventions produced a difference "
        f"shown to exceed the margin, although equivalence for the PZ under augmentation "
        f"remained unshown."))
    rowsT2 = []
    comp = {"baseline": None, "e1": ("val_E1_minus_baseline", "baseline"),
            "e2": ("val_E2_minus_baseline", "baseline"), "e3": ("val_E3_minus_E1", "E1")}
    for k, d in a.items():
        name = "Baseline" if k == "baseline" else k.upper()
        if comp[k] is None:
            diff, tp = "Reference", "—"
        else:
            t = tost[comp[k][0]]["macro"]
            diff = (f"{sg(t['mean_diff'])} vs {comp[k][1]} "
                    f"({d3(t['ci90'][0])} to {d3(t['ci90'][1])})")
            tp = fp(t["tost_p"])
        label = {"baseline": "Cross-entropy", "e1": "Baseline + soft Dice on CG and PZ",
                 "e2": "Baseline + augmentation", "e3": "E1 + class-weighted cross-entropy"}[k]
        rowsT2.append([name, label, str(d["epoch"]), d3(d["cg"]),
                       d3(d["pz"]), d3(d["macro"]), diff, tp])
    add(("tbl", (f"Table 2. Training configurations on the 20 validation cases. Each differs from "
                 f"its comparator in 1 factor. Difference: macro Dice, mean (90% CI). TOST P: "
                 f"equivalence within ±{margin} Dice (descriptive; this split was used for "
                 f"selection).",
        ["Configuration", "Change", "Best epoch", "CG Dice", "PZ Dice", "Macro Dice",
         "Difference (90% CI)", "TOST P"], rowsT2)))

    add(("h3", "Seed Replicates, 2.5D Input, and the Final Model"))
    ens, enst = sd_["ensemble"], sd_["ensemble_tta"]
    if sd_["adopted"] != "2d" or not sd_["tta_kept"]:
        raise RuntimeError("the text below describes a 2D ensemble with flip averaging kept")
    add(("p",
        f"Across {len(sd_['seed_list'])} seeds, E1 had a validation macro Dice of mean "
        f"{d3(arm2['mean_macro'])} (SD {d3(arm2['sd_macro'])}) with 2D input and "
        f"{d3(arm25['mean_macro'])} (SD {d3(arm25['sd_macro'])}) with 2.5D input (Table 3). The "
        f"difference ({sg(sd_['diff'])}) was below the prespecified margin of {sd_['margin']}, "
        f"so the 2D input was kept. The 2.5D input did not improve the PZ, the zone it was "
        f"meant to help ({d3(arm25['mean_pz'])} vs {d3(arm2['mean_pz'])}), and it varied more "
        f"between seeds. The range across the 3 seeds ({d3(sd_['range_2d'])} for the 2D input "
        f"and {d3(sd_['range_25d'])} for the 2.5D input) was similar to or larger than the "
        f"range across the 4 training configurations ({d3(N['arm_spread'])}), so the "
        f"differences in Table 2 are of a size that reseeding alone can produce. The ensemble of "
        f"the 2D seeds reached {d3(ens['macro'])}, above each of its members. Flip averaging "
        f"raised this to {d3(enst['macro'])} and was kept, and this model was frozen as the "
        f"final model."))
    rowsT3 = []
    for arm, lab, st_ in (("2d", "2D", arm2), ("2.5d", "2.5D", arm25)):
        for s_ in sd_["seed_list"]:
            v = sd_["per_seed"][arm][s_]
            rowsT3.append([f"{lab} input, seed {s_}", str(v["epoch"]), d3(v["cg"]), d3(v["pz"]),
                           d3(v["macro"])])
        rowsT3.append([f"{lab} input, mean (SD) across seeds", "—",
                       f"{d3(st_['mean_cg'])} ({d3(st_['sd_cg'])})",
                       f"{d3(st_['mean_pz'])} ({d3(st_['sd_pz'])})",
                       f"{d3(st_['mean_macro'])} ({d3(st_['sd_macro'])})"])
    rowsT3.append(["Ensemble of the 2D seeds", "—", d3(ens["cg"]), d3(ens["pz"]),
                   d3(ens["macro"])])
    rowsT3.append(["Ensemble with left-right flip averaging (final model)", "—", d3(enst["cg"]),
                   d3(enst["pz"]), d3(enst["macro"])])
    add(("tbl", ("Table 3. Seed replicates of the selected configuration (E1) with 2D and 2.5D "
                 "input, and the ensemble, on the 20 validation cases (case-level Dice).",
        ["Model", "Best epoch", "CG Dice", "PZ Dice", "Macro Dice"], rowsT3)))

    add(("h3", "Held-Out Performance and Inter-Reader Agreement"))
    cg1, pz1, ft = ts1["cg"], ts1["pz"], fvs["test"]

    def gap(key):
        """Model minus inter-reader, written as a shortfall with its 95% CI."""
        x = r2[key]
        return (f"{d3(-x['mean_diff'])} (95% CI {d3(-x['ci95'][1])} to {d3(-x['ci95'][0])}; "
                f"{fp(x['wilcoxon_p'])})")
    gcg, gpz = r2["dice_CG_model_minus_interreader"], r2["dice_PZ_model_minus_interreader"]
    hcg, hpz = r2["hd95_CG_model_minus_interreader"], r2["hd95_PZ_model_minus_interreader"]
    if min(x["wilcoxon_p"] for x in (gcg, gpz, hcg, hpz)) < 0.05:
        raise RuntimeError("the inter-reader text assumes no significant Dice or HD95 difference")
    add(("p",
        f"On the 19 held-out cases, the final model's Dice was mean {d3(cg['dice']['mean'])} (SD "
        f"{d3(cg['dice']['sd'])}) for the CG and {d3(pz['dice']['mean'])} (SD "
        f"{d3(pz['dice']['sd'])}) for the PZ (Table 4). For the single-seed model, the values "
        f"were {d3(cg1['dice']['mean'])} (SD {d3(cg1['dice']['sd'])}) and "
        f"{d3(pz1['dice']['mean'])} (SD {d3(pz1['dice']['sd'])}); the final model was higher in "
        f"macro Dice by {d3(ft['macro']['mean_diff'])} (95% CI {d3(ft['macro']['ci_low'])} to "
        f"{d3(ft['macro']['ci_high'])}; better in {ft['macro']['n_better']} of "
        f"{ft['macro']['n']} cases; {fp(ft['macro']['p'])}). The PZ combined high precision "
        f"({d3(pz['precision']['mean'])}) with lower recall ({d3(pz['recall']['mean'])}): the "
        f"model rarely labeled other tissue as PZ but labeled too little of it."))
    add(("p",
        f"Against reader 1, the final model's Dice was lower than reader 2's by "
        f"{gap('dice_CG_model_minus_interreader')} for the CG and by "
        f"{gap('dice_PZ_model_minus_interreader')} for the PZ, that is, "
        f"{gcg['ratio_of_means']*100:.1f}% and {gpz['ratio_of_means']*100:.1f}% of inter-reader "
        f"agreement. Neither difference was significant. Equivalence was not tested, however, "
        f"and with 19 cases the CIs do not exclude a shortfall of "
        f"{d3(-gcg['ci95'][0])} for the CG or {d3(-gpz['ci95'][0])} for the PZ. For the "
        f"single-seed model, both shortfalls had been significant "
        f"({d3(-r21['dice_CG_model_minus_interreader']['mean_diff'])}, "
        f"{fp(r21['dice_CG_model_minus_interreader']['wilcoxon_p'])}; "
        f"{d3(-r21['dice_PZ_model_minus_interreader']['mean_diff'])}, "
        f"{fp(r21['dice_PZ_model_minus_interreader']['wilcoxon_p'])}). Boundary distance was "
        f"also not significantly different from inter-reader variability: HD95 was "
        f"{mm(r2['hd95_CG_model_vs_r1']['mean'])} mm against {mm(r2['hd95_CG_r1_vs_r2']['mean'])} "
        f"mm for the CG ({fp(r2['hd95_CG_model_minus_interreader']['wilcoxon_p'])}) and "
        f"{mm(r2['hd95_PZ_model_vs_r1']['mean'])} mm against "
        f"{mm(r2['hd95_PZ_r1_vs_r2']['mean'])} mm for the PZ "
        f"({fp(r2['hd95_PZ_model_minus_interreader']['wilcoxon_p'])})."))
    m, mF = wil["macro"], wilF["macro"]
    add(("p",
        f"Against the baseline, the prespecified macro comparison, run with the single-seed "
        f"model, was not significant "
        f"({sg(m['mean_diff'])}, better in {m['n_better']} of {m['n']} cases; "
        f"{fp(m['p_raw'])}) and not equivalent either (90% CI {d3(t_e1b['macro']['ci90'][0])} to "
        f"{d3(t_e1b['macro']['ci90'][1])}), so the effect of the soft Dice term alone is "
        f"inconclusive. Post hoc, "
        f"CG Dice was equivalent ({fp(t_e1b['CG']['tost_p'])}) and PZ Dice was higher by "
        f"{d3(wil['PZ']['mean_diff'])} (better in {wil['PZ']['n_better']} of {wil['PZ']['n']} "
        f"cases; Holm-adjusted {fp(wil['PZ']['p_holm_zones'])}). The final model exceeded the "
        f"baseline by {d3(mF['mean_diff'])} macro Dice (better in {mF['n_better']} of {mF['n']} "
        f"cases; {fp(mF['p_raw'])}). That comparison was not prespecified, and it reflects "
        f"ensembling and flip averaging as well as the loss."))
    def rr(name, key, f=d3):
        return [name, msd(cg[key], f), "—", msd(pz[key], f), "—"]
    add(("tbl", ("Table 4. Held-out performance of the final model (19 cases, original voxel "
                 "space, per-case mean (SD)) and agreement between the 2 expert readers on the same "
                 "cases.",
        ["Metric", "CG, model vs reader 1", "CG, reader 2 vs reader 1", "PZ, model vs reader 1",
         "PZ, reader 2 vs reader 1"],
        [["Dice", msd(cg["dice"]), msd(r2["dice_CG_r1_vs_r2"]), msd(pz["dice"]),
          msd(r2["dice_PZ_r1_vs_r2"])],
         ["Dice, 95% CI", f"{d3(cg['dice']['ci_low'])}-{d3(cg['dice']['ci_high'])}", "—",
          f"{d3(pz['dice']['ci_low'])}-{d3(pz['dice']['ci_high'])}", "—"],
         ["Dice, single-seed model", msd(cg1["dice"]), "—", msd(pz1["dice"]), "—"],
         ["HD95, mm", msd(cg["hd95_mm"], mm), msd(r2["hd95_CG_r1_vs_r2"], mm),
          msd(pz["hd95_mm"], mm), msd(r2["hd95_PZ_r1_vs_r2"], mm)],
         ["Average surface distance, mm", msd(cg["asd_mm"], mm), msd(r2["asd_CG_r1_vs_r2"], mm),
          msd(pz["asd_mm"], mm), msd(r2["asd_PZ_r1_vs_r2"], mm)],
         rr("Intersection over union", "iou"), rr("Precision", "precision"),
         rr("Recall", "recall")])))

    add(("h3", "Reconstruction Verification and Fault Injection"))
    add(("p",
        f"All {g['n']} held-out reconstructions passed the geometry check, with an affine "
        f"identical to the source within single-precision rounding, and the round-trip test recovered every reference mask "
        f"exactly. In the fault-injection experiment, the correct pipeline returned all "
        f"{mf['v0_n']} references exactly (Table 5). Of the {mf['n_damaged']:,} damaged "
        f"reconstructions, the geometry check flagged {mf['damaged_caught_L1']:,} "
        f"({mf['damaged_caught_L1']/mf['n_damaged']*100:.1f}%), the round-trip test "
        f"{mf['damaged_caught_L2']:,} ({mf['damaged_caught_L2']/mf['n_damaged']*100:.1f}%), and "
        f"the 2 together all {mf['n_damaged']:,}, while neither flagged any of the "
        f"{mf['n_undamaged']} faulted reconstructions that caused no damage (Figure 4)."))
    add(("p",
        f"The 2 checks are complementary. A volume that was not reoriented carries a correct "
        f"header and wrong voxels, so the geometry check passed it in every cohort and only the "
        f"round-trip test caught it. On the held-out predictions this volume would have been "
        f"reported with CG Dice {d3(v1['cg']['dice_vs_gt'])} and PZ Dice "
        f"{d3(v1['pz']['dice_vs_gt'])}, with PZ centroids displaced by a mean of "
        f"{mm(v1['pz']['centroid_shift_mm'])} mm (maximum {mm(v1['pz']['centroid_shift_max_mm'])} "
        f"mm), while passing every header check (Figure 5). Conversely, an identity or "
        f"spacing-only affine leaves the voxels intact, so Dice is unchanged and only the "
        f"geometry check detects it. The spacing-only affine is the more dangerous of the 2: it "
        f"preserved every volume exactly (error 0%) while moving the gland "
        f"{mm(min(c['V6_spacing_only']['shift_CG_mm'] for c in S.values()))} to "
        f"{mm(max(c['V6_spacing_only']['shift_CG_mm'] for c in S.values()))} mm, so a "
        f"volume-based quality check would also have passed it."))
    add(("p",
        f"Whether a fault was visible depended on how the data were stored. Omitting "
        f"reorientation caused no damage in MSD, which is stored in the RAS orientation the "
        f"network uses; it produced only a left-right flip in PROSTATEx (LAS; CG Dice "
        f"{d3(px['V1_no_reorientation']['dice_CG'])}, PZ {d3(px['V1_no_reorientation']['dice_PZ'])}); "
        f"and it destroyed overlap in the 2 LPS cohorts (CG Dice "
        f"{d3(nci['V1_no_reorientation']['dice_CG'])} to "
        f"{d3(p158['V1_no_reorientation']['dice_CG'])}). A pipeline developed and tested on MSD "
        f"alone would therefore pass and then fail silently on LPS data. The anatomical checks, "
        f"which need no reference, flagged {mf['damaged_caught_L3']:,} of {mf['n_damaged']:,} "
        f"damaged cases ({mf['damaged_caught_L3']/mf['n_damaged']*100:.0f}%). They missed the "
        f"left-right flip, which preserves anterior-posterior order, and they raised "
        f"{mf['v0_L3b_false']} false alarms on the correct pipeline "
        f"({mf['v0_L3b_false']/mf['v0_n']*100:.1f}%), so they are a screen at inference time, not "
        f"a substitute for the round-trip test. Center cropping and loader-order stacking caused "
        f"no damage in any cohort, because this pipeline pads symmetrically and its loader emits "
        f"slices in axial order. These are properties of this implementation, not evidence that "
        f"the steps are unnecessary."))
    add(("p",
        f"The experiment also exposed a real geometry inconsistency. All {hv['n']} NCI-ISBI "
        f"masks from the 3 T site share size, spacing, and direction with their image series, "
        f"but their stored origin is offset by exactly half a voxel along every axis. The "
        f"annotations are aligned with the image voxels, so they were copied onto the image "
        f"grid. A physical-space resampling, the usual response when 2 origins differ, would "
        f"have moved the labels and reduced their agreement with the original annotation to "
        f"Dice {d3(hv['pz_dice_mean'])} for the PZ (minimum {d3(hv['pz_dice_min'])}) and "
        f"{d3(hv['cg_dice_mean'])} for the CG (minimum {d3(hv['cg_dice_min'])}) before any model "
        f"was applied. The same images distributed in MSD are offset from the DICOM-derived "
        f"geometry by half a voxel as well."))
    def cell(c, v):
        x = c[v]
        if x["dice_CG"] == 1 and x["dice_PZ"] == 1 and abs(x["vol_err_CG_pct"]) < .01 and x["shift_CG_mm"] < .01:
            return "No damage"
        if x["dice_CG"] == 1 and x["dice_PZ"] == 1:
            ve = x["vol_err_CG_pct"]
            vtxt = "0%" if abs(ve) < 0.5 else f"{ve:+.0f}%".replace("-", "−")
            return f"Dice 1.000; shift {x['shift_CG_mm']:.0f} mm; volume {vtxt}"
        return f"Dice {d3(x['dice_CG'])}/{d3(x['dice_PZ'])}"
    faults = [("V1_no_reorientation", "Reorientation not inverted", "Round-trip"),
              ("V3_identity_affine", "Identity affine", "Geometry"),
              ("V6_spacing_only", "Spacing-only affine", "Geometry"),
              ("V4b_shuffled_order", "Shuffled slice order", "Round-trip"),
              ("V2_centre_crop", "Center crop instead of stored offsets", "—"),
              ("V4_append_order", "Loader-order stacking", "—")]
    add(("tbl", (f"Table 5. Fault injection in {mf['v0_n']} reference segmentations. Cells give "
                 f"CG/PZ Dice against the reference or, where Dice is unchanged, the CG centroid "
                 f"shift and volume error (cohort means). The last column names the check that "
                 f"flagged 100% of damaged cases in every cohort.",
        ["Fault", f"Prostate158, LPS (n={mf['n_cases']['prostate158']})",
         f"PROSTATEx, LAS (n={mf['n_cases']['prostatex']})",
         f"MSD, RAS (n={mf['n_cases']['msd_task05']})",
         f"NCI-ISBI, LPS (n={mf['n_cases']['nci_isbi_2013']})", "Detected by"],
        [[lbl, cell(p158, k), cell(px, k), cell(msd_, k), cell(nci, k), det]
         for k, lbl, det in faults])))
    add(("fig", (f"{FIG_J}/F5_detection_matrix.png",
        'Figure 4. Share of damaged cases flagged by each check, per fault and cohort. Hatched: the fault left the reconstruction unchanged (no damage).')))
    add(("fig", (f"{FIG_J}/jmir_F4_mirrored_volume_passes.png",
        'Figure 5. Correct reconstruction (top) and a mirrored one that still passes every header check (bottom).')))

    add(("h3", "Zone Volumes and PSA Density"))
    w = vs["whole"]
    add(("p",
        f"The model underestimated whole-gland volume by {mm(-w['bias'])} mL "
        f"({abs(w['pct_of_gt']):.1f}% of the reference; 95% limits of agreement "
        f"{mm(w['loa_low'])} to {mm(w['loa_high'])} mL), mostly through the PZ "
        f"({mm(vs['pz']['bias'])} mL; {sg(vs['pz']['pct_of_gt'], 1)}%) (Figure 6). This PZ "
        f"shortfall is larger than, and opposite in sign to, the difference between readers "
        f"({sg(r2['vol_PZ_r2_minus_r1']['bias_ml'], 1)} mL), so label noise alone is unlikely to "
        f"explain it. It was also unchanged from the single-seed model "
        f"({mm(one['vol_s']['pz']['bias'])} mL): ensembling improved overlap without correcting "
        f"the underestimation. In the sensitivity analysis, {psa['n_overestimated']} of "
        f"{psa['n']} patients would have had PSA density overestimated, and "
        f"{psa['n_ever_reclassified']} of {psa['n']} would cross the 0.15 ng/mL/mL threshold at "
        f"some PSA value in the 4 to 10 ng/mL gray zone. Across that zone, "
        f"{psa['share_pairs_reclassified_pct']:.1f}% of patient-PSA pairs were reclassified; the "
        f"PSA window within which a patient was misclassified had a median width of "
        f"{psa['width_median']:.1f} ng/mL (maximum {psa['width_max']:.1f})."))
    add(("fig", (f"{FIG_J}/jmir_F2_zone_volume_agreement.png",
        'Figure 6. Zone-volume agreement and PSA density error on the held-out set.')))

    add(("h3", "External Testing"))
    pr, pr1 = nc["primary"], nc1["primary"]
    r1s = nc1["secondary"]["R1"]["cg"]                    # the rule's confirmatory test
    fx, fn_ = fvs["prostatex"], fvs["nci"]
    e1x = one["external"]
    if not (e["hd95_cg"]["n_over_30"] < e1x["hd95_cg"]["n_over_30"]
            and hd["overall"]["hd95_after_mean"] < hd["overall"]["hd95_mean"]
            and hd["tail"]["multi_component_frac"] > 0.9):
        raise RuntimeError("the PROSTATEx tail text no longer matches the data")
    add(("p",
        f"On PROSTATEx, Dice was {d3(es['cg']['dice']['mean'])} (SD {d3(es['cg']['dice']['sd'])}) "
        f"for the CG and {d3(es['pz']['dice']['mean'])} (SD {d3(es['pz']['dice']['sd'])}) for "
        f"the PZ (Table 6), against {d3(es1['cg']['dice']['mean'])} and "
        f"{d3(es1['pz']['dice']['mean'])} for the single-seed model (macro Dice "
        f"{sg(fx['macro']['mean_diff'])}; better in {fx['macro']['n_better']} of "
        f"{fx['macro']['n']} cases; {fp(fx['macro']['p'])}). CG HD95 had a median of "
        f"{mm(es['cg']['hd95_mm']['median'])} mm but a "
        f"mean of {mm(es['cg']['hd95_mm']['mean'])} mm, a tail produced by distant "
        f"false-positive components. The tail was shorter than with the single-seed model, for "
        f"which the mean was {mm(es1['cg']['hd95_mm']['mean'])} mm: cases above 30 mm fell from "
        f"{e1x['hd95_cg']['n_over_30']} to {e['hd95_cg']['n_over_30']} of {e['n_cases']}. "
        f"All {e['n_cases']} reconstructions passed the geometry "
        f"check, including {eg['n_cropped']} examinations large enough to be cropped rather than "
        f"padded, none of which lost annotated anatomy. Recomputing CG Dice separately for the "
        f"anterior and posterior halves showed that the gap to the internal result lies "
        f"posteriorly ({sg(stn['gap']['cg_dice_posterior'])}) rather than anteriorly "
        f"({sg(stn['gap']['cg_dice_anterior'])}), so the anterior stroma included in this "
        f"cohort's CG class does not explain it."))
    # The rule "counts as an improvement only if HD95 falls with P<.05 and the mean Dice
    # change is no lower than -0.005" (results/lcc_prespecification.md).
    def met(x):
        return x["hd95"]["mean_diff"] < 0 and x["hd95"]["wilcoxon_p"] < 0.05 and x["dice"]["mean_diff"] >= -0.005
    dec1 = nc1["decision"]["rules"]
    dR2 = dec["R2"]["pz"]["hd95"]["mean_diff"]
    if not (dec["R1"]["passed"] and dec1["R1"]["passed"] and dec1["R2"]["passed"]
            and not dec["R2"]["passed"] and isinstance(nc["secondary"]["R2"], str) and dR2 > 0
            and met(r1) and met(r1s) and met(r2pz)
            and bs[site15]["cg_dice_none"]["mean"] < bs[site3]["cg_dice_none"]["mean"]):
        raise RuntimeError("the NCI-ISBI text no longer matches the data; rewrite it")
    add(("p",
        f"On NCI-ISBI 2013, all {nc['geometry_ok']} reconstructions passed the geometry check. "
        f"CG Dice was {d3(pr['cg_dice_none']['mean'])} (SD {d3(pr['cg_dice_none']['sd'])}; 95% "
        f"CI {d3(pr['cg_dice_none']['ci95'][0])}-{d3(pr['cg_dice_none']['ci95'][1])}) and PZ "
        f"Dice {d3(pr['pz_dice_none']['mean'])} (SD {d3(pr['pz_dice_none']['sd'])}), lower at the "
        f"1.5 T endorectal-coil site (CG {d3(bs[site15]['cg_dice_none']['mean'])}) than at the "
        f"3 T site ({d3(bs[site3]['cg_dice_none']['mean'])}). For the single-seed model, the "
        f"values were {d3(pr1['cg_dice_none']['mean'])} and {d3(pr1['pz_dice_none']['mean'])} "
        f"(change with the final model: CG {sg(fn_['cg_dice_none']['mean_diff'])}, "
        f"{fp(fn_['cg_dice_none']['p'])}; PZ {sg(fn_['pz_dice_none']['mean_diff'])}, "
        f"{fp(fn_['pz_dice_none']['p'])})."))
    add(("p",
        f"In its confirmatory test, run with the single-seed model, the prespecified rule "
        f"reduced CG HD95 from {mm(pr1['cg_hd95_none']['mean'])} to "
        f"{mm(r1s['hd95_after']['mean'])} mm ({fp(r1s['hd95']['wilcoxon_p'])}) and increased CG "
        f"Dice by {d3(r1s['dice']['mean_diff'])} ({fp(r1s['dice']['wilcoxon_p'])}), meeting its "
        f"prespecified criterion. Repeated with the final model, the rule again passed the "
        f"internal harm check (CG Dice {sg(dec['R1']['cg']['dice']['mean_diff'])}, HD95 "
        f"{sg(dec['R1']['cg']['hd95']['mean_diff'], 1)} mm) and reduced CG HD95 "
        f"from {mm(pr['cg_hd95_none']['mean'])} to {mm(r1['hd95_after']['mean'])} mm "
        f"(median {mm(pr['cg_hd95_none']['median'])} to {mm(r1['hd95_after']['median'])}; "
        f"{fp(r1['hd95']['wilcoxon_p'])}), with a CG Dice change of "
        f"{sg(r1['dice']['mean_diff'])} ({fp(r1['dice']['wilcoxon_p'])}). Cases with CG HD95 "
        f"above 30 mm fell from {nc['cg_tail_before']} "
        f"to {nc['cg_tail_after']} of {nc['n']}. For the PZ, the secondary rule had reduced HD95 "
        f"in the confirmatory test from {mm(pr1['pz_hd95_none']['mean'])} to "
        f"{mm(r2pz['hd95_after']['mean'])} mm ({fp(r2pz['hd95']['wilcoxon_p'])}), with a Dice "
        f"change of {sg(r2pz['dice']['mean_diff'])}. With the final model it failed the internal "
        f"harm check, because validation PZ HD95 increased by {mm(dR2)} mm, and it was "
        f"therefore not applied, as prespecified."))
    add(("p",
        f"On the {ov['n_two']} external images annotated independently for both NCI-ISBI and "
        f"MSD, the 2 annotation sets agreed with Dice {d3(ov['msd_vs_nci_cg'])} for the CG and "
        f"{d3(ov['msd_vs_nci_pz'])} for the PZ, close to the Prostate158 inter-reader values. "
        f"The final model reached {d3(ov['model_vs_nci_cg'])} and {d3(ov['model_vs_nci_pz'])} "
        f"against the NCI-ISBI annotation. Unlike on the held-out set, its gap to "
        f"inter-annotator agreement was significant out of "
        f"domain, at {d3(-ov['gap_cg'])} for the CG and {d3(-ov['gap_pz'])} for the PZ (both "
        f"{fp(max(ov['gap_cg_p'], ov['gap_pz_p']))}"
        f"{'' if max(ov['gap_cg_p'], ov['gap_pz_p']) < 0.001 else ' or lower'})."))
    if max(ov["gap_cg_p"], ov["gap_pz_p"]) >= 0.05 or ov["gap_cg"] >= 0 or ov["gap_pz"] >= 0:
        raise RuntimeError("the 2-annotation text assumes a significant out-of-domain gap")
    def nrow(name, key, f=d3, src=None, bsrc=None):
        src, bsrc = src or (es, pr), bsrc or bs
        dd = lambda x: f"{f(x['mean'])} ({f(x['sd'])})"
        return [name, dd(src[0][key[0]][key[1]]) if key[0] in ("cg", "pz") else "",
                dd(src[1][key[2]]), dd(bsrc[site15][key[2]]), dd(bsrc[site3][key[2]])]
    one_src = dict(src=(es1, pr1), bsrc=nc1["by_site"])
    add(("tbl", (f"Table 6. External testing of the final model, mean (SD), with Dice of the "
                 f"single-seed model for comparison. The rule "
                 f"(largest connected CG component) was prespecified for NCI-ISBI 2013; for "
                 f"PROSTATEx the same computation is post hoc and shown for comparison only.",
        ["Metric", f"PROSTATEx (n={e['n_cases']})", f"NCI-ISBI, all (n={nc['n']})",
         f"NCI-ISBI, 1.5 T (n={nc['n_sites'][site15]})",
         f"NCI-ISBI, 3 T (n={nc['n_sites'][site3]})"],
        [nrow("CG Dice", ("cg", "dice", "cg_dice_none")),
         nrow("PZ Dice", ("pz", "dice", "pz_dice_none")),
         nrow("CG Dice, single-seed model", ("cg", "dice", "cg_dice_none"), **one_src),
         nrow("PZ Dice, single-seed model", ("pz", "dice", "pz_dice_none"), **one_src),
         nrow("CG HD95, mm", ("cg", "hd95_mm", "cg_hd95_none"), mm),
         nrow("PZ HD95, mm", ("pz", "hd95_mm", "pz_hd95_none"), mm),
         ["CG HD95 after the rule, mm", f"{mm(hd['overall']['hd95_after_mean'])} (post hoc)",
          f"{mm(r1['hd95_after']['mean'])} ({mm(r1['hd95_after']['sd'])})",
          mm(bs[site15]["cg_hd95_R1_mean"]), mm(bs[site3]["cg_hd95_R1_mean"])],
         ["CG Dice after the rule", f"{d3(hd['overall']['dice_after_mean'])} (post hoc)",
          f"{d3(r1['dice_after']['mean'])} ({d3(r1['dice_after']['sd'])})", "—", "—"]])))
    add(("fig", (f"{FIG_J}/jmir_F3_external_validation.png",
        'Figure 7. External testing on PROSTATEx. Error bars: SD.')))

    add(("h3", "Gland Size"))
    add(("p",
        f"On PROSTATEx, CG Dice correlated with reference CG volume (Spearman ρ="
        f"{gs['external']['cg']['corr']['spearman_r']:.2f}; {fp(gs['external']['cg']['corr']['spearman_p'])}), "
        f"from {d3(gs['external']['cg']['tertiles'][0]['dice_mean'])} in the smallest volume "
        f"tertile to {d3(gs['external']['cg']['tertiles'][2]['dice_mean'])} in the largest. The "
        f"PZ showed the same direction more weakly (ρ="
        f"{gs['external']['pz']['corr']['spearman_r']:.2f}; "
        f"{fp(gs['external']['pz']['corr']['spearman_p'])}). The internal cohort pointed the same "
        f"way for the CG (ρ={gs['internal']['cg']['corr']['spearman_r']:.2f}; "
        f"{fp(gs['internal']['cg']['corr']['spearman_p'])}) but is underpowered with 19 cases. "
        f"Relative volume error did not grow with gland size on the internal cohort (ρ="
        f"{gs['volume_error_vs_size']['spearman_r']:.2f}; "
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
            raise RuntimeError("the PI-CAI Results text no longer matches summary.json")
        add(("h3", "PSA Density on PI-CAI"))
        add(("p",
            f"Of {pc['n_studies']:,} studies from {pc['n_patients']:,} patients, "
            f"{ex['later study of the same patient']} later studies of the same patients and "
            f"{ex['no PSA']} patients without a PSA value were excluded, which left "
            f"{pc['n_eligible']:,} patients ({pc['centers']['RUMC']} from {cn['RUMC']}, "
            f"{pc['centers']['ZGT']} from {cn['ZGT']}, and {pc['centers']['PCNN']} from "
            f"{cn['PCNN']}), of whom {pc['n_cspca']} "
            f"({pc['n_cspca'] / pc['n_eligible'] * 100:.1f}%) had csPCa. All "
            f"{pc['n_converted']:,} images were converted, with a largest corner difference "
            f"below 0.001 mm. Every reconstruction passed the geometry check, and the round-trip "
            f"test restored {rt['passed']:,} of {rt['n_tested']:,} Bosma22b masks exactly. The "
            f"remaining mask extends beyond the foreground crop and is named in the label "
            f"repository as a faulty segmentation [@picai_labels]; the study stayed in every "
            f"analysis. The model returned an empty segmentation for {pcr['n_empty_model']} "
            f"patient. Inference ran in {len(mg_['runs'])} sessions from 1 archive of code and "
            f"weights, and {mg_['n_also_inferred_elsewhere']} studies that were inferred in 2 "
            f"sessions gave identical volumes."))
        add(("p",
            f"The primary comparison included the {pp['n']:,} patients with a reported volume "
            f"({pp['n_pos']} with csPCa); {ex['no reported prostate volume']} had none. The AUC "
            f"for csPCa was {d3(pp['auc_a'])} (95% CI {d3(pp['auc_a_ci95'][0])}-"
            f"{d3(pp['auc_a_ci95'][1])}) for PSA density from the model's volume and "
            f"{d3(pp['auc_b'])} ({d3(pp['auc_b_ci95'][0])}-{d3(pp['auc_b_ci95'][1])}) for PSA "
            f"density from the reported volume, a difference of {dci(pp)} (Table 7). The lower "
            f"bound exceeded the margin of {mgn}, so noninferiority was shown; it did not "
            f"exceed 0, so superiority was not. The difference was {dci(sv['rule_R1'])} with "
            f"rule R1. No study failed the geometry check, so that sensitivity analysis did not "
            f"apply."))
        add(("p",
            f"Of the {pc['n_eligible']:,} studies, {pdup['n_studies']} duplicated an image "
            f"already used in this study ({pdup['by_source']['PROSTATEx']} in PROSTATEx and "
            f"{pdup['by_source']['NCI-ISBI']} in NCI-ISBI 2013, with "
            f"{pdup['by_source']['PROSTATEx'] + pdup['by_source']['NCI-ISBI'] - pdup['n_studies']} "
            f"in both); all were from RUMC. None of these images was used for training. Without "
            f"the {sv['without_duplicates']['n_removed']} of them in the primary comparison, "
            f"the difference was {dci(sv['without_duplicates'])} in "
            f"{sv['without_duplicates']['n']:,} patients. By center, the difference was "
            f"{dci(bc['RUMC'])} at RUMC, {dci(bc['ZGT'])} at ZGT, and {dci(bc['PCNN'])} at PCNN. "
            f"At PCNN both AUCs were low ({d3(bc['PCNN']['auc_a'])} and "
            f"{d3(bc['PCNN']['auc_b'])}) and the lower bound fell below the margin; the "
            f"by-center analysis was descriptive and was not sized to test noninferiority."))
        add(("p",
            f"The model's volume was smaller than the reported volume by a mean of "
            f"{mm(-pv['bias'])} mL (95% limits of agreement {ci(pv['loa95'], mm)} mL; median "
            f"relative difference {mm(pv['median_rel_diff'] * 100)}%; Spearman ρ="
            f"{pv['spearman']:.2f}) and smaller than the Bosma22b volume by {mm(-pb['bias'])} mL "
            f"({ci(pb['loa95'], mm)} mL; {mm(pb['median_rel_diff'] * 100)}%; ρ="
            f"{pb['spearman']:.2f}). Whole-gland Dice against Bosma22b had a median of "
            f"{d3(dq['median'])} (IQR {d3(dq['q25_q75'][0])}-{d3(dq['q25_q75'][1])}; mean "
            f"{d3(dq['mean'])}). Smaller volumes give higher PSA density, so at "
            f"{pc['threshold']:.2f} ng/mL/mL the model's volume classified "
            f"{rc['model']['n_positive']} of {pp['n']:,} patients as positive and the reported "
            f"volume {rc['report']['n_positive']}. Sensitivity was "
            f"{d3(rc['model']['sensitivity'])} against {d3(rc['report']['sensitivity'])}, and "
            f"specificity {d3(rc['model']['specificity'])} against "
            f"{d3(rc['report']['specificity'])}. The 2 volumes classified {rc['discordant']} "
            f"patients differently: {rc['model_pos_report_neg']} were positive with the model's "
            f"volume only and {rc['model_neg_report_pos']} with the reported volume only. PSA "
            f"density from the Bosma22b volume had an AUC of {d3(bp['auc_a'])}, "
            f"{dci(bp)} above that from the reported volume."))

        def prow(name, auc, lo_hi, diff, r):
            return [name, f"{d3(auc)} ({d3(lo_hi[0])}-{d3(lo_hi[1])})", diff,
                    d3(r["sensitivity"]), d3(r["specificity"])]

        def dcell(r):
            return f"{d3(r['diff'])} ({ci(r['diff_ci95'])})"
        add(("tbl", (f"Table 7. PSA density for detecting clinically significant prostate cancer "
                     f"on PI-CAI ({pp['n']:,} patients, {pp['n_pos']} with cancer), by the source "
                     f"of the whole-gland volume. Sensitivity and specificity are at "
                     f"{pc['threshold']:.2f} ng/mL/mL.",
            ["Volume source", "AUC (95% CI)", "Difference from reported volume (95% CI)",
             "Sensitivity", "Specificity"],
            [prow("Final model", pp["auc_a"], pp["auc_a_ci95"], dcell(pp), rc["model"]),
             prow("Radiology report", pp["auc_b"], pp["auc_b_ci95"], "Reference", rc["report"]),
             prow("Bosma22b (public AI segmentation)", bp["auc_a"], bp["auc_a_ci95"], dcell(bp),
                  rc["bosma"])])))

    add(("h3", "Reporting of Reconstruction in Published Studies"))
    add(("p",
        f"Of {lit['n_records']} records, {lit['n']} studies met the criteria "
        f"({lit['dim'].get('2D', 0)} with 2D input, {lit['dim'].get('2D+3D', 0)} comparing 2D "
        f"with 3D input, and {lit['dim'].get('2.5D', 0)} with 2.5D input); the reason for each "
        f"exclusion is listed in Multimedia Appendix 4. None described how the slice outputs "
        f"were reassembled into a volume, and none of the {lit['inv_n']} that reported "
        f"resampling, resizing, or cropping described how it was undone. None reported a check "
        f"of the reconstructed volume (0/{lit['n']}; 95% CI 0%-{lit['geo_hi']*100:.1f}%). Only "
        f"2 stated the grid on which metrics were computed, in both cases the preprocessed grid "
        f"rather than that of the source image. Even so, {lit['n_depend']} reported per-volume "
        f"overlap or distances in millimeters, both of which presuppose a correctly assembled "
        f"volume."))

    # ------------------------------------------------------------- discussion
    add(("h2", "Discussion"))
    add(("h3", "Principal Results"))
    add(("p",
        f"The main finding concerns verification rather than accuracy. Across {mf['v0_n']} "
        f"reference segmentations in 4 cohorts and 3 orientation conventions, a header-based "
        f"geometry check and a voxel-level round-trip test each missed a different class of "
        f"reconstruction fault, and only their combination flagged every damaging fault, with no "
        f"false alarms. Whether a fault was visible depended on the data: the same missing "
        f"reorientation was harmless, a left-right flip, or a near-total loss of overlap in "
        f"different cohorts. A spacing-only affine kept every volume exact while moving the "
        f"gland by more than 10 cm, so neither overlap metrics nor volume plausibility would "
        f"have revealed it. In preparing a public cohort, the same reasoning revealed a "
        f"half-voxel origin convention that a standard resampling would have turned into a "
        f"Dice loss of up to {d3(1 - hv['pz_dice_min'])} for the PZ."))
    add(("p",
        f"The final 2D model performed close to expert agreement on the held-out set: "
        f"{r2['dice_CG_model_minus_interreader']['ratio_of_means']*100:.1f}% and "
        f"{r2['dice_PZ_model_minus_interreader']['ratio_of_means']*100:.1f}% of inter-reader "
        f"Dice for the CG and PZ, with neither Dice nor boundary distance significantly "
        f"different from the values between readers. With 19 cases, this is an absence of a "
        f"detected difference, not a demonstration of equivalence. Training interventions "
        f"mattered less than the variation between "
        f"patients and about as much as the variation between seeds: the 4 configurations "
        f"spanned {d3(N['arm_spread'])} macro Dice and 3 seeds of 1 configuration spanned "
        f"{d3(sd_['range_2d'])} to {d3(sd_['range_25d'])}, whereas "
        f"held-out CG Dice alone had an SD of {d3(cg['dice']['sd'])} across patients. "
        f"Augmentation and class weighting were equivalent to their comparators within "
        f"±{margin} Dice in most comparisons, the soft Dice term's overall effect was "
        f"inconclusive, with a post hoc gain confined to the PZ, and 2.5D input brought no "
        f"gain. The step that clearly improved held-out accuracy was ensembling 3 seeds with "
        f"flip averaging ({sg(ft['macro']['mean_diff'])} macro Dice over a single seed), and "
        f"the gain carried over to both external cohorts. Accuracy still "
        f"fell out of domain, most at a 1.5 T endorectal-coil site and most for the PZ. A "
        f"connected-component "
        f"rule, specified and checked internally before it was used, removed most of the "
        f"external boundary-error tail."))
    if pcr:
        add(("p",
            f"On {pp['n']:,} patients from 3 centers, PSA density computed from the model's "
            f"whole-gland volume discriminated clinically significant cancer no worse than PSA "
            f"density computed from the volume in the radiology report, within a margin fixed "
            f"before the data were downloaded. It was not better, and the model's volumes were "
            f"systematically smaller than both the reported volumes and those of a public AI "
            f"segmentation."))
    if not (fx["macro"]["mean_diff"] > 0 and fx["macro"]["p"] < 0.05
            and fn_["cg_dice_none"]["mean_diff"] > 0 and fn_["cg_dice_none"]["p"] < 0.05
            and fn_["pz_dice_none"]["mean_diff"] > 0):
        raise RuntimeError("'the gain carried over to both external cohorts' no longer holds")

    add(("h3", "Comparison With Prior Work"))
    add(("p",
        "The accuracy remains below that of the Prostate158 reference implementation, a 3D "
        "residual U-Net that reported Dice of approximately 0.877 for the CG and 0.754 for the "
        f"PZ on the same split [@adams_cbm]; the final model reached "
        f"{cg['dice']['mean']/0.877*100:.1f}% and "
        f"{pz['dice']['mean']/0.754*100:.1f}% of those values. The larger shortfall for the PZ "
        "is what the architecture predicts, since the PZ boundary runs obliquely to the axial "
        "plane, where a 2D network has no context. Supplying that context as 2.5D input did not "
        "help here, whereas earlier work on prostate MRI reported modest gains from 3 input "
        "slices [@zhang25d,hung_catnet]. Those gains were of about 0.002 to 0.025 Dice, a size "
        f"that 3 seeds per arm cannot separate from seed variation (SD "
        f"{d3(arm25['sd_macro'])} for the 2.5D arm), so our result is best read as no detectable "
        "benefit at this sample size, not as evidence against 2.5D input. It is consistent "
        "with the report that random effects during training alone produce differences in "
        "segmentation performance that often reach statistical significance [@akesson]. We do "
        "not present the model as "
        "competitive with 3D methods; its role here is a realistic source of slice-wise output."))
    add(("p",
        "Orientation errors in neuroimaging data are well documented: Glen et al [@glen] found "
        "left-right flips common enough in public MRI collections to warrant an automated "
        "check, and the conversion from DICOM to NIfTI is a recognized source of such errors "
        "[@li_dcm2niix]. Those studies concern the input data. This study extends the question "
        "to the output of a segmentation pipeline, where the error is introduced by the "
        "pipeline itself and cannot be found by inspecting the source images, and it measures "
        "which check detects which error across datasets rather than proposing a single check. "
        "Its finding that the anatomical check misses left-right flips is consistent with "
        "Glen et al, who used image-based comparison to detect such flips. Our cross-dataset "
        "result adds that the orientation convention of a benchmark determines whether a "
        "reorientation fault can be observed on it at all. The MSD prostate task, a widely used "
        "benchmark [@antonelli], is stored in the orientation most networks use internally, "
        "which makes it the cohort least able to reveal this fault."))
    add(("p",
        "Reporting guidelines ask for preprocessing details and code [@claim,tripod_ai], and a "
        "study can satisfy them while emitting a mirrored volume, because such a volume has a "
        "correct header, shape, spacing, and label set. In our audit, none of the "
        f"{lit['n']} published 2D or 2.5D prostate studies reported a check of its "
        f"reconstruction, although {lit['n_depend']} reported results that depend on one. Work "
        "that adapts 2D and video models to "
        "3D medical data [@yamagishi] ends in the same reassembly step. We suggest that studies "
        "deriving 3D results from slice-wise output report both a header check and a "
        "voxel-level round-trip test; the latter costs a single forward and inverse pass of an "
        "existing annotation."))

    add(("h3", "Clinical Implications"))
    pzb, pzb1 = vs["pz"]["bias"], one["vol_s"]["pz"]["bias"]
    if not (pzb < 0 and pzb1 < 0 and abs(pzb - pzb1) < 0.5 and fvs["test"]["pz_dice"]["mean_diff"] > 0
            and psa["n_ever_reclassified"] > psa["n"] / 2):
        raise RuntimeError("the Clinical Implications paragraph no longer matches the data")
    add(("p",
        "Dice is a research metric; volume is how zonal segmentation reaches clinical use. The "
        "model's PZ underestimation propagated into PSA density. Ensembling, which raised PZ "
        f"Dice, left this bias unchanged ({mm(pzb)} vs {mm(pzb1)} mL), so it appears to be "
        "systematic rather than an accident of 1 training run, and better overlap should not be "
        "assumed to correct volume. In the sensitivity analysis, "
        "most patients would cross the 0.15 ng/mL/mL threshold somewhere in the diagnostic "
        "gray zone, although the PSA window in which this happens was narrow for most of them. The error runs toward overestimated density. Combined with limits "
        "of agreement that are wide relative to the gland, this means volumes from this model "
        "should not be used for PSA density without review. Reporting volumes with limits of "
        "agreement makes this visible; reporting Dice alone does not."))
    if pcr:
        add(("p",
            f"The PI-CAI analysis, with measured PSA values, refines this conclusion without "
            f"reversing it. The AUC depends only on how patients are ranked, and an "
            f"underestimation that runs in the same direction for most patients leaves the ranking "
            f"almost intact (Spearman ρ={pv['spearman']:.2f} between the 2 volumes); this is why "
            f"discrimination was preserved. A threshold is different. Volumes that were "
            f"{mm(-pv['median_rel_diff'] * 100)}% smaller at the median moved many patients "
            f"above {pc['threshold']:.2f} ng/mL/mL, which raised sensitivity by "
            f"{(rc['model']['sensitivity'] - rc['report']['sensitivity']) * 100:.0f} percentage "
            f"points and lowered specificity by "
            f"{(rc['report']['specificity'] - rc['model']['specificity']) * 100:.0f}. The "
            f"model's volume could therefore stand in for the reported volume when patients are "
            f"ranked by risk, but a threshold established with reported volumes should not be "
            f"applied to it unchanged. We did not derive a corrected threshold or a volume "
            f"correction, because neither was prespecified and one fitted on these data would "
            f"need its own test. The reported volume is not a reference standard either: PSA "
            f"density from a second automatic segmentation was slightly better than PSA "
            f"density from the report, which suggests that the measurement in the report is "
            f"itself a source of error."))

    add(("h3", "Limitations"))
    add(("p",
        "The study has several strengths: selection rules that used validation data only, "
        "seed replicates under a rule fixed before they were run, a model frozen and "
        "identified by its weight hashes, an expert second "
        "reader as benchmark, a post-processing rule specified before its confirmatory test, "
        "and a fault-injection design in which every difference is attributable to the fault "
        "alone. It also has limitations. The development cohort comprises 158 examinations from 1 institution, and the held-out "
        "split has 19 cases, so internal estimates are imprecise; CIs are reported for that "
        "reason. Only the selected configuration was replicated across seeds, with 3 seeds per "
        "arm; the baseline, E2, and E3 were each trained once, so the equivalence results "
        "describe those runs rather than the configurations in general, and 3 seeds give only a "
        "rough estimate of seed variation [@akesson]. The held-out split had been used twice "
        "during earlier pipeline development with a different model, and once for the "
        "single-seed model before the seed extension was written. Every selection rule used "
        "validation data only, but the extension was designed with the earlier test result "
        "known, and the test set was not untouched. The connected-component rule was therefore "
        "tested on NCI-ISBI 2013, which had not been used before; that confirmatory test "
        "belongs to the single-seed model, and the final-model analysis repeats it on a cohort "
        "that was by then no longer unseen. The final model also costs 6 forward passes per "
        "slice (3 members, each on the slice and its mirror image), against 1 for a single "
        "network."))
    if not (abs(nc["excl_px"]["cg_dice"] - pr["cg_dice_none"]["mean"]) <= 0.001
            and abs(nc["excl_px"]["cg_hd95_R1"] - r1["hd95_after"]["mean"]) <= 0.1):
        raise RuntimeError("'changed ... by no more than 0.001 Dice and 0.1 mm' no longer holds")
    add(("p",
        f"The external cohorts were annotated under different protocols, so external "
        f"differences combine model generalization with annotation differences, and errors in a "
        f"reference standard can shift measured performance by themselves [@wang_labels]; the "
        f"2-annotation comparison bounds but does not remove this. The cohorts also partly "
        f"overlap ({ov['n_msd']} shared images between NCI-ISBI and MSD, and 1 between NCI-ISBI "
        f"and PROSTATEx); this affects no training data, and excluding the shared PROSTATEx case "
        f"changed the NCI-ISBI results by no more than 0.001 Dice and 0.1 mm (CG Dice "
        f"{d3(nc['excl_px']['cg_dice'])} vs {d3(pr['cg_dice_none']['mean'])}; "
        f"CG HD95 after the rule {mm(nc['excl_px']['cg_hd95_R1'])} vs "
        f"{mm(r1['hd95_after']['mean'])} mm). The fault injection uses a fixed set of faults "
        f"implemented in 1 pipeline; other pipelines may fail in other ways, and the 2 harmless "
        f"faults here may not be harmless elsewhere. "
        + ((f"The PSA density analysis on Prostate158 uses a range of PSA values; measured "
            f"values were available only on PI-CAI. That cohort comes from 3 centers in 1 "
            f"country. Part of its negative studies were confirmed by an MRI reading without "
            f"biopsy, so some cancers may have been missed; this verification bias applies "
            f"equally to both PSA densities, but its effect on the AUCs themselves is unknown. "
            f"The reported volumes were measured in routine care by methods that are not "
            f"recorded. {pdup['n_studies']} PI-CAI studies duplicate images of our external "
            f"cohorts, so those cohorts are not independent of PI-CAI, although excluding the "
            f"duplicates did not change the result. The noninferiority margin was our own "
            f"choice, noninferiority of the AUC says nothing about calibration, and at 1 of the "
            f"3 centers the lower confidence bound fell below the margin. No reader study was "
            f"performed, so no claim is made about patient outcomes. ") if pcr else
           (f"The PSA analysis uses a range of PSA values "
            f"rather than measured values, and no reader study was performed, so no claim is made "
            f"about clinical utility or patient outcomes. ")) +
        f"Accuracy also depended on gland size, "
        f"which limits use in small glands. Finally, the literature audit covered only "
        f"open-access articles in PubMed Central, used title-level search terms, and had 1 "
        f"reviewer, and it measures what studies reported, not what they did."))

    add(("h3", "Conclusions"))
    add(("p",
        "Reconstructing slice-wise segmentation into a 3D volume can fail in ways that leave a "
        "valid file, and in this study a header check and a voxel-level round-trip test each "
        "missed a different class of failure. Only their combination detected every damaging "
        "fault across 4 public cohorts, and whether a fault was observable at all depended on "
        "the orientation in which the data were stored. An ensemble of 3 seeds of the 2D U-Net "
        "segmented both zones without a detected difference from expert agreement on a small "
        "held-out set, while differences between training configurations were about as large "
        "as differences between seeds, and 2.5D input gave no benefit. A prespecified "
        "connected-component rule removed most of the external boundary error. Studies that "
        "derive 3D results from 2D segmentation should report both verification checks and "
        "should report volumes with limits of agreement alongside Dice."
        + (" On a multicenter cohort not used in development, PSA density from the model's volume was "
           "noninferior to PSA density from reported volumes for detecting clinically "
           "significant cancer, but the model's smaller volumes shifted the operating point at "
           "the usual threshold, so the 2 are interchangeable for ranking and not for "
           "threshold-based decisions." if pcr else "")))

    # ----------------------------------------------------------- back matter
    add(("h2", "Acknowledgments"))
    add(("p",
        "The authors thank the creators of Prostate158, PROSTATEx and its zonal annotations, "
        + ("NCI-ISBI 2013, the Medical Segmentation Decathlon, and the PI-CAI challenge for "
           "making their data public. " if pc else
           "NCI-ISBI 2013, and the Medical Segmentation Decathlon for making their data public. ") +
        "Generative artificial intelligence (Claude, Anthropic) was used to assist with code "
        "development, analysis scripting, figure preparation, and drafting and language editing "
        "of the manuscript text. All generated content was reviewed and verified by the authors, who "
        "take full responsibility for the work. The authors also thank Universitas Multimedia "
        "Nusantara for its support of this study."))
    add(("h2", "Funding"))
    add(("p", "This work was supported by Universitas Multimedia Nusantara. The funder had no role in the study design, data "
              "analysis, interpretation, or the decision to publish."))
    add(("h2", "Data Availability"))
    add(("p",
        "All datasets analyzed are publicly available from their original sources "
        f"[@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli{',picai_data,picai_labels' if pc else ''}]. "
        "They were used as provided; all "
        "preprocessing applied in this study is described in the Methods section."
        + (" The PI-CAI labels and clinical values were used at the repository commit recorded "
           "in Multimedia Appendix 5." if pc else "")))
    add(("h2", "Code Availability"))
    add(("p",
        f"The analysis code, configuration files, {'the 3' if pc else 'both'} prespecifications, and the scripts that "
        "generate every number in this article are publicly available in the project "
        "repository [@repo]. The frozen final model is identified there by the SHA-256 hashes "
        "of its 3 members' weights, together with the record of its selection. The 3 "
        "verification levels are also provided as a standalone script that "
        "depends only on NumPy and NiBabel (tools/verify_reconstruction.py), so that they can "
        "be applied to other pipelines."))
    add(("h2", "Authors' Contributions"))
    add(("p",
        "Conceptualization: BAS, EES. Methodology: BAS. Software: BAS. Formal analysis: BAS. "
        "Validation: BAS. Visualization: BAS. Writing - original draft: BAS. Writing - review "
        "and editing: EES, DA, MW, BAS. Supervision: EES."))
    add(("h2", "Conflicts of Interest"))
    add(("p", "None declared."))
    add(("h2", "Multimedia Appendix 1"))
    add(("p", "CLAIM 2024 checklist."))
    add(("h2", "Multimedia Appendix 2"))
    add(("p", "Prespecification of the connected-component rule and the NCI-ISBI 2013 analysis, "
              "with its erratum and the recorded file hash; and prespecification of the "
              "seed-replicate and 2.5D extension, with its recorded file hash and the selection "
              "record."))
    add(("h2", "Multimedia Appendix 3"))
    add(("p", "Supplementary results: training histories, per-case results, results of the "
              "single-seed and final models side by side, cohort geometry, and additional "
              "figures."))
    add(("h2", "Multimedia Appendix 4"))
    add(("p", "Audit of published reporting: protocol with its file hash, search record, "
              "screening decisions with reasons, and per-study coding with supporting "
              "quotations."))
    if pc:
        add(("h2", "Multimedia Appendix 5"))
        add(("p", "PSA density on PI-CAI: prespecification with its recorded file hash, erratum, "
                  "label repository commit, cohort flow, conversion and verification records, and "
                  "per-study results."))
    add(("h2", "References"))
    add(("refs", REFERENCES))
    add(("h2", "Abbreviations"))
    abbr = [("2D", "two-dimensional"), ("3D", "three-dimensional"),
            ("CG", "central gland"), ("CLAIM", "Checklist for Artificial Intelligence in Medical "
                                             "Imaging"),
            ("HD95", "95th-percentile Hausdorff distance"),
            ("LAS/LPS/RAS", "orientation conventions naming the directions of the first 3 array "
                            "axes (left/right, anterior/posterior, superior)"),
            ("MRI", "magnetic resonance imaging"), ("MSD", "Medical Segmentation Decathlon"),
            ("PSA", "prostate-specific antigen"), ("PZ", "peripheral zone"),
            ("TOST", "two one-sided tests")]
    if pc:
        abbr += [("AUC", "area under the receiver operating characteristic curve"),
                 ("csPCa", "clinically significant prostate cancer"),
                 ("ISUP", "International Society of Urological Pathology"),
                 ("PI-CAI", "Prostate Imaging: Cancer AI"),
                 ("PI-RADS", "Prostate Imaging-Reporting and Data System")]
        abbr.sort(key=lambda kv: kv[0].lower())
    for k, v in abbr:
        add(("p", f"{k}: {v}"))

    # Define the abbreviations shown inside each figure at the end of its caption,
    # as published JMIR articles do.
    tails = {
        1: ("AUC: area under the receiver operating characteristic curve; " if pc else "")
           + "CG: central gland; LAS/LPS/RAS: orientation conventions; MSD: Medical Segmentation "
           "Decathlon; " + ("PI-CAI: Prostate Imaging: Cancer AI; " if pc else "")
           + "PSA: prostate-specific antigen; PZ: peripheral zone; TOST: two one-sided "
           "tests.",
        2: "bg: background; CG: central gland; PZ: peripheral zone; T2W: T2-weighted.",
        3: "CG: central gland; NIfTI: Neuroimaging Informatics Technology Initiative; PZ: "
           "peripheral zone; RAS: right-anterior-superior orientation.",
        4: "CG: central gland; LAS/LPS/RAS: orientation conventions; MSD: Medical Segmentation "
           "Decathlon; PZ: peripheral zone.",
        5: "CG: central gland; PZ: peripheral zone.",
        6: "PSA: prostate-specific antigen.",
        7: "CG: central gland; HD95: 95th-percentile Hausdorff distance; PZ: peripheral zone.",
    }
    for i, (k, v) in enumerate(B):
        if k == "fig":
            n = int(v[1].split(".")[0].split()[-1])
            if n in tails:
                B[i] = (k, (v[0], f"{v[1]} {tails[n]}"))
    return B


import manuscript_content_en as _en  # noqa: E402

_KEEP = ["sung", "turkbey", "weinreb", "mcneal", "ronneberger", "isensee", "litjens_survey",
         "bardis", "astono", "cheng", "claim", "tripod_ai", "adams_cbm", "adams_dib", "litjens",
         "clark", "cuocolo_ejr", "cuocolo_jmri", "taha", "maierhein", "bland", "nordstrom",
         "monai", "yamagishi"]
# JMIR reference style: sentence-case titles, [doi] and [Medline] tags.
REFERENCES = {
    "sung": "Bray F, Laversanne M, Sung H, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin 2024 May;74(3):229-263. [doi: 10.3322/caac.21834] [Medline: 38572751]",
    "zhang25d": "Zhang Y, Liao Q, Ding L, Zhang J. Bridging 2D and 3D segmentation networks for computation-efficient volumetric medical image segmentation: an empirical study of 2.5D solutions. Comput Med Imaging Graph 2022 Jul;99:102088. [doi: 10.1016/j.compmedimag.2022.102088] [Medline: 35780703]",
    "hung_catnet": "Hung ALY, Zheng H, Miao Q, Raman SS, Terzopoulos D, Sung K. CAT-Net: a cross-slice attention transformer model for prostate zonal segmentation in MRI. IEEE Trans Med Imaging 2023 Jan;42(1):291-303. [doi: 10.1109/TMI.2022.3211764] [Medline: 36194719]",
    "cabello": "Cabello JB, Ruiz Garcia V, Torralba M, Maldonado Fernandez M, Ubeda M, Ansuategui E, et al. Critical appraisal tools for evaluating artificial intelligence in clinical studies: scoping review. J Med Internet Res 2025 Dec;27:e77110. [doi: 10.2196/77110] [Medline: 41359958]",
    "feng_mvi": "Feng W, Qu B, Han S. Accuracy of medical image–based deep learning for detecting microvascular invasion in hepatocellular carcinoma: systematic review and meta-analysis. J Med Internet Res 2026 Mar;28:e82000. [doi: 10.2196/82000] [Medline: 41814977]",
    "akesson": "Åkesson J, Töger J, Heiberg E. Random effects during training: implications for deep learning-based medical image segmentation. Comput Biol Med 2024 Sep;180:108944. [doi: 10.1016/j.compbiomed.2024.108944] [Medline: 39096609]",
    "nikolov": "Nikolov S, Blackwell S, Zverovitch A, Mendes R, Livne M, De Fauw J, et al. Clinically applicable segmentation of head and neck anatomy for radiotherapy: deep learning algorithm development and validation study. J Med Internet Res 2021 Jul;23(7):e26151. [doi: 10.2196/26151] [Medline: 34255661]",
    "wang_labels": "Wang Y, Han X, Li C, Luo L, Yin Q, Zhang J, et al. Impact of gold-standard label errors on evaluating performance of deep learning models in diabetic retinopathy screening: nationwide real-world validation study. J Med Internet Res 2024 Aug;26:e52506. [doi: 10.2196/52506] [Medline: 39141915]",
    "turkbey": "Turkbey B, Rosenkrantz AB, Haider MA, Padhani AR, Villeirs G, Macura KJ, et al. Prostate Imaging Reporting and Data System version 2.1: 2019 update of Prostate Imaging Reporting and Data System version 2. Eur Urol 2019 Sep;76(3):340-351. [doi: 10.1016/j.eururo.2019.02.033] [Medline: 30898406]",
    "weinreb": "Weinreb JC, Barentsz JO, Choyke PL, Cornud F, Haider MA, Macura KJ, et al. PI-RADS Prostate Imaging - Reporting and Data System: 2015, version 2. Eur Urol 2016 Jan;69(1):16-40. [doi: 10.1016/j.eururo.2015.08.052] [Medline: 26427566]",
    "mcneal": "McNeal JE. The zonal anatomy of the prostate. Prostate 1981;2(1):35-49. [doi: 10.1002/pros.2990020105] [Medline: 7279811]",
    "ronneberger": "Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: Navab N, Hornegger J, Wells W, Frangi A, editors. Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015. Cham: Springer; 2015:234-241. [doi: 10.1007/978-3-319-24574-4_28]",
    "isensee": "Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 2021 Feb;18(2):203-211. [doi: 10.1038/s41592-020-01008-z] [Medline: 33288961]",
    "litjens_survey": "Litjens G, Kooi T, Bejnordi BE, Setio AAA, Ciompi F, Ghafoorian M, et al. A survey on deep learning in medical image analysis. Med Image Anal 2017 Dec;42:60-88. [doi: 10.1016/j.media.2017.07.005] [Medline: 28778026]",
    "bardis": "Bardis M, Houshyar R, Chantaduly C, Tran-Harding K, Ushinsky A, Chahine C, et al. Segmentation of the prostate transition zone and peripheral zone on MR images with deep learning. Radiol Imaging Cancer 2021 May;3(3):e200024. [doi: 10.1148/rycan.2021200024] [Medline: 33929265]",
    "astono": "Astono IP, Welsh JS, Chalup S, Greer P. Optimisation of 2D U-Net model components for automatic prostate segmentation on MRI. Appl Sci 2020;10(7):2601. [doi: 10.3390/app10072601]",
    "cheng": "Cheng R, Lay N, Roth HR, Turkbey B, Jin D, Gandler W, et al. Fully automated prostate whole gland and central gland segmentation on MRI using holistically nested networks with short connections. J Med Imaging (Bellingham) 2019 Apr;6(2):024007. [doi: 10.1117/1.JMI.6.2.024007] [Medline: 31205977]",
    "claim": "Tejani AS, Klontzas ME, Gatti AA, Mongan JT, Moy L, Park SH, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. Radiol Artif Intell 2024 Jul;6(4):e240300. [doi: 10.1148/ryai.240300] [Medline: 38809149]",
    "tripod_ai": "Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024 Apr 16;385:e078378. [doi: 10.1136/bmj-2023-078378] [Medline: 38626948]",
    "adams_cbm": "Adams LC, Makowski MR, Engel G, Rattunde M, Busch F, Asbach P, et al. Prostate158 - an expert-annotated 3T MRI dataset and algorithm for prostate cancer detection. Comput Biol Med 2022 Sep;148:105817. [doi: 10.1016/j.compbiomed.2022.105817] [Medline: 35841780]",
    "adams_dib": "Adams LC, Makowski MR, Engel G, Rattunde M, Busch F, Asbach P, et al. Dataset of prostate MRI annotated for anatomical zones and cancer. Data Brief 2022 Dec;45:108739. [doi: 10.1016/j.dib.2022.108739] [Medline: 36426089]",
    "litjens": "Litjens G, Debats O, Barentsz J, Karssemeijer N, Huisman H. Computer-aided detection of prostate cancer in MRI. IEEE Trans Med Imaging 2014 May;33(5):1083-1092. [doi: 10.1109/TMI.2014.2303821] [Medline: 24770913]",
    "clark": "Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, et al. The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository. J Digit Imaging 2013 Dec;26(6):1045-1057. [doi: 10.1007/s10278-013-9622-7] [Medline: 23884657]",
    "cuocolo_ejr": "Cuocolo R, Stanzione A, Castaldo A, De Lucia DR, Imbriaco M. Quality control and whole-gland, zonal and lesion annotations for the PROSTATEx challenge public dataset. Eur J Radiol 2021 May;138:109647. [doi: 10.1016/j.ejrad.2021.109647] [Medline: 33721767]",
    "cuocolo_jmri": "Cuocolo R, Comelli A, Stefano A, Benfante V, Dahiya N, Stanzione A, et al. Deep learning whole-gland and zonal prostate segmentation on a public MRI dataset. J Magn Reson Imaging 2021 Aug;54(2):452-459. [doi: 10.1002/jmri.27585] [Medline: 33634932]",
    "taha": "Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging 2015 Aug 12;15:29. [doi: 10.1186/s12880-015-0068-x] [Medline: 26263899]",
    "maierhein": "Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods 2024 Feb;21(2):195-212. [doi: 10.1038/s41592-023-02151-z] [Medline: 38347141]",
    "bland": "Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986 Feb 8;1(8476):307-310. [Medline: 2868172]",
    "nordstrom": "Nordström T, Akre O, Aly M, Grönberg H, Eklund M. Prostate-specific antigen (PSA) density in the diagnostic algorithm of prostate cancer. Prostate Cancer Prostatic Dis 2018 Apr;21(1):57-63. [doi: 10.1038/s41391-017-0024-7] [Medline: 29259293]",
    "monai": "Cardoso MJ, Li W, Brown R, Ma N, Kerfoot E, Wang Y, et al. MONAI: an open-source framework for deep learning in healthcare. arXiv. Preprint posted online on November 4, 2022. [doi: 10.48550/arXiv.2211.02701]",
    "yamagishi": "Yamagishi Y, Hanaoka S, Kikuchi T, Nakao T, Nakamura Y, Nomura Y, et al. Using Segment Anything Model 2 for zero-shot 3D segmentation of abdominal organs in computed tomography scans to adapt video tracking capabilities for 3D medical imaging: algorithm development and validation. JMIR AI 2025 Apr 29;4:e72109. [doi: 10.2196/72109] [Medline: 40611505]",
    "glen": "Glen DR, Taylor PA, Buchsbaum BR, Cox RW, Reynolds RC. Beware (surprisingly common) left-right flips in your MRI data: an efficient and robust method to check MRI dataset consistency using AFNI. Front Neuroinform 2020;14:18. [doi: 10.3389/fninf.2020.00018] [Medline: 32528270]",
    "li_dcm2niix": "Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods 2016 May 1;264:47-56. [doi: 10.1016/j.jneumeth.2016.03.001] [Medline: 26945974]",
    "antonelli": "Antonelli M, Reinke A, Bakas S, Farahani K, Kopp-Schneider A, Landman BA, et al. The Medical Segmentation Decathlon. Nat Commun 2022 Jul 15;13(1):4128. [doi: 10.1038/s41467-022-30695-9] [Medline: 35840566]",
    "bloch_nci": "Bloch N, Madabhushi A, Huisman H, Freymann J, Kirby J, Grauer M, et al. NCI-ISBI 2013 challenge: automated segmentation of prostate structures. The Cancer Imaging Archive. 2015. [doi: 10.7937/K9/TCIA.2015.zF0vlOPv]",
    "holm": "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat 1979;6(2):65-70.",
    "schuirmann": "Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm 1987 Dec;15(6):657-680. [doi: 10.1007/BF01068419] [Medline: 3450848]",
    "lakens": "Lakens D. Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Soc Psychol Personal Sci 2017 May;8(4):355-362. [doi: 10.1177/1948550617697177] [Medline: 28736600]",
    "picai_lancet": "Saha A, Bosma JS, Twilt JJ, van Ginneken B, Bjartell A, Padhani AR, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. Lancet Oncol 2024 Jul;25(7):879-887. [doi: 10.1016/S1470-2045(24)00220-1] [Medline: 38876123]",
    "picai_data": "Saha A, Twilt JJ, Bosma JS, van Ginneken B, Yakar D, Elschot M, et al. The PI-CAI challenge: public training and development dataset (version 2.0). Zenodo. 2022 Jun 10. [doi: 10.5281/zenodo.6624726]",
    "picai_labels": "Diagnostic Image Analysis Group, Radboud University Medical Center. picai_labels: annotations and clinical information for the PI-CAI public training and development dataset. GitHub. URL: https://github.com/DIAGNijmegen/picai_labels [accessed 2026-11-12]",
    "repo": "Sandro BA. prostate-mri-segmentation-thesis. GitHub. URL: https://github.com/amadeussandro/prostate-mri-segmentation-thesis [accessed 2026-11-10] [[TO FILL: Zenodo DOI]]",
}
assert set(_KEEP) <= set(REFERENCES), set(_KEEP) - set(REFERENCES)
