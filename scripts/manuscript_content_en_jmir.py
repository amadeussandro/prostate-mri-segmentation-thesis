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

AUTHORS = ["Benedict Amadeus Sandro, BCS [[TO FILL: degree or omit]]",
           "Eunike Endariahna Surbakti, MTI"]
AFFIL = ["Department of Informatics, Faculty of Engineering and Informatics, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Corresponding Author: [[TO FILL: name of corresponding author]], Department of "
        "Informatics, Faculty of Engineering and Informatics, Universitas Multimedia Nusantara, "
        "Jl. Scientia Boulevard, Gading Serpong, Tangerang, Banten 15810, Indonesia. "
        "Phone: [[TO FILL]]. Email: [[TO FILL]]. ORCID: Benedict Amadeus Sandro "
        "0009-0000-1459-9139; Eunike Endariahna Surbakti [[TO FILL]]")

FIG_RQ2 = "results/figures_rq2"
FIG_SUP = "results/figures_supervisor"
FIG_J = "results/figures_jmir"


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


def sg(x: float, nd: int = 3) -> str:
    """Signed value with a true minus sign."""
    return f"{x:+.{nd}f}".replace("-", "−")


def msd(d: dict, f=d3) -> str:
    return f"{f(d['mean'])} ({f(d['sd'])})"


def build(N: dict) -> List[Tuple[str, object]]:
    a, ts, es, vs = N["arms"], N["t_s"], N["e_s"], N["vol_s"]
    st, tost, wil = N["stats"], N["stats"]["tost"], N["stats"]["wilcoxon_test_E1_vs_baseline"]
    r2, mf, nc, ov, psa = N["reader2"], N["mf"], N["nci"], N["ov"], N["psa"]
    g, ab, e = N["geometry"], N["ablation"], N["external"]
    eg, hd, gs, stn = e["geometry"], N["hd95_tail"], N["gland_size"], N["anterior_stroma"]
    margin = st["margin"]
    cg, pz = ts["cg"], ts["pz"]
    v1, v3 = ab["V1_no_reorientation"], ab["V3_identity_affine"]
    S = mf["summary"]
    p158, px, msd_, nci = S["prostate158"], S["prostatex"], S["msd_task05"], S["nci_isbi_2013"]
    r1 = nc["secondary"]["R1"]["cg"]
    r2pz = nc["secondary"]["R2"]["pz"]
    site15, site3 = "BMC 1.5T endorectal", "Radboud 3T"
    bs = nc["by_site"]
    hv = mf["nci_halfvoxel"]
    t_e1b = tost["test_E1_minus_baseline"]
    dec = nc["decision"]["rules"]

    B: List[Tuple[str, object]] = []
    add = B.append

    # ------------------------------------------------------------------ front
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
        "registration can use it. That step is routinely performed, and its errors can be silent: "
        "a volume can carry a valid header and still be mirrored or misplaced.")))
    add(("abs", ("Objective",
        "To develop a 2D U-Net for segmenting the central gland (CG) and peripheral zone (PZ), to "
        "reconstruct its output in the original 3D voxel space with explicit verification, and "
        "to measure which verification check detects which reconstruction error.")))
    add(("abs", ("Methods",
        f"Four single-factor training configurations were compared on Prostate158 (119 training, "
        f"20 validation, and 19 held-out test examinations), with equivalence tested against a "
        f"margin of ±{margin} Dice. The selected model was frozen and identified by its "
        f"checkpoint hash. Reconstruction was verified by a header-based geometry check and a "
        f"model-free round-trip test. Six reconstruction faults were then injected into "
        f"{mf['v0_n']} reference segmentations from 4 public cohorts stored in 3 orientation "
        f"conventions. Accuracy was compared with a second expert reader, and the frozen model "
        f"was applied to PROSTATEx ({e['n_cases']} examinations) and to NCI-ISBI 2013 "
        f"({nc['n']} examinations), the latter with a connected-component rule specified "
        f"before inference.")))
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
        f"{d3(r2['dice_PZ_r1_vs_r2']['mean'])}; boundary distance did not differ from "
        f"inter-reader variability. Training configurations were equivalent or inconclusive "
        f"within ±{margin} Dice. On NCI-ISBI 2013, CG Dice was "
        f"{d3(nc['primary']['cg_dice_none']['mean'])}, and the prespecified rule reduced CG "
        f"95th-percentile Hausdorff distance from {mm(nc['primary']['cg_hd95_none']['mean'])} to "
        f"{mm(r1['hd95_after']['mean'])} mm ({fp(r1['hd95']['wilcoxon_p'])}).")))
    add(("abs", ("Conclusions",
        "A header check and a voxel-level round-trip test detect different reconstruction "
        "errors, and only their combination caught every damaging fault. Whether a fault is "
        "visible at all depends on how the data were stored, so a pipeline tested on one public "
        "dataset can fail silently on another. Studies that derive 3D results from 2D "
        "segmentation should report both checks.")))
    add(("abs", ("Keywords",
        "prostatic neoplasms; magnetic resonance imaging; deep learning; image segmentation; "
        "image processing, computer-assisted; quality control; reproducibility of results; "
        "external testing")))

    # ----------------------------------------------------------- introduction
    add(("h2", "Introduction"))
    add(("p",
        "Prostate cancer is among the most frequently diagnosed cancers in men [@sung]. "
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
        "train, and prior work has segmented both zones this way [@bardis,astono]. Their output, "
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
        "milliliters. Reporting guidelines ask authors to describe preprocessing and to share "
        "code [@claim,tripod_ai], but they do not ask how a reconstructed volume was verified, "
        "and published segmentation studies rarely say."))
    add(("p",
        "What is missing is evidence about which verification actually detects which error, and "
        "whether that answer holds across datasets. This study had 2 aims. The first was to "
        "develop a 2D U-Net for CG and PZ segmentation under a selection protocol fixed before "
        "the held-out data were examined, and to benchmark it against a second expert reader and "
        "on 2 external cohorts. The second, which we regard as the more transferable "
        "contribution, was to reconstruct its output in the original voxel space with explicit "
        "verification and to measure, by injecting known faults into reference segmentations "
        "from 4 public cohorts, which check detects which fault."))

    # ---------------------------------------------------------------- methods
    add(("h2", "Methods"))
    add(("h3", "Study Design and Reporting"))
    add(("p",
        "This was a retrospective algorithm development and validation study on public, "
        "de-identified data. It is reported following the Checklist for Artificial Intelligence "
        "in Medical Imaging (CLAIM) 2024 [@claim] (Multimedia Appendix 1). The workflow is shown "
        "in Figure 1. The study had 2 levels of model selection, both fixed before the held-out "
        "data were examined, and 2 analyses that were specified in writing before they were run: "
        "the equivalence margin and a connected-component post-processing rule (Multimedia "
        "Appendix 2)."))
    add(("fig", (f"{FIG_SUP}/fig_alur_penelitian.png",
        "Figure 1. Study workflow. Four single-factor training configurations were compared on "
        "the validation split; the selected configuration was frozen and evaluated once on the "
        "held-out test split, then used for reconstruction, volumetric analysis, and external "
        "validation.")))

    add(("h3", "Data Sources"))
    add(("p",
        "Four public cohorts were used (Table 1). Prostate158 [@adams_cbm,adams_dib] provided "
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
        f"because it is stored in a third orientation convention."))
    add(("p",
        f"Overlap between the external cohorts was checked by voxel-wise image correlation. "
        f"{ov['n_msd']} NCI-ISBI 3 T examinations are MSD images with independent annotations, "
        f"and 1 is also a PROSTATEx case. No external image overlaps with Prostate158, which comes "
        f"from a different institution. The overlap does not affect any training data; it is "
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
          f"{nc['inplane_min']:.3f}-{nc['inplane_max']:.3f}", "Zonal (CG, PZ)"],
         ["MSD Task05", "Fault injection only", f"{mf['n_cases']['msd_task05']} of 32",
          "3 T, 1 institution", "RAS", "0.60-0.75", "Zonal (transition zone, PZ)"]])))

    add(("h3", "Preprocessing"))
    add(("p",
        "Preprocessing is summarized in Figure 2. Volumes were reoriented to a canonical RAS "
        "frame. Intensities were clipped at the 0.5th and 99.5th percentiles and z-score "
        "normalized per volume; the mask was never normalized. The in-plane grid was "
        "standardized to 442×442 by symmetric zero-padding or center-cropping, and the "
        "through-plane axis was not resampled. The mask path therefore involves no "
        "interpolation, and every parameter needed to invert it (orientation transform, per-axis "
        "pad or crop offsets, and depths) was stored per case."))
    add(("fig", (f"{FIG_RQ2}/F6_preprocessing.png",
        "Figure 2. Preprocessing. Each row gives an operation, its purpose, and the quantity "
        "stored per case so that the operation can be inverted. Two rows are deliberate "
        "omissions.")))

    add(("h3", "Model and Training"))
    add(("p",
        "The network was a 4-level 2D U-Net [@ronneberger] with 32 initial feature maps, 1 input "
        "and 3 output channels, and 7,762,531 parameters (Figure 3). Each level applied 2 "
        "3×3 convolutions with batch normalization and rectified linear units; "
        "downsampling used 2×2 max pooling and upsampling used transposed convolutions with "
        "skip connections. Dropout of 0.2 was applied at the bottleneck. The prediction was the "
        "arg-max over 3 classes, so the output is a partition by construction. A standard "
        "architecture was chosen deliberately: self-configuring pipelines show that an "
        "unmodified U-Net remains competitive [@isensee], and a novel architecture would "
        "confound the reconstruction analysis with model design."))
    add(("fig", (f"{FIG_SUP}/fig_arsitektur_unet.png",
        "Figure 3. The 2D U-Net. The encoder halves the spatial resolution and doubles the "
        "channel width over 4 levels to a 512-channel bottleneck; the decoder mirrors it, with "
        "skip connections (dashed). Input: 1 T2-weighted channel. Output: background, central "
        "gland, and peripheral zone.")))
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
        "decision. The selected checkpoint was identified thereafter by the SHA-256 hash of its "
        "weights."))

    add(("h3", "3D Reconstruction and Verification"))
    add(("p",
        "Each predicted slice was written back to the axial index it carried, never in "
        "loader order; a missing or duplicated index raised an error. The volume was then "
        "inverse-transformed through the preprocessing chain in reverse order with "
        "nearest-neighbor interpolation and written as NIfTI with the source affine and spacing "
        "(Figure 4). Four checks were defined, at 3 levels."))
    add(("p",
        "Level 1, the geometry check, reads the written file's own header and compares array "
        "shape, affine, voxel spacing, orientation codes, and label set with the source. Level "
        "2, the round-trip test, uses no model: the reference mask is passed forward through "
        "preprocessing and back through the same reconstruction code and must be recovered "
        "voxel for voxel. Because the mask transform is a signed axis permutation followed by a "
        "pad or crop, exact recovery is expected, and the outcome is reported as pass or fail, "
        "never as an accuracy. Level 3 comprises 2 anatomical checks that need no reference and "
        "can therefore run at inference time: the PZ centroid must lie posterior to the CG "
        "centroid in scanner coordinates, and the gland centroid must not move more than 15 mm "
        "between adjacent slices."))
    add(("fig", (f"{FIG_SUP}/fig_alur_rekonstruksi.png",
        "Figure 4. Reconstruction and verification. Left: the forward path from source volume "
        "to reassembled prediction. Right: the geometry check, which reads the written header, "
        "and the model-free round-trip test, which compares voxels.")))

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
        "same code, so that the model's error could be set against expert disagreement. Zone "
        "volumes were computed from the absolute determinant of the affine and compared by "
        "Bland-Altman analysis [@bland]."))
    add(("p",
        "Because no PSA values are available, the clinical consequence of volume error was "
        "examined as a sensitivity analysis. For each patient, the range of PSA values at which "
        "the reference and predicted volumes place PSA density on opposite sides of 0.15 "
        "ng/mL/mL was computed, and the share of patient-PSA pairs reclassified was summarized "
        "over the 4 to 10 ng/mL diagnostic gray zone."))

    add(("h3", "External Testing"))
    add(("p",
        "The frozen model was applied to both external cohorts with no training, tuning, "
        "threshold search, or test-time augmentation, through the same inference and "
        "reconstruction code. PROSTATEx had been analyzed earlier, and a connected-component "
        "decomposition there showed that its CG boundary-error tail consisted of small, distant "
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
        "carrying a single whole-gland label was excluded."))

    add(("h3", "Statistical Analysis"))
    add(("p",
        f"Paired per-case differences were tested with the Wilcoxon signed-rank test. The "
        f"prespecified comparison was macro Dice of the selected configuration against the "
        f"baseline on the held-out set. Per-zone tests were post hoc and adjusted with the Holm "
        f"method [@holm] across the 2 zones. Because a nonsignificant difference does not show "
        f"absence of an effect, equivalence was tested with 2 one-sided tests (TOST) "
        f"[@schuirmann,lakens] on paired differences with a margin of ±{margin} Dice, fixed "
        f"before the tests were run; equivalence was concluded when the 90% CI lay within the "
        f"margin. The margin is smaller than the SD of inter-reader Dice for either zone "
        f"({d3(r2['dice_CG_r1_vs_r2']['sd'])} for the CG and "
        f"{d3(r2['dice_PZ_r1_vs_r2']['sd'])} for the PZ). Tests on the validation split are "
        f"descriptive, because that split was used for selection. Dispersion is the sample SD, "
        f"and CIs are bootstrap percentile intervals (10,000 resamples, fixed seed). Training "
        f"used a single NVIDIA Tesla T4 GPU; all other analyses ran on a CPU with Python "
        f"(version 3.13; Python Software Foundation), PyTorch (version 2.14; Linux Foundation), "
        f"NiBabel (version 5.4), SimpleITK (version 2.4), and SciPy (version 1.18). The label "
        f"convention follows MONAI [@monai]."))

    add(("h3", "Ethical Considerations"))
    add(("p",
        "This study used only publicly available, de-identified data released for research "
        "use under their respective open licenses, which permit reuse with attribution (for "
        "example, MSD under CC BY-SA 4.0, the PROSTATEx zonal annotations under CC BY 4.0, and "
        "NCI-ISBI 2013 under CC BY 3.0). No human participants were "
        "recruited and no identifiable information was accessed, so ethics review was not "
        "required [[TO FILL: cite the Universitas Multimedia Nusantara policy or exemption "
        "reference]]."))

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
        f"lower). Two interventions that are commonly recommended thus produced no difference "
        f"larger than the margin."))
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

    add(("h3", "Held-Out Performance and Inter-Reader Agreement"))
    add(("p",
        f"On the 19 held-out cases, Dice was mean {d3(cg['dice']['mean'])} (SD "
        f"{d3(cg['dice']['sd'])}) for the CG and {d3(pz['dice']['mean'])} (SD "
        f"{d3(pz['dice']['sd'])}) for the PZ (Table 3). The PZ combined high precision "
        f"({d3(pz['precision']['mean'])}) with lower recall ({d3(pz['recall']['mean'])}): the "
        f"model rarely labeled other tissue as PZ but labeled too little of it. Against reader "
        f"1, the model's Dice was lower than reader 2's by "
        f"{d3(-r2['dice_CG_model_minus_interreader']['mean_diff'])} for the CG (95% CI "
        f"{d3(-r2['dice_CG_model_minus_interreader']['ci95'][1])}-"
        f"{d3(-r2['dice_CG_model_minus_interreader']['ci95'][0])}; "
        f"{fp(r2['dice_CG_model_minus_interreader']['wilcoxon_p'])}) and by "
        f"{d3(-r2['dice_PZ_model_minus_interreader']['mean_diff'])} for the PZ "
        f"({d3(-r2['dice_PZ_model_minus_interreader']['ci95'][1])}-"
        f"{d3(-r2['dice_PZ_model_minus_interreader']['ci95'][0])}; "
        f"{fp(r2['dice_PZ_model_minus_interreader']['wilcoxon_p'])}), that is, "
        f"{r2['dice_CG_model_minus_interreader']['ratio_of_means']*100:.1f}% and "
        f"{r2['dice_PZ_model_minus_interreader']['ratio_of_means']*100:.1f}% of inter-reader "
        f"agreement. Boundary distance did not differ from inter-reader variability: HD95 was "
        f"{mm(r2['hd95_CG_model_vs_r1']['mean'])} mm against {mm(r2['hd95_CG_r1_vs_r2']['mean'])} "
        f"mm for the CG ({fp(r2['hd95_CG_model_minus_interreader']['wilcoxon_p'])}) and "
        f"{mm(r2['hd95_PZ_model_vs_r1']['mean'])} mm against "
        f"{mm(r2['hd95_PZ_r1_vs_r2']['mean'])} mm for the PZ "
        f"({fp(r2['hd95_PZ_model_minus_interreader']['wilcoxon_p'])})."))
    m = wil["macro"]
    add(("p",
        f"Against the baseline, the prespecified macro comparison was not significant "
        f"({sg(m['mean_diff'])}, better in {m['n_better']} of {m['n']} cases; "
        f"{fp(m['p_raw'])}) and not equivalent either (90% CI {d3(t_e1b['macro']['ci90'][0])} to "
        f"{d3(t_e1b['macro']['ci90'][1])}), so the overall effect is inconclusive. Post hoc, "
        f"CG Dice was equivalent ({fp(t_e1b['CG']['tost_p'])}) and PZ Dice was higher by "
        f"{d3(wil['PZ']['mean_diff'])} (better in {wil['PZ']['n_better']} of {wil['PZ']['n']} "
        f"cases; Holm-adjusted {fp(wil['PZ']['p_holm_zones'])})."))
    def rr(name, key, f=d3):
        return [name, msd(cg[key], f), "—", msd(pz[key], f), "—"]
    add(("tbl", ("Table 3. Held-out performance of the selected model (19 cases, original voxel "
                 "space, per-case mean (SD)) and agreement between the 2 expert readers on the same "
                 "cases.",
        ["Metric", "CG, model vs reader 1", "CG, reader 2 vs reader 1", "PZ, model vs reader 1",
         "PZ, reader 2 vs reader 1"],
        [["Dice", msd(cg["dice"]), msd(r2["dice_CG_r1_vs_r2"]), msd(pz["dice"]),
          msd(r2["dice_PZ_r1_vs_r2"])],
         ["Dice, 95% CI", f"{d3(cg['dice']['ci_low'])}-{d3(cg['dice']['ci_high'])}", "—",
          f"{d3(pz['dice']['ci_low'])}-{d3(pz['dice']['ci_high'])}", "—"],
         ["HD95, mm", msd(cg["hd95_mm"], mm), msd(r2["hd95_CG_r1_vs_r2"], mm),
          msd(pz["hd95_mm"], mm), msd(r2["hd95_PZ_r1_vs_r2"], mm)],
         ["Average surface distance, mm", msd(cg["asd_mm"], mm), msd(r2["asd_CG_r1_vs_r2"], mm),
          msd(pz["asd_mm"], mm), msd(r2["asd_PZ_r1_vs_r2"], mm)],
         rr("Intersection over union", "iou"), rr("Precision", "precision"),
         rr("Recall", "recall")])))

    add(("h3", "Reconstruction Verification and Fault Injection"))
    add(("p",
        f"All {g['n']} held-out reconstructions passed the geometry check, with a maximum "
        f"absolute affine difference of {g['affine_max_abs_diff']:.1f} as read back from the "
        f"single-precision header, and the round-trip test recovered every reference mask "
        f"exactly. In the fault-injection experiment, the correct pipeline returned all "
        f"{mf['v0_n']} references exactly (Table 4). Of the {mf['n_damaged']:,} damaged "
        f"reconstructions, the geometry check flagged {mf['damaged_caught_L1']:,} "
        f"({mf['damaged_caught_L1']/mf['n_damaged']*100:.1f}%), the round-trip test "
        f"{mf['damaged_caught_L2']:,} ({mf['damaged_caught_L2']/mf['n_damaged']*100:.1f}%), and "
        f"the 2 together all {mf['n_damaged']:,}, while neither flagged any of the "
        f"{mf['n_undamaged']} faulted reconstructions that caused no damage (Figure 5)."))
    add(("p",
        f"The 2 checks are complementary. A volume that was not reoriented carries a correct "
        f"header and wrong voxels, so the geometry check passed it in every cohort and only the "
        f"round-trip test caught it. On the held-out predictions this volume would have been "
        f"reported with CG Dice {d3(v1['cg']['dice_vs_gt'])} and PZ Dice "
        f"{d3(v1['pz']['dice_vs_gt'])}, with PZ centroids displaced by a mean of "
        f"{mm(v1['pz']['centroid_shift_mm'])} mm (maximum {mm(v1['pz']['centroid_shift_max_mm'])} "
        f"mm), while passing every header check (Figure 6). Conversely, an identity or "
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
    add(("tbl", (f"Table 4. Fault injection in {mf['v0_n']} reference segmentations. Cells give "
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
        f"Figure 5. Which check detects which fault. Each cell is the share of damaged cases "
        f"that a check flagged, per cohort. Left of the dashed line: the 2 checks that use the "
        f"reference and their combination. Right: the 2 anatomical checks that need no "
        f"reference. Hatched: the fault caused no damage in that cohort. The correct pipeline "
        f"was flagged in 0 of {mf['v0_n']} cases by the first 3 columns.")))
    add(("fig", (f"{FIG_RQ2}/F4_mirrored_volume_passes.png",
        "Figure 6. One held-out case reconstructed correctly (top) and without inverse "
        "reorientation (bottom). Both pass every header check; the lower volume is "
        "anatomically mirrored and its peripheral-zone Dice collapses.")))

    add(("h3", "Zone Volumes and PSA Density"))
    w = vs["whole"]
    add(("p",
        f"The model underestimated whole-gland volume by {mm(-w['bias'])} mL "
        f"({abs(w['pct_of_gt']):.1f}% of the reference; 95% limits of agreement "
        f"{mm(w['loa_low'])} to {mm(w['loa_high'])} mL), mostly through the PZ "
        f"({mm(vs['pz']['bias'])} mL; {sg(vs['pz']['pct_of_gt'], 1)}%) (Figure 7). This PZ "
        f"shortfall is larger than, and opposite in sign to, the difference between readers "
        f"({sg(r2['vol_PZ_r2_minus_r1']['bias_ml'], 1)} mL), so it is a property of the model "
        f"rather than of label noise. In the sensitivity analysis, {psa['n_overestimated']} of "
        f"{psa['n']} patients would have had PSA density overestimated, and "
        f"{psa['n_ever_reclassified']} of {psa['n']} would cross the 0.15 ng/mL/mL threshold at "
        f"some PSA value in the 4 to 10 ng/mL gray zone. Across that zone, "
        f"{psa['share_pairs_reclassified_pct']:.1f}% of patient-PSA pairs were reclassified; the "
        f"PSA window within which a patient was misclassified had a median width of "
        f"{psa['width_median']:.1f} ng/mL (maximum {psa['width_max']:.1f})."))
    add(("fig", (f"{FIG_RQ2}/F2_zone_volume_agreement.png",
        "Figure 7. Zone-volume agreement on the held-out set. Bland-Altman plots for the central "
        "gland and peripheral zone, and the resulting per-case error in PSA density.")))

    add(("h3", "External Testing"))
    pr = nc["primary"]
    add(("p",
        f"On PROSTATEx, Dice was {d3(es['cg']['dice']['mean'])} (SD {d3(es['cg']['dice']['sd'])}) "
        f"for the CG and {d3(es['pz']['dice']['mean'])} (SD {d3(es['pz']['dice']['sd'])}) for "
        f"the PZ (Table 5). CG HD95 had a median of {mm(es['cg']['hd95_mm']['median'])} mm but a "
        f"mean of {mm(es['cg']['hd95_mm']['mean'])} mm, a heavy tail produced by distant "
        f"false-positive components. All {e['n_cases']} reconstructions passed the geometry "
        f"check, including {eg['n_cropped']} examinations large enough to be cropped rather than "
        f"padded, none of which lost annotated anatomy. Recomputing CG Dice separately for the "
        f"anterior and posterior halves showed that the gap to the internal result lies "
        f"posteriorly ({sg(stn['gap']['cg_dice_posterior'])}) rather than anteriorly "
        f"({sg(stn['gap']['cg_dice_anterior'])}), so the anterior stroma included in this "
        f"cohort's CG class does not explain it."))
    add(("p",
        f"On NCI-ISBI 2013, all {nc['geometry_ok']} reconstructions passed the geometry check. "
        f"CG Dice was {d3(pr['cg_dice_none']['mean'])} (SD {d3(pr['cg_dice_none']['sd'])}; 95% "
        f"CI {d3(pr['cg_dice_none']['ci95'][0])}-{d3(pr['cg_dice_none']['ci95'][1])}) and PZ "
        f"Dice {d3(pr['pz_dice_none']['mean'])} (SD {d3(pr['pz_dice_none']['sd'])}), lower at the "
        f"1.5 T endorectal-coil site (CG {d3(bs[site15]['cg_dice_none']['mean'])}) than at the "
        f"3 T site ({d3(bs[site3]['cg_dice_none']['mean'])}). The prespecified rule passed the "
        f"internal harm check (CG Dice {sg(dec['R1']['cg']['dice']['mean_diff'])}, HD95 "
        f"{sg(dec['R1']['cg']['hd95']['mean_diff'], 1)} mm). On NCI-ISBI 2013, it reduced CG HD95 "
        f"from {mm(pr['cg_hd95_none']['mean'])} to {mm(r1['hd95_after']['mean'])} mm "
        f"(median {mm(pr['cg_hd95_none']['median'])} to {mm(r1['hd95_after']['median'])}; "
        f"{fp(r1['hd95']['wilcoxon_p'])}) and increased CG Dice by "
        f"{d3(r1['dice']['mean_diff'])} ({fp(r1['dice']['wilcoxon_p'])}), meeting its "
        f"prespecified criterion. Cases with CG HD95 above 30 mm fell from {nc['cg_tail_before']} "
        f"to {nc['cg_tail_after']} of {nc['n']}. For the PZ, the secondary rule reduced HD95 "
        f"from {mm(pr['pz_hd95_none']['mean'])} to {mm(r2pz['hd95_after']['mean'])} mm "
        f"({fp(r2pz['hd95']['wilcoxon_p'])}) with Dice unchanged ({sg(r2pz['dice']['mean_diff'])})."))
    add(("p",
        f"On the {ov['n_two']} external images annotated independently for both NCI-ISBI and "
        f"MSD, the 2 annotation sets agreed with Dice {d3(ov['msd_vs_nci_cg'])} for the CG and "
        f"{d3(ov['msd_vs_nci_pz'])} for the PZ, close to the Prostate158 inter-reader values. "
        f"The model reached {d3(ov['model_vs_nci_cg'])} and {d3(ov['model_vs_nci_pz'])} against "
        f"the NCI-ISBI annotation. Its gap to inter-annotator agreement therefore widened out of "
        f"domain, to {d3(-ov['gap_cg'])} for the CG and {d3(-ov['gap_pz'])} for the PZ (both "
        f"{fp(max(ov['gap_cg_p'], ov['gap_pz_p']))})."))
    def nrow(name, key, f=d3):
        dd = lambda x: f"{f(x['mean'])} ({f(x['sd'])})"
        return [name, dd(es[key[0]][key[1]]) if key[0] in ("cg", "pz") else "",
                dd(pr[key[2]]), dd(bs[site15][key[2]]), dd(bs[site3][key[2]])]
    add(("tbl", (f"Table 5. External testing of the frozen model, mean (SD). The rule "
                 f"(largest connected CG component) was prespecified for NCI-ISBI 2013; for "
                 f"PROSTATEx the same computation is post hoc and shown for comparison only.",
        ["Metric", f"PROSTATEx (n={e['n_cases']})", f"NCI-ISBI, all (n={nc['n']})",
         f"NCI-ISBI, 1.5 T (n={nc['n_sites'][site15]})",
         f"NCI-ISBI, 3 T (n={nc['n_sites'][site3]})"],
        [nrow("CG Dice", ("cg", "dice", "cg_dice_none")),
         nrow("PZ Dice", ("pz", "dice", "pz_dice_none")),
         nrow("CG HD95, mm", ("cg", "hd95_mm", "cg_hd95_none"), mm),
         nrow("PZ HD95, mm", ("pz", "hd95_mm", "pz_hd95_none"), mm),
         ["CG HD95 after the rule, mm", f"{mm(hd['overall']['hd95_after_mean'])} (post hoc)",
          f"{mm(r1['hd95_after']['mean'])} ({mm(r1['hd95_after']['sd'])})",
          mm(bs[site15]["cg_hd95_R1_mean"]), mm(bs[site3]["cg_hd95_R1_mean"])],
         ["CG Dice after the rule", f"{d3(hd['overall']['dice_after_mean'])} (post hoc)",
          f"{d3(r1['dice_after']['mean'])} ({d3(r1['dice_after']['sd'])})", "—", "—"]])))
    add(("fig", (f"{FIG_RQ2}/F3_external_validation.png",
        "Figure 8. PROSTATEx external testing. Internal and external Dice; the distribution "
        "of central-gland HD95, showing a heavy tail rather than a uniform shift; and the "
        "relation between HD95 and precision.")))

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
        f"The 2D U-Net itself performed close to, but below, expert agreement: "
        f"{r2['dice_CG_model_minus_interreader']['ratio_of_means']*100:.0f}% and "
        f"{r2['dice_PZ_model_minus_interreader']['ratio_of_means']*100:.0f}% of inter-reader "
        f"Dice for the CG and PZ, with boundary distances that did not differ from those between "
        f"readers. Training interventions mattered less than the variation between patients: "
        f"augmentation and class weighting were equivalent to their comparators within "
        f"±{margin} Dice in most comparisons, and the soft Dice term's overall effect was "
        f"inconclusive, with a post hoc gain confined to the PZ. On 2 external cohorts, accuracy "
        f"fell, most at a 1.5 T endorectal-coil site and most for the PZ. A connected-component "
        f"rule, specified and checked internally before it was used, removed almost the entire "
        f"external boundary-error tail."))

    add(("h3", "Comparison With Prior Work"))
    add(("p",
        "The accuracy is below that of the Prostate158 reference implementation, a 3D residual "
        "U-Net that reported Dice of approximately 0.877 for the CG and 0.754 for the PZ on the "
        f"same split [@adams_cbm], that is, {cg['dice']['mean']/0.877*100:.1f}% and "
        f"{pz['dice']['mean']/0.754*100:.1f}% of those values. The larger shortfall for the PZ "
        "is what the architecture predicts, since the PZ boundary runs obliquely to the axial "
        "plane, where a 2D network has no context. We therefore do not present the model as "
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
        "correct header, shape, spacing, and label set. Work that adapts 2D and video models to "
        "3D medical data [@yamagishi] ends in the same reassembly step. We suggest that studies "
        "deriving 3D results from slice-wise output report both a header check and a "
        "voxel-level round-trip test; the latter costs a single forward and inverse pass of an "
        "existing annotation."))

    add(("h3", "Clinical Implications"))
    add(("p",
        "Dice is a research metric; volume is how zonal segmentation reaches clinical use. The "
        "model's PZ underestimation propagated into PSA density, and in the sensitivity "
        "analysis most patients would cross the 0.15 ng/mL/mL threshold somewhere in the "
        "diagnostic gray zone. The error runs toward overestimated density. Combined with limits "
        "of agreement that are wide relative to the gland, this means volumes from this model "
        "should not be used for PSA density without review. Reporting volumes with limits of "
        "agreement makes this visible; reporting Dice alone does not."))

    add(("h3", "Limitations"))
    add(("p",
        "The development cohort comprises 158 examinations from 1 institution, and the held-out "
        "split has 19 cases, so internal estimates are imprecise; CIs are reported for that "
        "reason. Each configuration was trained once with 1 seed, so seed-to-seed variation is "
        "not estimated, and the equivalence results describe these runs rather than the "
        "configurations in general. The held-out split had been used twice during earlier "
        "pipeline development with a different model; the selection rule used validation data "
        "only, but the test set was not untouched. For that reason, the analyses added in this "
        "work that required a confirmatory cohort were run on NCI-ISBI 2013, which had not been "
        "used before."))
    add(("p",
        f"The external cohorts were annotated under different protocols, so external "
        f"differences combine model generalization with annotation differences; the "
        f"2-annotation comparison bounds but does not remove this. The cohorts also partly "
        f"overlap ({ov['n_msd']} shared images between NCI-ISBI and MSD, and 1 between NCI-ISBI "
        f"and PROSTATEx); this affects no training data, and excluding the shared PROSTATEx case "
        f"changed no NCI-ISBI result materially. The fault injection uses a fixed set of faults "
        f"implemented in 1 pipeline; other pipelines may fail in other ways, and the 2 harmless "
        f"faults here may not be harmless elsewhere. The PSA analysis uses a range of PSA values "
        f"rather than measured values, and no reader study was performed, so no claim is made "
        f"about clinical utility or patient outcomes. Accuracy also depended on gland size, "
        f"which limits use in small glands."))

    add(("h3", "Conclusions"))
    add(("p",
        "Reconstructing slice-wise segmentation into a 3D volume can fail in ways that leave a "
        "valid file, and in this study a header check and a voxel-level round-trip test each "
        "missed a different class of failure. Only their combination detected every damaging "
        "fault across 4 public cohorts, and whether a fault was observable at all depended on "
        "the orientation in which the data were stored. The 2D U-Net segmented the central "
        "gland close to expert agreement and the peripheral zone less well, and a prespecified "
        "connected-component rule removed most of its external boundary error. Studies that "
        "derive 3D results from 2D segmentation should report both verification checks and "
        "should report volumes with limits of agreement alongside Dice."))

    # ----------------------------------------------------------- back matter
    add(("h2", "Acknowledgments"))
    add(("p",
        "The authors thank the creators of Prostate158, PROSTATEx and its zonal annotations, "
        "NCI-ISBI 2013, and the Medical Segmentation Decathlon for making their data public. "
        "Generative artificial intelligence (Claude, Anthropic) was used to assist with code "
        "development, analysis scripting, and language editing [[TO FILL: confirm and complete "
        "this statement]]. All generated content was reviewed and verified by the authors, who "
        "take full responsibility for the work. Transcripts are provided in Multimedia "
        "Appendix 3 [[TO FILL: attach or remove]]."))
    add(("h2", "Funding"))
    add(("p", "[[TO FILL: funding source for the article processing fee, or 'No external "
              "financial support or grants were received from any public, commercial, or "
              "not-for-profit entities for the research, authorship, or publication of this "
              "article.']]"))
    add(("h2", "Data Availability"))
    add(("p",
        "All datasets analyzed are publicly available from their original sources "
        "[@adams_dib,cuocolo_ejr,clark,bloch_nci,antonelli]. The analysis code, configuration "
        "files, the prespecified rule, and the scripts that generate every number in this "
        "article are available in the project repository [@repo]. The frozen model is "
        "identified there by the SHA-256 hash of its weights."))
    add(("h2", "Authors' Contributions"))
    add(("p",
        "Conceptualization: BAS, EES. Methodology: BAS. Software: BAS. Formal analysis: BAS. "
        "Validation: BAS. Visualization: BAS. Writing - original draft: BAS. Writing - review "
        "and editing: EES, BAS. Supervision: EES. [[TO FILL: confirm roles]]"))
    add(("h2", "Conflicts of Interest"))
    add(("p", "None declared."))
    add(("h2", "Multimedia Appendix 1"))
    add(("p", "CLAIM 2024 checklist."))
    add(("h2", "Multimedia Appendix 2"))
    add(("p", "Prespecification of the connected-component rule and the NCI-ISBI 2013 analysis, "
              "with its erratum and the recorded file hash."))
    add(("h2", "Multimedia Appendix 3"))
    add(("p", "Supplementary results: training histories, per-case results, cohort geometry, and "
              "additional figures."))
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
    add(("p", "; ".join(f"{k}: {v}" for k, v in abbr)))
    add(("h2", "References"))
    add(("refs", REFERENCES))
    return B


import manuscript_content_en as _en  # noqa: E402

_KEEP = ["sung", "turkbey", "weinreb", "mcneal", "ronneberger", "isensee", "litjens_survey",
         "bardis", "astono", "cheng", "claim", "tripod_ai", "adams_cbm", "adams_dib", "litjens",
         "clark", "cuocolo_ejr", "cuocolo_jmri", "taha", "maierhein", "bland", "nordstrom",
         "monai", "yamagishi"]
# JMIR reference style: sentence-case titles, [doi] and [Medline] tags.
REFERENCES = {
    "sung": "Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin 2021 May;71(3):209-249. [doi: 10.3322/caac.21660] [Medline: 33538338]",
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
    "repo": "Sandro BA. prostate-mri-segmentation-thesis. GitHub. URL: https://github.com/amadeussandro/prostate-mri-segmentation-thesis [accessed [[TO FILL: date]]] [[TO FILL: Zenodo DOI]]",
}
assert set(_KEEP) <= set(REFERENCES), set(_KEEP) - set(REFERENCES)
