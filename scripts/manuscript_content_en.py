"""English JMIR manuscript content.

The document is expressed as an ordered list of blocks that the builder renders
into Word. Numbers come from `manuscript_numbers`, never from the prose, so a
value cannot disagree with the artifact that produced it.

Block grammar:
    ("title", str) ("authors", [str]) ("affil", [str]) ("corr", str)
    ("h2", str) ("h3", str) ("p", str)
    ("abs", (label, text))
    ("fig", (path, caption))
    ("tbl", (caption, [headers], [[cells]]))
    ("refs", [str])
"""

from __future__ import annotations

from typing import List, Tuple

TITLE = ("Segmentation of Anatomical Zones in Prostate MRI Using a 2D U-Net "
         "and Spatially Validated 3D Reconstruction")

AUTHORS = ["Benedict Amadeus Sandro", "Eunike Endariahna Surbakti, S.Kom., M.T.I."]
AFFIL = ["Department of Informatics, Faculty of Engineering and Informatics, "
         "Universitas Multimedia Nusantara, Tangerang, Indonesia"]
CORR = ("Corresponding author: Benedict Amadeus Sandro, Department of Informatics, "
        "Faculty of Engineering and Informatics, Universitas Multimedia Nusantara, "
        "Tangerang, Banten, Indonesia")

FIG_DIR_RQ1 = "results/figures_rq1"
FIG_DIR_RQ2 = "results/figures_rq2"
VIZ = "results/rq2_e1_epoch88/visualizations"


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def build(N: dict) -> List[Tuple[str, object]]:
    a = N["arms"]
    vt = N["validation_tests"]
    t = N["test"]
    tb = N["test_vs_baseline"]
    g = N["geometry"]
    ab = N["ablation"]
    v = N["volumes"]
    e = N["external"]
    eg = e["geometry"]

    cg_d = t["per_class"]["cg"]["dice"]
    pz_d = t["per_class"]["pz"]["dice"]
    bg_d = t["per_class"]["background"]["dice"]
    v1, v3 = ab["V1_no_reorientation"], ab["V3_identity_affine"]
    # Phrase the pass count honestly: "All N" only when none failed.
    geo_all = (f"All {g['n']}" if g['n_all_ok'] == g['n']
               else f"{g['n_all_ok']} of {g['n']}")

    B: List[Tuple[str, object]] = []
    add = B.append

    # ---------------------------------------------------------------- front
    add(("title", TITLE))
    add(("authors", AUTHORS))
    add(("affil", AFFIL))
    add(("corr", CORR))

    # ------------------------------------------------------------- abstract
    add(("h2", "Abstract"))
    add(("abs", ("Background",
        "Prostate cancer is among the most frequently diagnosed malignancies in men, and "
        "multiparametric MRI has become central to its detection and staging. Interpretation "
        "depends on the zonal anatomy of the gland, because the central gland (CG) and the "
        "peripheral zone (PZ) differ in both appearance and in the prevalence and significance "
        "of tumours arising in them. Automated zonal segmentation is therefore a practical "
        "requirement for computer-assisted reading. Two-dimensional networks remain attractive "
        "for anisotropic prostate MRI, but they produce slice-wise output, and reassembling that "
        "output into a spatially valid 3D volume is an implementation step that is widely "
        "performed and rarely verified.")))
    add(("abs", ("Objective",
        "This study had two aims: to evaluate a 2D U-Net for anatomical zone segmentation on "
        "prostate MRI under a pre-specified model-selection protocol, and to implement and "
        "validate the reconstruction of its slice-wise predictions into the original 3D voxel "
        "space. A secondary aim was to quantify what the reconstruction step costs when its "
        "individual operations are omitted, and to express the result in units that carry "
        "clinical meaning.")))
    add(("abs", ("Methods",
        f"We used the Prostate158 dataset with its official patient-level split "
        f"({a['baseline']['n'] and 119} training, {a['e1']['n']} validation, {t['n_cases']} "
        "held-out test). Four single-factor training configurations were compared: a "
        "cross-entropy baseline, the addition of a soft Dice term on CG and PZ, the addition of "
        "geometric and intensity augmentation, and the addition of class-weighted cross-entropy. "
        "The configuration with the highest macro Dice on the validation split was selected "
        "before the test set was examined, frozen, and identified thereafter by the SHA-256 hash "
        "of its checkpoint. Slice-wise predictions were reassembled by explicit axial index, "
        "inverse-transformed, and written with the source affine restored, then scored per case "
        "in original voxel space against the untouched expert annotation. Reconstruction "
        "correctness was assessed in two independent ways: a per-case geometric checklist, and a "
        "model-free round-trip fidelity test. An ablation rebuilt the same predictions with "
        "individual reconstruction operations omitted. Zone volumes were computed in millilitres "
        "and compared by Bland-Altman analysis. The frozen model was finally applied once to an "
        f"independent cohort of {e['n_cases']} PROSTATEx examinations.")))
    add(("abs", ("Results",
        f"On the validation split the four configurations spanned {N['arm_spread']:.4f} macro "
        f"Dice, smaller than the per-case standard deviation within any single configuration "
        f"({N['arm_sd_range'][0]:.2f}-{N['arm_sd_range'][1]:.2f}). The selected configuration "
        f"achieved a mean (SD) volume-level Dice of {cg_d['mean']:.4f} ({cg_d['sd']:.4f}) for CG "
        f"and {pz_d['mean']:.4f} ({pz_d['sd']:.4f}) for PZ on the held-out set, with a macro Dice "
        f"of {t['macro']['mean']:.4f}. {geo_all} reconstructions passed "
        f"every geometric check, with a maximum absolute affine difference of "
        f"{g['affine_max_abs_diff']:.1f}. The ablation showed that omitting the inverse "
        f"orientation step produced an anatomically mirrored volume that passed the entire "
        f"geometric checklist while its reported Dice fell to {v1['cg']['dice_vs_gt']:.4f} (CG) "
        f"and {v1['pz']['dice_vs_gt']:.4f} (PZ); omitting affine restoration left Dice unchanged "
        f"while displacing the gland by {v3['cg']['centroid_shift_mm']:.0f} mm and inflating "
        f"every volume by {v3['cg']['volume_pct_error']:.0f}%. Whole-gland volume was "
        f"under-estimated by {abs(v['whole']['bias']):.2f} mL "
        f"({abs(v['whole']['pct_of_gt']):.1f}% of the reference), which inflates PSA density by "
        f"{v['psad']['mean_pct']:.1f}% on average. On the external cohort macro Dice was "
        f"{e['macro']['mean']:.4f}.")))
    add(("abs", ("Conclusions",
        "A standard 2D U-Net segments the central gland reliably and the peripheral zone less "
        "so, and three single-factor training interventions moved performance by less than the "
        "variation between patients. The reconstruction findings are the more transferable "
        "result: geometric validation and overlap metrics are each blind to a different class of "
        "reconstruction error, so neither alone is sufficient, and a model-free round-trip "
        "fidelity test is what detects a volume whose header is correct and whose voxels are "
        "not. Because zone volumes enter PSA density as a denominator, a reconstruction error "
        "that leaves Dice untouched can still change a clinical quantity substantially.")))
    add(("abs", ("Keywords",
        "prostate cancer; magnetic resonance imaging; image segmentation; deep learning; U-Net; "
        "3D reconstruction; peripheral zone; external validation; reproducibility; PSA density")))

    # ---------------------------------------------------------- introduction
    add(("h2", "Introduction"))
    add(("p",
        "Prostate cancer is one of the most frequently diagnosed malignancies in men worldwide "
        "and a leading cause of cancer-related death [1]. Multiparametric MRI has become central to "
        "detection, localization and staging, and structured reporting frameworks have made "
        "image interpretation more consistent [2,3]. That interpretation is organized around the zonal "
        "anatomy of the gland."))
    add(("p",
        "The prostate is conventionally divided into the peripheral zone and the central gland, "
        "the latter comprising the transition and central zones [4]. The distinction matters "
        "clinically: most carcinomas arise in the peripheral zone, lesion assessment criteria "
        "differ between zones, and the ratio of zonal volumes informs the interpretation of "
        "benign enlargement. Zonal segmentation is therefore not an abstract partitioning task "
        "but a prerequisite for several downstream clinical quantities."))
    add(("p",
        "Deep learning has become the standard approach to this segmentation problem [5,6]. Prostate "
        "MRI is strongly anisotropic, with in-plane resolution an order of magnitude finer than "
        "slice thickness, and two-dimensional networks remain a reasonable choice under that "
        "geometry: they avoid interpolating along the axis where information is sparsest, and "
        "they are inexpensive to train. Their output, however, is slice-wise. Anything that "
        "consumes the segmentation as a three-dimensional object - a volume measurement, a "
        "surface rendering, registration to another series, or a biopsy plan - requires those "
        "slices to be reassembled into a volume that occupies the correct physical space."))
    add(("p",
        "That reassembly is routinely treated as an implementation detail. It is not a single "
        "operation but a chain of them: slices must be placed at their correct axial index, "
        "spatial preprocessing must be inverted in the reverse order it was applied, and the "
        "source affine must be restored so the volume sits where the scanner put it. Each step "
        "can fail independently, and the failures do not announce themselves. A volume with a "
        "correct header and mirrored contents is still a valid NIfTI file; a volume with perfect "
        "voxels and a default affine still yields a number when its volume is measured. Papers "
        "report the metrics that follow from this step far more often than they report how the "
        "step was verified."))
    add(("p",
        "This study addresses both halves of that situation on a public dataset. We first "
        "evaluate a 2D U-Net for zonal segmentation under a selection protocol fixed before the "
        "held-out data were examined, comparing four single-factor training configurations. We "
        "then implement the 2D-to-3D reconstruction with explicit verification at every stage, "
        "and - this is the part we believe is most transferable - we measure what each "
        "verification step is actually worth by rebuilding the same predictions with individual "
        "operations omitted and reporting what a reader would have seen in each case."))
    add(("p",
        "We frame the contribution conservatively. This study proposes no new architecture and "
        "does not claim state-of-the-art accuracy; the 2D U-Net is a component, and the "
        "reproducibility of the pipeline is a property rather than a finding. The contributions "
        "are the empirical comparison of training configurations under a pre-specified rule, "
        "including the negative result that three interventions failed to move performance "
        "beyond between-patient variation, and the measured demonstration that geometric "
        "validation alone is insufficient to establish that a reconstructed volume is correct."))

    # -------------------------------------------------------------- methods
    add(("h2", "Methods"))

    add(("h3", "Study Design and Overview"))
    add(("p",
        "This was an experimental, quantitative study conducted entirely on retrospective, "
        "publicly available, de-identified imaging. The workflow comprised preprocessing, 2D "
        "slice extraction, training of four single-factor configurations, model selection on the "
        "validation split, a single evaluation on the held-out test split, reconstruction of the "
        "selected model's predictions into original voxel space with verification, an ablation "
        "of the reconstruction stage, volumetric analysis, and one application to an independent "
        "external cohort."))
    add(("fig", (f"{FIG_DIR_RQ2}/F0_study_pipeline.png",
        "Figure 1. Overall study workflow. Four single-factor training configurations are "
        "compared on the validation split; the selected configuration is frozen and evaluated "
        "once on the held-out test split, then used for reconstruction, ablation, volumetric "
        "analysis and external validation.")))

    add(("h3", "Dataset"))
    add(("p",
        "Prostate158 was used as the methodological foundation. It comprises 158 biparametric 3T "
        "prostate MRI examinations with expert anatomical annotations distributed in NIfTI "
        "format [7]. The input to the model was the axial T2-weighted series, a single channel. The "
        "segmentation target was the reader-1 anatomical mask with integer labels, where 0 is "
        "background, 1 is the central gland and 2 is the peripheral zone."))
    add(("p",
        "The meaning of the integer labels was verified rather than assumed, because the mapping "
        "is reported inconsistently in the literature and an inverted reading would silently "
        "exchange the two zones in every reported metric. Verification combined multi-planar "
        "visual inspection of individual cases against the known anatomy - the peripheral zone "
        "forms a thin crescent on the posterior aspect of the gland at the rectal interface, "
        "while the central gland is the bulky, often nodular mass filling the centre and anterior "
        "- with morphometric consistency across the cohort. All lines of evidence agreed."))

    add(("h3", "Data Split"))
    add(("p",
        f"The official patient-level split was used unchanged: 119 training, {a['e1']['n']} "
        f"validation and {t['n_cases']} held-out test examinations. Splitting at the patient "
        "level, rather than the slice level, prevents slices from one examination appearing in "
        "more than one split. We verified that the training and validation identifier sets were "
        "disjoint and that neither intersected the test identifiers."))

    add(("h3", "Image Preprocessing"))
    add(("p",
        "Every geometric operation was applied identically to the image and its mask, differing "
        "only in interpolation order: linear for the image, nearest-neighbour for the mask, so "
        "that integer labels are never blended. Volumes were reoriented to a canonical RAS frame, "
        "intensity was z-score normalized per volume after percentile clipping, and the in-plane "
        "grid was standardized to 442x442 by symmetric zero-padding or centre-cropping. No "
        "resampling was performed along the through-plane axis, whose spacing is already uniform "
        "across the cohort; avoiding it keeps the slice grid identical to the acquisition and "
        "removes a source of interpolation error from the reconstruction path."))
    add(("p",
        "Every parameter needed to invert these operations - the orientation transform, the pad "
        "or crop offsets per axis, and the pre- and post-transform depths - was retained per case "
        "rather than recomputed later."))

    add(("h3", "2D Slice Preparation"))
    add(("p",
        "Each preprocessed volume was decomposed into axial slices, and every slice was tagged "
        "with the axial index it came from. The index travels with the slice through inference "
        "and is what the slice is written back to during reconstruction, so correctness does not "
        "depend on the order in which a data loader happens to emit samples. All slices were "
        "retained for evaluation, including those containing no annotated anatomy."))

    add(("h3", "2D U-Net Architecture"))
    add(("p",
        "The network was a four-level encoder-decoder U-Net [5] with 32 initial feature maps, a "
        "single input channel and three output channels, totalling 7,762,531 parameters. Each "
        "level applied two 3x3 convolutions with batch normalization and ReLU; downsampling used "
        "2x2 max pooling and upsampling used transposed convolutions with skip connections from "
        "the matching encoder level. Dropout of 0.3 was applied at the bottleneck. The final "
        "layer produced per-class logits, and the per-slice prediction was the arg-max over the "
        "three classes."))
    add(("fig", ("results/figures_manuscript/from_original/image5.png",
        "Figure 2. The 2D U-Net architecture: a four-level encoder-decoder with 32 initial "
        "feature maps, skip connections at each level, and a three-channel output corresponding "
        "to background, central gland and peripheral zone.")))

    add(("h3", "Training Configurations and Model Selection"))
    add(("p",
        "Four configurations were trained. Each differed from the baseline in exactly one "
        "factor, so that any difference in performance is attributable to that factor rather "
        "than to a combination of changes. The baseline used cross-entropy loss. The second "
        "configuration added a soft Dice term computed on the two foreground classes, weighted "
        "equally with cross-entropy. The third added geometric and intensity augmentation to the "
        "baseline while keeping the loss unchanged. The fourth added class-weighted "
        "cross-entropy to the second configuration, up-weighting the peripheral zone to address "
        "its under-representation."))
    add(("p",
        "All other settings were held fixed across configurations: identical preprocessing, "
        "identical architecture, AdamW with a learning rate of 0.001 and weight decay of 0.0001, "
        "batch size 8, 100 epochs, full 32-bit precision, and a fixed random seed of 42."))
    add(("p",
        "The selection rule was fixed before the held-out data were examined: the configuration "
        "with the highest macro Dice on the 20 validation cases would be selected, and no test "
        "metric would contribute to that decision. After selection the checkpoint was frozen and "
        "identified by the SHA-256 hash of its weights thereafter, so that every subsequent "
        "analysis could be shown to use the same model rather than a file with the same name."))

    add(("h3", "Segmentation Evaluation"))
    add(("p",
        "Evaluation was performed per case at the volume level in original voxel space, against "
        "the expert annotation read from disk without modification. Scoring in original space, "
        "rather than in the preprocessed grid, means the reported figures describe the volume a "
        "downstream consumer would actually receive. Aggregation was per case and never pooled "
        "over slices, which would weight large examinations more heavily than small ones."))
    add(("p",
        "Reported metrics were Dice, intersection over union, 95th-percentile Hausdorff distance "
        "(HD95), average surface distance (ASD), precision and recall [14,15]. Surface distances were "
        "computed in millimetres using each case's own voxel spacing. A class absent from both "
        "prediction and annotation was treated as trivial agreement; a class absent from only "
        "one leaves surface distances undefined, and those cases were excluded from that metric "
        "rather than being assigned an arbitrary value. Macro Dice is the mean of the two "
        "foreground classes; background Dice is reported separately because including it would "
        "inflate the summary with a class that is trivially easy."))
    add(("p",
        "Standard deviations are population standard deviations throughout. Confidence intervals "
        "are bootstrap percentile intervals over 2,000 resamples with a fixed seed."))

    add(("h3", "3D Reconstruction"))
    add(("p",
        "Predicted slices were assembled into a volume by writing each slice to the axial index "
        "it carries, never by appending in iteration order; a missing or duplicated index raises "
        "an error rather than producing a plausible-looking volume. The assembled volume was then "
        "inverse-transformed through the preprocessing chain in reverse order, using "
        "nearest-neighbour interpolation throughout so that integer labels are preserved exactly. "
        "The result was written as a NIfTI image carrying the original source affine and voxel "
        "spacing, never a default identity affine."))
    add(("fig", (f"{FIG_DIR_RQ2}/F5_reconstruction_workflow.png",
        "Figure 3. The 2D-to-3D reconstruction and verification workflow. Predictions are "
        "assembled by explicit axial index, inverse-transformed in reverse order with "
        "nearest-neighbour interpolation, and written with the source affine restored. Two "
        "independent checks follow: a per-case geometric checklist and a model-free round-trip "
        "fidelity test.")))

    add(("h3", "Geometric and Fidelity Validation"))
    add(("p",
        "Reconstruction correctness was assessed in two independent ways, which we report "
        "separately because they detect different things."))
    add(("p",
        "The first is a per-case geometric checklist. The geometry is read back from the output "
        "file's own header, not from the variables used to construct it, and compared against the "
        "untouched source: array shape, affine matrix, voxel spacing, orientation codes, and the "
        "set of label values present. Reading back from the header also catches precision loss "
        "introduced by the file format itself, which stores the affine in single precision."))
    add(("p",
        "The second is a round-trip fidelity test, which involves no model at all. The expert "
        "annotation is passed forward through the same preprocessing, reassembled and "
        "inverse-transformed by the same code, and compared with the original annotation. Under "
        "this configuration the mask transform is a symmetric pad followed by a signed axis "
        "permutation, both of which are exactly invertible, so the expected outcome is exact "
        "recovery. We therefore report this as a pass or fail verification outcome and "
        "deliberately keep it out of any results table, because a Dice of 1.0 reported alongside "
        "model accuracy invites the reading that the model is perfect. It is a property of the "
        "transform, not of the model."))

    add(("h3", "Ablation of the Reconstruction Stage"))
    add(("p",
        "To establish what each verification step is worth, we rebuilt the same saved 2D "
        "predictions with individual reconstruction operations omitted. Because every variant "
        "starts from identical predictions, any difference is attributable to the reconstruction "
        "stage alone, with no model variation involved. Four shortcuts were examined: omitting "
        "the inverse orientation step; centre-cropping back to the native in-plane size instead "
        "of using the recorded offsets; writing the volume with an identity affine instead of the "
        "source affine; and stacking slices in loader order instead of by explicit index. A fifth "
        "variant combined all four."))
    add(("p",
        "Each variant was measured in three ways: its agreement with the correctly reconstructed "
        "volume, which isolates the damage; the Dice it would have reported against the expert "
        "annotation, which is what a reader would have seen; and whether the geometric checklist "
        "flagged it. As a correctness gate, the reference reconstruction was rebuilt from the "
        "saved predictions and metadata alone and required to be bit-identical to the volume the "
        "main run had written, so that the comparison is made against a reference that can be "
        "reproduced."))

    add(("h3", "Zone Volumes"))
    add(("p",
        "Zone volumes were computed as the voxel count of each class multiplied by the physical "
        "volume of a voxel, taken as the absolute determinant of the affine's linear part rather "
        "than the product of header spacings, so that a rotated or sheared acquisition is handled "
        "exactly. Agreement with the volumes derived from the expert annotation was assessed by "
        "Bland-Altman analysis [12], reporting bias and 95% limits of agreement."))
    add(("p",
        "Volumes were examined because they are the quantity through which this segmentation "
        "reaches clinical use. Whole-gland volume is the denominator of PSA density [13], so an error "
        "in the volume propagates directly into a quantity used in biopsy decisions. Volumes also "
        "make the geometric work legible: a volume in millilitres is a voxel count multiplied by "
        "a quantity read from the affine, so a pipeline that loses the affine reports a wrong "
        "clinical number from entirely correct voxels."))

    add(("h3", "External Validation"))
    add(("p",
        f"The frozen model was applied once to an independent cohort of {e['n_cases']} PROSTATEx "
        "examinations [8,11] with publicly released expert zonal annotations [9,10]. No training, fine-tuning, "
        "threshold search or model selection was performed on this cohort; it was used purely for "
        "inference, through the same inference-and-reconstruction code path as the internal "
        "evaluation."))
    add(("p",
        "The source annotations distribute the peripheral zone and the remainder of the gland as "
        "two separate binary masks. These were combined into the model's three-label convention "
        "using a greater-than-zero test rather than equality, after one case was found to carry a "
        "stray voxel of value 2 in its peripheral-zone mask that an equality test would have "
        "discarded. Mutual exclusivity of the two masks was re-measured for every case rather "
        "than assumed. Cases were resolved through the cohort manifest, because 15 of them use "
        "file-naming that cannot be derived from the case identifier."))
    add(("p",
        f"This cohort differs from the development data in ways that were measured rather than "
        f"described qualitatively. Its in-plane resolution spans {eg['inplane_min']}-"
        f"{eg['inplane_max']} mm against {0.46875} mm internally, its through-plane spacing spans "
        f"{eg['through_min']}-{eg['through_max']} mm, and its axial field of view averages "
        f"{eg['fov_mean']:.0f} mm against {eg['fov_internal']:.1f} mm, a factor of "
        f"{eg['fov_ratio']:.2f}. Because the pipeline performs no in-plane resampling, the "
        "anatomy is presented to the model at a different pixel scale than it was trained on, and "
        "a wider field of view means the model sees surrounding structures it never encountered "
        "during training. Both are genuine covariate shifts and are reported alongside the "
        "accuracy figures rather than treated as nuisances."))

    add(("h3", "Statistical Analysis"))
    add(("p",
        "Differences between configurations were assessed with the Wilcoxon signed-rank test on "
        "paired per-case values, which does not assume normality and is appropriate for the "
        "sample sizes involved. The comparison specified in advance was the macro Dice of the "
        "selected configuration against the baseline on the held-out set."))
    add(("p",
        "Two classes of test are reported here with explicit caveats, because reporting them "
        "without those caveats would overstate what they establish. Tests computed on the "
        "validation split are descriptive rather than confirmatory: that split is what the model "
        "was selected on, so a significance claim derived from it is circular. Per-zone tests on "
        "the held-out set are post hoc; they were not pre-specified, and with three comparisons "
        "a Bonferroni-corrected threshold would be 0.0167. We report all of these values because "
        "they are informative about where the difference between configurations lies, and we "
        "label them so that they are not read as confirmatory evidence."))

    add(("h3", "Visualization and Computational Environment"))
    add(("p",
        "Qualitative figures show, for selected cases, the T2-weighted image, the expert "
        "annotation, the prediction, their overlay and an error map. Representative cases were "
        "chosen by an objective rule - the cases with the highest, median and lowest mean "
        "foreground Dice - rather than by visual appeal. Multi-planar projections are rendered "
        "with the display aspect ratio set from the voxel spacing, so that the strongly "
        "anisotropic grid is not displayed as if it were isotropic."))
    add(("p",
        "Training was performed on a single NVIDIA Tesla T4 GPU under PyTorch; the label "
        "convention follows the MONAI ordering [16]. Inference, "
        "reconstruction, ablation and volumetric analysis require no GPU and were executed on "
        "CPU. All analysis code, configuration files and the exact evaluation scripts are "
        "available in the public repository cited below."))

    add(("h3", "Ethics"))
    add(("p",
        "This study used publicly available, de-identified datasets and involved no human "
        "participants, no recruitment and no intervention. Institutional review board approval "
        "and informed consent were therefore not required. The datasets were used in accordance "
        "with their respective licences, and the annotations are credited to their original "
        "authors."))

    add(("tbl", ("Table 1. Dataset characteristics and experimental configuration.",
        ["Item", "Value"],
        [
            ["Dataset", "Prostate158 (158 examinations, 3T MRI, NIfTI)"],
            ["Input modality", "T2-weighted, single channel"],
            ["Segmentation target", "Reader-1 anatomical mask, labels {0,1,2}"],
            ["Classes", "0 = background, 1 = central gland (CG), 2 = peripheral zone (PZ)"],
            ["Official split", f"119 training / {a['e1']['n']} validation / {t['n_cases']} held-out test"],
            ["Split unit", "Patient (case) level"],
            ["Preprocessing", "RAS orientation; per-volume z-score after percentile clipping; "
                              "crop/pad to 442x442; no through-plane resampling"],
            ["Architecture", "2D U-Net, 32 initial features, 1 input / 3 output channels, "
                             "7,762,531 parameters"],
            ["Optimizer", "AdamW, learning rate 0.001, weight decay 0.0001"],
            ["Batch size / epochs", "8 / 100"],
            ["Random seed", "42"],
            ["Precision", "FP32"],
            ["Selection rule", "Highest macro Dice on the validation split, fixed in advance"],
            ["Model identity", "SHA-256 of the frozen checkpoint"],
            ["External cohort", f"PROSTATEx, {e['n_cases']} examinations, inference only"],
        ])))

    # -------------------------------------------------------------- results
    add(("h2", "Results"))

    add(("h3", "Model Selection on the Validation Split"))
    add(("p",
        f"Table 2 reports the four configurations on the {a['e1']['n']} validation cases. The "
        f"configuration adding a soft Dice term achieved the highest macro Dice "
        f"({a['e1']['macro']:.4f}) and was selected. Its best epoch was {a['e1']['epoch']}."))
    add(("p",
        f"The more informative observation is the size of the differences. The four "
        f"configurations spanned {N['arm_spread']:.4f} macro Dice, while the per-case standard "
        f"deviation within any single configuration ranged from {N['arm_sd_range'][0]:.4f} to "
        f"{N['arm_sd_range'][1]:.4f}. The spread between interventions is thus several times "
        f"smaller than the variation between patients within one of them. Augmentation and "
        f"class-weighted cross-entropy did not improve on the configurations they modified "
        f"(augmentation {vt['e2_vs_baseline']['delta']:+.4f} against the baseline, "
        f"P={vt['e2_vs_baseline']['p']:.2f}; class weighting "
        f"{vt['e3_vs_e1']['delta']:+.4f} against the selected configuration, "
        f"P={vt['e3_vs_e1']['p']:.2f}). We report this as a negative result rather than omitting "
        f"it: on this dataset, at this scale, two interventions that are routinely recommended "
        f"produced no measurable benefit."))
    add(("p",
        f"Exploratory paired tests on this split favour the selected configuration over both the "
        f"baseline ({vt['e1_vs_baseline']['delta']:+.4f}, P={vt['e1_vs_baseline']['p']:.4f}) and "
        f"the augmentation arm ({vt['e1_vs_e2']['delta']:+.4f}, "
        f"P={vt['e1_vs_e2']['p']:.4f}). These values are descriptive only. The validation split "
        f"is the split on which selection was performed, so a significance claim computed on it "
        f"is circular and is not offered as evidence of superiority."))

    add(("tbl", (f"Table 2. The four training configurations and their performance on the "
                 f"{a['e1']['n']} validation cases. Each configuration differs from the baseline "
                 f"in exactly one factor. Macro Dice is the mean of CG and PZ.",
        ["Configuration", "Intervention", "Best epoch", "CG Dice", "PZ Dice", "Macro Dice"],
        [[k.upper() if k != "baseline" else "Baseline", d["label"], str(d["epoch"]),
          f"{d['cg']:.4f}", f"{d['pz']:.4f}", f"{d['macro']:.4f}"]
         for k, d in a.items()])))

    add(("h3", "Segmentation Performance on the Held-Out Test Set"))
    add(("p",
        f"The frozen model was evaluated once on the {t['n_cases']} held-out cases. Mean (SD) "
        f"Dice was {cg_d['mean']:.4f} ({cg_d['sd']:.4f}) for the central gland and "
        f"{pz_d['mean']:.4f} ({pz_d['sd']:.4f}) for the peripheral zone, giving a macro Dice of "
        f"{t['macro']['mean']:.4f}. Background Dice was {bg_d['mean']:.4f} ({bg_d['sd']:.4f}) and "
        f"is reported separately. Full metrics are given in Table 3."))
    add(("p",
        f"The two zones behave differently, and the pattern is consistent. The peripheral zone "
        f"combines high precision ({t['per_class']['pz']['precision']['mean']:.4f}) with "
        f"markedly lower recall ({t['per_class']['pz']['recall']['mean']:.4f}), whereas the "
        f"central gland is approximately balanced "
        f"({t['per_class']['cg']['precision']['mean']:.4f} and "
        f"{t['per_class']['cg']['recall']['mean']:.4f}). What the model labels as peripheral "
        f"zone is usually correct; it simply labels too little of it. This is the expected "
        f"difficulty profile for a thin peripheral structure whose boundary occupies a small "
        f"fraction of the image."))
    add(("p",
        f"Per-case macro Dice ranged from {t['macro']['min']:.4f} to {t['macro']['max']:.4f}. "
        f"Central-gland Dice ranged from {t['cg_range'][0]:.4f} to {t['cg_range'][1]:.4f} and "
        f"peripheral-zone Dice from {t['pz_range'][0]:.4f} to {t['pz_range'][1]:.4f}. The "
        f"variation between patients is substantially larger than the variation between the "
        f"training configurations compared in Table 2."))
    add(("p",
        f"Against the baseline on the same cases, the selected configuration improved macro Dice "
        f"by {tb['macro']['delta']:+.4f}, better in {tb['macro']['better_in']} of {tb['n']} cases "
        f"(W={tb['macro']['W']:.0f}, P={tb['macro']['p']:.3f}). This pre-specified comparison "
        f"does not reach the conventional threshold, and we therefore describe the improvement as "
        f"consistent in direction rather than statistically established. A post hoc comparison "
        f"restricted to the peripheral zone gives {tb['pz']['delta']:+.4f}, better in "
        f"{tb['pz']['better_in']} of {tb['n']} cases (W={tb['pz']['W']:.0f}, "
        f"P={tb['pz']['p']:.3f}), suggesting that what difference exists is concentrated in the "
        f"harder zone; with three comparisons a corrected threshold would be 0.0167, so this "
        f"value is reported as exploratory."))

    add(("tbl", (f"Table 3. Segmentation performance of the selected model on the {t['n_cases']} "
                 f"held-out test cases. Volume-level, original voxel space, aggregated per case. "
                 f"Values are mean (SD), with 95% bootstrap confidence intervals for Dice.",
        ["Metric", "Central gland", "Peripheral zone"],
        [
            ["Dice", f"{cg_d['mean']:.4f} ({cg_d['sd']:.4f})",
                     f"{pz_d['mean']:.4f} ({pz_d['sd']:.4f})"],
            ["Dice, 95% CI", f"{cg_d['ci_low']:.4f}-{cg_d['ci_high']:.4f}",
                             f"{pz_d['ci_low']:.4f}-{pz_d['ci_high']:.4f}"],
        ] + [
            [name,
             f"{t['per_class']['cg'][m]['mean']:.4f} ({t['per_class']['cg'][m]['sd']:.4f})",
             f"{t['per_class']['pz'][m]['mean']:.4f} ({t['per_class']['pz'][m]['sd']:.4f})"]
            for m, name in (("iou", "IoU"), ("hd95_mm", "HD95 (mm)"), ("asd_mm", "ASD (mm)"),
                            ("precision", "Precision"), ("recall", "Recall"))
        ] + [["Macro Dice", f"{t['macro']['mean']:.4f}", ""],
             ["Background Dice", f"{bg_d['mean']:.4f} ({bg_d['sd']:.4f})", ""]])))

    add(("h3", "3D Reconstruction and Geometric Validation"))
    add(("p",
        f"{geo_all} reconstructions passed every geometric check. Array "
        f"shape, voxel spacing and orientation codes matched the source in every case, the label "
        f"set was confined to the three expected values, and the maximum absolute difference "
        f"between the reconstructed and source affine matrices was {g['affine_max_abs_diff']:.1f} "
        f"across the cohort - exact agreement rather than agreement within tolerance, which is "
        f"meaningful because the comparison is made against the affine as stored in the output "
        f"file's own single-precision header."))
    add(("p",
        "The round-trip fidelity test passed: the expert annotation, passed forward through "
        "preprocessing and back through the same reconstruction code, was recovered exactly. We "
        "report this as a verification outcome rather than as a metric. The accuracy of the "
        "segmentation itself is the figure given in Table 3, and the two should not appear in the "
        "same table."))

    add(("h3", "What the Reconstruction Steps Are Worth"))
    add(("p",
        "The ablation rebuilt the same predictions with individual reconstruction operations "
        "omitted. The reference reconstruction, rebuilt independently from the saved predictions "
        "and metadata, was bit-identical to the volume written by the main run in every case, so "
        "the comparisons below are made against a reproducible reference."))
    add(("p",
        f"Omitting the inverse orientation step produced a volume that is anatomically mirrored. "
        f"Its reported Dice against the expert annotation fell from {cg_d['mean']:.4f} to "
        f"{v1['cg']['dice_vs_gt']:.4f} for the central gland and from {pz_d['mean']:.4f} to "
        f"{v1['pz']['dice_vs_gt']:.4f} for the peripheral zone, and the peripheral zone was "
        f"displaced by up to {v1['pz']['centroid_shift_max_mm']:.0f} mm. The geometric checklist "
        f"flagged {v1['cg']['n_caught']} of {v1['cg']['n_cases']} such cases. This is the central "
        f"finding of the ablation: the checklist inspects the file header, the header is "
        f"correct, and only the voxel data is wrong, so the error is invisible to it."))
    add(("p",
        f"Omitting affine restoration produces the opposite pattern. The voxels are untouched, so "
        f"Dice against the annotation is unchanged at {v3['cg']['dice_vs_gt']:.4f} and "
        f"{v3['pz']['dice_vs_gt']:.4f} - Dice is computed on voxel indices and cannot see an "
        f"affine at all. Yet the gland is displaced by {v3['cg']['centroid_shift_mm']:.0f} mm in "
        f"scanner coordinates and every zone volume is inflated by "
        f"{v3['cg']['volume_pct_error']:.0f}%, because a unit isotropic voxel is not the voxel "
        f"the scanner acquired. Here the checklist flagged all "
        f"{v3['cg']['n_caught']} of {v3['cg']['n_cases']} cases."))
    add(("p",
        "The two failure modes are therefore exactly complementary. Overlap metrics detect the "
        "mirrored volume and are blind to the lost affine; the geometric checklist detects the "
        "lost affine and is blind to the mirrored volume. A pipeline reporting only accuracy "
        "metrics and a pipeline validating only headers are each blind to one of them, which is "
        "the argument for performing both checks and for the round-trip fidelity test that "
        "compares voxels rather than headers."))
    add(("p",
        "Two of the four shortcuts caused no measurable damage on this cohort, and the reasons "
        "differ in a way worth distinguishing. Centre-cropping is safe structurally, because this "
        "pipeline always places padding at the centre, so a centre crop cannot diverge from the "
        "recorded offsets for a padded case - but it still cannot invert a forward crop, which a "
        "cohort with larger images would produce. Stacking in loader order is safe only "
        "circumstantially, because this loader happens to emit slices in axial order; that is a "
        "property of the loader, not of the reconstruction, and it can change without any change "
        "to the reconstruction code. Neither should be read as evidence that the step is "
        "unnecessary."))
    add(("fig", (f"{FIG_DIR_RQ2}/F1_reconstruction_ablation.png",
        "Figure 4. The measured cost of each omitted reconstruction step. Panel A gives the Dice "
        "each shortcut would have reported against the expert annotation, from identical "
        "predictions, next to the correct pipeline. Panel B shows which check detects which "
        "failure.")))
    add(("fig", (f"{FIG_DIR_RQ2}/F4_mirrored_volume_passes.png",
        "Figure 5. The same case reconstructed correctly (upper row) and with the inverse "
        "orientation step omitted (lower row). Both volumes pass every geometric check, because "
        "the header is correct in each and only the voxel data differs. The lower volume is "
        "anatomically mirrored and its peripheral-zone Dice collapses.")))

    add(("tbl", ("Table 4. Reconstruction ablation. Every variant is rebuilt from the same 2D "
                 "predictions, so differences are attributable to the reconstruction stage alone. "
                 "Reported Dice is what a reader would have seen against the expert annotation.",
        ["Shortcut", "Reported CG Dice", "Reported PZ Dice", "Centroid shift (mm)",
         "Volume error (%)", "Flagged by checklist"],
        [[lbl,
          f"{ab[k]['cg']['dice_vs_gt']:.4f}", f"{ab[k]['pz']['dice_vs_gt']:.4f}",
          f"{ab[k]['cg']['centroid_shift_mm']:.1f}",
          f"{ab[k]['cg']['volume_pct_error']:.0f}",
          f"{ab[k]['cg']['n_caught']}/{ab[k]['cg']['n_cases']}"]
         for k, lbl in [("V1_no_reorientation", "Inverse orientation omitted"),
                        ("V2_naive_centre_crop", "Centre crop instead of recorded offsets"),
                        ("V3_identity_affine", "Affine not restored"),
                        ("V4_append_order", "Slices stacked in loader order"),
                        ("V5_all_naive", "All four combined")]
         ] + [["None (correct pipeline)",
               f"{v1['cg']['dice_vs_gt_correct']:.4f}", f"{v1['pz']['dice_vs_gt_correct']:.4f}",
               "0.0", "0", "0/19"]])))

    add(("h3", "Anatomical Zone Volumes"))
    add(("p",
        f"Across the {v['n']} held-out cases the expert annotation gives a mean whole-gland "
        f"volume of {v['whole']['gt_mean']:.2f} mL, of which {v['cg']['gt_mean']:.2f} mL is "
        f"central gland and {v['pz']['gt_mean']:.2f} mL peripheral zone. The model "
        f"under-estimates all three. Whole-gland volume is biased by {v['whole']['bias']:.2f} mL "
        f"({v['whole']['pct_of_gt']:.1f}% of the reference), under-estimated in "
        f"{v['whole']['under_in']} of {v['n']} cases, with 95% limits of agreement from "
        f"{v['whole']['loa_low']:.2f} to {v['whole']['loa_high']:.2f} mL."))
    add(("p",
        f"The peripheral zone is affected most: a bias of {v['pz']['bias']:.2f} mL, "
        f"{abs(v['pz']['pct_of_gt']):.1f}% of its true volume, under-estimated in "
        f"{v['pz']['under_in']} of {v['n']} cases. This is the low recall of Table 3 expressed in "
        f"millilitres. The central gland is comparatively well estimated, with a bias of "
        f"{v['cg']['bias']:.2f} mL ({abs(v['cg']['pct_of_gt']):.1f}%)."))
    add(("p",
        f"The consequence follows from the arithmetic of PSA density, which divides serum PSA by "
        f"gland volume. Under-estimating the denominator inflates the quotient: across these "
        f"cases PSA density would be over-estimated by {v['psad']['mean_pct']:.1f}% on average "
        f"and by up to {v['psad']['max_pct']:.0f}% in the worst case, biased upward in "
        f"{v['psad']['over_in']} of {v['n']} cases. Against a commonly cited threshold of 0.15 "
        f"ng/mL/cc, a patient whose true density is {v['psad']['threshold_equivalent']:.3f} would "
        f"read at the threshold. The direction is toward more biopsies rather than fewer, which "
        f"is the safer direction clinically but is nonetheless a systematic mis-calibration."))
    add(("p",
        "We emphasise the limits of agreement over the bias. A bias of a few millilitres averages "
        "out across a cohort; limits of agreement spanning more than 27 mL on a gland of roughly "
        "54 mL describe what can happen to an individual patient, and it is the individual "
        "patient for whom PSA density is computed."))
    add(("fig", (f"{FIG_DIR_RQ2}/F2_zone_volume_agreement.png",
        "Figure 6. Zone-volume agreement. Bland-Altman plots for the central gland and peripheral "
        "zone, and the resulting per-case error in PSA density.")))

    add(("tbl", (f"Table 5. Zone volumes on the {v['n']} held-out cases, in millilitres, with "
                 f"Bland-Altman agreement against the expert annotation. Limits of agreement are "
                 f"bias +/- 1.96 SD.",
        ["Quantity", "Reference, mean (SD)", "Predicted, mean (SD)", "Bias", "% of reference",
         "95% limits of agreement"],
        [[name,
          f"{v[k]['gt_mean']:.2f} ({v[k]['gt_sd']:.2f})",
          f"{v[k]['pred_mean']:.2f} ({v[k]['pred_sd']:.2f})",
          f"{v[k]['bias']:+.2f}", f"{v[k]['pct_of_gt']:+.1f}",
          f"{v[k]['loa_low']:.2f} to {v[k]['loa_high']:.2f}"]
         for k, name in (("cg", "Central gland"), ("pz", "Peripheral zone"),
                         ("whole", "Whole gland"))])))

    add(("h3", "External Validation"))
    add(("p",
        f"Applied once to {e['n_cases']} PROSTATEx examinations without any adaptation, the "
        f"frozen model achieved a mean (SD) Dice of {e['cg']['dice']['mean']:.4f} "
        f"({e['cg']['dice']['sd']:.4f}) for the central gland and {e['pz']['dice']['mean']:.4f} "
        f"({e['pz']['dice']['sd']:.4f}) for the peripheral zone, a macro Dice of "
        f"{e['macro']['mean']:.4f}. Relative to the internal held-out result this is a decrease "
        f"of {t['macro']['mean'] - e['macro']['mean']:.4f} in macro Dice."))
    add(("p",
        f"The zonal pattern observed internally persists and intensifies. Peripheral-zone recall "
        f"falls to {e['pz']['recall']['mean']:.4f} while its precision rises to "
        f"{e['pz']['precision']['mean']:.4f}: on unfamiliar data the model becomes more "
        f"conservative, labelling less and being right more often about what it does label."))
    add(("p",
        f"Overlap transfers, but the boundary metric does not transfer in the same way, and the "
        f"distinction is informative. Central-gland HD95 has a median of "
        f"{e['hd95_cg']['median']:.1f} mm - close to the internal value - but a mean of "
        f"{e['hd95_cg']['mean']:.1f} mm and a maximum of {e['hd95_cg']['max']:.1f} mm. "
        f"{e['hd95_cg']['n_over_30']} of {e['n_cases']} cases exceed 30 mm. This is a heavy tail "
        f"rather than a uniform degradation, and the affected cases retain high Dice, which "
        f"indicates small volumes of spurious prediction far from the gland rather than a "
        f"boundary that is uniformly wrong. HD95 correlates negatively with precision across the "
        f"cohort (r={e['hd95_cg']['corr_with_precision']:+.2f}), consistent with that reading."))
    add(("p",
        f"The most likely mechanism is the field-of-view difference. The external examinations "
        f"cover {eg['fov_mean']:.0f} mm in plane against {eg['fov_internal']:.1f} mm internally, "
        f"a factor of {eg['fov_ratio']:.2f}, so they contain pelvic structures the model never "
        f"encountered during training, and it is in that additional territory that spurious "
        f"predictions can appear. We state this as a hypothesis consistent with the measurements "
        f"rather than as an established cause; confirming it would require connected-component "
        f"analysis of the affected volumes, which we did not perform."))
    add(("p",
        f"Reconstruction behaved correctly on this cohort despite its different geometry. "
        f"{eg['n_cropped']} of {e['n_cases']} examinations are large enough that the in-plane "
        f"standardization crops rather than pads them, exercising a branch of the transform that "
        f"the development data never triggers, and no case lost annotated anatomy to that crop. "
        f"The cohort is also acquired in a different orientation convention from the development "
        f"data, which the reconstruction handles through the same inverse transform."))
    add(("p",
        "These figures are not a like-for-like comparison with the internal result. The "
        "annotators, scanners, acquisition geometry and zonal definitions all differ, and the "
        "external annotations were produced by a different group under a different protocol. The "
        "quantity of interest is the size of the gap, interpreted alongside the covariate shift "
        "that partly explains it."))
    add(("fig", (f"{FIG_DIR_RQ2}/F3_external_validation.png",
        "Figure 7. External validation. Internal and external Dice side by side; the distribution "
        "of central-gland HD95 on the external cohort, showing a heavy tail rather than a uniform "
        "shift; and the relationship between HD95 and precision.")))

    add(("tbl", (f"Table 6. Internal versus external performance. Internal: {t['n_cases']} "
                 f"Prostate158 held-out cases. External: {e['n_cases']} PROSTATEx examinations, "
                 f"inference only. Values are mean (SD).",
        ["Metric", "Internal, CG", "External, CG", "Internal, PZ", "External, PZ"],
        [[name,
          f"{t['per_class']['cg'][m]['mean']:.4f} ({t['per_class']['cg'][m]['sd']:.4f})",
          f"{e['cg'][m]['mean']:.4f} ({e['cg'][m]['sd']:.4f})",
          f"{t['per_class']['pz'][m]['mean']:.4f} ({t['per_class']['pz'][m]['sd']:.4f})",
          f"{e['pz'][m]['mean']:.4f} ({e['pz'][m]['sd']:.4f})"]
         for m, name in (("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)"),
                         ("asd_mm", "ASD (mm)"), ("precision", "Precision"),
                         ("recall", "Recall"))])))

    add(("h3", "Qualitative Results"))
    add(("p",
        f"Representative cases were selected by mean foreground Dice rather than by appearance: "
        f"case {t['representatives']['good']} is the highest-scoring, case "
        f"{t['representatives']['median']} the median, and case "
        f"{t['representatives']['challenging']} the lowest. In the highest-scoring case both "
        f"zones are well delineated and disagreement is confined to a thin boundary band. In the "
        f"lowest-scoring case the central gland is recovered while the peripheral zone is "
        f"substantially under-segmented, with the central gland extending into territory the "
        f"annotation assigns to the peripheral zone - a class confusion at the zonal interface "
        f"rather than a failure to find the prostate."))
    add(("fig", (f"{VIZ}/patient_{t['representatives']['good']}/"
                 f"patient_{t['representatives']['good']}_slice_016.png",
        f"Figure 8. Representative high-performing case (case {t['representatives']['good']}). "
        f"From left: T2-weighted image, expert annotation, prediction, overlay, and error map.")))
    add(("fig", (f"{VIZ}/patient_{t['representatives']['challenging']}/"
                 f"patient_{t['representatives']['challenging']}_slice_015.png",
        f"Figure 9. Representative challenging case (case "
        f"{t['representatives']['challenging']}). The central gland is recovered while the "
        f"peripheral zone is under-segmented at the zonal interface.")))

    # ----------------------------------------------------------- discussion
    add(("h2", "Discussion"))

    add(("h3", "Principal Results"))
    add(("p",
        f"A standard 2D U-Net segmented the central gland with a Dice of {cg_d['mean']:.4f} and "
        f"the peripheral zone with a Dice of {pz_d['mean']:.4f} on a held-out split, and three "
        f"single-factor training interventions changed performance by less than the variation "
        f"between patients. We regard the second half of that sentence as the more useful finding "
        f"for RQ1. The four configurations spanned {N['arm_spread']:.4f} macro Dice against a "
        f"within-configuration per-case spread several times larger, which means that on this "
        f"dataset, at this scale, the loss function and the augmentation policy are not where the "
        f"remaining error lives. The limiting factor is boundary delineation in the peripheral "
        f"zone, and the consistent pattern of high precision with low recall locates it "
        f"specifically: the model is not mistaking other tissue for peripheral zone, it is "
        f"failing to claim enough of it."))
    add(("p",
        "For RQ2 the principal result is that geometric validation, which is the verification "
        "most commonly reported when it is reported at all, is not sufficient. A reconstruction "
        "with a perfectly correct header and mirrored voxel data passes every check in the "
        "checklist - shape, affine, spacing, orientation codes and label set - while its reported "
        "Dice falls by roughly a quarter for the central gland and by three quarters for the "
        "peripheral zone. The complementary failure, a correct voxel array written with a default "
        "affine, is invisible to overlap metrics because those metrics are computed on voxel "
        "indices, yet it displaces the gland by nearly two hundred millimetres and inflates every "
        "volume by almost half."))
    add(("p",
        "This is what gives the round-trip fidelity test its purpose. Passing the expert "
        "annotation forward through preprocessing and back through the same reconstruction code, "
        "and requiring it back unchanged, is the only check in this pipeline that compares voxels "
        "rather than headers and therefore the only one that detects the mirrored volume. Its "
        "outcome is a pass or a fail, not an accuracy, and we deliberately keep it out of the "
        "results tables for that reason."))

    add(("h3", "Comparison With Prior Work"))
    add(("p",
        "The accuracy reported here is below that of the Prostate158 reference implementation, "
        "which used a three-dimensional residual U-Net and reported a central-gland Dice of "
        "approximately 0.877 and a peripheral-zone Dice of approximately 0.754 [7]. The difference is "
        "consistent with the architectural difference: a 3D network can use through-plane context "
        "that a 2D network does not see. We therefore do not present the present results as "
        "competitive with that reference, and the comparison is offered as context rather than as "
        "a claim."))
    add(("p",
        "The relationship between the two zones is also reported inconsistently across studies on "
        "this dataset. Several report the central gland as the easier zone; at least one reports "
        "the reverse. Our results place the central gland as clearly easier, and the ordering is "
        "stable across both cohorts examined here. Part of the inter-study inconsistency is "
        "plausibly attributable to protocol rather than to model: aggregation per slice rather "
        "than per case, evaluation in a resampled grid rather than in original voxel space, and "
        "the reading of the label integers themselves all shift the reported figure. We verified "
        "the label mapping explicitly for that reason."))
    add(("p",
        "On the reconstruction side, the literature treats spatial consistency predominantly as a "
        "model property, addressed through cross-slice attention or post-processing. The "
        "reassembly step itself, and its verification, is usually described in a sentence if at "
        "all. We are not aware of prior work that measures what the individual operations in that "
        "step are worth by omitting them. That measurement is what we consider this study's "
        "transferable contribution, and it is not specific to prostate imaging: any pipeline that "
        "produces slice-wise predictions and reassembles them is exposed to the same two failure "
        "modes."))

    add(("h3", "Clinical Implications"))
    add(("p",
        "Dice is a research metric. The quantity through which zonal segmentation reaches "
        "clinical practice is volume, and the analysis here shows that the translation is not "
        "neutral. Under-segmentation of the peripheral zone by roughly a quarter of its volume, "
        "and of the whole gland by about an eighth, inflates PSA density by approximately a "
        "seventh on average. That error runs toward more biopsies rather than fewer, which is the "
        "safer direction, but it is systematic rather than random and it is large enough to move "
        "patients across a decision threshold."))
    add(("p",
        "The limits of agreement are the operative quantity for individual use, and they are wide "
        "relative to the gland. On that basis we would not propose that volumes produced by this "
        "model be used for PSA density without correction or human review. The constructive "
        "reading is that reporting volumes in millilitres with explicit limits of agreement makes "
        "this visible, whereas reporting Dice alone does not."))

    add(("h3", "Limitations"))
    add(("p",
        "The development cohort is a single public dataset of 158 examinations, and the held-out "
        "split contains 19 cases. Estimates from that sample are imprecise, and we report "
        "confidence intervals rather than point estimates alone for that reason. The architecture "
        "is standard and no architectural search was performed."))
    add(("p",
        "The held-out split was not pristine. It had been used twice previously during pipeline "
        "development, before the four training configurations compared here were designed, and "
        "those earlier uses involved a different model. The selection rule for the present "
        "comparison was fixed in advance and used validation data only, and no test metric "
        "contributed to it. We disclose the earlier use rather than claiming an untouched test "
        "set."))
    add(("p",
        "The statistical comparisons are limited by sample size and by the structure of the "
        "study. The pre-specified test of the selected configuration against the baseline did not "
        "reach the conventional threshold. Tests computed on the validation split are circular "
        "and reported as descriptive only; per-zone tests on the held-out split are post hoc and "
        "would not survive correction for multiple comparisons. None of these should be read as "
        "establishing superiority."))
    add(("p",
        "The external cohort was annotated by a different group under a different protocol, so "
        "the external figures confound model generalization with annotation differences and "
        "cannot be decomposed into the two. The pipeline performs no in-plane resampling, so part "
        "of the external gap is attributable to pixel-scale shift rather than to model failure; "
        "separating the two would require a second run with a resampling adapter, which we did "
        "not perform. The mechanism proposed for the heavy tail in external boundary error is a "
        "hypothesis consistent with the measurements, not a demonstrated cause."))
    add(("p",
        "Finally, no reader study was conducted, and no claim is made about clinical utility, "
        "deployment readiness or effect on patient outcomes. The volumetric analysis establishes "
        "the magnitude of a potential error, not its clinical consequence in practice."))

    add(("h3", "Conclusions"))
    add(("p",
        f"A 2D U-Net segmented the central gland with a Dice of {cg_d['mean']:.4f} and the "
        f"peripheral zone with a Dice of {pz_d['mean']:.4f} on a held-out split of Prostate158, "
        f"and retained a macro Dice of {e['macro']['mean']:.4f} on an independent cohort of "
        f"{e['n_cases']} examinations without adaptation. Three single-factor training "
        f"interventions produced no improvement exceeding between-patient variation, which "
        f"locates the remaining error in boundary delineation rather than in the training "
        f"objective."))
    add(("p",
        "The reconstruction of slice-wise predictions into a spatially valid volume was "
        "implemented with verification at every stage and validated on both cohorts. Measuring "
        "what each verification step is worth shows that geometric validation and overlap metrics "
        "are each blind to a different class of reconstruction error, so neither is sufficient "
        "alone, and that a model-free round-trip fidelity test is what detects a volume whose "
        "header is correct and whose voxels are not. Because zone volumes enter PSA density as a "
        "denominator, a reconstruction error invisible to Dice can still change a clinical "
        "quantity substantially. We therefore recommend that studies reporting 3D results derived "
        "from 2D segmentation state how the reconstruction was verified, and report volumes with "
        "limits of agreement alongside overlap metrics."))

    # ----------------------------------------------------------- back matter
    add(("h2", "Acknowledgments"))
    add(("p",
        "The authors thank the creators of the Prostate158 dataset and of the PROSTATEx zonal "
        "annotations for making their data and expert annotations publicly available. No external "
        "funding was received for this work."))

    add(("h2", "Conflicts of Interest"))
    add(("p", "None declared."))

    add(("h2", "Data and Code Availability"))
    add(("p",
        "Both datasets used in this study are publicly available from their original sources "
        "under their respective licences. All analysis code, configuration files, evaluation "
        "scripts and the figure-generation code are available at "
        "https://github.com/amadeussandro/prostate-mri-segmentation-thesis. Every quantitative "
        "value reported in this manuscript is derived programmatically from the result artifacts "
        "produced by that code. The frozen model checkpoint is identified by the SHA-256 hash "
        "recorded in the repository, so that the model used for every analysis reported here can "
        "be verified rather than assumed."))

    add(("h2", "Abbreviations"))
    add(("p",
        "ASD: average surface distance. CG: central gland. HD95: 95th-percentile Hausdorff "
        "distance. IoU: intersection over union. MRI: magnetic resonance imaging. PSA: "
        "prostate-specific antigen. PZ: peripheral zone. SD: standard deviation."))

    add(("h2", "References"))
    add(("refs", [
        "Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, Bray F. Global "
        "Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 "
        "Cancers in 185 Countries. CA Cancer J Clin. 2021;71(3):209-249. doi:10.3322/caac.21660",

        "Turkbey B, Rosenkrantz AB, Haider MA, Padhani AR, Villeirs G, Macura KJ, et al. Prostate "
        "Imaging Reporting and Data System Version 2.1: 2019 Update of Prostate Imaging Reporting "
        "and Data System Version 2. Eur Urol. 2019;76(3):340-351. doi:10.1016/j.eururo.2019.02.033",

        "Weinreb JC, Barentsz JO, Choyke PL, Cornud F, Haider MA, Macura KJ, et al. PI-RADS "
        "Prostate Imaging - Reporting and Data System: 2015, Version 2. Eur Urol. "
        "2016;69(1):16-40. doi:10.1016/j.eururo.2015.08.052",

        "McNeal JE. The zonal anatomy of the prostate. Prostate. 1981;2(1):35-49. "
        "doi:10.1002/pros.2990020105",

        "Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image "
        "Segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI). "
        "Lecture Notes in Computer Science, vol 9351. Springer; 2015:234-241. "
        "doi:10.1007/978-3-319-24574-4_28",

        "Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring "
        "method for deep learning-based biomedical image segmentation. Nat Methods. "
        "2021;18(2):203-211. doi:10.1038/s41592-020-01008-z",

        "Adams LC, Makowski MR, Engel G, Rattunde M, Busch F, Asbach P, et al. Prostate158 - An "
        "expert-annotated 3T MRI dataset and algorithm for prostate cancer detection. Comput Biol "
        "Med. 2022;148:105817. doi:10.1016/j.compbiomed.2022.105817",

        "Litjens G, Debats O, Barentsz J, Karssemeijer N, Huisman H. Computer-aided detection of "
        "prostate cancer in MRI. IEEE Trans Med Imaging. 2014;33(5):1083-1092. "
        "doi:10.1109/TMI.2014.2303821",

        "Cuocolo R, Stanzione A, Castaldo A, De Lucia DR, Imbriaco M. Quality control and "
        "whole-gland, zonal and lesion annotations for the PROSTATEx challenge public dataset. "
        "Eur J Radiol. 2021;146:109647. doi:10.1016/j.ejrad.2021.109647",

        "Cuocolo R, Comelli A, Stefano A, Benfante V, Dahiya N, Stanzione A, et al. Deep Learning "
        "Whole-Gland and Zonal Prostate Segmentation on a Public MRI Dataset. J Magn Reson "
        "Imaging. 2021;54(2):452-459. doi:10.1002/jmri.27585",

        "Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, et al. The Cancer Imaging "
        "Archive (TCIA): maintaining and operating a public information repository. J Digit "
        "Imaging. 2013;26(6):1045-1057. doi:10.1007/s10278-013-9622-7",

        "Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of "
        "clinical measurement. Lancet. 1986;1(8476):307-310. doi:10.1016/S0140-6736(86)90837-8",

        "Nordstrom T, Akre O, Aly M, Gronberg H, Eklund M. Prostate-specific antigen (PSA) "
        "density in the diagnostic algorithm of prostate cancer. Prostate Cancer Prostatic Dis. "
        "2018;21(1):57-63. doi:10.1038/s41391-017-0024-7",

        "Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, "
        "selection, and tool. BMC Med Imaging. 2015;15:29. doi:10.1186/s12880-015-0068-x",

        "Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E, et al. Metrics "
        "reloaded: recommendations for image analysis validation. Nat Methods. "
        "2024;21(2):195-212. doi:10.1038/s41592-023-02151-z",

        "Cardoso MJ, Li W, Brown R, Ma N, Kerfoot E, Wang Y, et al. MONAI: An open-source "
        "framework for deep learning in healthcare. arXiv. 2022. doi:10.48550/arXiv.2211.02701",
    ]))

    return B
