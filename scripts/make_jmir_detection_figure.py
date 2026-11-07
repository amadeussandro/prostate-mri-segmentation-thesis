"""Figure: which check detects which reconstruction fault, in 4 public cohorts.

Reads results/modelfree_ablation/per_case_all.csv. Cell = share of DAMAGED
cases flagged by the check; a fault that did no damage in a cohort is shown
as "no damage" rather than as a 0% miss.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = pd.read_csv(os.path.join(ROOT, "results/modelfree_ablation/per_case_all.csv"))
p = p[p.variant != "V0_correct"].copy()
p["damaged"] = (p.dice_CG < 1) | (p.dice_PZ < 1) | (p.shift_CG_mm > 0.01) | (p.shift_PZ_mm > 0.01) | (p.vol_err_CG_pct.abs() > 0.01)
p["either"] = p.L1_geometry_flags | p.L2_roundtrip_flags

faults = [("V1_no_reorientation", "Reorientation\nnot inverted"),
          ("V3_identity_affine", "Identity affine\nwritten"),
          ("V6_spacing_only", "Spacing-only\naffine"),
          ("V4b_shuffled_order", "Slices stacked\nout of order")]
checks = [("L1_geometry_flags", "Geometry\ncheck"), ("L2_roundtrip_flags", "Round-trip\ntest"),
          ("either", "Either of\nthe two"), ("L3a_flags", "PZ posterior\nto CG"),
          ("L3b_flags", "Slice\ncontinuity")]
cohorts = [("prostate158", "Prostate158 (LPS, n=158)"), ("prostatex", "PROSTATEx (LAS, n=204)"),
           ("msd_task05", "MSD Task05 (RAS, n=30)"), ("nci_isbi_2013", "NCI-ISBI 2013 (LPS, n=78)")]

ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue", ramp)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6), dpi=150)
for ax, (key, title) in zip(axes.ravel(), cohorts):
    for i, (fv, _) in enumerate(faults):
        g = p[(p.dataset == key) & (p.variant == fv)]
        dm = g[g.damaged]
        for j, (ck, _) in enumerate(checks):
            if len(dm) == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec", edgecolor="#fcfcfb",
                                           lw=2, hatch="////"))
                ax.text(j, i, "no\ndamage", ha="center", va="center", color="#52514e", fontsize=7.5)
                continue
            v = dm[ck].mean()
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(v), edgecolor="#fcfcfb", lw=2))
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    color="#fcfcfb" if v > 0.45 else "#0b0b0b", fontsize=8.5,
                    fontweight="bold" if ck == "either" else "normal")
    ax.set_xlim(-.5, len(checks) - .5); ax.set_ylim(len(faults) - .5, -.5)
    ax.set_xticks(range(len(checks))); ax.set_xticklabels([c[1] for c in checks], fontsize=7.5, color="#52514e")
    ax.set_yticks(range(len(faults))); ax.set_yticklabels([f[1] for f in faults], fontsize=7.5, color="#52514e")
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(2.5, color="#52514e", lw=0.8, ls=(0, (2, 2)))
    ax.set_title(title, fontsize=9, color="#0b0b0b", pad=30, loc="left")
fig.tight_layout(rect=(0, 0, 1, 1))
out = os.path.join(ROOT, "results/figures_jmir/F5_detection_matrix.png")
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(out)
