"""Build the additional skripsi figures and tables from frozen artifacts.

The thesis presented ten figures while the project had already computed far
more, and research question 1 - the four-arm comparison - carried no figure at
all. This script closes that gap. It generates two new plots, adopts the seven
RQ1 figures that were built but never used, composes the 3D renders the
reconstruction chapter never showed, and emits every new table as LaTeX from the
result CSVs, so no number in the thesis is typed by hand.

Read-only with respect to results/. Everything it writes lands in the skripsi
template folder.

    python scripts/make_skripsi_extra.py
"""
import csv
import io
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")

SKRIPSI = r"D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840"
PICS = os.path.join(SKRIPSI, "assets", "pics")
GEN = os.path.join(SKRIPSI, "src", "_generated")
# The canonical copy of every figure this script draws lives in the repo, so
# the extended manuscript can use it; the thesis receives a copy.
FIG_EXT = os.path.join(RESULTS, "figures_extended")

INK = "#1b1b1b"
GRID = "#d9d9d9"
# Distinguishable without relying on hue alone: each arm also gets its own
# line style, so the figure survives greyscale printing and colour blindness.
ARMS = [
    ("E1", "exp_e1_dice_ce/metrics.csv", "#1f4e79", "-", "E1 (+SoftDice)"),
    ("E2", "exp_e2_ce_aug/metrics.csv", "#b06000", "--", "E2 (+augmentasi)"),
    ("E3", "exp_e3_dice_ce_pzweight/exp_e3_dice_ce_pzweight/metrics.csv",
     "#4a7c22", ":", "E3 (+bobot kelas)"),
]

SELECTED_EPOCH = 88


def load_metrics(relpath):
    rows = []
    with io.open(os.path.join(RESULTS, relpath), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss_total"]),
                "val_loss": float(r["val_loss"]),
                "val_dice": float(r["val_dice_proxy"]),
                "cg": float(r["val_dice_class1_CG"]),
                "pz": float(r["val_dice_class2_PZ"]),
            })
    rows.sort(key=lambda d: d["epoch"])
    return rows


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=INK)


# ---------------------------------------------------------------- figure 1
# Each arm optimises a different objective, so their losses are on different
# scales by construction and must not share an axis: E2 has no Dice term at
# all, which makes its total loss look an order of magnitude "better" than E3's
# while meaning nothing of the kind. Losses therefore get one panel each, and
# only validation Dice - the same quantity for every arm - is overlaid.
LOSS_NAME = {
    "E1": "CE + SoftDice",
    "E2": "CE only, augmented data",
    "E3": "class-weighted CE + SoftDice",
}


def fig_training_curves(data, out):
    """Per-arm training loss (own scale) and shared validation Dice."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(11.6, 7.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.25],
                  hspace=0.52, wspace=0.32)

    first_e1 = data["E1"][0]["epoch"]
    letters = "ABC"
    for i, (key, _, colour, dash, label) in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, i])
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                color=colour, linestyle=dash, linewidth=1.7)
        ax.set_xlim(0, 101)
        # The objective goes in the title, not inside the axes: E1's loss spikes
        # to the top of its own range at epoch 84 and ran straight through a
        # label placed in the plot area.
        style(ax, "%s  %s" % (letters[i], key), "Epoch", "Training loss")
        ax.set_title(LOSS_NAME[key], loc="right", fontsize=8, color="#444444")
        if key == "E1":
            ax.axvspan(0, first_e1, color="#bdbdbd", alpha=0.30, lw=0)
            ax.text(first_e1 / 2.0, 0.5, "epochs 1-%d\nnot logged" % (first_e1 - 1),
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="center", fontsize=7.8, color="#555555")

    ax = fig.add_subplot(gs[1, :])
    for key, _, colour, dash, label in ARMS:
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["val_dice"] for r in rows],
                color=colour, linestyle=dash, linewidth=1.7, label=label)
    ax.axvspan(0, first_e1, color="#bdbdbd", alpha=0.22, lw=0)

    sel = [r for r in data["E1"] if r["epoch"] == SELECTED_EPOCH][0]
    ax.plot([SELECTED_EPOCH], [sel["val_dice"]], marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor="#1f4e79",
            markeredgewidth=1.9, zorder=5)
    ax.annotate("E1, epoch %d: checkpoint frozen" % SELECTED_EPOCH,
                xy=(SELECTED_EPOCH, sel["val_dice"]),
                xytext=(SELECTED_EPOCH - 14, 0.30),
                fontsize=8.5, color="#1f4e79", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0))
    ax.set_xlim(0, 101)
    style(ax, "D  Batch-level validation Dice, all arms (the selection criterion)",
          "Epoch", "Batch-level validation Dice")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#bdbdbd", alpha=0.4,
                         label="E1 epochs 1-%d not logged" % (first_e1 - 1)))
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              loc="lower right", ncol=2)

    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 2
def fig_epoch_selection(data, out):
    """Why epoch 88 - and what a different criterion would have chosen.

    The training loop maximised the slice-level validation Dice. Epoch 88 is
    that maximum. The macro of the two per-class Dice values, which is what the
    thesis reports everywhere else, peaks one epoch group earlier. Stating this
    is cheaper than having an examiner notice it in the curve.
    """
    rows = data["E1"]
    ep = [r["epoch"] for r in rows]
    proxy = [r["val_dice"] for r in rows]
    macro = [(r["cg"] + r["pz"]) / 2.0 for r in rows]

    best_proxy = max(rows, key=lambda r: r["val_dice"])["epoch"]
    best_macro = max(rows, key=lambda r: (r["cg"] + r["pz"]) / 2.0)["epoch"]

    by_ep = {r["epoch"]: r for r in rows}
    y_proxy = by_ep[best_proxy]["val_dice"]
    y_macro = (by_ep[best_macro]["cg"] + by_ep[best_macro]["pz"]) / 2.0

    # Both curves are batch-level 2D quantities from the training loop. Neither
    # is the case-level 3D Dice the thesis reports in its tables, so neither
    # label may claim to be. The red curve is the slice-level ANALOGUE of the
    # reported macro, and only epochs 88 (best) and the final epoch were kept as
    # checkpoints - so whether epoch 81 would also rank higher at case level
    # cannot be tested, and the figure does not say it would.
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(ep, proxy, color="#1f4e79", linewidth=2.0,
            label="Batch-level foreground Dice (the selection criterion)")
    ax.plot(ep, macro, color="#9b2226", linewidth=1.7, linestyle="--",
            label="Batch-level macro of CG and PZ Dice")
    ax.axvline(best_proxy, color="#1f4e79", linewidth=1.0, alpha=0.45)
    ax.axvline(best_macro, color="#9b2226", linewidth=1.0, alpha=0.45,
               linestyle="--")
    ax.plot([best_proxy], [y_proxy], marker="o", markersize=7,
            markerfacecolor="#1f4e79", markeredgecolor="white", zorder=5)
    ax.plot([best_macro], [y_macro], marker="o", markersize=7,
            markerfacecolor="#9b2226", markeredgecolor="white", zorder=5)

    # Headroom above the data so both labels sit in empty space and never
    # cross a curve or the legend.
    lo = min(min(proxy), min(macro))
    hi = max(max(proxy), max(macro))
    span = hi - lo
    ax.set_ylim(lo - 0.04 * span, hi + 0.50 * span)
    top = hi + 0.34 * span

    ax.annotate("epoch %d: maximum of the criterion\ncheckpoint frozen here"
                % best_proxy,
                xy=(best_proxy, y_proxy), xytext=(best_proxy + 4.5, top),
                fontsize=8.3, color="#1f4e79", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0))
    ax.annotate("epoch %d: higher on this\nbatch-level macro (%.4f vs %.4f)"
                % (best_macro, y_macro,
                   (by_ep[best_proxy]["cg"] + by_ep[best_proxy]["pz"]) / 2.0),
                xy=(best_macro, y_macro), xytext=(best_macro - 6.5, top),
                fontsize=8.3, color="#9b2226", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.0))

    style(ax, "Model selection on the 20 validation cases, E1, epochs %d-%d"
          % (ep[0], ep[-1]), "Epoch", "Validation Dice")
    ax.legend(fontsize=8.3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out, best_proxy, best_macro


# ---------------------------------------------------------------- figure 3
# The reconstructed volumes, which the thesis never actually showed. The three
# cases are the objective selection made by run_rq2_reconstruction.py - highest,
# median and lowest mean foreground Dice - so they are not hand-picked. The
# thesis previously showed only the two extremes; the median is the typical
# outcome and is the one a reader most needs to see.
#
# One figure per case: stacked, the three renders are 1:2.2 and would run past
# the bottom of an A4 page at full text width.
PROJECTION_CASES = [
    ("002", "baik"),
    ("006", "median"),
    ("016", "sulit"),
]


def adopt_3d_projections():
    out = []
    for case, tag in PROJECTION_CASES:
        src = os.path.join(RESULTS, "rq2_e1_epoch88", "visualizations",
                           "patient_%s" % case,
                           "patient_%s_3d_projections.png" % case)
        if not os.path.exists(src):
            print("  LEWAT (tidak ada): proyeksi 3D kasus %s" % case)
            continue
        dst = os.path.join(PICS, "fig_proyeksi_3d_%s.png" % tag)
        shutil.copyfile(src, dst)
        out.append((case, tag, dst))
    return out


# ---------------------------------------------------------------- tables
def num(x, nd=4):
    """Indonesian decimal comma, fixed places."""
    return ("%.*f" % (nd, x)).replace(".", ",")


def write_tex(name, lines):
    path = os.path.join(GEN, name)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% Dibangkitkan oleh scripts/make_skripsi_extra.py.\n")
        fh.write("% Jangan disunting tangan: jalankan ulang skripnya.\n")
        fh.write("\n".join(lines) + "\n")
    return path


def table_per_case(out_name):
    """Per-case internal test metrics, 19 cases, for the appendix."""
    path = os.path.join(RESULTS, "rq2_e1_epoch88", "metrics_per_case.csv")
    per = {}
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cid = int(r["class_id"])
            if cid == 0:
                continue
            per.setdefault(r["patient_id"], {})[cid] = r
    cases = sorted(per)

    L = [r"\begin{longtable}{l c c c c}",
         r"\caption{Metrik per kasus model E1 pada 19 kasus data uji yang"
         r" dipisahkan, dihitung pada ruang voxel asli setelah rekonstruksi"
         r" 3D.}\label{tab:perkasus}\\",
         r"\hline",
         r"\textbf{Kasus} & \textbf{Dice CG} & \textbf{Dice PZ} &"
         r" \textbf{HD95 CG (mm)} & \textbf{HD95 PZ (mm)} \\",
         r"\hline",
         r"\endfirsthead",
         r"\hline",
         r"\textbf{Kasus} & \textbf{Dice CG} & \textbf{Dice PZ} &"
         r" \textbf{HD95 CG (mm)} & \textbf{HD95 PZ (mm)} \\",
         r"\hline",
         r"\endhead",
         r"\hline",
         r"\endfoot"]

    cg_d, pz_d = [], []
    for c in cases:
        a, b = per[c][1], per[c][2]
        cg_d.append(float(a["dice"]))
        pz_d.append(float(b["dice"]))
        L.append("%s & %s & %s & %s & %s \\\\" % (
            c, num(float(a["dice"])), num(float(b["dice"])),
            num(float(a["hd95_mm"]), 2), num(float(b["hd95_mm"]), 2)))

    mean_cg = sum(cg_d) / len(cg_d)
    mean_pz = sum(pz_d) / len(pz_d)
    L.append(r"\hline")
    L.append(r"\textbf{Rerata} & \textbf{%s} & \textbf{%s} & & \\"
             % (num(mean_cg), num(mean_pz)))
    L.append(r"\end{longtable}")
    return write_tex(out_name, L), len(cases), mean_cg, mean_pz


def table_external_full(out_name):
    """Every reported external metric with its confidence interval."""
    path = os.path.join(RESULTS, "external_prostatex_e1_v2", "metrics_summary.csv")
    rows = []
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["class_id"]) == 0:
                continue
            rows.append(r)

    labels = {"dice": "Dice", "iou": "IoU", "hd95_mm": "HD95 (mm)",
              "asd_mm": "ASD (mm)", "precision": r"\textit{Precision}",
              "recall": r"\textit{Recall}"}
    order = ["dice", "iou", "hd95_mm", "asd_mm", "precision", "recall"]
    by = {}
    for r in rows:
        by[(int(r["class_id"]), r["metric"])] = r

    n_cases = int(rows[0]["n_cases"])

    # The interval goes on its own line beneath each estimate. On one line a
    # cell reads "0,8137 (0,0935) [0,7999--0,8263]", and two such cells plus
    # the label column run to about 16.5 cm against a 14 cm text width.
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Seluruh metrik yang dilaporkan pada kohort eksternal"
         r" PROSTATEx (%d pemeriksaan, inferensi saja). Baris atas: rerata"
         r" (simpangan baku populasi). Baris bawah: selang kepercayaan 95\%%"
         r" \textit{bootstrap}.}" % n_cases,
         r"\label{tab:eksternal-lengkap}",
         r"\begin{tabular}{l c c}",
         r"\hline",
         r"\textbf{Metrik} & \textbf{\textit{Central gland}} &"
         r" \textbf{\textit{Peripheral zone}} \\",
         r"\hline"]
    for m in order:
        est, ci = [], []
        for cid in (1, 2):
            r = by[(cid, m)]
            nd = 2 if m.endswith("_mm") else 4
            est.append("%s (%s)" % (num(float(r["mean"]), nd),
                                    num(float(r["std"]), nd)))
            ci.append(r"{\footnotesize [%s--%s]}" % (
                num(float(r["ci95_low"]), nd), num(float(r["ci95_high"]), nd)))
        L.append("%s & %s & %s \\\\" % (labels[m], est[0], est[1]))
        L.append(" & %s & %s \\\\[2pt]" % (ci[0], ci[1]))
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return write_tex(out_name, L)


def network_input_size():
    """The in-plane size the network takes, read from the frozen E1 config."""
    import yaml
    path = os.path.join(PROJECT, "configs", "experiments", "exp_e1_dice_ce.yaml")
    with io.open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def find(node):
        if isinstance(node, dict):
            if "target_size" in node:
                return node["target_size"]
            for v in node.values():
                hit = find(v)
                if hit is not None:
                    return hit
        return None

    size = find(cfg)
    if not size:
        raise RuntimeError("target_size not found in %s" % path)
    return int(size[0]), int(size[1])


def table_cohort_geometry(out_name):
    """How the external cohort's geometry compares with the training data.

    Two columns of cohort_geometry.csv are easy to misread, so their labels
    say what they measure. `inplane_exceeds_target` is about the image MATRIX
    exceeding the network input, which forces a crop - not about resolution.
    `scale_vs_prostate158` is Prostate158's spacing over the case's spacing,
    i.e. how large the anatomy appears relative to the training data.
    """
    path = os.path.join(RESULTS, "external_prostatex_e1_v2", "cohort_geometry.csv")
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    n = len(rows)

    inplane = sorted({float(r["inplane_mm"]) for r in rows})
    through = sorted({float(r["through_plane_mm"]) for r in rows})
    orients = sorted({r["orientation"] for r in rows})
    scales = sorted({float(r["scale_vs_prostate158"]) for r in rows})
    ok = sum(1 for r in rows if r["geometry_all_ok"] == "True")
    cropped = [r for r in rows if r["inplane_exceeds_target"] == "True"]
    lost = sum(int(r["class1_lost"]) + int(r["class2_lost"]) for r in rows)
    lost_in_cropped = sum(int(r["class1_lost"]) + int(r["class2_lost"])
                          for r in cropped)
    affine = max(float(r["affine_max_abs_diff"]) for r in rows)

    # The training spacing is recoverable from every row (scale x spacing is
    # constant), so it is derived here rather than copied from a script.
    ref = {round(float(r["scale_vs_prostate158"]) * float(r["inplane_mm"]), 6)
           for r in rows}
    if len(ref) != 1:
        raise RuntimeError("inconsistent Prostate158 reference spacing: %s" % ref)
    ref_mm = ref.pop()

    matrix = sorted({int(r["shape"].strip("[]").split(",")[0]) for r in rows})
    net_h, net_w = network_input_size()
    crop_matrix = sorted({int(r["shape"].strip("[]").split(",")[0]) for r in cropped})
    crop_mm = sorted({float(r["inplane_mm"]) for r in cropped})

    def rng(vals, nd=4):
        if len(vals) == 1:
            return num(vals[0], nd)
        return "%s--%s" % (num(vals[0], nd), num(vals[-1], nd))

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Geometri kohort eksternal PROSTATEx, dihitung dari berkas"
         r" sumber setiap pemeriksaan. Faktor skala adalah spasi"
         r" \textit{in-plane} Prostate158 (%s mm) dibagi spasi kasus: nilai di"
         r" atas 1 berarti anatomi tampak lebih besar daripada pada data"
         r" pelatihan.}" % num(ref_mm, 5),
         r"\label{tab:geometri-kohort}",
         r"\small",
         r"\begin{tabular}{l c}",
         r"\hline",
         r"\textbf{Properti} & \textbf{Nilai (%d pemeriksaan)} \\" % n,
         r"\hline",
         r"Resolusi \textit{in-plane} (mm) & %s \\" % rng(inplane, 3),
         r"Ketebalan irisan (mm) & %s \\" % rng(through, 2),
         r"Faktor skala anatomi terhadap Prostate158 & %s \\" % rng(scales, 2),
         r"Ukuran matriks \textit{in-plane} (piksel) & %d--%d \\"
         % (matrix[0], matrix[-1]),
         r"Matriks melebihi masukan jaringan (%d $\times$ %d), dipotong"
         r" & %d dari %d \\" % (net_h, net_w, len(cropped), n),
         r"\quad di antaranya: matriks %s, resolusi %s mm & semua %d \\"
         % (" / ".join(str(m) for m in crop_matrix),
            " / ".join(num(m, 1) for m in crop_mm), len(cropped)),
         r"Voxel label yang hilang akibat pemotongan & %d \\" % lost_in_cropped,
         r"Voxel label yang hilang, seluruh kohort & %d \\" % lost,
         r"Orientasi & %s \\" % ", ".join(orients),
         r"Lolos seluruh pemeriksaan geometri & %d dari %d \\" % (ok, n),
         r"Selisih affine absolut maksimum & %s \\" % num(affine, 1),
         r"\hline", r"\end{tabular}", r"\end{table}"]
    return write_tex(out_name, L), n, ok, lost


# ---------------------------------------------------------------- adopt
ADOPT = [
    ("figures_rq1/01_validation_dice_overview.png", "fig_rq1_ikhtisar.png"),
    ("figures_rq1/02_validation_percase_spread.png", "fig_rq1_sebaran.png"),
    ("figures_rq1/03_validation_all_metrics.png", "fig_rq1_semua_metrik.png"),
    ("figures_rq1/04_e3_mechanism.png", "fig_e3_mekanisme.png"),
    ("figures_rq1/05_validation_vs_test.png", "fig_validasi_vs_uji.png"),
    ("figures_rq1/06_test_percase.png", "fig_uji_perkasus.png"),
    ("figures_rq1/07_rq1_journey.png", "fig_rq1_perjalanan.png"),
]


def main():
    os.makedirs(PICS, exist_ok=True)
    os.makedirs(GEN, exist_ok=True)

    data = {k: load_metrics(p) for k, p, _, _, _ in ARMS}
    print("riwayat pelatihan yang tersedia:")
    for k in ("E1", "E2", "E3"):
        print("  %-3s epoch %3d-%3d (%d epoch)"
              % (k, data[k][0]["epoch"], data[k][-1]["epoch"], len(data[k])))
    print("  Baseline: tidak ada riwayat pelatihan yang tersimpan")
    print()

    os.makedirs(FIG_EXT, exist_ok=True)
    p1 = fig_training_curves(data, os.path.join(FIG_EXT, "fig_training_curves.png"))
    shutil.copyfile(p1, os.path.join(PICS, "fig_kurva_pelatihan.png"))
    print("gambar: fig_kurva_pelatihan.png   (kanonik: %s)"
          % os.path.relpath(p1, PROJECT))

    p2, bp, bm = fig_epoch_selection(
        data, os.path.join(FIG_EXT, "fig_epoch_selection.png"))
    shutil.copyfile(p2, os.path.join(PICS, "fig_pemilihan_epoch.png"))
    print("gambar: fig_pemilihan_epoch.png   (kriteria -> epoch %d, makro -> epoch %d)"
          % (bp, bm))

    for case, tag, dst in adopt_3d_projections():
        print("gambar: %-28s <- kasus %s (%s)" % (os.path.basename(dst), case, tag))

    # The stacked composite from the first run is superseded and would not fit
    # on a page; remove it so it cannot be referenced by mistake.
    stale = os.path.join(PICS, "fig_proyeksi_3d.png")
    if os.path.exists(stale):
        os.remove(stale)
        print("dihapus: fig_proyeksi_3d.png (gabungan lama, terlalu tinggi)")

    for src, dst in ADOPT:
        s = os.path.join(RESULTS, src)
        if not os.path.exists(s):
            print("  LEWAT (tidak ada): %s" % src)
            continue
        shutil.copyfile(s, os.path.join(PICS, dst))
        print("diadopsi: %-28s <- %s" % (dst, src))

    print()
    t1, ncase, mcg, mpz = table_per_case("tabel_perkasus_internal.tex")
    print("tabel: %s  (%d kasus, rerata CG %s / PZ %s)"
          % (os.path.basename(t1), ncase, num(mcg), num(mpz)))

    t2 = table_external_full("tabel_eksternal_lengkap.tex")
    print("tabel: %s" % os.path.basename(t2))

    t3, n, ok, lost = table_cohort_geometry("tabel_geometri_kohort.tex")
    print("tabel: %s  (%d pemeriksaan, %d lolos geometri, %d voxel hilang)"
          % (os.path.basename(t3), n, ok, lost))
    return 0


if __name__ == "__main__":
    sys.exit(main())
