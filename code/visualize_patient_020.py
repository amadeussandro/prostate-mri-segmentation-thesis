import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless saving
import matplotlib.pyplot as plt

def visualize_patient_020():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    patient_dir = os.path.join(project_root, "dataset", "prostate158_train", "train", "020")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_png = os.path.join(results_dir, "020_patient_overview.png")

    print(f"Loading data from {patient_dir}...")
    # Load volumes
    t2_img = nib.load(os.path.join(patient_dir, "t2.nii.gz")).get_fdata()
    adc_img = nib.load(os.path.join(patient_dir, "adc.nii.gz")).get_fdata()
    dwi_img = nib.load(os.path.join(patient_dir, "dwi.nii.gz")).get_fdata()

    # Load masks
    t2_anat = nib.load(os.path.join(patient_dir, "t2_anatomy_reader1.nii.gz")).get_fdata()
    t2_tum = nib.load(os.path.join(patient_dir, "t2_tumor_reader1.nii.gz")).get_fdata()
    adc_tum1 = nib.load(os.path.join(patient_dir, "adc_tumor_reader1.nii.gz")).get_fdata()
    adc_tum2 = nib.load(os.path.join(patient_dir, "adc_tumor_reader2.nii.gz")).get_fdata()

    # Select representative slice: slice 7 contains non-zero voxels across all masks
    slice_idx = 7
    print(f"Selected slice index: {slice_idx} (axial)")

    # Extract 2D slices
    t2_s = np.rot90(t2_img[:, :, slice_idx])
    adc_s = np.rot90(adc_img[:, :, slice_idx])
    dwi_s = np.rot90(dwi_img[:, :, slice_idx])

    t2_anat_s = np.rot90(t2_anat[:, :, slice_idx])
    t2_tum_s = np.rot90(t2_tum[:, :, slice_idx])
    adc_tum1_s = np.rot90(adc_tum1[:, :, slice_idx])
    adc_tum2_s = np.rot90(adc_tum2[:, :, slice_idx])

    # Setup figure: 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Patient 020 MRI & Segmentation Overview (Slice {slice_idx}/24)", fontsize=16, fontweight="bold")

    # 1. T2 MRI raw
    ax = axes[0, 0]
    ax.imshow(t2_s, cmap="gray")
    ax.set_title("T2-weighted MRI (Raw)")
    ax.axis("off")

    # 2. T2 + Anatomy Mask
    ax = axes[0, 1]
    ax.imshow(t2_s, cmap="gray")
    # Mask out background (0)
    masked_anat = np.ma.masked_where(t2_anat_s == 0, t2_anat_s)
    im2 = ax.imshow(masked_anat, cmap="autumn", alpha=0.5, vmin=1, vmax=2)
    ax.set_title("T2 + Anatomy Mask (Reader 1)")
    ax.axis("off")

    # 3. T2 + T2 Tumor Mask
    ax = axes[0, 2]
    ax.imshow(t2_s, cmap="gray")
    masked_t2_tum = np.ma.masked_where(t2_tum_s == 0, t2_tum_s)
    ax.imshow(masked_t2_tum, cmap="Reds", alpha=0.6)
    ax.set_title("T2 + Tumor Mask (Reader 1, Val=3)")
    ax.axis("off")

    # 4. DWI MRI raw
    ax = axes[0, 3]
    ax.imshow(dwi_s, cmap="gray")
    ax.set_title("DWI MRI (Raw)")
    ax.axis("off")

    # 5. ADC MRI raw
    ax = axes[1, 0]
    ax.imshow(adc_s, cmap="gray")
    ax.set_title("ADC Map (Raw)")
    ax.axis("off")

    # 6. ADC + ADC Tumor Reader 1
    ax = axes[1, 1]
    ax.imshow(adc_s, cmap="gray")
    masked_adc_tum1 = np.ma.masked_where(adc_tum1_s == 0, adc_tum1_s)
    ax.imshow(masked_adc_tum1, cmap="cool", alpha=0.6)
    ax.set_title("ADC + Tumor Mask (Reader 1)")
    ax.axis("off")

    # 7. ADC + ADC Tumor Reader 2
    ax = axes[1, 2]
    ax.imshow(adc_s, cmap="gray")
    masked_adc_tum2 = np.ma.masked_where(adc_tum2_s == 0, adc_tum2_s)
    ax.imshow(masked_adc_tum2, cmap="spring", alpha=0.6)
    ax.set_title("ADC + Tumor Mask (Reader 2)")
    ax.axis("off")

    # 8. Comparison overlay on T2
    ax = axes[1, 3]
    ax.imshow(t2_s, cmap="gray")
    # Contour overlays for reader comparison
    if np.any(adc_tum1_s > 0):
        ax.contour(adc_tum1_s > 0, colors="cyan", linewidths=1.5, linestyles="-")
    if np.any(adc_tum2_s > 0):
        ax.contour(adc_tum2_s > 0, colors="magenta", linewidths=1.5, linestyles="--")
    if np.any(t2_tum_s > 0):
        ax.contour(t2_tum_s > 0, colors="yellow", linewidths=1.5, linestyles=":")
    ax.set_title("Comparison on T2 (Cyan:ADC R1, Mag:ADC R2, Yel:T2 R1)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated visualization and saved to: {output_png}")
    print(f"File size: {os.path.getsize(output_png)} bytes")

if __name__ == "__main__":
    visualize_patient_020()

