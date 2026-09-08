import os
import sys
import numpy as np
import nibabel as nib

def inspect_patient_020():
    # Resolve project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    patient_dir = os.path.join(project_root, "dataset", "prostate158_train", "train", "020")
    
    print("=" * 80)
    print(f"INSPECTING PATIENT 020 AT: {patient_dir}")
    print("=" * 80)
    
    if not os.path.exists(patient_dir):
        print(f"ERROR: Directory does not exist: {patient_dir}")
        return

    files = sorted([f for f in os.listdir(patient_dir) if f.endswith(".nii.gz")])
    print(f"Total .nii.gz files found: {len(files)}")
    for f in files:
        print(f" - {f}")
    print("=" * 80)

    for fname in files:
        fpath = os.path.join(patient_dir, fname)
        print(f"\n>>> FILE: {fname}")
        print("-" * 50)
        
        # Load NIfTI
        img = nib.load(fpath)
        header = img.header
        affine = img.affine
        
        # Get raw data array
        data = img.get_fdata()
        
        # Determine orientation codes
        try:
            orientation_codes = nib.orientations.aff2axcodes(affine)
            orientation_str = "".join(orientation_codes)
        except Exception as e:
            orientation_str = f"Unknown ({e})"
            orientation_codes = ()
        
        # Basic properties
        shape = data.shape
        ndim = data.ndim
        raw_dtype = str(header.get_data_dtype())
        loaded_dtype = str(data.dtype)
        zooms = tuple(float(z) for z in header.get_zooms())
        
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        mean_val = float(np.mean(data))
        nonzero_count = int(np.count_nonzero(data))
        total_voxels = int(data.size)
        nonzero_pct = (nonzero_count / total_voxels) * 100.0
        
        unique_vals = np.unique(data)
        
        print("[VERIFIED FROM FILE]")
        print(f"  Filename:             {fname}")
        print(f"  File size (bytes):    {os.path.getsize(fpath)}")
        print(f"  Shape:                {shape}")
        print(f"  Number of dimensions: {ndim}")
        print(f"  Header Data Type:     {raw_dtype}")
        print(f"  Loaded Array Type:    {loaded_dtype}")
        print(f"  Voxel Spacing (zooms):{zooms}")
        print(f"  Orientation:          {orientation_str} {orientation_codes}")
        print(f"  Affine matrix:\n{affine}")
        print(f"  Min value:            {min_val}")
        print(f"  Max value:            {max_val}")
        print(f"  Mean value:           {mean_val:.4f}")
        print(f"  Non-zero voxels:      {nonzero_count} / {total_voxels} ({nonzero_pct:.3f}%)")
        
        # Mask check
        is_discrete = False
        if len(unique_vals) <= 20:
            print(f"  Unique values count:  {len(unique_vals)}")
            print(f"  Unique values:        {unique_vals.tolist()}")
            if np.all(np.isclose(unique_vals, np.round(unique_vals))):
                is_discrete = True
                counts = {float(val): int(np.count_nonzero(data == val)) for val in unique_vals}
                print(f"  Value distribution:   {counts}")
        else:
            print(f"  Unique values count:  {len(unique_vals)} (continuous spectrum)")
        
        print("\n[VERIFIED MASK / IMAGE CLASSIFICATION]")
        if is_discrete and min_val >= 0 and len(unique_vals) <= 10:
            print(f"  VERIFIED FROM FILE: Discrete values {unique_vals.tolist()}, acting as categorical/binary mask.")
        else:
            print(f"  VERIFIED FROM FILE: Continuous intensity distribution, acting as anatomical/functional MRI volume.")

        print("\n[ASSUMPTION / NOT VERIFIED]")
        if "tumor" in fname.lower():
            print("  ASSUMPTION: Filename suggests lesion/tumor annotation.")
            print("  NOT VERIFIED FROM RAW FILE: Clinical grading (PI-RADS), reader identity credentials, or histology confirmation.")
        elif "anatomy" in fname.lower():
            print("  ASSUMPTION: Filename suggests whole-gland or zonal anatomical segmentation.")
            print("  NOT VERIFIED FROM RAW FILE: Specific anatomical sub-regions without examining protocol.")
        elif "adc" in fname.lower():
            print("  ASSUMPTION: Apparent Diffusion Coefficient map.")
        elif "dwi" in fname.lower():
            print("  ASSUMPTION: High b-value Diffusion-Weighted Imaging.")
        elif "t2" in fname.lower():
            print("  ASSUMPTION: T2-weighted anatomical MRI.")

if __name__ == "__main__":
    inspect_patient_020()

