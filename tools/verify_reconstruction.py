"""Verify a 3D volume reassembled from slice-wise segmentation output.

Standalone (numpy + nibabel only), so it can be dropped into any pipeline. It
implements the 3 verification levels of the accompanying article:

  Level 1  geometry   the written file's own header against the source image:
                      shape, affine, voxel spacing, orientation codes, label set
  Level 2  roundtrip  a reference mask passed forward through YOUR preprocessing
                      and back through YOUR reconstruction must come back voxel
                      for voxel (Python API: `roundtrip`, you supply the 2 steps)
  Level 3  anatomy    reference-free plausibility: PZ centroid posterior to the
                      CG centroid, gland present on a contiguous run of slices,
                      and no in-plane centroid jump above 15 mm between slices

No single level is sufficient. In the article's fault-injection experiment the
geometry check missed every orientation and slice-order fault and the round-trip
test missed every affine fault; only both together caught every damaging fault.
Level 3 misses left-right flips and is a screen, not a substitute.

Command line:
    python tools/verify_reconstruction.py geometry --source image.nii.gz --written pred.nii.gz
    python tools/verify_reconstruction.py anatomy  --written pred.nii.gz --cg 1 --pz 2
Exit status is 0 when every requested check passes and 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Dict, Iterable, Optional

import nibabel as nib
import numpy as np

JUMP_MM = 15.0


# --------------------------------------------------------------------- level 1
def geometry(source: nib.Nifti1Image, written: nib.Nifti1Image,
             labels: Optional[Iterable[int]] = (0, 1, 2), atol: float = 1e-3) -> Dict[str, object]:
    """Compare the header of the written segmentation with the source image."""
    a_src, a_out = source.affine, written.affine
    zs, zo = source.header.get_zooms()[:3], written.header.get_zooms()[:3]
    found = sorted(int(v) for v in np.unique(np.asarray(written.dataobj)))
    res = {
        "shape": tuple(written.shape[:3]) == tuple(source.shape[:3]),
        "affine": bool(np.allclose(a_src, a_out, atol=atol)),
        "spacing": bool(np.allclose(zs, zo, atol=atol)),
        "orientation": nib.aff2axcodes(a_src) == nib.aff2axcodes(a_out),
        "labels": labels is None or set(found) <= set(labels),
    }
    res["passed"] = all(res.values())
    res["affine_max_abs_diff"] = float(np.abs(a_src - a_out).max())
    res["labels_found"] = found
    return res


# --------------------------------------------------------------------- level 2
def roundtrip(mask: np.ndarray, forward: Callable[[np.ndarray], object],
              inverse: Callable[[object], np.ndarray]) -> Dict[str, object]:
    """Pass a reference mask through the pipeline's own forward and inverse steps.

    `forward` must be the preprocessing and slicing applied to images (without
    intensity steps, which do not apply to a mask); `inverse` must be the exact
    reconstruction code applied to predictions. Exact recovery is expected,
    because a mask transform without resampling is a permutation plus pad/crop.
    The result is pass or fail, never an accuracy.
    """
    back = np.asarray(inverse(forward(mask)))
    same_shape = back.shape == mask.shape
    n_diff = int((back != mask).sum()) if same_shape else int(mask.size)
    return {"passed": bool(same_shape and n_diff == 0), "same_shape": same_shape,
            "voxels_different": n_diff}


# --------------------------------------------------------------------- level 3
def _world_centroid(vol: np.ndarray, aff: np.ndarray, label: int) -> Optional[np.ndarray]:
    idx = np.argwhere(vol == label)
    if idx.size == 0:
        return None
    return aff[:3, :3] @ idx.mean(axis=0) + aff[:3, 3]


def anatomy(written: nib.Nifti1Image, cg: int = 1, pz: int = 2,
            jump_mm: float = JUMP_MM) -> Dict[str, object]:
    """Reference-free checks. nibabel affines map to RAS+, so posterior = smaller y."""
    vol = np.round(np.asarray(written.dataobj)).astype(np.int16)
    aff = written.affine
    c_cg, c_pz = _world_centroid(vol, aff, cg), _world_centroid(vol, aff, pz)
    pz_posterior = bool(c_cg is not None and c_pz is not None and c_pz[1] < c_cg[1])

    zaxis = int(np.argmax(np.abs(aff[:3, :3][2, :])))  # voxel axis closest to world z
    moved = np.moveaxis(vol > 0, zaxis, -1)
    present = np.where(moved.reshape(-1, moved.shape[-1]).any(axis=0))[0]
    gaps = 0 if present.size == 0 else int(present.max() - present.min() + 1 - present.size)
    zooms = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    inplane = [i for i in range(3) if i != zaxis]
    jumps, prev = [], None
    for k in range(moved.shape[-1]):
        sl = moved[..., k]
        if not sl.any():
            prev = None
            continue
        c = np.argwhere(sl).mean(axis=0) * zooms[inplane]
        if prev is not None:
            jumps.append(float(np.linalg.norm(c - prev)))
        prev = c
    max_jump = max(jumps) if jumps else 0.0
    continuity = bool(gaps == 0 and max_jump <= jump_mm)
    return {"passed": pz_posterior and continuity, "pz_posterior_to_cg": pz_posterior,
            "slice_continuity": continuity, "empty_slices_inside_gland": gaps,
            "max_centroid_jump_mm": round(max_jump, 2)}


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry", help="level 1: header of written file vs source")
    g.add_argument("--source", required=True)
    g.add_argument("--written", required=True)
    g.add_argument("--labels", default="0,1,2", help="allowed labels, or 'any'")
    a = sub.add_parser("anatomy", help="level 3: reference-free plausibility")
    a.add_argument("--written", required=True)
    a.add_argument("--cg", type=int, default=1)
    a.add_argument("--pz", type=int, default=2)
    a.add_argument("--jump-mm", type=float, default=JUMP_MM)
    args = ap.parse_args(argv)
    if args.cmd == "geometry":
        labels = None if args.labels == "any" else [int(x) for x in args.labels.split(",")]
        res = geometry(nib.load(args.source), nib.load(args.written), labels)
    else:
        res = anatomy(nib.load(args.written), args.cg, args.pz, args.jump_mm)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
