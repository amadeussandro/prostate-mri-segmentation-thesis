# Round-trip failure: 11050_1001070 (fold 3)

Status 2026-11-12, local round-trip after fold 3: 1152 of 1153 non-empty Bosma22b masks pass.

| Item | Value |
|---|---|
| Case | 11050_1001070 (picai_public_images_fold3.zip) |
| Grid | 640 x 640 x 37, 0.28125 x 0.28125 x 3.45 mm, LPS; mask on the same grid as T2W |
| Bosma22b foreground | 184,476 voxels (50.3 mL), one 3D connected component |
| Lost by the round-trip | 59,326 voxels (32%), all outside the central 442 x 442 crop |
| Slices affected | 23-29; slice 22 is fully inside the crop |

The conversion is correct: the image corners match to 4.5e-6 mm, and the mask and the image share one grid.
The loss is caused by the frozen preprocessing (centre crop to 442 x 442), which the round-trip test exists to detect.

Visual review (roundtrip_failure_11050_1001070.png, red = Bosma22b, yellow = crop):
- The part of the mask inside the crop sits on the gland.
- The part outside the crop runs as a thin extension to the anterior image edge.
- This looks like a leak of the Bosma22b AI mask into anterior tissue rather than gland that the crop cuts away.
- This is a non-radiologist reading. It is reported as observed, and the case stays in every analysis, as the prespecification requires.

Confirmed by the data provider: the picai_labels README (commit ce4a4723d7c46d882a6cbaacb40ed6c4be86282f,
row for `anatomical_delineations/whole_gland/AI/Bosma22b`) names this exact case, 11050_1001070,
as its example of a faulty AI whole-gland segmentation. Found 2026-11-12, after the failure had been detected and
reviewed above.

Effect on the endpoints: the model only sees the crop. If the leak reading is right, the model volume for this case is unaffected, while Bosma22b over-states the volume. This affects the model-vs-Bosma22b agreement for one case only.
