"""Training-time data augmentation for the RQ-1 controlled experiments (E2).

Baseline preservation contract
------------------------------
This module is ADDITIVE and DEFAULT-OFF. A config with no top-level
`augmentation:` block resolves to `AugmentationConfig(enabled=False)`, and
`build_train_augmentor` then returns ``None`` -- so the training dataset is
constructed with ``transform=None``, exactly as the Baseline and E1 arms were.
No existing experiment changes behavior because this file exists.

What E2 turns on
----------------
The transform set is NOT invented here. It is the pre-committed E2
specification from ``Outputs/rq1_experiment_specification.md`` section 6
("E2 -- Data Augmentation"), reproduced verbatim:

    Applied to TRAINING slices only, seeded (42).
    Geometric (image bilinear, mask nearest, applied identically to both):
      * random rotation      +/- 10 deg    p = 0.5
      * random scale / zoom  0.9 - 1.1     p = 0.3
      * horizontal flip                    p = 0.5
    Intensity (image only):
      * random gamma         0.7 - 1.5     p = 0.3
    Excluded on purpose: vertical flip, heavy elastic deformation, large
    rotations (they would distort zonal shape).

Correctness requirements enforced here
--------------------------------------
1. Image and mask receive the IDENTICAL spatial transform -- the same drawn
   angle, the same drawn scale factor, the same flip decision -- differing
   only in interpolation order (image order=1, mask ALWAYS order=0). This is
   the same image/mask discipline `src/transforms.py` already applies to
   preprocessing.
2. Labels stay valid: nearest-neighbour resampling cannot invent a label, and
   the output is re-cast to int64. A post-condition assertion rejects any
   label outside the set present before augmentation.
3. Rotation and scaling are composed into ONE affine matrix and applied with a
   SINGLE interpolation per array. Applying them as two sequential resamples
   would blur the image twice and erode thin PZ structures in the mask for no
   scientific reason.
4. Geometry is preserved: the output slice has exactly the input's (H, W), so
   the fixed 442x442 batching contract established by preprocessing is
   untouched and the model input shape never changes.
5. Areas rotated/zoomed in from outside the slice are filled with the image's
   own minimum (the same padding convention `src/transforms.py` uses for
   crop/pad) for the image, and with label 0 = background for the mask.
6. Intensity gamma is applied on a per-slice min-max rescaling to [0, 1] and
   then mapped back to the slice's original value range. The preprocessed
   slices are z-scored (`normalize_intensity`), so they contain NEGATIVE
   values and a raw ``x ** gamma`` would produce NaNs. The rescale-apply-
   restore form is the standard way to keep gamma well-defined on z-scored
   data, and it is an identity when gamma == 1.

Reproducibility
---------------
Each augmentor owns a single `numpy.random.Generator` seeded from
`augmentation.seed` (42 for E2). With `num_workers = 0` -- the project default
and the setting E2 uses -- the whole training run draws from that one stream,
so the run is seed-reproducible. `src/train.py` REFUSES to start an augmented
run with `num_workers > 0`, because forked DataLoader workers would each
inherit a copy of the generator and silently destroy both reproducibility and
the intended augmentation distribution.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

# The pre-committed E2 specification (Outputs/rq1_experiment_specification.md
# section 6). These are the DEFAULTS; a config may restate them but E2's own
# scientific-control test asserts the committed config matches them exactly.
SPEC_ROTATION_DEGREES = 10.0
SPEC_ROTATION_P = 0.5
SPEC_SCALE_MIN = 0.9
SPEC_SCALE_MAX = 1.1
SPEC_SCALE_P = 0.3
SPEC_HFLIP_P = 0.5
SPEC_GAMMA_MIN = 0.7
SPEC_GAMMA_MAX = 1.5
SPEC_GAMMA_P = 0.3

# Interpolation orders. NEVER make these configurable: a linear interpolation
# on a label map is a correctness bug, not a tuning knob.
IMAGE_INTERPOLATION_ORDER = 1  # bilinear
MASK_INTERPOLATION_ORDER = 0   # nearest-neighbour -- mandatory for labels

_ALLOWED_KEYS = frozenset({
    "enabled", "seed", "rotation_degrees", "rotation_p", "scale_min",
    "scale_max", "scale_p", "hflip_p", "gamma_min", "gamma_max", "gamma_p",
})


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"augmentation.{name} must be in [0, 1], got {value}.")
    return value


@dataclass(frozen=True)
class AugmentationConfig:
    """Fully-resolved augmentation specification for one experiment arm.

    `enabled=False` (the default) means the arm trains with NO augmentation,
    which is the Baseline / E1 behavior.
    """

    enabled: bool = False
    seed: int = 42
    rotation_degrees: float = SPEC_ROTATION_DEGREES
    rotation_p: float = SPEC_ROTATION_P
    scale_min: float = SPEC_SCALE_MIN
    scale_max: float = SPEC_SCALE_MAX
    scale_p: float = SPEC_SCALE_P
    hflip_p: float = SPEC_HFLIP_P
    gamma_min: float = SPEC_GAMMA_MIN
    gamma_max: float = SPEC_GAMMA_MAX
    gamma_p: float = SPEC_GAMMA_P

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "rotation_degrees", float(self.rotation_degrees))
        object.__setattr__(self, "scale_min", float(self.scale_min))
        object.__setattr__(self, "scale_max", float(self.scale_max))
        object.__setattr__(self, "gamma_min", float(self.gamma_min))
        object.__setattr__(self, "gamma_max", float(self.gamma_max))
        for field_name in ("rotation_p", "scale_p", "hflip_p", "gamma_p"):
            object.__setattr__(
                self, field_name, _validate_probability(field_name, getattr(self, field_name))
            )

        if self.rotation_degrees < 0:
            raise ValueError(
                f"augmentation.rotation_degrees must be >= 0, got {self.rotation_degrees}."
            )
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("augmentation.scale_min/scale_max must be > 0.")
        if self.scale_min > self.scale_max:
            raise ValueError(
                f"augmentation.scale_min ({self.scale_min}) must be <= "
                f"scale_max ({self.scale_max})."
            )
        if self.gamma_min <= 0 or self.gamma_max <= 0:
            raise ValueError("augmentation.gamma_min/gamma_max must be > 0.")
        if self.gamma_min > self.gamma_max:
            raise ValueError(
                f"augmentation.gamma_min ({self.gamma_min}) must be <= "
                f"gamma_max ({self.gamma_max})."
            )

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "AugmentationConfig":
        """Resolve the augmentation spec from a full experiment config dict.

        A config with no `augmentation:` block -> disabled, i.e. the exact
        Baseline / E1 behavior. `augmentation.seed` defaults to
        `training.seed` so an arm cannot accidentally augment off a different
        random stream than the one the rest of the run is seeded with.
        """
        block = (config or {}).get("augmentation")
        training_seed = ((config or {}).get("training") or {}).get("seed", 42)
        if not block:
            return AugmentationConfig(enabled=False, seed=int(training_seed))

        if not isinstance(block, dict):
            raise ValueError(
                f"`augmentation:` must be a mapping, got {type(block).__name__}."
            )

        unknown = set(block) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in `augmentation:` block: {sorted(unknown)}. "
                "Refusing to run an experiment against a misspelled augmentation "
                "setting that would silently be ignored."
            )

        return AugmentationConfig(
            enabled=block.get("enabled", False),
            seed=block.get("seed", training_seed),
            rotation_degrees=block.get("rotation_degrees", SPEC_ROTATION_DEGREES),
            rotation_p=block.get("rotation_p", SPEC_ROTATION_P),
            scale_min=block.get("scale_min", SPEC_SCALE_MIN),
            scale_max=block.get("scale_max", SPEC_SCALE_MAX),
            scale_p=block.get("scale_p", SPEC_SCALE_P),
            hflip_p=block.get("hflip_p", SPEC_HFLIP_P),
            gamma_min=block.get("gamma_min", SPEC_GAMMA_MIN),
            gamma_max=block.get("gamma_max", SPEC_GAMMA_MAX),
            gamma_p=block.get("gamma_p", SPEC_GAMMA_P),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable spec -- embedded in checkpoints and the run fingerprint."""
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "rotation_degrees": self.rotation_degrees,
            "rotation_p": self.rotation_p,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_p": self.scale_p,
            "hflip_p": self.hflip_p,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "gamma_p": self.gamma_p,
            "image_interpolation_order": IMAGE_INTERPOLATION_ORDER,
            "mask_interpolation_order": MASK_INTERPOLATION_ORDER,
            "applied_to": "train" if self.enabled else "none",
        }

    def describe(self) -> str:
        """One-line human-readable description, for logs and fingerprints."""
        if not self.enabled:
            return "OFF (no augmentation)"
        return (
            f"ON (train only, seed={self.seed}): "
            f"rot +/-{self.rotation_degrees:g}deg p={self.rotation_p:g} | "
            f"scale {self.scale_min:g}-{self.scale_max:g} p={self.scale_p:g} | "
            f"hflip p={self.hflip_p:g} | "
            f"gamma {self.gamma_min:g}-{self.gamma_max:g} p={self.gamma_p:g} "
            f"(image bilinear, mask nearest)"
        )


def _affine_matrix_and_offset(
    shape_hw: Tuple[int, int], angle_degrees: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (matrix, offset) pair `ndimage.affine_transform` expects for a
    rotation by `angle_degrees` and a zoom by `scale`, both about the slice
    center.

    `affine_transform` evaluates ``output[o] = input[matrix @ o + offset]``, so
    `matrix` is the INVERSE of the forward map. Forward map:

        out = scale * R(angle) @ (in - center) + center
    =>  in  = R(-angle) @ (out - center) / scale + center
    """
    theta = np.deg2rad(float(angle_degrees))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # R(-theta) / scale
    matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64) / float(scale)
    center = (np.asarray(shape_hw, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center
    return matrix, offset


def _apply_gamma(image_hw: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction on a z-scored slice, via min-max rescale and restore.

    Exactly the identity when the slice is constant (zero dynamic range) or
    when gamma == 1, so it can never introduce NaNs or shift a flat slice.
    """
    v_min = float(image_hw.min())
    v_max = float(image_hw.max())
    span = v_max - v_min
    if span <= 0.0:
        return image_hw.astype(np.float32)
    unit = (image_hw - v_min) / span
    # Clip guards against tiny out-of-range values from floating-point
    # round-off before the fractional power, which would otherwise give NaN.
    unit = np.clip(unit, 0.0, 1.0)
    return (np.power(unit, float(gamma)) * span + v_min).astype(np.float32)


class SliceAugmentor:
    """Seeded, train-only augmentation callable for `ProstateZonal2DDataset`.

    Consumes and returns the dataset's sample dict:
      * ``image`` : float32, shape (1, H, W)
      * ``mask``  : int64,   shape (H, W)
    and leaves every other key (``patient_id``, ``slice_idx``) untouched.
    Shapes are preserved exactly.

    This object is STATEFUL (it owns the RNG). It must be attached to the
    TRAINING dataset only -- never to a validation or test dataset, whose
    evaluation must stay deterministic. `src/train.py` is the only wiring
    point and it enforces that.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        if not config.enabled:
            raise ValueError(
                "SliceAugmentor was constructed from a disabled AugmentationConfig. "
                "Use build_train_augmentor(), which returns None when augmentation "
                "is off, instead of instantiating this directly."
            )
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        # Purely diagnostic counters (printed in the notebook's verification
        # cell); they never influence a draw.
        self.counts = {"calls": 0, "rotation": 0, "scale": 0, "hflip": 0, "gamma": 0}

    def reset(self) -> None:
        """Re-seed the stream. Used by tests to assert determinism."""
        self._rng = np.random.default_rng(self.config.seed)
        self.counts = {key: 0 for key in self.counts}

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        rng = self._rng

        image = np.asarray(sample["image"], dtype=np.float32)
        mask = np.asarray(sample["mask"])

        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                f"SliceAugmentor expects image of shape (1, H, W), got {image.shape}."
            )
        if mask.ndim != 2 or mask.shape != image.shape[1:]:
            raise ValueError(
                f"SliceAugmentor expects mask of shape {image.shape[1:]}, got {mask.shape}."
            )

        image_hw = image[0]
        labels_before = set(int(v) for v in np.unique(mask))
        self.counts["calls"] += 1

        # Draws. All four are drawn on EVERY call, unconditionally, so the
        # random stream advances by a fixed amount per sample regardless of
        # which transforms actually fire. That keeps the sequence reproducible
        # and independent of the probability values.
        do_rotate = bool(rng.random() < cfg.rotation_p)
        angle = float(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
        do_scale = bool(rng.random() < cfg.scale_p)
        scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
        do_hflip = bool(rng.random() < cfg.hflip_p)
        do_gamma = bool(rng.random() < cfg.gamma_p)
        gamma = float(rng.uniform(cfg.gamma_min, cfg.gamma_max))

        angle = angle if do_rotate else 0.0
        scale = scale if do_scale else 1.0

        # 1+2. Rotation and scaling, composed into ONE affine so each array is
        #      interpolated exactly once. Identical geometry for image and
        #      mask; only the interpolation order differs.
        if do_rotate or do_scale:
            matrix, offset = _affine_matrix_and_offset(image_hw.shape, angle, scale)
            image_hw = ndimage.affine_transform(
                image_hw,
                matrix,
                offset=offset,
                order=IMAGE_INTERPOLATION_ORDER,
                mode="constant",
                cval=float(image_hw.min()),
                output_shape=image_hw.shape,
            ).astype(np.float32)
            mask = ndimage.affine_transform(
                mask.astype(np.float32),
                matrix,
                offset=offset,
                order=MASK_INTERPOLATION_ORDER,
                mode="constant",
                cval=0.0,
                output_shape=mask.shape,
            )
            mask = np.round(mask).astype(np.int64)
            if do_rotate:
                self.counts["rotation"] += 1
            if do_scale:
                self.counts["scale"] += 1

        # 3. Horizontal flip (left-right, i.e. the W axis). Identical for image
        #    and mask; lossless, no interpolation.
        if do_hflip:
            image_hw = np.ascontiguousarray(image_hw[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            self.counts["hflip"] += 1

        # 4. Gamma. IMAGE ONLY -- the mask is never touched by an intensity
        #    transform.
        if do_gamma:
            image_hw = _apply_gamma(image_hw, gamma)
            self.counts["gamma"] += 1

        mask = np.ascontiguousarray(mask.astype(np.int64))
        labels_after = set(int(v) for v in np.unique(mask))
        if not labels_after.issubset(labels_before | {0}):
            raise AssertionError(
                "Augmentation produced label(s) absent from the input mask: "
                f"before={sorted(labels_before)} after={sorted(labels_after)}. "
                "This indicates a non-nearest-neighbour resample on the label map."
            )

        out = dict(sample)
        out["image"] = np.ascontiguousarray(image_hw[np.newaxis, ...].astype(np.float32))
        out["mask"] = mask
        return out


def build_train_augmentor(config: Dict[str, Any]) -> Optional[SliceAugmentor]:
    """Return the TRAINING augmentation callable for an experiment config.

    Returns ``None`` when the config has no `augmentation:` block or sets
    `enabled: false` -- so Baseline / E1 keep building their training dataset
    with ``transform=None``, unchanged.
    """
    aug_config = AugmentationConfig.from_config(config)
    if not aug_config.enabled:
        return None
    return SliceAugmentor(aug_config)
