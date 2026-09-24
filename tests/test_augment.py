"""Correctness tests for the E2 training-time augmentation (src/augment.py).

These are the properties E2's validity depends on:
  * augmentation is DEFAULT-OFF, so Baseline / E1 are byte-identical;
  * image and mask receive the SAME spatial transform;
  * the mask is only ever resampled nearest-neighbour, so labels stay in the
    original label set and no interpolated "1.5" label can appear;
  * intensity transforms touch the image ONLY;
  * shapes are preserved, so the 442x442 batching contract holds;
  * the stream is reproducible from the seed;
  * the committed parameters match the pre-committed specification.
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.augment import (  # noqa: E402
    IMAGE_INTERPOLATION_ORDER,
    MASK_INTERPOLATION_ORDER,
    AugmentationConfig,
    SliceAugmentor,
    _apply_gamma,
    build_train_augmentor,
)

H = W = 64


def make_sample(seed: int = 0):
    """A z-scored-looking image plus a 3-class mask with a CG core and PZ ring."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 1.0, size=(1, H, W)).astype(np.float32)

    mask = np.zeros((H, W), dtype=np.int64)
    yy, xx = np.mgrid[0:H, 0:W]
    radius = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    mask[radius < 18] = 2   # PZ ring (outer)
    mask[radius < 11] = 1   # CG core (inner)
    return {"image": image, "mask": mask, "patient_id": "000", "slice_idx": 7}


ALWAYS_ON = dict(
    enabled=True, seed=42,
    rotation_p=1.0, scale_p=1.0, hflip_p=1.0, gamma_p=1.0,
)


class TestDefaultOff(unittest.TestCase):
    """Baseline / E1 preservation: no `augmentation:` block -> nothing happens."""

    def test_config_without_block_is_disabled(self):
        cfg = AugmentationConfig.from_config({"training": {"seed": 42}})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.describe(), "OFF (no augmentation)")

    def test_build_train_augmentor_returns_none_when_absent(self):
        self.assertIsNone(build_train_augmentor({"training": {"seed": 42}}))
        self.assertIsNone(build_train_augmentor({}))

    def test_build_train_augmentor_returns_none_when_explicitly_disabled(self):
        self.assertIsNone(build_train_augmentor({"augmentation": {"enabled": False}}))

    def test_augmentation_seed_defaults_to_training_seed(self):
        cfg = AugmentationConfig.from_config(
            {"training": {"seed": 7}, "augmentation": {"enabled": True}}
        )
        self.assertEqual(cfg.seed, 7)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            AugmentationConfig.from_config(
                {"augmentation": {"enabled": True, "rotaton_p": 0.5}}
            )

    def test_disabled_config_cannot_build_an_augmentor_directly(self):
        with self.assertRaises(ValueError):
            SliceAugmentor(AugmentationConfig(enabled=False))


class TestConfigValidation(unittest.TestCase):
    def test_probability_out_of_range_rejected(self):
        for field in ("rotation_p", "scale_p", "hflip_p", "gamma_p"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    AugmentationConfig(enabled=True, **{field: 1.5})

    def test_inverted_ranges_rejected(self):
        with self.assertRaises(ValueError):
            AugmentationConfig(enabled=True, scale_min=1.2, scale_max=0.9)
        with self.assertRaises(ValueError):
            AugmentationConfig(enabled=True, gamma_min=1.6, gamma_max=0.7)

    def test_negative_rotation_rejected(self):
        with self.assertRaises(ValueError):
            AugmentationConfig(enabled=True, rotation_degrees=-5.0)

    def test_interpolation_orders_are_correct(self):
        self.assertEqual(IMAGE_INTERPOLATION_ORDER, 1, "image must be bilinear")
        self.assertEqual(MASK_INTERPOLATION_ORDER, 0, "mask must be nearest-neighbour")


class TestGeometryAndLabels(unittest.TestCase):
    def setUp(self):
        self.augmentor = SliceAugmentor(AugmentationConfig(**ALWAYS_ON))

    def test_shapes_preserved(self):
        for _ in range(20):
            out = self.augmentor(make_sample())
            self.assertEqual(out["image"].shape, (1, H, W))
            self.assertEqual(out["mask"].shape, (H, W))

    def test_dtypes_preserved(self):
        out = self.augmentor(make_sample())
        self.assertEqual(out["image"].dtype, np.float32)
        self.assertEqual(out["mask"].dtype, np.int64)

    def test_labels_stay_in_the_original_label_set(self):
        """No interpolated label may appear -- the signature of a linear
        resample on a label map."""
        for _ in range(50):
            out = self.augmentor(make_sample())
            labels = set(int(v) for v in np.unique(out["mask"]))
            self.assertTrue(
                labels.issubset({0, 1, 2}),
                f"augmentation produced labels outside {{0,1,2}}: {sorted(labels)}",
            )

    def test_non_spatial_keys_are_carried_through(self):
        out = self.augmentor(make_sample())
        self.assertEqual(out["patient_id"], "000")
        self.assertEqual(out["slice_idx"], 7)

    def test_input_sample_is_not_mutated(self):
        sample = make_sample()
        image_before = sample["image"].copy()
        mask_before = sample["mask"].copy()
        self.augmentor(sample)
        np.testing.assert_array_equal(sample["image"], image_before)
        np.testing.assert_array_equal(sample["mask"], mask_before)

    def test_rejects_wrong_image_shape(self):
        bad = make_sample()
        bad["image"] = bad["image"][0]  # (H, W) instead of (1, H, W)
        with self.assertRaises(ValueError):
            self.augmentor(bad)

    def test_rejects_mismatched_mask_shape(self):
        bad = make_sample()
        bad["mask"] = np.zeros((H, W + 1), dtype=np.int64)
        with self.assertRaises(ValueError):
            self.augmentor(bad)


class TestImageMaskCoupling(unittest.TestCase):
    """The single most important property: image and mask move TOGETHER."""

    def test_hflip_applies_to_both(self):
        cfg = AugmentationConfig(
            enabled=True, seed=1, rotation_p=0.0, scale_p=0.0, hflip_p=1.0, gamma_p=0.0
        )
        sample = make_sample()
        out = SliceAugmentor(cfg)(sample)
        np.testing.assert_array_equal(out["image"][0], sample["image"][0][:, ::-1])
        np.testing.assert_array_equal(out["mask"], sample["mask"][:, ::-1])

    def test_rotation_moves_image_and_mask_consistently(self):
        """Build an image that IS the mask, rotate, and confirm the two outputs
        still agree on where the foreground is. If image and mask were given
        different angles (or different centers), the overlap would collapse."""
        cfg = AugmentationConfig(
            enabled=True, seed=3, rotation_degrees=10.0, rotation_p=1.0,
            scale_p=0.0, hflip_p=0.0, gamma_p=0.0,
        )
        sample = make_sample()
        # Image := foreground indicator, so image and mask carry the same shape.
        sample["image"] = (sample["mask"] > 0).astype(np.float32)[np.newaxis, ...]

        out = SliceAugmentor(cfg)(sample)
        image_fg = out["image"][0] > 0.5
        mask_fg = out["mask"] > 0
        intersection = np.logical_and(image_fg, mask_fg).sum()
        union = np.logical_or(image_fg, mask_fg).sum()
        self.assertGreater(union, 0)
        # Bilinear (image) vs nearest (mask) differ only on the boundary ring.
        self.assertGreater(intersection / union, 0.90)

    def test_scale_moves_image_and_mask_consistently(self):
        cfg = AugmentationConfig(
            enabled=True, seed=5, rotation_p=0.0,
            scale_min=1.1, scale_max=1.1, scale_p=1.0, hflip_p=0.0, gamma_p=0.0,
        )
        sample = make_sample()
        sample["image"] = (sample["mask"] > 0).astype(np.float32)[np.newaxis, ...]

        out = SliceAugmentor(cfg)(sample)
        image_fg = out["image"][0] > 0.5
        mask_fg = out["mask"] > 0
        iou = np.logical_and(image_fg, mask_fg).sum() / max(
            1, np.logical_or(image_fg, mask_fg).sum()
        )
        self.assertGreater(iou, 0.90)
        # A 1.1x zoom must actually enlarge the foreground.
        self.assertGreater(mask_fg.sum(), (sample["mask"] > 0).sum())

    def test_identity_when_all_probabilities_are_zero(self):
        cfg = AugmentationConfig(
            enabled=True, seed=9, rotation_p=0.0, scale_p=0.0, hflip_p=0.0, gamma_p=0.0
        )
        sample = make_sample()
        out = SliceAugmentor(cfg)(sample)
        np.testing.assert_array_equal(out["image"], sample["image"])
        np.testing.assert_array_equal(out["mask"], sample["mask"])


class TestIntensityIsImageOnly(unittest.TestCase):
    def test_gamma_does_not_touch_the_mask(self):
        cfg = AugmentationConfig(
            enabled=True, seed=11, rotation_p=0.0, scale_p=0.0, hflip_p=0.0, gamma_p=1.0
        )
        sample = make_sample()
        out = SliceAugmentor(cfg)(sample)
        np.testing.assert_array_equal(out["mask"], sample["mask"])
        self.assertFalse(np.array_equal(out["image"], sample["image"]))

    def test_gamma_is_finite_on_z_scored_input(self):
        """The preprocessed slices are z-scored and contain NEGATIVE values; a
        naive x**gamma would return NaN."""
        image = np.random.default_rng(0).normal(0.0, 1.0, size=(H, W)).astype(np.float32)
        self.assertLess(image.min(), 0.0)
        for gamma in (0.7, 1.0, 1.5):
            out = _apply_gamma(image, gamma)
            self.assertTrue(np.isfinite(out).all(), f"gamma={gamma} produced non-finite values")

    def test_gamma_one_is_the_identity(self):
        image = np.random.default_rng(1).normal(0.0, 1.0, size=(H, W)).astype(np.float32)
        np.testing.assert_allclose(_apply_gamma(image, 1.0), image, atol=1e-5)

    def test_gamma_preserves_the_value_range(self):
        image = np.random.default_rng(2).normal(0.0, 1.0, size=(H, W)).astype(np.float32)
        out = _apply_gamma(image, 0.7)
        self.assertAlmostEqual(float(out.min()), float(image.min()), places=4)
        self.assertAlmostEqual(float(out.max()), float(image.max()), places=4)

    def test_gamma_on_a_constant_slice_is_the_identity(self):
        flat = np.full((H, W), -0.5, dtype=np.float32)
        np.testing.assert_array_equal(_apply_gamma(flat, 0.7), flat)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_the_same_sequence(self):
        samples = [make_sample(i) for i in range(12)]

        run_a = SliceAugmentor(AugmentationConfig(**ALWAYS_ON))
        first = [run_a(s) for s in samples]

        run_b = SliceAugmentor(AugmentationConfig(**ALWAYS_ON))
        second = [run_b(s) for s in samples]
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a["image"], b["image"])
            np.testing.assert_array_equal(a["mask"], b["mask"])

    def test_different_seeds_give_a_different_sequence(self):
        samples = [make_sample(i) for i in range(12)]
        spec = dict(ALWAYS_ON)
        spec.update(rotation_p=0.5, scale_p=0.3, hflip_p=0.5, gamma_p=0.3)

        a = SliceAugmentor(AugmentationConfig(**{**spec, "seed": 42}))
        b = SliceAugmentor(AugmentationConfig(**{**spec, "seed": 2024}))
        outs_a = [a(s)["image"] for s in samples]
        outs_b = [b(s)["image"] for s in samples]
        self.assertFalse(
            all(np.array_equal(x, y) for x, y in zip(outs_a, outs_b)),
            "two different seeds produced an identical augmentation sequence",
        )

    def test_reset_restarts_the_stream(self):
        augmentor = SliceAugmentor(AugmentationConfig(**ALWAYS_ON))
        sample = make_sample(0)
        first = augmentor(sample)["image"]
        augmentor(make_sample(1))
        augmentor.reset()
        again = augmentor(sample)["image"]
        np.testing.assert_array_equal(first, again)

    def test_stream_advances_by_a_fixed_amount_per_call(self):
        """Every draw happens unconditionally, so the sequence of decisions does
        not depend on which transforms happened to fire earlier."""
        never = AugmentationConfig(
            enabled=True, seed=42, rotation_p=0.0, scale_p=0.0, hflip_p=0.0, gamma_p=0.0
        )
        always = AugmentationConfig(**ALWAYS_ON)
        a, b = SliceAugmentor(never), SliceAugmentor(always)
        sample = make_sample(0)
        for _ in range(5):
            a(sample)
            b(sample)
        # Both consumed 7 variates per call from the same seed; the next draw
        # from each generator must therefore agree.
        self.assertAlmostEqual(
            float(a._rng.random()), float(b._rng.random()), places=12
        )


class TestApproximateRates(unittest.TestCase):
    """The configured probabilities really are the firing rates (within noise)."""

    def test_firing_rates_match_the_specification(self):
        cfg = AugmentationConfig(
            enabled=True, seed=42,
            rotation_p=0.5, scale_p=0.3, hflip_p=0.5, gamma_p=0.3,
        )
        augmentor = SliceAugmentor(cfg)
        sample = make_sample(0)
        n = 2000
        for _ in range(n):
            augmentor(sample)

        counts = augmentor.counts
        self.assertEqual(counts["calls"], n)
        for key, expected in (
            ("rotation", 0.5), ("scale", 0.3), ("hflip", 0.5), ("gamma", 0.3)
        ):
            rate = counts[key] / n
            self.assertAlmostEqual(
                rate, expected, delta=0.05,
                msg=f"{key} fired at {rate:.3f}, expected ~{expected}",
            )


if __name__ == "__main__":
    unittest.main()
