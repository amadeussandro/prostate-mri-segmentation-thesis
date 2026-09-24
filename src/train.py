"""Training loop for 2D prostate zonal anatomy segmentation."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.augment import AugmentationConfig, build_train_augmentor
from src.dataset import ProstateZonal2DDataset
from src.evaluate import compute_dice, compute_iou
from src.losses import LossConfig, compute_loss_components
from src.model import build_model
from src.splits import load_official_split
from src.transforms import PreprocessingConfig
from src.utils import (
    build_config_fingerprint,
    ensure_dir,
    format_config_fingerprint,
    get_device,
    load_config,
    set_seed,
)


def _resolve_split_ids(config: Dict[str, Any]):
    """Resolve (train_ids, val_ids, test_ids) for this run.

    If the config points at the official train.csv/valid.csv (research
    configs, e.g. config_baseline.yaml), the official case-level 119/20/19
    split is used. Otherwise falls back to an explicit `patient_ids` /
    `val_patient_ids` list (used by the Case-020 smoke-test config and by
    quarantined legacy configs) -- this path is NOT the official research
    split and must never be used for reporting thesis results.
    """
    data_cfg = config.get("data", {})
    validation_cfg = config.get("validation", {})

    if "train_csv" in data_cfg and "valid_csv" in data_cfg:
        split = load_official_split(
            train_csv=data_cfg["train_csv"],
            valid_csv=data_cfg["valid_csv"],
            test_csv=data_cfg.get("test_csv"),
            test_dir=data_cfg.get("test_dir"),
        )
        return split.train_ids, split.val_ids, split.test_ids

    train_ids = data_cfg.get("patient_ids", [])
    val_ids = validation_cfg.get("val_patient_ids", [])
    return train_ids, val_ids, None


def compute_loss(
    predictions,
    targets,
    loss_type="cross_entropy",
    dice_weight=1.0,
    bce_weight=1.0,
    loss_config=None,
):
    """
    Compute loss for multiclass prostate zonal segmentation.

    Backward compatible: called with the defaults (or loss_type="cross_entropy")
    this returns EXACTLY `nn.CrossEntropyLoss()(predictions, targets.long())`,
    the original baseline loss, unchanged.

    Parameters
    ----------
    predictions : torch.Tensor
        Raw logits with shape (B, C, H, W).
    targets : torch.Tensor
        Integer class labels with shape (B, H, W).
    loss_type : str
        Legacy selector ("cross_entropy" / "dice_ce"). Ignored when
        `loss_config` is supplied.
    dice_weight, bce_weight : float
        Legacy arguments, kept for signature compatibility. Prefer passing a
        fully-resolved `loss_config`.
    loss_config : Optional[src.losses.LossConfig]
        Fully-resolved loss spec (E1 path). Takes precedence over `loss_type`.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    cfg = loss_config or LossConfig(name=loss_type, dice_weight=dice_weight)
    total, _ = compute_loss_components(predictions, targets, cfg)
    return total


def _move_batch_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move image and mask tensors from a dataset batch to the target device.

    `non_blocking=True` only has an effect when the source tensors are in pinned
    memory (DataLoader pin_memory=True, used for CUDA) -- it overlaps the host->
    device copy with compute. It is a NO-OP for CPU/non-pinned tensors and does
    NOT change any value that is transferred, so it is numerically identical to a
    blocking copy; it only affects transfer scheduling.
    """
    non_blocking = device == "cuda"
    images = batch["image"].float().to(device, non_blocking=non_blocking)
    masks = batch["mask"].long().to(device, non_blocking=non_blocking)

    return images, masks


def _dataloader_perf_kwargs(training_cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Build performance-only DataLoader kwargs from the config, with defaults
    that EXACTLY preserve the historical behavior (num_workers=0, pin_memory on
    CUDA). These knobs never change which samples are drawn, their order, or
    their values -- the WeightedRandomSampler's seeded generator and the
    deterministic dataset decide all of that -- so they cannot affect the
    scientific result. They only affect how batches are fetched/overlapped.

    Recognized optional `training:` keys (all default to the prior behavior, so
    an unchanged config trains identically to before):
      - num_workers (int, default 0)
      - pin_memory (bool, default: device == 'cuda')
      - persistent_workers (bool, default: num_workers > 0)
      - prefetch_factor (int, default 4; only applied when num_workers > 0)
    """
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", device == "cuda"))
    kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            training_cfg.get("persistent_workers", True)
        )
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 4))
    return kwargs


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: str,
    loss_type: str = "cross_entropy",
    use_amp: bool = False,
    scaler: Any = None,
    loss_config: Any = None,
    return_components: bool = False,
) -> Any:
    """Execute one training epoch.

    AMP (mixed precision) is OPT-IN and off by default. When `use_amp` is False
    (the default, and every existing FP32 experiment) the loop below is the
    exact original FP32 path -- no autocast, no GradScaler -- so FP32 behavior is
    byte-for-byte unchanged. When `use_amp` is True, the forward pass runs under
    `torch.autocast` and the backward/step go through the provided GradScaler.
    AMP is a training-only speedup for the pilot; evaluation/inference stays FP32.

    `loss_config` (optional, E1) selects the compound Dice+CE objective; when it
    is None the legacy `loss_type` path is used, i.e. plain cross entropy.

    Returns
    -------
    float
        Sample-weighted mean total loss (default -- unchanged contract).
    Tuple[float, Dict[str, float]]
        When `return_components=True`: the same mean total loss plus the
        sample-weighted mean of each term ({"ce", "dice", "total"}), for
        per-epoch logging.
    """

    model.train()

    cfg = loss_config or LossConfig(name=loss_type)

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_samples = 0

    autocast_device = "cuda" if device == "cuda" else "cpu"

    for batch in dataloader:
        images, masks = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.autocast(device_type=autocast_device, enabled=True):
                predictions = model(images)
                loss, components = compute_loss_components(predictions, masks, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Original FP32 path -- unchanged.
            predictions = model(images)
            loss, components = compute_loss_components(predictions, masks, cfg)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += components["ce"] * batch_size
        total_dice += components["dice"] * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return (0.0, {"ce": 0.0, "dice": 0.0, "total": 0.0}) if return_components else 0.0

    mean_total = total_loss / total_samples
    if not return_components:
        return mean_total

    return mean_total, {
        "ce": total_ce / total_samples,
        "dice": total_dice / total_samples,
        "total": mean_total,
    }


def _multiclass_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float]:
    """Compute mean foreground Dice and IoU for multiclass segmentation."""

    predicted_classes = torch.argmax(predictions, dim=1)

    dice_scores = []
    iou_scores = []

    # Class 0 = background.
    # Evaluate foreground classes only: class 1 and class 2.
    for class_id in range(1, num_classes):
        pred_mask = (predicted_classes == class_id).cpu().numpy()
        true_mask = (targets == class_id).cpu().numpy()

        pred_has_foreground = bool(pred_mask.any())
        true_has_foreground = bool(true_mask.any())

        # Skip a class if neither prediction nor ground truth contains it.
        if not pred_has_foreground and not true_has_foreground:
            continue

        dice_scores.append(
            compute_dice(pred_mask, true_mask)
        )
        iou_scores.append(
            compute_iou(pred_mask, true_mask)
        )

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return mean_dice, mean_iou


def validate(
    model: Any,
    dataloader: Any,
    device: str,
    loss_type: str = "cross_entropy",
    num_classes: int = 3,
    loss_config: Any = None,
) -> Dict[str, float]:
    """Execute validation over the validation dataset.

    IMPORTANT (checkpoint-selection contract): the returned "dice" is the
    batch-level foreground-macro Dice proxy that has ALWAYS driven best-model
    selection in this project, and it is unchanged by the E1 work. Adding a
    `loss_config` only changes the reported validation LOSS terms (so the E1 run
    logs the same objective it trains on); it does NOT change which epoch is
    saved as best. Baseline and E1 therefore use an identical within-run
    checkpoint-selection rule, which is what makes them comparable.
    """

    model.eval()

    cfg = loss_config or LossConfig(name=loss_type)

    total_loss = 0.0
    total_ce = 0.0
    total_dice_term = 0.0
    total_samples = 0

    dice_values = []
    iou_values = []
    # Per-class slice-level Dice, accumulated separately for reporting ONLY --
    # this does not feed the primary `dice` mean used for checkpoint selection,
    # so it cannot change which epoch is saved as best.
    per_class_dice_values: Dict[int, list] = {c: [] for c in range(1, num_classes)}

    with torch.no_grad():
        for batch in dataloader:
            images, masks = _move_batch_to_device(batch, device)

            predictions = model(images)

            loss, components = compute_loss_components(predictions, masks, cfg)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_ce += components["ce"] * batch_size
            total_dice_term += components["dice"] * batch_size
            total_samples += batch_size

            dice, iou = _multiclass_metrics(
                predictions,
                masks,
                num_classes=num_classes,
            )

            dice_values.append(dice)
            iou_values.append(iou)

            predicted_classes = torch.argmax(predictions, dim=1)
            for class_id in range(1, num_classes):
                pred_mask = (predicted_classes == class_id).cpu().numpy()
                true_mask = (masks == class_id).cpu().numpy()
                if not pred_mask.any() and not true_mask.any():
                    continue  # class absent from both: skip (same rule as the mean)
                per_class_dice_values[class_id].append(compute_dice(pred_mask, true_mask))

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    mean_dice = (
        float(np.mean(dice_values))
        if dice_values
        else 0.0
    )

    mean_iou = (
        float(np.mean(iou_values))
        if iou_values
        else 0.0
    )

    per_class_dice = {
        class_id: (float(np.mean(vals)) if vals else 0.0)
        for class_id, vals in per_class_dice_values.items()
    }

    return {
        "loss": average_loss,
        # Per-term breakdown of the validation objective (reporting only --
        # never used for checkpoint selection).
        "ce_loss": (total_ce / total_samples) if total_samples > 0 else 0.0,
        "dice_loss": (total_dice_term / total_samples) if total_samples > 0 else 0.0,
        "dice": mean_dice,
        "iou": mean_iou,
        "per_class_dice": per_class_dice,
    }


def _build_dataset(
    config: Dict[str, Any],
    patient_ids,
    augment: bool = False,
) -> ProstateZonal2DDataset:
    """Build the current thesis-core zonal dataset.

    `augment` (E2) attaches the seeded training-time augmentation callable from
    `src/augment.py`. It defaults to False, and even when True it is a NO-OP
    unless the config carries an `augmentation:` block with `enabled: true` --
    so Baseline / E1 and every validation/test dataset keep `transform=None`,
    exactly as before. Callers must NEVER pass `augment=True` for a validation
    or test split: evaluation has to stay deterministic and unaugmented.
    """

    data_cfg = config.get("data", {})

    dataset_root = data_cfg["dataset_root"]

    slice_sampling = data_cfg.get(
        "slice_sampling",
        "all",
    )

    preprocessing_config = PreprocessingConfig.from_dict(config.get("preprocessing", {}))

    # Thesis-safe default: current thesis target is t2_anatomy_reader1.nii.gz
    # (labels {0,1,2}). `data.target_mask` is honored if a config sets it, but
    # this can never silently select the retired lesion target -- any mask
    # whose labels are not a subset of {0,1,2} (e.g. t2_tumor_reader1.nii.gz,
    # labels {0,3}) is rejected by ProstateZonal2DDataset's own label
    # validation (src/dataset.py), regardless of what this config key says.
    mask_name = data_cfg.get("target_mask", "t2_anatomy_reader1.nii.gz")

    transform = build_train_augmentor(config) if augment else None

    return ProstateZonal2DDataset(
        dataset_root=dataset_root,
        patient_ids=patient_ids,
        image_name="t2.nii.gz",
        mask_name=mask_name,
        slice_sampling=slice_sampling,
        preprocessing_config=preprocessing_config,
        transform=transform,
        cache_data=True,
    )


def _ensure_global_stats(config: Dict[str, Any], train_patient_ids) -> bool:
    """Experiment 2-B safeguard: when preprocessing.normalization == 'global'
    and the stats are not fully supplied, compute them from the TRAINING split
    ONLY and inject them into config['preprocessing']['global_stats'].

    Returns True if stats were computed and injected, False otherwise (no-op).

    Why this lives here and is gated:
      * Gated strictly on normalization == 'global'. The Experiment-1 baseline
        (and Exp02-A / Exp02-C) use 'per_volume' and are therefore COMPLETELY
        untouched by this function -- it returns immediately.
      * Stats are pooled over `train_patient_ids` only -- never validation or
        test cases -- so no evaluation-set information leaks into normalization.
      * Because run_training saves `config` inside the checkpoint, the resolved
        stats travel with the checkpoint, so a later run_evaluation reuses the
        IDENTICAL train-derived stats (no recomputation, no divergence).
    """
    preproc = config.get("preprocessing", {})
    if preproc.get("normalization") != "global":
        return False

    existing = preproc.get("global_stats") or {}
    required = ("p_low", "p_high", "mean", "std")
    if all(existing.get(k) is not None for k in required):
        return False  # already fully provided -- respect explicit values

    from src.transforms import compute_global_intensity_stats

    data_cfg = config.get("data", {})
    stats = compute_global_intensity_stats(
        dataset_root=data_cfg["dataset_root"],
        case_ids=list(train_patient_ids),
        image_name="t2.nii.gz",
    )
    preproc["global_stats"] = stats
    config["preprocessing"] = preproc
    return True


def _build_train_sampler(
    train_dataset: ProstateZonal2DDataset,
    seed: int,
    use_weighted_sampling: bool = True,
) -> Any:
    """Build the TRAINING-only foreground/background-balanced sampler.

    Returns a `WeightedRandomSampler` (deterministic given `seed`, via a
    dedicated `torch.Generator`) when weighted sampling is enabled, or `None`
    when disabled -- callers should then fall back to `shuffle=True`.

    This must NEVER be used for validation/test loaders: those must see the
    complete, unweighted slice population (per the thesis evaluation
    protocol, which requires the full slice population to remain available
    for evaluation -- see `ProstateZonal2DDataset.compute_foreground_sample_weights`).
    """
    if not use_weighted_sampling:
        return None

    weights = train_dataset.compute_foreground_sample_weights()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )


def _capture_rng_state() -> Dict[str, Any]:
    """Snapshot the RNG state of every stream this training loop consumes.

    Persisted inside `latest_checkpoint.pt` so a Colab-interrupted run resumes
    with the same random stream rather than silently re-drawing from a fresh
    seed. CUDA state is captured only when CUDA is actually in use.
    """
    import random

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:  # pragma: no cover - defensive, never fail a save
            state["torch_cuda_all"] = None
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> bool:
    """Restore RNG streams saved by `_capture_rng_state`. Returns True if applied.

    Never raises: a checkpoint written by an older version (no RNG block), or on
    a machine with a different CUDA device count, simply skips restoration.
    """
    import random

    if not state:
        return False
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
        cuda_state = state.get("torch_cuda_all")
        if cuda_state and torch.cuda.is_available():
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARNING: could not fully restore RNG state ({type(exc).__name__}: {exc}).")
        return False
    return True


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """`torch.save` via a temp file + atomic replace.

    Critical for the Colab/Drive workflow: a session that dies mid-write must
    never leave a truncated `latest_checkpoint.pt` behind, because that would
    make the run unresumable. The previous good checkpoint stays intact until
    the new one is completely written.
    """
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# Config keys that MUST match for a resume to be scientifically valid. Note
# `num_epochs` is deliberately ABSENT: extending an interrupted run's epoch
# budget is a legitimate recovery action (and `_load_resume_checkpoint` already
# refuses to resume a run that is already complete).
_COMPAT_MODEL_KEYS = ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate")
_COMPAT_TRAINING_KEYS = ("batch_size", "learning_rate", "seed", "optimizer", "weight_decay", "amp")
_COMPAT_PREPROCESSING_KEYS = ("normalization", "z_resample", "spatial_mode", "target_size")
_COMPAT_DATA_KEYS = ("train_csv", "valid_csv", "target_mask", "slice_sampling")


def verify_checkpoint_compatibility(
    checkpoint_config: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    checkpoint_path: str = "",
) -> Dict[str, Any]:
    """Refuse to resume a run from a checkpoint belonging to a DIFFERENT experiment.

    This is the guard that makes it impossible to, say, continue the E1 DiceCE
    arm from the baseline's `best_model.pt`: experiment name, model
    architecture, loss specification, and the controlled training
    hyperparameters must all agree.

    Verification is skipped (with a printed note, not an error) when the
    checkpoint predates config embedding or carries no `experiment` block --
    older checkpoints stay loadable, exactly as before.

    Returns
    -------
    Dict[str, Any]
        {"verified": bool, "skipped_reason": Optional[str], "mismatches": List[str]}

    Raises
    ------
    ValueError
        If any controlled field differs. The message lists every mismatch.
    """
    result: Dict[str, Any] = {"verified": False, "skipped_reason": None, "mismatches": []}

    if not isinstance(checkpoint_config, dict) or "experiment" not in checkpoint_config:
        result["skipped_reason"] = (
            "checkpoint carries no `experiment` block (pre-dates config embedding); "
            "compatibility verification skipped"
        )
        return result

    mismatches = []

    ckpt_name = (checkpoint_config.get("experiment") or {}).get("name")
    cfg_name = (config.get("experiment") or {}).get("name")
    if ckpt_name != cfg_name:
        mismatches.append(f"experiment.name: checkpoint='{ckpt_name}' vs config='{cfg_name}'")

    ckpt_model = checkpoint_config.get("model") or {}
    cfg_model = config.get("model") or {}
    for key in _COMPAT_MODEL_KEYS:
        if key in ckpt_model or key in cfg_model:
            if ckpt_model.get(key) != cfg_model.get(key):
                mismatches.append(
                    f"model.{key}: checkpoint={ckpt_model.get(key)!r} vs config={cfg_model.get(key)!r}"
                )

    ckpt_training = checkpoint_config.get("training") or {}
    cfg_training = config.get("training") or {}
    for key in _COMPAT_TRAINING_KEYS:
        ckpt_value = ckpt_training.get(key, False if key == "amp" else None)
        cfg_value = cfg_training.get(key, False if key == "amp" else None)
        if ckpt_value != cfg_value:
            mismatches.append(f"training.{key}: checkpoint={ckpt_value!r} vs config={cfg_value!r}")

    ckpt_loss = LossConfig.from_config(checkpoint_config).to_dict()
    cfg_loss = LossConfig.from_config(config).to_dict()
    if ckpt_loss != cfg_loss:
        mismatches.append(f"loss: checkpoint={ckpt_loss} vs config={cfg_loss}")

    # Augmentation (E2) is an experimental factor exactly like the loss, so a
    # resume must not silently switch it on or off, or change any of its
    # parameters, half-way through a run. An older checkpoint carrying no
    # `augmentation:` block resolves to "disabled", which is the correct
    # interpretation of Baseline / E1 checkpoints.
    ckpt_aug = AugmentationConfig.from_config(checkpoint_config).to_dict()
    cfg_aug = AugmentationConfig.from_config(config).to_dict()
    if ckpt_aug != cfg_aug:
        mismatches.append(f"augmentation: checkpoint={ckpt_aug} vs config={cfg_aug}")

    # Preprocessing must be identical too: resuming across a changed
    # normalization / resample / spatial policy would mix two input pipelines.
    ckpt_preproc = checkpoint_config.get("preprocessing") or {}
    cfg_preproc = config.get("preprocessing") or {}
    for key in _COMPAT_PREPROCESSING_KEYS:
        ckpt_value = ckpt_preproc.get(key)
        cfg_value = cfg_preproc.get(key)
        if isinstance(ckpt_value, list):
            ckpt_value = tuple(ckpt_value)
        if isinstance(cfg_value, list):
            cfg_value = tuple(cfg_value)
        if ckpt_value != cfg_value:
            mismatches.append(
                f"preprocessing.{key}: checkpoint={ckpt_value!r} vs config={cfg_value!r}"
            )

    # Dataset split identity: same CSVs, same target mask, same slice policy.
    ckpt_data = checkpoint_config.get("data") or {}
    cfg_data = config.get("data") or {}
    for key in _COMPAT_DATA_KEYS:
        if ckpt_data.get(key) != cfg_data.get(key):
            mismatches.append(
                f"data.{key}: checkpoint={ckpt_data.get(key)!r} vs config={cfg_data.get(key)!r}"
            )

    if mismatches:
        result["mismatches"] = mismatches
        raise ValueError(
            "Refusing to resume: the checkpoint was produced by a DIFFERENT experiment "
            "configuration, so continuing would silently mix two experiments.\n"
            f"  checkpoint: {checkpoint_path or '<unknown>'}\n"
            + "".join(f"  - {m}\n" for m in mismatches)
            + "Start this experiment fresh in its own output_dir, or point "
            "`resume_from` at a checkpoint belonging to THIS experiment."
        )

    result["verified"] = True
    return result


_METRICS_CSV_FIELDS = (
    "epoch",
    "train_loss_total",
    "train_loss_ce",
    "train_loss_dice",
    "val_loss",
    "val_dice_proxy",
    "val_iou",
    "val_dice_class1_CG",
    "val_dice_class2_PZ",
    "learning_rate",
    "epoch_seconds",
    "amp",
)


def _write_metrics_csv(path: str, history: list) -> None:
    """Rewrite the per-epoch metrics CSV from the in-memory history.

    Written after EVERY epoch (whole-file rewrite, then atomic replace) so a
    Colab timeout still leaves a complete, readable record of everything
    finished so far.
    """
    import csv

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_METRICS_CSV_FIELDS))
        writer.writeheader()
        for row in history:
            per_class = row.get("val_per_class_dice") or {}
            writer.writerow({
                "epoch": row.get("epoch"),
                "train_loss_total": row.get("train_loss"),
                "train_loss_ce": row.get("train_loss_ce"),
                "train_loss_dice": row.get("train_loss_dice"),
                "val_loss": row.get("val_loss"),
                "val_dice_proxy": row.get("val_dice"),
                "val_iou": row.get("val_iou"),
                "val_dice_class1_CG": per_class.get("1"),
                "val_dice_class2_PZ": per_class.get("2"),
                "learning_rate": row.get("learning_rate"),
                "epoch_seconds": row.get("epoch_seconds"),
                "amp": row.get("amp"),
            })
    os.replace(tmp_path, path)


def _atomic_json_dump(payload: Any, path: str) -> None:
    """Write JSON via a temp file + atomic replace.

    Same motivation as `_atomic_torch_save`: a Colab session killed mid-write
    must never leave a truncated `training_history.json`, because that file is
    the only complete record of the epochs that came before a resume.
    """
    import json

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _history_from_metrics_csv(path: str) -> list:
    """Reconstruct per-epoch history records from a `metrics.csv` written by
    `_write_metrics_csv`. Used as the FALLBACK history source when no JSON
    history file exists (e.g. an arm first run without `history_path`).

    Never raises on a malformed/partial file: an unreadable row is skipped, so
    a half-written CSV can still contribute the epochs it does hold.
    """
    import csv

    if not path or not os.path.isfile(path):
        return []

    def _num(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    records = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                try:
                    epoch = int(raw["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                cg = _num(raw.get("val_dice_class1_CG"))
                pz = _num(raw.get("val_dice_class2_PZ"))
                per_class = {}
                if cg is not None:
                    per_class["1"] = cg
                if pz is not None:
                    per_class["2"] = pz
                records.append({
                    "epoch": epoch,
                    "train_loss": _num(raw.get("train_loss_total")),
                    "train_loss_ce": _num(raw.get("train_loss_ce")),
                    "train_loss_dice": _num(raw.get("train_loss_dice")),
                    "val_loss": _num(raw.get("val_loss")),
                    "val_loss_ce": None,
                    "val_loss_dice": None,
                    "val_dice": _num(raw.get("val_dice_proxy")),
                    "val_iou": _num(raw.get("val_iou")),
                    "val_per_class_dice": per_class or None,
                    "learning_rate": _num(raw.get("learning_rate")),
                    "epoch_seconds": _num(raw.get("epoch_seconds")),
                    "amp": str(raw.get("amp", "")).strip().lower() == "true",
                })
    except OSError:
        return []
    return records


def load_existing_history(
    history_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> list:
    """Load the per-epoch history already on disk for an experiment arm.

    THIS IS THE FIX FOR THE HISTORY-TRUNCATION BUG. Before this existed,
    `run_training` always started from an empty `epoch_history`, so a run
    resumed at epoch 57 rewrote `metrics.csv` / `training_history.json` with
    epochs 57..100 only and epochs 1..56 were lost -- even though the training
    itself resumed correctly.

    Sources, in order of preference:
      1. `history_path` (JSON written by run_training) -- richest record, it
         carries the per-class val Dice dict and the val loss breakdown;
      2. `metrics_csv_path` (metrics.csv) -- same epochs, slightly fewer
         fields; used when the JSON is missing, unreadable, or covers fewer
         epochs than the CSV.

    Records from both sources are merged by epoch (JSON wins on conflict),
    returned sorted by epoch. Returns `[]` when nothing exists on disk, which
    is exactly the correct state for a fresh run. Never raises: a corrupted
    history file must not be able to abort a training run, so it is reported
    and skipped rather than propagated.
    """
    import json

    json_records: list = []
    if history_path and os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("epochs", []) if isinstance(payload, dict) else payload
            if isinstance(raw, list):
                json_records = [r for r in raw if isinstance(r, dict) and "epoch" in r]
        except (OSError, ValueError) as exc:
            print(
                f"  WARNING: could not read existing history JSON '{history_path}' "
                f"({type(exc).__name__}: {exc}); falling back to metrics.csv."
            )

    csv_records = _history_from_metrics_csv(metrics_csv_path) if metrics_csv_path else []

    merged: Dict[int, Dict[str, Any]] = {}
    for record in csv_records:
        merged[int(record["epoch"])] = record
    for record in json_records:  # JSON wins: it is the richer record
        try:
            merged[int(record["epoch"])] = record
        except (TypeError, ValueError):
            continue

    return [merged[e] for e in sorted(merged)]


def upsert_epoch_record(history: list, record: Dict[str, Any]) -> list:
    """Insert `record` into `history`, keyed by epoch, keeping epochs sorted.

    Epoch is the unique key: re-running an epoch that already exists REPLACES
    that one record and leaves every other epoch intact. Nothing is ever
    truncated, and the history can only grow or be corrected in place.

    Mutates and returns `history` so callers can use it either way.
    """
    epoch = int(record["epoch"])
    for index, existing in enumerate(history):
        try:
            existing_epoch = int(existing.get("epoch"))
        except (AttributeError, TypeError, ValueError):
            continue
        if existing_epoch == epoch:
            history[index] = record
            break
    else:
        history.append(record)

    history.sort(key=lambda r: int(r.get("epoch", 0)))
    return history


def _load_resume_checkpoint(
    resume_from: str,
    model: Any,
    optimizer: Any,
    device: str,
    num_epochs: int,
    scaler: Any = None,
    sampler: Any = None,
    restore_rng: bool = False,
    verify_against_config: Optional[Dict[str, Any]] = None,
    state_out: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float, int]:
    """Restore model + optimizer state from a checkpoint to resume training.

    Returns (start_epoch, best_metric, checkpoint_epoch), where start_epoch is
    checkpoint_epoch + 1 -- training continues from there, never from epoch 1.

    The checkpoint format is exactly the one written by run_training: a dict
    with "epoch", "model_state_dict", "optimizer_state_dict", "best_metric"
    (and "config"). Backward-compatible with checkpoints carrying just those
    keys (e.g. the Experiment-1 epoch-60 best_model.pt), and the format written
    on resume is unchanged, so resumed checkpoints stay loadable the same way.

    AMP resume (optional, pilot only): if `scaler` is provided AND the checkpoint
    carries a "scaler_state_dict" (only AMP-mode runs write one), the GradScaler
    state is restored so mixed-precision resume is correct. An FP32 checkpoint
    (no such key) loads exactly as before -- `scaler` is simply left fresh -- so
    all existing FP32 checkpoints remain fully loadable and AMP fields are never
    required.

    `weights_only=False` is required because the checkpoint embeds the config
    dict (arbitrary Python), not just tensors.

    Fails clearly (ValueError) if the checkpoint is already at or past the
    configured total epochs, instead of silently doing nothing / retraining.
    """
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")

    checkpoint = torch.load(resume_from, map_location=device, weights_only=False)

    for required_key in ("epoch", "model_state_dict", "optimizer_state_dict", "best_metric"):
        if required_key not in checkpoint:
            raise KeyError(
                f"Resume checkpoint '{resume_from}' is missing required key "
                f"'{required_key}'. Keys present: {sorted(checkpoint.keys())}"
            )

    # Experiment-identity guard: verified BEFORE any state is loaded into the
    # model, so an incompatible checkpoint cannot half-apply. Raises on mismatch.
    if verify_against_config is not None:
        compat = verify_checkpoint_compatibility(
            checkpoint.get("config"), verify_against_config, checkpoint_path=resume_from
        )
        if compat["verified"]:
            print("  Checkpoint compatibility: VERIFIED (same experiment, model, loss, hyperparameters).")
        elif compat["skipped_reason"]:
            print(f"  Checkpoint compatibility: SKIPPED -- {compat['skipped_reason']}.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Optional reproducibility state (only written by newer checkpoints; an
    # older checkpoint simply has neither key and is loaded exactly as before).
    if restore_rng and _restore_rng_state(checkpoint.get("rng_state")):
        print("  Restored RNG state (python/numpy/torch) from checkpoint.")

    sampler_state = checkpoint.get("sampler_generator_state")
    if sampler is not None and sampler_state is not None:
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            try:
                generator.set_state(torch.as_tensor(sampler_state, dtype=torch.uint8))
                print("  Restored WeightedRandomSampler generator state from checkpoint.")
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"  WARNING: could not restore sampler generator state "
                    f"({type(exc).__name__}: {exc}); sampling continues from the seeded state."
                )

    checkpoint_epoch = int(checkpoint["epoch"])
    best_metric = float(checkpoint["best_metric"])
    start_epoch = checkpoint_epoch + 1

    # Optional extras for callers that want more than the 3-tuple (the tuple
    # shape is part of this function's existing contract and stays as it is).
    # `best_epoch` is written by current checkpoints; older ones lack it and
    # report None, which callers must treat as "unknown", never as epoch 0.
    if state_out is not None:
        state_out["best_epoch"] = checkpoint.get("best_epoch")
        state_out["config"] = checkpoint.get("config")

    if start_epoch > num_epochs:
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} is already >= configured total "
            f"epochs ({num_epochs}); there is nothing to resume. Increase "
            f"training.num_epochs for a longer run, or start a fresh run instead."
        )

    return start_epoch, best_metric, checkpoint_epoch


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Read-only inspection of a training checkpoint, for orchestration/recovery.

    NEVER trains, mutates, or deletes the checkpoint -- it only `torch.load`s it
    (map_location='cpu') and reports what it stores. Safe to call on a
    Drive-backed checkpoint before deciding whether to resume. Never raises: a
    missing file, or an unreadable (corrupted/truncated) or structurally
    incompatible one, is reported through the returned dict, not an exception.

    Returns a dict with keys:
      exists, path, size_bytes, readable, error,
      epoch, best_metric,
      has_model_state, has_optimizer_state, has_scheduler_state, has_rng_state,
      resumable

    Note on `has_scheduler_state` / `has_rng_state`: the training design uses no
    LR scheduler, so `has_scheduler_state` is always False. `has_rng_state` is
    True for `latest_checkpoint.pt` files written by the current run_training
    (which persists python/numpy/torch RNG state for interruption-safe resume)
    and False for older checkpoints and for `best_model.pt`; neither is required
    for a safe resume.
    """
    info: Dict[str, Any] = {
        "exists": False,
        "path": checkpoint_path,
        "size_bytes": None,
        "readable": False,
        "error": None,
        "epoch": None,
        "best_metric": None,
        "best_epoch": None,
        "has_model_state": False,
        "has_optimizer_state": False,
        "has_scheduler_state": False,
        "has_scaler_state": False,
        "has_rng_state": False,
        "resumable": False,
    }

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return info

    info["exists"] = True
    try:
        info["size_bytes"] = os.path.getsize(checkpoint_path)
    except OSError:
        info["size_bytes"] = None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupted / truncated / unpicklable
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    if not isinstance(checkpoint, dict):
        info["error"] = f"Checkpoint is not a dict (got {type(checkpoint).__name__})."
        return info

    info["readable"] = True
    info["epoch"] = checkpoint.get("epoch")
    info["best_metric"] = checkpoint.get("best_metric")
    # Written by current checkpoints; None for older ones (means "unknown").
    info["best_epoch"] = checkpoint.get("best_epoch")
    info["has_model_state"] = "model_state_dict" in checkpoint
    info["has_optimizer_state"] = "optimizer_state_dict" in checkpoint
    info["has_scheduler_state"] = "scheduler_state_dict" in checkpoint
    info["has_scaler_state"] = "scaler_state_dict" in checkpoint
    info["has_rng_state"] = any("rng" in str(k).lower() for k in checkpoint.keys())

    required = ("epoch", "model_state_dict", "optimizer_state_dict", "best_metric")
    missing = [k for k in required if k not in checkpoint]
    if missing:
        info["error"] = f"Missing required key(s): {missing}"
    elif not isinstance(info["epoch"], int):
        info["error"] = f"'epoch' is not an int (got {type(info['epoch']).__name__})."
    else:
        info["resumable"] = True

    return info


def plan_training_run(checkpoint_path: str, num_epochs: int) -> Dict[str, Any]:
    """Decide -- WITHOUT training or mutating anything -- whether an arm should
    start fresh, resume, is already complete, or has an unusable checkpoint, and
    build a human-readable log message.

    This is pure orchestration/recovery logic; the actual resume mechanics still
    live in `_load_resume_checkpoint` / `run_training`. A Colab (or any) caller
    uses `action` to choose between `run_training(config)` and
    `run_training(config, resume_from=checkpoint_path)` -- so the notebook never
    duplicates training logic.

    `action` is one of:
      "fresh"            -- no checkpoint present; start at epoch 1.
      "resume"           -- checkpoint at epoch < num_epochs; continue at epoch+1.
      "already_complete" -- checkpoint epoch >= num_epochs; do NOT train again.
      "corrupt"          -- checkpoint exists but is unreadable/incompatible;
                            fail clearly, do NOT start fresh, do NOT delete it.
    """
    info = inspect_checkpoint(checkpoint_path)

    if not info["exists"]:
        return {
            "action": "fresh",
            "resume_from": None,
            "checkpoint_epoch": None,
            "best_metric": None,
            "start_epoch": 1,
            "num_epochs": num_epochs,
            "checkpoint_info": info,
            "message": (
                "No checkpoint found.\n"
                f"Starting fresh from epoch 1/{num_epochs}."
            ),
        }

    if not info["resumable"]:
        reason = info["error"] or "unreadable checkpoint"
        return {
            "action": "corrupt",
            "resume_from": None,
            "checkpoint_epoch": info["epoch"],
            "best_metric": info["best_metric"],
            "start_epoch": None,
            "num_epochs": num_epochs,
            "checkpoint_info": info,
            "message": (
                "Existing checkpoint detected but NOT usable:\n"
                f"{checkpoint_path}\n"
                f"Reason: {reason}\n"
                "Refusing to start fresh (that would risk discarding a real run). "
                "The checkpoint has NOT been modified or deleted -- investigate or "
                "restore a good checkpoint before retrying."
            ),
        }

    epoch = int(info["epoch"])
    if epoch >= num_epochs:
        return {
            "action": "already_complete",
            "resume_from": None,
            "checkpoint_epoch": epoch,
            "best_metric": info["best_metric"],
            "best_epoch": info.get("best_epoch"),
            "start_epoch": None,
            "num_epochs": num_epochs,
            "checkpoint_info": info,
            "message": (
                "Existing checkpoint detected:\n"
                f"{checkpoint_path}\n"
                f"Checkpoint epoch: {epoch} (>= {num_epochs}).\n"
                "Training already complete for this arm -- NOT retraining. "
                "Proceed to this arm's validation-evaluation cell."
            ),
        }

    best = info["best_metric"]
    best_str = f"{best:.6f}" if isinstance(best, (int, float)) else str(best)
    best_epoch = info.get("best_epoch")
    return {
        "action": "resume",
        "resume_from": checkpoint_path,
        "checkpoint_epoch": epoch,
        "best_metric": best,
        "best_epoch": best_epoch,
        "start_epoch": epoch + 1,
        "num_epochs": num_epochs,
        "checkpoint_info": info,
        "message": (
            "Existing checkpoint detected:\n"
            f"{checkpoint_path}\n"
            "Resume requested.\n"
            f"Checkpoint epoch: {epoch}\n"
            f"Best validation Dice: {best_str}\n"
            f"Continuing from epoch {epoch + 1}/{num_epochs}."
        ),
    }


def run_training(
    config_path: str,
    resume_from: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run (or resume) a training experiment from a YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to the experiment YAML. Defines the split, model, optimizer,
        epoch target, preprocessing, output_dir, etc. -- unchanged by resume.
    resume_from : Optional[str], default=None
        If given, path to a checkpoint (e.g. best_model.pt) whose model,
        optimizer, and best_metric are restored; training then continues from
        the checkpoint's stored epoch + 1 through training.num_epochs. If None
        (the default), a fresh run starts at epoch 1 with best_metric = -inf --
        identical to the prior behavior.
    history_path : Optional[str], default=None
        If given, per-epoch metrics (train loss, val loss, val Dice, per-class
        val Dice, epoch runtime seconds) are appended to this JSON file. Purely
        additive telemetry for the AMP pilot; None (the default) means no history
        file is written, so Exp01/Exp02 behavior is unchanged.

    Mixed precision (AMP) is opt-in via `training.amp: true` in the config and is
    OFF by default; every FP32 config trains exactly as before. AMP affects
    training only -- validation/evaluation stays FP32.

    Returns
    -------
    Dict[str, Any]
        Small run summary: {"resumed", "start_epoch", "num_epochs",
        "best_metric", "best_checkpoint", "amp"}. Callers may ignore it.
    """
    import json
    import time

    config = load_config(config_path)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    validation_cfg = config.get("validation", {})

    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()

    # Mixed precision: OPT-IN, off by default. Only truly active on CUDA; a
    # GradScaler is created only in AMP mode and its state is checkpointed so
    # AMP resume is correct. FP32 runs never create a scaler and never write a
    # scaler_state_dict, so their checkpoints are byte-compatible with before.
    use_amp = bool(training_cfg.get("amp", False))
    scaler = torch.amp.GradScaler(enabled=(use_amp and device == "cuda")) if use_amp else None
    if use_amp:
        print(f"AMP (mixed precision) ENABLED for TRAINING "
              f"(effective on CUDA only; device={device}). Evaluation stays FP32.")

    # Augmentation (E2). Resolved from the config: absent block -> disabled, so
    # Baseline / E1 are untouched. Applied to the TRAINING dataset only.
    augmentation_config = AugmentationConfig.from_config(config)
    if augmentation_config.enabled:
        print(f"Augmentation ENABLED (training slices only): {augmentation_config.describe()}")
        if int(training_cfg.get("num_workers", 0)) > 0:
            raise ValueError(
                "Augmentation is enabled together with training.num_workers > 0 "
                f"({training_cfg.get('num_workers')}). Forked DataLoader workers would each "
                "inherit a COPY of the augmentation RNG, which destroys seed-exact "
                "reproducibility and distorts the augmentation distribution.\n"
                "Fix: set training.num_workers: 0 (the project default) for any augmented arm."
            )

    # Per-epoch history. Populated from disk BELOW once the output paths are
    # known, so a resumed run appends to the existing record instead of
    # starting an empty one (see `load_existing_history`).
    epoch_history: list = []

    output_dir = ensure_dir(
        config.get(
            "experiment",
            {},
        ).get(
            "output_dir",
            "./results",
        )
    )

    # Loss selection (E1). Defaults to plain cross entropy, so every config
    # without a `loss:` block trains exactly as the baseline did.
    loss_config = LossConfig.from_config(config)

    print(f"Loaded config: {config_path}")
    print(f"Target compute device: {device}")
    print(f"Outputs will be saved to: {output_dir}")

    # ---------------------------------------------------------
    # Experiment fingerprint -- printed AND persisted next to the
    # results, so it is always obvious which single factor this
    # run changed relative to the baseline.
    # ---------------------------------------------------------
    fingerprint = build_config_fingerprint(config, device=device)
    print()
    print(format_config_fingerprint(fingerprint))
    print()

    with open(os.path.join(output_dir, "config_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, default=str)
    with open(os.path.join(output_dir, "config_fingerprint.txt"), "w", encoding="utf-8") as f:
        f.write(format_config_fingerprint(fingerprint) + "\n")
    try:
        import yaml

        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    except Exception as exc:  # pragma: no cover - snapshot is best-effort
        print(f"WARNING: could not write config.yaml snapshot ({type(exc).__name__}: {exc}).")

    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    latest_checkpoint = os.path.join(output_dir, "latest_checkpoint.pt")
    training_state_path = os.path.join(ensure_dir(os.path.join(output_dir, "state")),
                                       "training_state.json")

    # Optional periodic epoch snapshots (`checkpoints/epoch_XXX.pt`). OFF by
    # default (0), because each snapshot of this model is ~93 MB and 100 of them
    # would be ~9 GB of Google Drive. `latest_checkpoint.pt` (every epoch) plus
    # `best_model.pt` (on improvement) already make the run fully resumable; the
    # snapshots are only for post-hoc inspection of intermediate epochs.
    snapshot_every = int(training_cfg.get("checkpoint_every_n_epochs", 0) or 0)
    snapshot_dir = ensure_dir(os.path.join(output_dir, "checkpoints")) if snapshot_every > 0 else None
    if snapshot_every > 0:
        print(f"Periodic epoch snapshots: every {snapshot_every} epoch(s) -> {snapshot_dir}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_patient_ids, val_patient_ids, test_patient_ids = _resolve_split_ids(config)

    if set(train_patient_ids) & set(val_patient_ids):
        raise ValueError(
            "Training and validation patient IDs overlap."
        )

    if test_patient_ids is None:
        print(
            "WARNING: official 19-case test set not available locally; "
            "this run has no held-out test evaluation."
        )

    # Exp02-B: if global normalization is requested without explicit stats,
    # derive them from the TRAINING split only (leakage-safe) and inject them
    # so both train+val datasets and the saved checkpoint share identical stats.
    if _ensure_global_stats(config, train_patient_ids):
        gs = config["preprocessing"]["global_stats"]
        print(
            "Computed GLOBAL normalization stats from the training split only "
            f"(n={len(train_patient_ids)}): "
            f"mean={gs['mean']:.4f} std={gs['std']:.4f} "
            f"p_low={gs['p_low']:.2f} p_high={gs['p_high']:.2f}"
        )

    train_dataset = _build_dataset(
        config,
        train_patient_ids,
        augment=True,   # NO-OP unless the config enables augmentation (E2)
    )

    print(
        f"Training patients: {train_patient_ids}"
    )
    print(
        f"Training slices: {len(train_dataset)}"
    )

    # Training-time foreground/background balancing (Phase 3 Step 10). Does
    # NOT filter any slice out of train_dataset -- it only changes sampling
    # frequency. Enabled by default; set training.use_weighted_sampling:
    # false in a config to fall back to plain shuffling.
    use_weighted_sampling = training_cfg.get("use_weighted_sampling", True)
    train_sampler = _build_train_sampler(train_dataset, seed, use_weighted_sampling)

    loader_perf_kwargs = _dataloader_perf_kwargs(training_cfg, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get(
            "batch_size",
            4,
        ),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_perf_kwargs,
    )

    val_loader = None

    if val_patient_ids:
        # augment=False, ALWAYS. The validation set is never augmented -- the
        # evaluation signal that drives checkpoint selection has to stay
        # deterministic and comparable across arms.
        val_dataset = _build_dataset(
            config,
            val_patient_ids,
            augment=False,
        )

        print(
            f"Validation patients: {val_patient_ids}"
        )
        print(
            f"Validation slices: {len(val_dataset)}"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_cfg.get(
                "batch_size",
                4,
            ),
            shuffle=False,
            **loader_perf_kwargs,
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(config).to(device)

    print(
        f"Model: {model_cfg.get('architecture', 'ProstateUNet2D')}"
    )
    print(
        f"Input channels: {model_cfg.get('in_channels', 1)}"
    )
    print(
        f"Output classes: {model_cfg.get('out_channels', 3)}"
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer_name = training_cfg.get(
        "optimizer",
        "AdamW",
    ).lower()

    learning_rate = training_cfg.get(
        "learning_rate",
        0.001,
    )

    weight_decay = training_cfg.get(
        "weight_decay",
        0.0001,
    )

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    num_epochs = training_cfg.get(
        "num_epochs",
        5,
    )

    loss_type = training_cfg.get(
        "loss_function",
        "cross_entropy",
    )

    num_classes = model_cfg.get(
        "out_channels",
        3,
    )

    best_checkpoint = os.path.join(
        output_dir,
        "best_model.pt",
    )

    # ---------------------------------------------------------
    # Resume (optional).
    # A fresh run (resume_from=None) starts at epoch 1 with
    # best_metric = -inf, exactly as before. Resuming restores
    # model + optimizer + best_metric and continues from the
    # checkpoint's stored epoch + 1 -- it never restarts at 1,
    # and best_metric is never reset, so best_model.pt is still
    # only overwritten when validation Dice improves on the
    # restored best.
    # ---------------------------------------------------------
    if resume_from is not None:
        restored: Dict[str, Any] = {}
        start_epoch, best_metric, checkpoint_epoch = _load_resume_checkpoint(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            scaler=scaler,
            sampler=train_sampler,
            restore_rng=True,
            verify_against_config=config,
            state_out=restored,
        )
        best_epoch = restored.get("best_epoch")

        # HISTORY PRESERVATION. Load everything already recorded for this arm
        # BEFORE the epoch loop, so new epochs are appended to epochs 1..N
        # rather than replacing them. Without this the resumed session would
        # rewrite metrics.csv / training_history.json containing only the
        # epochs it ran itself.
        epoch_history = load_existing_history(history_path, metrics_csv_path)
        prior_epochs = [int(r["epoch"]) for r in epoch_history]

        print(f"Resumed from checkpoint: {resume_from}")
        print(
            f"  Checkpoint epoch: {checkpoint_epoch} "
            f"| restored best_metric: {best_metric:.6f}"
            f" | best epoch: {best_epoch if best_epoch is not None else 'unknown (pre-dates best_epoch tracking)'}"
        )
        print(f"  Continuing from epoch {start_epoch} through {num_epochs}")
        if epoch_history:
            print(
                f"  Restored training history: {len(epoch_history)} epoch record(s), "
                f"epochs {min(prior_epochs)}..{max(prior_epochs)} -- these are PRESERVED "
                "and will be appended to, never overwritten."
            )
            missing = sorted(set(range(1, checkpoint_epoch + 1)) - set(prior_epochs))
            if missing:
                print(
                    f"  NOTE: no history record on disk for epoch(s) {missing}. Training "
                    "resumes correctly regardless; those rows simply were never written "
                    "(or were lost before this run) and cannot be reconstructed."
                )
        else:
            print(
                "  WARNING: no existing history found on disk for this arm "
                f"(looked in {history_path!r} and {metrics_csv_path!r}). "
                "The rebuilt history will start at the resumed epoch."
            )
    else:
        start_epoch = 1
        best_metric = -float("inf")
        best_epoch = None

        # A FRESH run must not silently inherit a previous run's history. If
        # records exist here, the caller pointed a fresh run at a directory
        # that already holds a run -- refuse rather than blend two runs.
        stale = load_existing_history(history_path, metrics_csv_path)
        if stale:
            stale_epochs = [int(r["epoch"]) for r in stale]
            raise ValueError(
                "Refusing to start a FRESH run in an output directory that already "
                f"contains training history ({len(stale)} epoch record(s), epochs "
                f"{min(stale_epochs)}..{max(stale_epochs)}) in:\n"
                f"  {metrics_csv_path}\n"
                f"  {history_path}\n"
                "Starting fresh here would overwrite that record. Either pass "
                "`resume_from=<latest_checkpoint.pt>` to continue that run, or point "
                "`experiment.output_dir` at a new directory."
            )

    for epoch in range(start_epoch, num_epochs + 1):

        epoch_start = time.perf_counter()

        train_loss, train_components = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=loss_type,
            use_amp=use_amp,
            scaler=scaler,
            loss_config=loss_config,
            return_components=True,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])

        if loss_config.is_baseline_cross_entropy:
            print(
                f"Epoch {epoch}/{num_epochs} "
                f"- train_loss: {train_loss:.6f}"
            )
        else:
            print(
                f"Epoch {epoch}/{num_epochs} "
                f"- train_loss: {train_loss:.6f} "
                f"(ce: {train_components['ce']:.6f} | dice: {train_components['dice']:.6f})"
            )

        val_metrics = None
        if val_loader is not None:

            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_type=loss_type,
                num_classes=num_classes,
                loss_config=loss_config,
            )

            per_class = val_metrics.get("per_class_dice", {})
            print(
                f"  val_loss: {val_metrics['loss']:.6f} "
                f"| val_dice: {val_metrics['dice']:.6f} "
                f"| val_iou: {val_metrics['iou']:.6f}"
            )
            if per_class:
                print(
                    "  val_dice per class (slice-level proxy) -- "
                    + " | ".join(
                        f"class {cid}: {value:.6f}" for cid, value in sorted(per_class.items())
                    )
                )

            current_metric = val_metrics["dice"]

        else:
            # For the current single-patient sanity check,
            # there is no validation set.
            current_metric = -train_loss

        epoch_seconds = time.perf_counter() - epoch_start

        # Per-epoch telemetry. `upsert_epoch_record` keys on `epoch`, so this
        # appends a new epoch (or corrects an existing one in place) and can
        # never drop a previously recorded epoch.
        upsert_epoch_record(epoch_history, {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_ce": train_components["ce"],
            "train_loss_dice": train_components["dice"],
            "val_loss": (val_metrics["loss"] if val_metrics else None),
            "val_loss_ce": (val_metrics["ce_loss"] if val_metrics else None),
            "val_loss_dice": (val_metrics["dice_loss"] if val_metrics else None),
            "val_dice": (val_metrics["dice"] if val_metrics else None),
            "val_iou": (val_metrics["iou"] if val_metrics else None),
            "val_per_class_dice": (
                {str(k): v for k, v in val_metrics.get("per_class_dice", {}).items()}
                if val_metrics else None
            ),
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "amp": use_amp,
        })

        def _build_checkpoint_payload(include_resume_state: bool) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": config,
            }
            # Only AMP runs persist scaler state -- FP32 checkpoints stay in the
            # exact prior 5-key format (backward-compatible on load).
            if use_amp and scaler is not None:
                payload["scaler_state_dict"] = scaler.state_dict()
            if include_resume_state:
                # Reproducibility state lives ONLY in latest_checkpoint.pt, so
                # best_model.pt keeps the historical 5-key format byte-for-byte.
                payload["rng_state"] = _capture_rng_state()
                generator = getattr(train_sampler, "generator", None)
                if generator is not None:
                    payload["sampler_generator_state"] = generator.get_state()
            return payload

        if current_metric > best_metric:

            best_metric = current_metric
            best_epoch = epoch

            _atomic_torch_save(_build_checkpoint_payload(include_resume_state=False), best_checkpoint)

            print(
                f"  Saved best checkpoint: {best_checkpoint} (best epoch: {best_epoch})"
            )

        # Interruption safety: latest_checkpoint.pt is refreshed after EVERY
        # epoch (atomically) and carries model + optimizer + epoch + best_metric
        # + config + RNG/sampler state, so a Colab disconnect resumes from the
        # next epoch rather than restarting. It is written AFTER the best-model
        # block so its `best_metric` reflects this epoch's outcome.
        _atomic_torch_save(_build_checkpoint_payload(include_resume_state=True), latest_checkpoint)

        if snapshot_every > 0 and epoch % snapshot_every == 0:
            snapshot_path = os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pt")
            _atomic_torch_save(_build_checkpoint_payload(include_resume_state=False), snapshot_path)
            print(f"  Saved epoch snapshot: {snapshot_path}")

        # Persist the running history each epoch so a timeout still leaves a
        # usable partial record (atomic-ish: written whole each time).
        _write_metrics_csv(metrics_csv_path, epoch_history)

        # Human/orchestrator-readable run state, refreshed every epoch. A
        # notebook can read this WITHOUT loading a 93 MB checkpoint to find out
        # where the run got to.
        _atomic_json_dump(
            {
                "experiment": (config.get("experiment") or {}).get("name"),
                "output_dir": output_dir,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_completed_epoch": epoch,
                "num_epochs": num_epochs,
                "epochs_remaining": max(0, num_epochs - epoch),
                "complete": epoch >= num_epochs,
                "epoch_records_on_disk": len(epoch_history),
                "first_recorded_epoch": epoch_history[0]["epoch"] if epoch_history else None,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "resumed_this_session": resume_from is not None,
                "session_start_epoch": start_epoch,
                "device": device,
                "amp": use_amp,
                "loss": loss_config.to_dict(),
                "augmentation": augmentation_config.to_dict(),
                "best_checkpoint": best_checkpoint,
                "latest_checkpoint": latest_checkpoint,
                "metrics_csv": metrics_csv_path,
                "history_json": history_path,
            },
            training_state_path,
        )

        if history_path is not None:
            _atomic_json_dump(
                {
                    "experiment": (config.get("experiment") or {}).get("name"),
                    "amp": use_amp,
                    "device": device,
                    "num_epochs": num_epochs,
                    "augmentation": augmentation_config.to_dict(),
                    "loss": loss_config.to_dict(),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "epochs_recorded": len(epoch_history),
                    "epochs": epoch_history,
                },
                history_path,
            )

    print("Training completed.")
    print(f"Best checkpoint:   {best_checkpoint} (epoch {best_epoch})")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"Per-epoch metrics: {metrics_csv_path} ({len(epoch_history)} epoch record(s))")

    return {
        "resumed": resume_from is not None,
        "start_epoch": start_epoch,
        "num_epochs": num_epochs,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_recorded": len(epoch_history),
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "metrics_csv": metrics_csv_path,
        "history_path": history_path,
        "training_state": training_state_path,
        "amp": use_amp,
        "loss": loss_config.to_dict(),
        "augmentation": augmentation_config.to_dict(),
    }


if __name__ == "__main__":
    run_training("configs/config_020.yaml")