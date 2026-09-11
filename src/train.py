"""Training loop for 2D prostate zonal anatomy segmentation."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import ProstateZonal2DDataset
from src.evaluate import compute_dice, compute_iou
from src.model import build_model
from src.splits import load_official_split
from src.transforms import PreprocessingConfig
from src.utils import ensure_dir, get_device, load_config, set_seed


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
):
    """
    Compute loss for multiclass prostate zonal segmentation.

    Parameters
    ----------
    predictions : torch.Tensor
        Raw logits with shape (B, C, H, W).
    targets : torch.Tensor
        Integer class labels with shape (B, H, W).

    Returns
    -------
    torch.Tensor
        Scalar CrossEntropy loss.
    """
    if loss_type.lower() in ("cross_entropy", "ce"):
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, targets.long())

    raise ValueError(
        f"Unsupported loss_type: {loss_type}. "
        "Currently supported: cross_entropy"
    )


def _move_batch_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move image and mask tensors from a dataset batch to the target device."""
    images = batch["image"].float().to(device)
    masks = batch["mask"].long().to(device)

    return images, masks


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: str,
    loss_type: str = "cross_entropy",
) -> float:
    """Execute one training epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images, masks = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        loss = compute_loss(
            predictions,
            masks,
            loss_type=loss_type,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return total_loss / total_samples


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
) -> Dict[str, float]:
    """Execute validation over the validation dataset."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    dice_values = []
    iou_values = []

    with torch.no_grad():
        for batch in dataloader:
            images, masks = _move_batch_to_device(batch, device)

            predictions = model(images)

            loss = compute_loss(
                predictions,
                masks,
                loss_type=loss_type,
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            dice, iou = _multiclass_metrics(
                predictions,
                masks,
                num_classes=num_classes,
            )

            dice_values.append(dice)
            iou_values.append(iou)

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

    return {
        "loss": average_loss,
        "dice": mean_dice,
        "iou": mean_iou,
    }


def _build_dataset(
    config: Dict[str, Any],
    patient_ids,
) -> ProstateZonal2DDataset:
    """Build the current thesis-core zonal dataset."""

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

    return ProstateZonal2DDataset(
        dataset_root=dataset_root,
        patient_ids=patient_ids,
        image_name="t2.nii.gz",
        mask_name=mask_name,
        slice_sampling=slice_sampling,
        preprocessing_config=preprocessing_config,
        cache_data=True,
    )


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


def _load_resume_checkpoint(
    resume_from: str,
    model: Any,
    optimizer: Any,
    device: str,
    num_epochs: int,
) -> Tuple[int, float, int]:
    """Restore model + optimizer state from a checkpoint to resume training.

    Returns (start_epoch, best_metric, checkpoint_epoch), where start_epoch is
    checkpoint_epoch + 1 -- training continues from there, never from epoch 1.

    The checkpoint format is exactly the one written by run_training: a dict
    with "epoch", "model_state_dict", "optimizer_state_dict", "best_metric"
    (and "config"). Backward-compatible with checkpoints carrying just those
    keys (e.g. the Experiment-1 epoch-60 best_model.pt), and the format written
    on resume is unchanged, so resumed checkpoints stay loadable the same way.

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

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    checkpoint_epoch = int(checkpoint["epoch"])
    best_metric = float(checkpoint["best_metric"])
    start_epoch = checkpoint_epoch + 1

    if start_epoch > num_epochs:
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} is already >= configured total "
            f"epochs ({num_epochs}); there is nothing to resume. Increase "
            f"training.num_epochs for a longer run, or start a fresh run instead."
        )

    return start_epoch, best_metric, checkpoint_epoch


def run_training(config_path: str, resume_from: Optional[str] = None) -> Dict[str, Any]:
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

    Returns
    -------
    Dict[str, Any]
        Small run summary: {"resumed", "start_epoch", "num_epochs",
        "best_metric", "best_checkpoint"}. Callers may ignore it.
    """

    config = load_config(config_path)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    validation_cfg = config.get("validation", {})

    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()

    output_dir = ensure_dir(
        config.get(
            "experiment",
            {},
        ).get(
            "output_dir",
            "./results",
        )
    )

    print(f"Loaded config: {config_path}")
    print(f"Target compute device: {device}")
    print(f"Outputs will be saved to: {output_dir}")

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

    train_dataset = _build_dataset(
        config,
        train_patient_ids,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get(
            "batch_size",
            4,
        ),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    val_loader = None

    if val_patient_ids:
        val_dataset = _build_dataset(
            config,
            val_patient_ids,
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
            num_workers=0,
            pin_memory=(device == "cuda"),
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
        start_epoch, best_metric, checkpoint_epoch = _load_resume_checkpoint(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
        )
        print(f"Resumed from checkpoint: {resume_from}")
        print(
            f"  Checkpoint epoch: {checkpoint_epoch} "
            f"| restored best_metric: {best_metric:.6f}"
        )
        print(f"  Continuing from epoch {start_epoch} through {num_epochs}")
    else:
        start_epoch = 1
        best_metric = -float("inf")

    for epoch in range(start_epoch, num_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=loss_type,
        )

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss: {train_loss:.6f}"
        )

        if val_loader is not None:

            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                loss_type=loss_type,
                num_classes=num_classes,
            )

            print(
                f"  val_loss: {val_metrics['loss']:.6f} "
                f"| val_dice: {val_metrics['dice']:.6f} "
                f"| val_iou: {val_metrics['iou']:.6f}"
            )

            current_metric = val_metrics["dice"]

        else:
            # For the current single-patient sanity check,
            # there is no validation set.
            current_metric = -train_loss

        if current_metric > best_metric:

            best_metric = current_metric

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                best_checkpoint,
            )

            print(
                f"  Saved best checkpoint: {best_checkpoint}"
            )

    print("Training completed.")
    print(f"Best checkpoint: {best_checkpoint}")

    return {
        "resumed": resume_from is not None,
        "start_epoch": start_epoch,
        "num_epochs": num_epochs,
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint,
    }


if __name__ == "__main__":
    run_training("configs/config_020.yaml")