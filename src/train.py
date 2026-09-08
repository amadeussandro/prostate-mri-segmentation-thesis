"""Training loop for 2D prostate zonal anatomy segmentation."""

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ProstateZonal2DDataset
from src.evaluate import compute_dice, compute_iou
from src.model import build_model
from src.utils import ensure_dir, get_device, load_config, set_seed


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

    return ProstateZonal2DDataset(
        dataset_root=dataset_root,
        patient_ids=patient_ids,
        image_name="t2.nii.gz",
        mask_name="t2_anatomy_reader1.nii.gz",
        slice_sampling=slice_sampling,
        cache_data=True,
    )


def run_training(config_path: str) -> None:
    """Run a training experiment from a YAML configuration."""

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

    train_patient_ids = data_cfg.get(
        "patient_ids",
        [],
    )

    val_patient_ids = validation_cfg.get(
        "val_patient_ids",
        [],
    )

    if set(train_patient_ids) & set(val_patient_ids):
        raise ValueError(
            "Training and validation patient IDs overlap."
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get(
            "batch_size",
            4,
        ),
        shuffle=True,
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

    best_metric = -float("inf")
    best_checkpoint = os.path.join(
        output_dir,
        "best_model.pt",
    )

    for epoch in range(1, num_epochs + 1):

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


if __name__ == "__main__":
    run_training("configs/config_020.yaml")