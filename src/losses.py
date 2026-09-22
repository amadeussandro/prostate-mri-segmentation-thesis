"""Configurable segmentation losses for the RQ-1 controlled experiments.

Baseline preservation contract
------------------------------
This module is ADDITIVE. The default is `cross_entropy`, and in that mode the
returned loss is EXACTLY `nn.CrossEntropyLoss()(logits, targets.long())` -- the
same object the original `src/train.py::compute_loss` constructed, computed the
same way, with the soft-Dice term never even evaluated. So a config with no
`loss:` block (every pre-existing config, including
`configs/config_baseline.yaml`) trains bit-for-bit as before.

Experiment E1 (`loss.name: dice_ce`) adds an overlap-based term:

    L = ce_weight * CE(logits, target)  +  dice_weight * SoftDice(logits, target)

    SoftDice = 1 - mean_{c in dice_classes} (2*sum(p_c * g_c) + eps)
                                            / (sum(p_c) + sum(g_c) + eps)

where `p_c` is the softmax probability of class c and `g_c` the one-hot ground
truth. `dice_classes` defaults to the two FOREGROUND anatomical zones
(1 = CG, 2 = PZ); background (class 0) is deliberately EXCLUDED so that the
94.2%-of-voxels background cannot dominate -- and mask -- the 1.5%-of-voxels PZ
term. See the RQ-1 audit for the class-frequency census that motivates this.

Configuration sources
---------------------
Two spellings are accepted, and a conflict between them is a hard error rather
than a silent precedence rule:

  * preferred (E1 and later):   top-level  `loss: {name, ce_weight, ...}`
  * legacy  (baseline/Exp02):   `training.loss_function: "cross_entropy"`

If both are present and name a DIFFERENT loss, `LossConfig.from_config` raises,
so an experiment can never be run against two disagreeing sources of truth.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

# Accepted spellings, mapped to canonical names.
_CROSS_ENTROPY_ALIASES = frozenset({"cross_entropy", "crossentropy", "cross-entropy", "ce"})
_DICE_CE_ALIASES = frozenset({"dice_ce", "dicece", "dice-ce", "dice+ce", "ce_dice", "dice_cross_entropy"})

CROSS_ENTROPY = "cross_entropy"
DICE_CE = "dice_ce"

SUPPORTED_LOSSES: Tuple[str, ...] = (CROSS_ENTROPY, DICE_CE)


def canonical_loss_name(name: str) -> str:
    """Map any accepted spelling to its canonical name.

    Raises
    ------
    ValueError
        If `name` is not a supported loss.
    """
    key = str(name).strip().lower()
    if key in _CROSS_ENTROPY_ALIASES:
        return CROSS_ENTROPY
    if key in _DICE_CE_ALIASES:
        return DICE_CE
    raise ValueError(
        f"Unsupported loss_type: {name}. "
        f"Currently supported: {', '.join(SUPPORTED_LOSSES)}"
    )


@dataclass(frozen=True)
class LossConfig:
    """Fully-resolved loss specification for one experiment arm.

    Attributes
    ----------
    name : str
        Canonical loss name ("cross_entropy" or "dice_ce").
    ce_weight, dice_weight : float
        Term weights for the compound loss. IGNORED for "cross_entropy", which
        is always the bare CE loss so the baseline stays exactly reproducible.
    dice_classes : Tuple[int, ...]
        Class indices included in the soft-Dice term (E1: (1, 2) = CG, PZ).
    epsilon : float
        Smoothing constant, added to BOTH numerator and denominator.
    """

    name: str = CROSS_ENTROPY
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    dice_classes: Tuple[int, ...] = (1, 2)
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_loss_name(self.name))
        object.__setattr__(self, "dice_classes", tuple(int(c) for c in self.dice_classes))
        object.__setattr__(self, "ce_weight", float(self.ce_weight))
        object.__setattr__(self, "dice_weight", float(self.dice_weight))
        object.__setattr__(self, "epsilon", float(self.epsilon))

        if self.name == DICE_CE:
            if not self.dice_classes:
                raise ValueError("loss.dice_classes must list at least one class id for dice_ce.")
            if len(set(self.dice_classes)) != len(self.dice_classes):
                raise ValueError(f"loss.dice_classes contains duplicates: {self.dice_classes}")
            if any(c < 0 for c in self.dice_classes):
                raise ValueError(f"loss.dice_classes must be non-negative: {self.dice_classes}")
            if self.ce_weight < 0 or self.dice_weight < 0:
                raise ValueError("loss.ce_weight and loss.dice_weight must be non-negative.")
            if self.ce_weight == 0 and self.dice_weight == 0:
                raise ValueError("loss.ce_weight and loss.dice_weight cannot both be zero.")
        if self.epsilon <= 0:
            raise ValueError(f"loss.epsilon must be > 0, got {self.epsilon}.")

    @property
    def is_baseline_cross_entropy(self) -> bool:
        return self.name == CROSS_ENTROPY

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form, used in the config fingerprint and checkpoints."""
        if self.is_baseline_cross_entropy:
            return {"name": self.name}
        return {
            "name": self.name,
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "dice_classes": list(self.dice_classes),
            "epsilon": self.epsilon,
        }

    def describe(self) -> str:
        """One-line human description for logs / the experiment fingerprint."""
        if self.is_baseline_cross_entropy:
            return "Cross Entropy (baseline)"
        classes = ", ".join(str(c) for c in self.dice_classes)
        return (
            f"{self.ce_weight:g} x CE + {self.dice_weight:g} x SoftDice"
            f" (dice_classes=[{classes}], eps={self.epsilon:g})"
        )

    @staticmethod
    def from_config(config: Optional[Mapping[str, Any]]) -> "LossConfig":
        """Resolve the loss spec from a full experiment config dict.

        Precedence: the top-level `loss:` block wins, but if
        `training.loss_function` also names a DIFFERENT loss this raises --
        two disagreeing sources of truth are never silently resolved.
        """
        config = config or {}
        loss_block = config.get("loss") or {}
        if not isinstance(loss_block, Mapping):
            raise ValueError(f"`loss:` must be a mapping, got {type(loss_block).__name__}.")

        legacy_name = (config.get("training") or {}).get("loss_function")

        if loss_block:
            name = loss_block.get("name", legacy_name or CROSS_ENTROPY)
            if legacy_name is not None:
                if canonical_loss_name(legacy_name) != canonical_loss_name(name):
                    raise ValueError(
                        "Conflicting loss configuration: top-level `loss.name` = "
                        f"'{name}' but `training.loss_function` = '{legacy_name}'. "
                        "Remove one of them so the experiment has a single source of truth."
                    )
            defaults = LossConfig()
            return LossConfig(
                name=name,
                ce_weight=loss_block.get("ce_weight", defaults.ce_weight),
                dice_weight=loss_block.get("dice_weight", defaults.dice_weight),
                dice_classes=loss_block.get("dice_classes", defaults.dice_classes),
                epsilon=loss_block.get("epsilon", defaults.epsilon),
            )

        return LossConfig(name=legacy_name or CROSS_ENTROPY)


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_ids: Sequence[int] = (1, 2),
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Multi-class soft Dice loss over the selected classes only.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output, shape (B, C, H, W).
    targets : torch.Tensor
        Integer class labels, shape (B, H, W).
    class_ids : Sequence[int]
        Classes included in the mean (E1: foreground 1 and 2; background excluded).
    epsilon : float
        Smoothing constant added to numerator AND denominator, so a class that
        is absent from both prediction and ground truth yields Dice = 1 (loss
        contribution 0) instead of 0/0.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]: 1 - mean per-class soft Dice.

    Notes
    -----
    Sums run over the whole batch and both spatial axes (a "batch-soft-Dice"),
    which is the standard formulation and is markedly more stable than a
    per-image mean when a class is missing from individual slices -- relevant
    here because PZ is absent from 53.9% of training slices.

    `logits` is upcast to float32 before softmax so the term stays numerically
    stable; under FP32 (the only mode these experiments use) this is a no-op.
    """
    if logits.dim() != 4:
        raise ValueError(f"soft_dice_loss expects logits of shape (B, C, H, W), got {tuple(logits.shape)}.")

    probs = torch.softmax(logits.float(), dim=1)
    targets = targets.long()
    num_classes = probs.shape[1]

    dice_per_class = []
    for class_id in class_ids:
        if not 0 <= class_id < num_classes:
            raise ValueError(
                f"dice class id {class_id} is out of range for a {num_classes}-class model."
            )
        prob_c = probs[:, class_id]
        target_c = (targets == class_id).to(probs.dtype)

        intersection = torch.sum(prob_c * target_c)
        denominator = torch.sum(prob_c) + torch.sum(target_c)
        dice_per_class.append((2.0 * intersection + epsilon) / (denominator + epsilon))

    mean_dice = torch.stack(dice_per_class).mean()
    return 1.0 - mean_dice


def compute_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_config: Optional[LossConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the configured loss and its per-term breakdown.

    Returns
    -------
    (total, components)
        total : torch.Tensor
            Scalar differentiable loss to call `.backward()` on.
        components : Dict[str, float]
            Detached {"ce", "dice", "total"} values for per-epoch logging. For
            the baseline cross-entropy path "dice" is 0.0 and the soft-Dice
            term is never computed.
    """
    cfg = loss_config or LossConfig()
    targets_long = targets.long()

    ce_tensor = nn.CrossEntropyLoss()(predictions, targets_long)

    if cfg.is_baseline_cross_entropy:
        # EXACT original baseline path: bare CE, no weighting, no Dice term.
        ce_value = float(ce_tensor.detach())
        return ce_tensor, {"ce": ce_value, "dice": 0.0, "total": ce_value}

    dice_tensor = soft_dice_loss(
        predictions, targets_long, class_ids=cfg.dice_classes, epsilon=cfg.epsilon
    )
    total = cfg.ce_weight * ce_tensor + cfg.dice_weight * dice_tensor

    return total, {
        "ce": float(ce_tensor.detach()),
        "dice": float(dice_tensor.detach()),
        "total": float(total.detach()),
    }
