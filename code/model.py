"""Model architecture definitions and factory for 2D Prostate MRI Segmentation.

This module provides scaffolding for 2D segmentation architectures (e.g. 2D U-Net,
Attention U-Net) designed for multi-channel input (T2, ADC, DWI) and binary tumor masks.
"""

from typing import Any, Dict, Optional

# Safe PyTorch import: allow module to import cleanly without PyTorch installed in environment
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    BaseModule = nn.Module
except ImportError:
    BaseModule = object


class ProstateUNet2D(BaseModule):
    """2D U-Net architecture for prostate lesion segmentation.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g. 3 for stacked T2 + ADC + DWI).
    out_channels : int, default=1
        Number of output prediction channels (1 for binary lesion segmentation).
    init_features : int, default=32
        Base number of feature channels in the first encoder layer.
    dropout_rate : float, default=0.2
        Dropout probability in bottleneck layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        dropout_rate: float = 0.2,
    ) -> None:
        if BaseModule is not object:
            super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.dropout_rate = dropout_rate

        # TODO: Define encoder blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
        # TODO: Define bottleneck block with dropout
        # TODO: Define decoder blocks with upsampling (ConvTranspose2d / Bilinear + Conv2d)
        # TODO: Implement skip connection concatenation
        # TODO: Define final 1x1 classification head (out_channels)

    def forward(self, x: Any) -> Any:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C=3, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (B, 1, H, W).
        """
        # Scaffolding placeholder: not implemented yet
        raise NotImplementedError(
            "ProstateUNet2D forward pass is a scaffolding placeholder. "
            "Model architecture will be implemented in the training pipeline phase."
        )


def build_model(config: Dict[str, Any]) -> Any:
    """Factory function to instantiate a segmentation model based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    Any
        Instantiated model instance.
    """
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    out_channels = model_cfg.get("out_channels", 1)
    init_features = model_cfg.get("init_features", 32)
    dropout_rate = model_cfg.get("dropout_rate", 0.2)

    return ProstateUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        dropout_rate=dropout_rate,
    )

