"""2D U-Net Model Architecture for Prostate MRI Tumor Segmentation.

This module implements a standard 2D U-Net architecture (ProstateUNet2D) tailored
for multi-parametric / bi-parametric Prostate MRI lesion segmentation.

Key Architectural Characteristics:
- Multi-channel input (T2 + ADC + DWI -> 3 channels)
- Binary segmentation output (1 channel, raw logits without sigmoid)
- Convolution block: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
- Downsampling: MaxPool2d(2, 2)
- Upsampling: ConvTranspose2d(2, 2)
- Robust spatial padding on skip-connection concatenation to support odd dimensions
  (e.g. Prostate158 axial slice dimensions: 270 x 270)
- Bottleneck with Dropout2d for regularization
"""

from typing import Any, Dict

# Safe PyTorch import: allow module to import cleanly in lightweight environments
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    BaseModule = nn.Module
except ImportError:
    BaseModule = object  # type: ignore


class ConvBlock(BaseModule):
    """Double convolution block: [Conv2d -> BatchNorm2d -> ReLU] x 2.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        if BaseModule is not object:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = None

    def forward(self, x: Any) -> Any:
        """Forward pass through double convolution block."""
        return self.block(x)


class ProstateUNet2D(BaseModule):
    """Standard 2D U-Net architecture for prostate lesion segmentation.

    Accepts multi-modal 2D axial slices [B, 3, H, W] and predicts binary lesion
    logits [B, 1, H, W].

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g. 3 for stacked T2 + ADC + DWI).
    out_channels : int, default=1
        Number of output channels (1 for binary tumor segmentation logits).
    init_features : int, default=32
        Base number of feature channels in the first encoder block.
    dropout_rate : float, default=0.2
        Dropout probability applied in the bottleneck block.
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

        if BaseModule is object:
            return

        f = init_features  # 32

        # ---------------------------------------------------------------------
        # Encoder (Downsampling Path)
        # ---------------------------------------------------------------------
        # Stage 1: [B, in_channels, H, W] -> [B, f, H, W]
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: [B, f, H/2, W/2] -> [B, 2f, H/2, W/2]
        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: [B, 2f, H/4, W/4] -> [B, 4f, H/4, W/4]
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: [B, 4f, H/8, W/8] -> [B, 8f, H/8, W/8]
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        # [B, 8f, H/16, W/16] -> [B, 16f, H/16, W/16]
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # ---------------------------------------------------------------------
        # Decoder (Upsampling Path + Skip Connections)
        # ---------------------------------------------------------------------
        # Stage 4 Up: 16f -> 8f
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        # Stage 3 Up: 8f -> 4f
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        # Stage 2 Up: 4f -> 2f
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        # Stage 1 Up: 2f -> f
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # ---------------------------------------------------------------------
        # Final Output Head (1x1 Convolution)
        # ---------------------------------------------------------------------
        # [B, f, H, W] -> [B, out_channels, H, W]
        # Raw logits output (no sigmoid applied here)
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    @staticmethod
    def _pad_to_match(x: Any, target: Any) -> Any:
        """Pad tensor x to match the spatial dimensions (H, W) of target tensor.

        This ensures robust skip-connection concatenation for non-power-of-2
        or odd spatial sizes such as Prostate158's 270x270 slices.
        """
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        return x

    def forward(self, x: Any) -> Any:
        """Forward pass through the 2D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W), e.g. [B, 3, 270, 270].

        Returns
        -------
        torch.Tensor
            Predicted raw logits tensor of shape (B, out_channels, H, W),
            e.g. [B, 1, 270, 270].
        """
        if BaseModule is object:
            raise NotImplementedError(
                "PyTorch is not available. Please install PyTorch to run ProstateUNet2D."
            )

        # Encoder path
        s1 = self.enc1(x)                # [B, 32, H, W]
        p1 = self.pool1(s1)              # [B, 32, H/2, W/2]

        s2 = self.enc2(p1)               # [B, 64, H/2, W/2]
        p2 = self.pool2(s2)              # [B, 64, H/4, W/4]

        s3 = self.enc3(p2)               # [B, 128, H/4, W/4]
        p3 = self.pool3(s3)              # [B, 128, H/8, W/8]

        s4 = self.enc4(p3)               # [B, 256, H/8, W/8]
        p4 = self.pool4(s4)              # [B, 256, H/16, W/16]

        # Bottleneck with dropout
        b = self.bottleneck(p4)          # [B, 512, H/16, W/16]
        b = self.dropout(b)

        # Decoder path with skip connections and spatial alignment
        d4 = self.up4(b)                 # [B, 256, ~H/8, ~W/8]
        d4 = self._pad_to_match(d4, s4)
        d4 = torch.cat([d4, s4], dim=1)  # [B, 512, H/8, W/8]
        d4 = self.dec4(d4)               # [B, 256, H/8, W/8]

        d3 = self.up3(d4)                # [B, 128, ~H/4, ~W/4]
        d3 = self._pad_to_match(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # [B, 256, H/4, W/4]
        d3 = self.dec3(d3)               # [B, 128, H/4, W/4]

        d2 = self.up2(d3)                # [B, 64, ~H/2, ~W/2]
        d2 = self._pad_to_match(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # [B, 128, H/2, W/2]
        d2 = self.dec2(d2)               # [B, 64, H/2, W/2]

        d1 = self.up1(d2)                # [B, 32, ~H, ~W]
        d1 = self._pad_to_match(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # [B, 64, H, W]
        d1 = self.dec1(d1)               # [B, 32, H, W]

        # Final 1x1 convolution producing raw logits
        logits = self.final_conv(d1)     # [B, 1, H, W]

        # Defensive assertion: final output (H, W) must exactly match input (H, W)
        if logits.shape[2:] != x.shape[2:]:
            logits = self._pad_to_match(logits, x)

        return logits


def build_model(config: Dict[str, Any]) -> Any:
    """Factory function to instantiate a segmentation model based on configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    ProstateUNet2D
        Instantiated ProstateUNet2D model instance.
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
