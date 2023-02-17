"""Model definition for ZeroDose."""
import torch
import torch.nn as nn

from .networks import DummyGenerator
from .networks import UNet3D


class ZeroDose(nn.Module):
    """ZeroDose model."""

    def __init__(self, model_type: str = "standard") -> None:
        """Initialize the model."""
        self.model_type = model_type
        self.generator: nn.Module
        super().__init__()
        if model_type == "standard":
            self.generator = UNet3D(use_dropout=True)
        elif model_type == "test":
            self.generator = DummyGenerator()

    def forward(self, mrs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.generator(mrs)
