"""Model definition for ZeroDose."""
import torch
import torch.nn as nn

from zerodose import utils

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
        elif model_type == "dummy":
            self.generator = DummyGenerator()
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

    def forward(self, mrs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.generator(mrs)


class AbnormalityMap(nn.Module):
    """Abnormality map generator."""

    def __init__(self, sigma_smooth=3) -> None:
        """Initialize the Abnormality map."""
        super().__init__()
        self.smooth = utils.GaussianSmoothing(
            channels=1, kernel_size=5 * sigma_smooth, sigma=sigma_smooth, dim=3
        )

    def forward(
        self, pet: torch.Tensor, sbpet: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate the abnormality map."""
        pet_blurred = self.smooth(pet)
        sbpet_blurred = self.smooth(sbpet)
        abnormality_map = (pet_blurred - sbpet_blurred) / (sbpet_blurred + 1e-7) * 100
        abnormality_map[torch.isnan(abnormality_map)] = 0
        return abnormality_map
