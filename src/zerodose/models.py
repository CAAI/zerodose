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

    @staticmethod
    def get_default():
        """Get the default model."""
        return utils.get_model()


class AbnormalityMap(nn.Module):
    """Abnormality map generator."""

    def __init__(self, sigma_smooth=3) -> None:
        """Initialize the Abnormality map."""
        super().__init__()
        self.smooth = utils.GaussianSmoothing(
            channels=1, kernel_size=5 * sigma_smooth, sigma=sigma_smooth, dim=3
        )

    @staticmethod
    def get_default():
        """Get the default model."""
        return AbnormalityMap(sigma_smooth=3)

    def forward(
        self, pet: torch.Tensor, sbpet: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate the abnormality map."""
        pet_blurred = self.smooth(pet)
        sbpet_blurred = self.smooth(sbpet)
        abnormality_map = (pet_blurred - sbpet_blurred) / (sbpet_blurred + 1e-7) * 100
        abnormality_map[torch.isnan(abnormality_map)] = 0
        abnormality_map *= mask
        return abnormality_map


class QuantileNormalization(nn.Module):
    """Quantile normalization of the sbPET image."""

    def __init__(
        self,
        quantile,
        sigma_normalization=3,
    ) -> None:
        """Initialize the normalization module."""
        super().__init__()
        self.quantile = quantile
        self.smooth_normalization = utils.GaussianSmoothing(
            channels=1,
            kernel_size=5 * sigma_normalization,
            sigma=sigma_normalization,
            dim=3,
        )

    def forward(
        self, pet: torch.Tensor, sbpet: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize the sbPET image."""
        return self._scale_sbpet(pet, sbpet, mask)

    def _get_normalization_mask(self, pet, mask):
        qt = torch.quantile(pet[mask], self.quantile)
        norm_mask = mask & (pet > qt)
        return norm_mask

    def _scale_sbpet(self, pet, sbpet, mask):
        for _i in range(2):
            pet_blurred = self.smooth_normalization(pet)
            norm_mask = self._get_normalization_mask(pet_blurred, mask)
            norm_const = torch.mean(pet[norm_mask]) / torch.mean(sbpet[norm_mask])
            sbpet *= norm_const
            sbpet[~mask] = pet[~mask]

        return sbpet

    @staticmethod
    def get_default():
        """Get the default model."""
        return QuantileNormalization(quantile=0.97, sigma_normalization=3)
