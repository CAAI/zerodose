"""Processing functions for the ZeroDose model."""

import torch
import torchio as tio
from torchio import SpatialTransform
from torchio.data.subject import Subject
import torch.nn as nn

from zerodose import processing
from zerodose import utils

_X_MIN_MNI = 0
_X_MAX_MNI = 192
_Y_MIN_MNI = 21
_Y_MAX_MNI = 213
_Z_MIN_MNI = -7
_Z_MAX_MNI = 185


def _crop_mni_to_192(arr: torch.Tensor) -> torch.Tensor:
    """Crop the MNI image to 192x192x192."""
    parr = torch.zeros((1, _X_MAX_MNI - _X_MIN_MNI, _Y_MAX_MNI - _Y_MIN_MNI, 192))
    parr[:, :, :, abs(_Z_MIN_MNI) : 192] = arr[
        :, _X_MIN_MNI:_X_MAX_MNI, _Y_MIN_MNI:_Y_MAX_MNI, :_Z_MAX_MNI
    ]

    return parr


def _crop_192_to_mni(arr: torch.Tensor) -> torch.Tensor:
    """Crop the 192x192x192 image to MNI."""
    parr = torch.zeros((1, 197, 233, 189))
    parr[:, _X_MIN_MNI:_X_MAX_MNI, _Y_MIN_MNI:_Y_MAX_MNI, :_Z_MAX_MNI] = arr[
        :, :, :, abs(_Z_MIN_MNI) : 192
    ]

    return parr


def postprocess(arr: torch.Tensor) -> torch.Tensor:
    """Postprocess the 192x192x192 image to MNI."""
    return _crop_192_to_mni(arr)


class Pad(SpatialTransform):
    """Pad the MNI image to 192x192x192."""

    def __init__(self, **kwargs) -> None:
        """Initialize the transform."""
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        """Apply the transform to the subject."""
        for image in self.get_images(subject):
            _pad(image)

        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the transform is invertible."""
        return False


def _pad(image: tio.Image) -> None:
    data = processing._crop_mni_to_192(image.data)
    image.set_data(data)


class ToFloat32(SpatialTransform):
    """Convert the image to float32."""

    def __init__(self, **kwargs) -> None:
        """Initialize the transform."""
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        """Apply the transform to the subject."""
        for image in self.get_images(subject):
            _to_float(image)

        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Return whether the transform is invertible."""
        return False


def _to_float(image: tio.Image) -> None:
    _data = image.numpy().astype("f")
    data = torch.as_tensor(_data)
    image.set_data(data)


class QuantileNormalization(nn.Module):
    def __init__(self,
                 quantile, 
                 sigma_normalization=3, 
                 ) -> None:
        super().__init__()
        self.quantile = quantile
        self.smooth_normalization = utils.GaussianSmoothing(channels=1,kernel_size=5*sigma_normalization,sigma=sigma_normalization,dim=3)
    
    def forward(self, pet: torch.Tensor, sbpet: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._scale_sbpet(pet,sbpet,mask)
        
    def _get_normalization_mask(self, pet, mask):
        qt = torch.quantile(pet[mask],self.quantile)
        norm_mask = mask & (pet>qt)
        return norm_mask
    
    def _scale_sbpet(self, pet,sbpet,mask):
        for i in range(2):
            pet_blurred = self.smooth_normalization(pet)
            norm_mask = self._get_normalization_mask(pet_blurred, mask)
            norm_const = torch.mean(pet[norm_mask])/torch.mean(sbpet[norm_mask])
            sbpet *= norm_const
            sbpet[~mask] = pet[~mask]

        return sbpet