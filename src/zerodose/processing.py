"""Processing functions for the ZeroDose model."""

import torch
import torchio as tio
from torchio import SpatialTransform
from torchio.data.subject import Subject

from zerodose import processing


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
