"""Dataset class for the ZeroDose project."""
from typing import Any
from typing import Dict

import torchio as tio

from zerodose.processing import Binarize
from zerodose.processing import PadAndCropMNI
from zerodose.processing import ToFloat32


class SubjectDataset(tio.data.SubjectsDataset):
    """Dataset class for the ZeroDose project."""

    def __init__(self, mri_fnames, mask_fnames, pet_fnames=None):
        """Initialize the dataset."""
        transforms = self._get_augmentation_transform_val()

        subjects = [
            self._make_subject_predict(mr_f, ma_f)
            for mr_f, ma_f in zip(  # noqa
                mri_fnames,
                mask_fnames,
            )
        ]

        if pet_fnames is not None:
            for sub, pet_fname in zip(subjects, pet_fnames):  # noqa
                sub.add_image(tio.ScalarImage(pet_fname), "pet")

        super().__init__(subjects, transforms)

    def _make_subject_predict(self, mr_path, mask_path) -> tio.Subject:
        subject_dict: Dict[Any, Any] = {}
        subject_dict["mr"] = tio.ScalarImage(mr_path)
        subject_dict["mask"] = tio.LabelMap(mask_path)
        return tio.Subject(subject_dict)

    def _get_augmentation_transform_val(self) -> tio.Compose:
        augmentations = [
            Binarize(include=["mask"]),
            tio.transforms.ZNormalization(include=["mr"], masking_method="mask"),
            PadAndCropMNI(),
            ToFloat32(include=["mr"]),
        ]

        return tio.Compose(augmentations)
