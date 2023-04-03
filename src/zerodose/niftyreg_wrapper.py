"""Niftyreg wrapper for torchio."""
import os
import tempfile

import nibabel as nib
import niftyreg  # type: ignore
import numpy as np
import torch
from torchio import Subject
from torchio.data import io
from torchio.transforms import SpatialTransform


def _save_matrix_nifty(affine_mat, file_name):
    s = ""
    for i in range(affine_mat.shape[0]):
        for j in range(affine_mat.shape[1]):
            affine_mat[i, j] = affine_mat[i, j].item()
            s += str(affine_mat[i, j]) + " "
        s = s[:-1]
        s += "\n"
    with open(file_name, "w") as f:
        f.write(s)


def _read_matrix_nifty(file_name):
    with open(file_name) as f:
        s = f.read()
    s = s.split("\n")
    s = s[:-1]
    s = [i.split(" ") for i in s]
    s = [[float(j) for j in i] for i in s]
    s = np.array(s)
    return s


def _register_mri_to_mni(mri_fname, ref):
    temp_aff = tempfile.NamedTemporaryFile(delete=False)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    out_mri_fname = temp_out.name
    mni_template = ref
    niftyreg.main(
        [
            "aladin",
            "-flo",
            mri_fname,
            "-ref",
            mni_template,
            "-res",
            out_mri_fname,
            "-aff",
            temp_aff.name,
            "-speeeeed",
        ]
    )
    affine_mat = _read_matrix_nifty(temp_aff.name)
    temp_aff.close()
    temp_out.close()
    os.remove(temp_aff.name)
    os.remove(temp_out.name)
    return affine_mat


def _nifty_reg_resample(ref_path, flo_img, affine_mat):
    temp_flo = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    temp_res = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
    temp_aff = tempfile.NamedTemporaryFile(delete=False)

    io.write_image(flo_img.tensor, flo_img.affine, temp_flo.name)
    _save_matrix_nifty(affine_mat, temp_aff.name)

    niftyreg.main(
        [
            "resample",
            "-ref",
            ref_path,
            "-flo",
            temp_flo.name,
            "-res",
            temp_res.name,
            "-aff",
            temp_aff.name,
        ]
    )

    # After the other process has finished writing to the files, read from them
    res = nib.load(temp_res.name)
    data = res.get_fdata()
    affine = res.affine

    temp_flo.close()
    temp_res.close()
    temp_aff.close()

    os.remove(temp_flo.name)
    os.remove(temp_res.name)
    os.remove(temp_aff.name)

    return data, affine


class NiftyRegistration(SpatialTransform):
    """Nifty Registration for torchio."""

    def __init__(
        self,
        floating_image=None,
        ref=None,
        **kwargs,
    ):
        """Initialize the niftyreg registration transform."""
        self.floating_image = floating_image
        super().__init__(**kwargs)
        self.ref = ref

    def apply_transform(self, subject: Subject) -> Subject:
        """Apply the registration and coregistration."""
        temp_flo = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")

        io.write_image(
            subject[self.floating_image].tensor,
            subject[self.floating_image].affine,
            temp_flo.name,
        )

        affine = _register_mri_to_mni(temp_flo.name, self.ref)

        temp_flo.close()
        os.remove(temp_flo.name)
        inverse_ref = subject[self.floating_image].path
        transformer = NiftyResample(
            ref=self.ref, affine=affine, inverse_ref=inverse_ref
        )
        transformed = transformer(subject)
        return transformed  # type: ignore


class NiftyResample(SpatialTransform):
    """Nifty resample (coreregistration)."""

    def __init__(self, affine, ref=None, is_inverse=False, inverse_ref=None, **kwargs):
        """Initialize the nifty resample."""
        self.affine = affine
        self.ref = ref
        self.is_inverse = is_inverse
        self.inverse_ref = inverse_ref
        super().__init__(**kwargs)
        self.args_names = ("affine", "ref", "is_inverse", "inverse_ref")

    def apply_transform(self, subject: Subject) -> Subject:
        """Apply the transform."""
        for image in self.get_images(subject):
            if self.is_inverse:
                if image.path is not None:
                    ref = image.path
                else:
                    ref = self.inverse_ref
                _apply_niftyreg_resample(image, ref, np.linalg.inv(self.affine))
            else:
                _apply_niftyreg_resample(image, self.ref, self.affine)
        return subject

    @staticmethod
    def is_invertible():
        """Whether the transform is invertible."""
        return True

    def inverse(self):
        """Return the inverse resample."""
        return NiftyResample(
            affine=self.affine, is_inverse=True, inverse_ref=self.inverse_ref
        )


def _apply_niftyreg_resample(image, ref_path, affine_mat):
    data, affine = _nifty_reg_resample(ref_path, image, affine_mat)
    image.affine = affine
    data = data.copy()
    data = torch.as_tensor(data)
    image.set_data(data.unsqueeze(0))
