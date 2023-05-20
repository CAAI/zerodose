"""Niftyreg wrapper for torchio."""
import os
import tempfile

import niftyreg  # type: ignore
import numpy as np

from zerodose.utils import get_mni_template


def _rigid_pet_to_ref(pet_path, ref_path):
    with tempfile.TemporaryDirectory() as tempdir:
        out_rigid = os.path.join(tempdir, "rigid.txt")
        out_temp = os.path.join(tempdir, "out.nii.gz")
        niftyreg.main(
            [
                "aladin",
                "-flo",
                pet_path,
                "-ref",
                ref_path,
                "-res",
                out_temp,
                "-aff",
                out_rigid,
                "-voff",
                "-rigOnly",
            ],
        )
        mat_rig = _read_matrix_nifty(out_rigid)
        return mat_rig


def from_mni(
    ref_fname,
    in_affine_fname,
    float_fname,
    out_fname,
):
    """Resample a floating image to a reference image using an affine matrix."""
    with tempfile.NamedTemporaryFile() as temp_inv_affine:
        affine_mat = _read_matrix_nifty(in_affine_fname)
        inv_affine_mat = np.linalg.inv(affine_mat)
        inv_affine = temp_inv_affine.name
        _save_matrix_nifty(inv_affine_mat, inv_affine)

        niftyreg.main(
            [
                "resample",
                "-ref",
                ref_fname,
                "-flo",
                float_fname,
                "-res",
                out_fname,
                "-aff",
                inv_affine,
                "-voff",
            ]
        )


def to_mni(
    in_mri_fname,
    in_mask_fname,
    in_pet_fname,
    out_mri_fname,
    out_mask_fname,
    out_pet_fname,
    out_affine_fname,
    reg_pet_to_mri=True,
):
    """Resample a floating image to a reference image using an affine matrix."""
    ref = get_mni_template("mask")

    niftyreg.main(
        [
            "aladin",
            "-flo",
            in_mask_fname,
            "-ref",
            ref,
            "-res",
            out_mask_fname,
            "-aff",
            out_affine_fname,
            "-speeeeed",
            "-voff",
        ],
    )

    in_affine = out_affine_fname

    niftyreg.main(
        [
            "resample",
            "-ref",
            ref,
            "-flo",
            in_mask_fname,
            "-res",
            out_mask_fname,
            "-aff",
            in_affine,
            "-inter",
            "0",
        ]
    )

    niftyreg.main(
        [
            "resample",
            "-ref",
            ref,
            "-flo",
            in_mri_fname,
            "-res",
            out_mri_fname,
            "-aff",
            in_affine,
            "-voff",
        ]
    )

    if reg_pet_to_mri and in_pet_fname is not None:
        mat_rig = _rigid_pet_to_ref(in_pet_fname, in_mri_fname)
        mat_aff = _read_matrix_nifty(in_affine)
        mat_aff = mat_rig @ mat_aff
        _save_matrix_nifty(mat_aff, in_affine)

    if in_pet_fname is not None:
        niftyreg.main(
            [
                "resample",
                "-ref",
                ref,
                "-flo",
                in_pet_fname,
                "-res",
                out_pet_fname,
                "-aff",
                in_affine,
            ]
        )


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
