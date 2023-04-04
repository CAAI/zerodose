"""Tests that are slow to run."""
import os
import shutil
from urllib.request import urlopen

import nibabel as nib
import numpy as np
import pytest
from nibabel.processing import resample_from_to

from zerodose import __main__
from zerodose.pipeline import run_full
from zerodose.pipeline import synthesize_baselines


def _download_file(url: str, filename: str) -> None:
    data = urlopen(url).read()  # noqa
    with open(filename, "wb") as f:
        f.write(data)


def _get_mni_dir():
    test_dir = os.path.dirname(__file__)
    mni_dir = os.path.join(test_dir, "mni_test_data")
    return mni_dir


def _augment_mni_img_for_tests(image_in_path, image_out_path):
    rad = np.deg2rad(10)
    cos_gamma = np.cos(rad)
    sin_gamma = np.sin(rad)
    rot_mat = np.array(
        [
            [1, 0, 0, 0],
            [0, cos_gamma, -sin_gamma, 0],
            [0, sin_gamma, cos_gamma, 0],
            [0, 0, 0, 1],
        ]
    )

    translation_mat = np.array(
        [[1, 0, 0, 12], [0, 1, 0, -5], [0, 0, 1, 7], [0, 0, 0, 1]]
    )

    scale_mat = np.array(
        [[1.02, 0, 0, 0], [0, 1.3, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]]
    )

    affine_mat = rot_mat @ translation_mat @ scale_mat
    img_in = nib.load(image_in_path)

    after_rot = resample_from_to(
        img_in, ((190, 200, 170), affine_mat.dot(img_in.affine))
    )

    ornt = np.array([[0, 1], [1, 1], [2, 1]])

    img_orient = after_rot.as_reoriented(ornt)

    nib.save(img_orient, image_out_path)


def _maybe_download_and_extract_mni():
    mni_dir = _get_mni_dir()
    if not os.path.isdir(mni_dir):
        os.mkdir(mni_dir)

    mri_path = os.path.join(
        mni_dir, "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a.nii"
    )
    mask_path = os.path.join(
        mni_dir, "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
    )
    if not os.path.isfile(mri_path) or not os.path.isfile(mask_path):
        url = (
            "https://www.bic.mni.mcgill.ca/~vfonov/icbm/"
            "2009/mni_icbm152_nlin_sym_09a_nifti.zip"
        )
        filename = os.path.join(mni_dir, "mni_icbm152_nlin_sym_09a_nifti.zip")
        _download_file(url, filename)
        shutil.unpack_archive(filename, mni_dir)

    pet_path = os.path.join(mni_dir, "MNI152_PET_1mm.nii")
    if not os.path.exists(pet_path):
        url = "https://static-curis.ku.dk/portal/files/55751238/MNI152_PET_1mm.nii"
        _download_file(url, pet_path)
        _mri = nib.load(mri_path)
        _pet = nib.load(pet_path)
        _pet = resample_from_to(_pet, _mri)
        nib.save(_pet, pet_path)

    agumented_images_dir = os.path.join(_get_mni_dir(), "augmented")
    if not os.path.exists(agumented_images_dir):
        os.makedirs(agumented_images_dir)

    augmented_mri_path = os.path.join(agumented_images_dir, "t1.nii.gz")
    augmented_pet_path = os.path.join(agumented_images_dir, "pet.nii.gz")
    augmented_mask_path = os.path.join(agumented_images_dir, "mask.nii.gz")

    if not os.path.exists(augmented_mri_path):
        _augment_mni_img_for_tests(mri_path, augmented_mri_path)
    if not os.path.exists(augmented_pet_path):
        _augment_mni_img_for_tests(pet_path, augmented_pet_path)
    if not os.path.exists(augmented_mask_path):
        _augment_mni_img_for_tests(mask_path, augmented_mask_path)

    images_dict = {
        "mri_mni": mri_path,
        "pet_mni": pet_path,
        "mask_mni": mask_path,
        "mri_aug": augmented_mri_path,
        "pet_aug": augmented_pet_path,
        "mask_aug": augmented_mask_path,
    }

    return images_dict


def get_test_image_path(image_id):
    """Return the path to test images."""
    img_dict = _maybe_download_and_extract_mni()
    return img_dict[image_id]


@pytest.fixture(scope="session")
def pet_mni_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("pet_mni")


@pytest.fixture(scope="session")
def mask_mni_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("mask_mni")


@pytest.fixture(scope="session")
def mri_aug_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("mri_aug")


@pytest.fixture(scope="session")
def pet_aug_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("pet_aug")


@pytest.fixture(scope="session")
def mask_aug_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("mask_aug")


@pytest.fixture(scope="session")
def mri_mni_file():
    """Returns the path to the MNI PET file."""
    return get_test_image_path("mri_mni")


@pytest.fixture(scope="session")
def sbpet_outputfile():
    """Returns the path to an sbPET output file."""
    fn = os.path.join(_get_mni_dir(), "mni_sbpet.nii.gz")
    return fn


@pytest.fixture(scope="session")
def abn_outputfile():
    """Returns the path to an sbPET output file."""
    fn = os.path.join(_get_mni_dir(), "abn.nii.gz")
    return fn


@pytest.mark.slow
def test_syn_mni(runner, mri_mni_file, mask_mni_file, sbpet_outputfile) -> None:
    """Test the syn command with the standard model and MNI files."""
    result = runner.invoke(
        __main__.main,
        [
            "syn",
            "-i",
            mri_mni_file,
            "-m",
            mask_mni_file,
            "-o",
            sbpet_outputfile,
        ],
    )

    assert result.exit_code == 0


@pytest.mark.slow
@pytest.mark.usefixtures("use_dummy_model")
def test_syn_niftyreg(mri_aug_file, mask_aug_file, sbpet_outputfile) -> None:
    """Test the syn command."""
    mni_shape = (197, 233, 189)
    start_shape = nib.load(mri_aug_file).get_fdata().shape
    assert start_shape != mni_shape

    synthesize_baselines(
        mri_aug_file,
        mask_aug_file,
        sbpet_outputfile,
        device="cuda:0",
    )

    assert nib.load(sbpet_outputfile).get_fdata().shape == start_shape


@pytest.mark.slow
def test_run(
    pet_mni_file, mask_mni_file, mri_mni_file, sbpet_outputfile, abn_outputfile
):
    """Test the run command with the standard model and MNI files."""
    run_full(
        mri_fname=mri_mni_file,
        mask_fname=mask_mni_file,
        out_sbpet=sbpet_outputfile,
        pet_fname=pet_mni_file,
        out_abn=abn_outputfile,
        device="cuda:0",
        do_image=False,
        do_registration=False,
        verbose=True,
    )


@pytest.mark.slow
def test_run_register(
    pet_aug_file, mask_aug_file, mri_aug_file, sbpet_outputfile, abn_outputfile
):
    """Test the run command with the standard model and MNI files."""
    run_full(
        mri_fname=mri_aug_file,
        mask_fname=mask_aug_file,
        out_sbpet=sbpet_outputfile,
        pet_fname=pet_aug_file,
        out_abn=abn_outputfile,
        device="cuda:0",
        do_image=False,
        verbose=True,
    )


@pytest.mark.slow
def test_run_no_pet(mask_aug_file, mri_aug_file, sbpet_outputfile):
    """Test the run command with the standard model and MNI files."""
    run_full(
        mri_fname=mri_aug_file,
        mask_fname=mask_aug_file,
        out_sbpet=sbpet_outputfile,
        device="cuda:0",
        verbose=True,
    )
