import sys
import pytest


import os
import pytest
import nibabel as nib
from nibabel.processing import resample_from_to
from urllib.request import urlopen
from zerodose import __main__
import shutil

def _download_file(url: str, filename: str) -> None:
    data = urlopen(url).read()  # noqa
    with open(filename, "wb") as f:
        f.write(data)

def _get_mni_dir():
    test_dir = os.path.dirname(__file__)
    mni_dir = os.path.join(test_dir, "mni_test_data")
    return mni_dir

def _maybe_download_and_extract_mni():

    mni_dir = _get_mni_dir()
    if not os.path.isdir(mni_dir):
        os.mkdir(mni_dir)

    mri_path = os.path.join(mni_dir, "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a.nii")
    mask_path = os.path.join(mni_dir, "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii")
    if not os.path.isfile(mri_path) or not os.path.isfile(mask_path):
        url = "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip"
        filename = os.path.join(mni_dir, "mni_icbm152_nlin_sym_09a_nifti.zip")
        _download_file(url, filename)
        shutil.unpack_archive(filename, mni_dir)

    pet_path = os.path.join(mni_dir, "MNI152_PET_1mm.nii")
    if not os.path.exists(pet_path):
        url = "https://static-curis.ku.dk/portal/files/55751238/MNI152_PET_1mm.nii"
        _download_file(url, pet_path)
        _mri = nib.load(mri_path)
        _pet = nib.load(pet_path)
        _pet = resample_from_to(_pet,_mri)
        nib.save(_pet,pet_path)

    return mri_path, mask_path, pet_path

@pytest.fixture(scope="session")
def pet_mni_file():
    [_,_,pet_fname] = _maybe_download_and_extract_mni()
    return pet_fname

@pytest.fixture(scope="session")
def mask_mni_file():
    [_,mask_fname,_] = _maybe_download_and_extract_mni()
    return mask_fname

@pytest.fixture(scope="session")
def mri_mni_file():
    [mri_fname,_,_] = _maybe_download_and_extract_mni()
    return mri_fname

@pytest.fixture(scope="session")
def sbpet_outputfile():
    fn = os.path.join("_get_mni_dir", "mni_sbpet.nii.gz")
    return fn


@pytest.mark.slow
def test_syn(runner,mri_mni_file,mask_mni_file, sbpet_outputfile) -> None:
    """It exits with a status code of zero."""
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


