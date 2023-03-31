"""Test cases for the __main__ module."""
import os

import nibabel as nib
import numpy as np
import pytest
import torch

from zerodose import __main__
from zerodose.run_abnormality import create_abnormality_maps
from zerodose.run_synthesize import synthesize_baselines
from zerodose.utils import get_model


@pytest.fixture()
def use_dummy_model():
    """Use the dummy model for testing."""
    os.environ["ZERODOSE_USE_DUMMY_MODEL"] = "1"
    yield
    del os.environ["ZERODOSE_USE_DUMMY_MODEL"]


def _create_random_image_small(seed) -> nib.Nifti1Image:
    """Create a random image."""
    np.random.seed = seed
    data = np.random.rand(30, 30, 30)
    return nib.Nifti1Image(data, np.eye(4))


def _create_random_mask_small(seed) -> nib.Nifti1Image:
    """Create a random mask."""
    np.random.seed = seed
    data = np.random.rand(30, 30, 30)
    data = (data > 0.5).astype("uint8")
    return nib.Nifti1Image(data, np.eye(4))


def _create_random_image(seed) -> nib.Nifti1Image:
    """Create a random image."""
    np.random.seed = seed
    data = np.random.rand(197, 233, 189)
    return nib.Nifti1Image(data, np.eye(4))


def _create_random_mask(seed) -> nib.Nifti1Image:
    """Create a random mask."""
    np.random.seed = seed
    data = np.random.rand(197, 233, 189)
    data = (data > 0.5).astype("uint8")
    return nib.Nifti1Image(data, np.eye(4))


@pytest.fixture(scope="session")
def pet_file(tmp_path_factory):
    """Create a random PET image."""
    img = _create_random_image(seed=0)
    fn = str(tmp_path_factory.mktemp("input_images") / "pet0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def sbpet_file(tmp_path_factory):
    """Create a random PET image.""" ""
    img = _create_random_image(seed=1)
    fn = str(tmp_path_factory.mktemp("input_images") / "sbpet0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def mri_file(tmp_path_factory):
    """Create a random MRI image."""
    img = _create_random_image(seed=2)
    fn = str(tmp_path_factory.mktemp("input_images") / "mri0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def mask_file(tmp_path_factory):
    """Create a random mask image."""
    img = _create_random_mask(seed=3)
    fn = str(tmp_path_factory.mktemp("input_images") / "mask0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def pet_file_small(tmp_path_factory):
    """Create a random PET image."""
    img = _create_random_image_small(seed=0)
    fn = str(tmp_path_factory.mktemp("input_images") / "pet0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def sbpet_file_small(tmp_path_factory):
    """Create a random PET image.""" ""
    img = _create_random_image_small(seed=1)
    fn = str(tmp_path_factory.mktemp("input_images") / "sbpet0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def mri_file_small(tmp_path_factory):
    """Create a random MRI image."""
    img = _create_random_image_small(seed=2)
    fn = str(tmp_path_factory.mktemp("input_images") / "mri0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def mask_file_small(tmp_path_factory):
    """Create a random mask image."""
    img = _create_random_mask_small(seed=3)
    fn = str(tmp_path_factory.mktemp("input_images") / "mask0.nii.gz")
    nib.save(img, fn)
    return fn


@pytest.fixture(scope="session")
def sbpet_outputfile(tmp_path_factory):
    """Create a random mask image."""
    fn = str(tmp_path_factory.mktemp("output_images") / "output0.nii.gz")
    return fn


@pytest.fixture(scope="session")
def abn_outputfile(tmp_path_factory):
    """Create a random mask image."""
    fn = str(tmp_path_factory.mktemp("output_images") / "mni_abn.nii.gz")
    return fn


def test_main_succeeds(runner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.main)
    assert result.exit_code == 0


def test_main_help_succeeds(runner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.main, ["--help"])
    assert result.exit_code == 0


# Tests


@pytest.mark.parametrize("model_type", ["standard", "dummy"])
def test_forward_pass(model_type) -> None:
    """Test the initialize_model command."""
    model = get_model(model_type)
    with torch.no_grad():
        model.eval()
        outs = model.forward(torch.zeros(1, 1, 192, 192, 32))
        assert outs.shape == (1, 1, 192, 192, 32)


@pytest.mark.usefixtures("use_dummy_model")
def test_syn(runner, mri_file, mask_file, sbpet_outputfile) -> None:
    """Test the syn command."""
    cmd = [
        "syn",
        "-i",
        mri_file,
        "-m",
        mask_file,
        "-o",
        sbpet_outputfile,
        "--device",
        "cpu",
    ]

    result = runner.invoke(__main__.main, cmd)

    assert result.exit_code == 0


@pytest.mark.usefixtures("use_dummy_model")
def test_syn_no_output_fname(runner, mri_file, mask_file) -> None:
    """Test the syn command."""
    cmd = ["syn", "-i", mri_file, "-m", mask_file, "--device", "cpu"]

    result = runner.invoke(__main__.main, cmd)

    assert result.exit_code == 0


@pytest.mark.usefixtures("use_dummy_model")
@pytest.mark.parametrize("save_output", [True, False])
def test_syn_function(mri_file, mask_file, sbpet_outputfile, save_output) -> None:
    """Test the syn command."""
    synthesize_baselines(
        mri_file,
        mask_file,
        sbpet_outputfile,
        verbose=True,
        save_output=save_output,
        device="cpu",
    )


@pytest.mark.usefixtures("use_dummy_model")
@pytest.mark.parametrize("save_output", [True, False])
def test_abn_function(
    pet_file_small, sbpet_file_small, mask_file_small, abn_outputfile, save_output
) -> None:
    """Test the abn command."""
    create_abnormality_maps(
        sbpet_file_small,
        pet_file_small,
        mask_file_small,
        abn_outputfile,
        save_output=save_output,
        verbose=True,
        device="cpu",
    )


@pytest.mark.usefixtures("use_dummy_model")
def test_syn_multiple(runner, mri_file, mask_file, sbpet_outputfile) -> None:
    """Test the syn command."""
    cmd = [
        "syn",
        "-i",
        mri_file,
        "-m",
        mask_file,
        "-o",
        sbpet_outputfile,
        "-i",
        mri_file,
        "-m",
        mask_file,
        "-o",
        sbpet_outputfile,
        "--device",
        "cpu",
    ]

    result = runner.invoke(__main__.main, cmd)

    assert result.exit_code == 0


@pytest.mark.usefixtures("use_dummy_model")
def test_syn_input_validation(runner, mri_file, mask_file, sbpet_outputfile) -> None:
    """Test the syn command."""
    cmd = [
        "syn",
        "-i",
        mri_file,
        "-m",
        mask_file,
        "-o",
        sbpet_outputfile,
        "-i",
        mri_file,
        "--device",
        "cpu",
    ]

    result = runner.invoke(__main__.main, cmd)

    assert result.exit_code == 1


@pytest.mark.usefixtures("use_dummy_model")
def test_abn(runner, pet_file_small, mask_file_small, sbpet_file_small, abn_outputfile):
    """Test the abn command."""
    result = runner.invoke(
        __main__.main,
        [
            "abn",
            "-p",
            pet_file_small,
            "-m",
            mask_file_small,
            "-s",
            sbpet_file_small,
            "-o",
            abn_outputfile,
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
