"""Utility functions."""
import math
import os
import re
import shutil
import tempfile
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union
from urllib.request import urlopen

import nibabel as nib  # type: ignore
import torch
import torch.nn as nn
from torch.nn import functional

from zerodose.models import ZeroDose
from zerodose.paths import folder_with_parameter_files
from zerodose.paths import folder_with_templates


def save_nifty(data: torch.Tensor, filename_out: str, affine_ref: str) -> None:
    """Saves a nifty file."""
    save_directory = os.path.dirname(filename_out)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    func = nib.load(affine_ref).affine  # type: ignore
    data_npy = data.squeeze().cpu().detach().numpy()
    ni_img = nib.Nifti1Image(data_npy, func)
    nib.save(ni_img, filename_out)


def load_nifty(fname: str) -> torch.Tensor:
    """Loads a nifty file and returns it as a torch tensor."""
    return torch.from_numpy(nib.load(fname).get_fdata()).float()  # type: ignore


def get_mni_template(img="mr") -> str:
    """Returns the path to the t1 MNI template. If missing, downloads the template."""
    if os.environ.get("GITHUB_ACTIONS"):
        raise Exception(
            """MNI templates should not be downloaded
            when running GitHub Actions"""
        )
    else:
        _maybe_download_mni_template()
        return _get_mni_template_fname(img=img)


def binarize(img_path, out_path, threshold=0.5):
    """Binarize an image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    data[data > threshold] = 1
    img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(img, out_path)


def _get_model_fname() -> str:
    """Returns the path to the parameter file for the given fold."""
    return os.path.join(folder_with_parameter_files, "model_1.pt")


def _get_mni_template_fname(img="mr"):
    if img == "mr":
        return os.path.join(folder_with_templates, "t1.nii")
    elif img == "mask":
        return os.path.join(folder_with_templates, "mask.nii")


def _download_file(url: str, filename: str) -> None:
    data = urlopen(url).read()  # noqa
    with open(filename, "wb") as f:
        f.write(data)


def _maybe_download_parameters(
    force_overwrite: bool = False, verbose: bool = True
) -> None:
    """Downloads the parameters for some fold if it is not present yet.

    :param force_overwrite: if True the old parameter file will be\
          deleted (if present) prior to download
    :return:
    """
    if not os.path.isdir(folder_with_parameter_files):
        _maybe_mkdir_p(folder_with_parameter_files)

    out_filename = _get_model_fname()

    if force_overwrite and os.path.isfile(out_filename):
        os.remove(out_filename)

    if not os.path.isfile(out_filename):
        url = "http://sandbox.zenodo.org/record/1165160/files/gen2.pt?download=1"
        if verbose:
            print("Downloading model parameters", url, "...")
        _download_file(url, out_filename)


def _maybe_download_mni_template(
    force_overwrite: bool = False, verbose: bool = True
) -> None:
    if not os.path.isdir(folder_with_templates):
        _maybe_mkdir_p(folder_with_templates)

    out_mri = _get_mni_template_fname(img="mr")
    out_mask = _get_mni_template_fname(img="mask")

    if force_overwrite and os.path.isfile(out_mri):
        os.remove(out_mri)

    if force_overwrite and os.path.isfile(out_mri):
        os.remove(out_mask)

    if not os.path.isfile(out_mri) or not os.path.isfile(out_mask):
        with tempfile.TemporaryDirectory() as tempdirname:
            url = (
                "https://www.bic.mni.mcgill.ca/~vfonov/icbm/"
                "2009/mni_icbm152_nlin_sym_09a_nifti.zip"
            )
            filename = os.path.join(tempdirname, "mni_icbm152_nlin_sym_09a_nifti.zip")
            if verbose:
                print("Downloading MNI template", url, "...")
            _download_file(url, filename)
            if verbose:
                print("Unpacking MNI template", url, "...")
            shutil.unpack_archive(filename, tempdirname)
            t1 = os.path.join(
                tempdirname,
                "mni_icbm152_nlin_sym_09a",
                "mni_icbm152_t1_tal_nlin_sym_09a.nii",
            )
            mask = os.path.join(
                tempdirname,
                "mni_icbm152_nlin_sym_09a",
                "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii",
            )
            shutil.copy(t1, out_mri)
            shutil.copy(mask, out_mask)


def _maybe_mkdir_p(directory: str) -> None:
    """Creates a directory if it does not exist yet."""
    if not os.path.exists(directory):
        os.makedirs(directory)


class GaussianSmoothing(nn.Module):
    """Apply gaussian smoothing on a tensor."""

    def __init__(self, channels, kernel_size, sigma, dim=2):
        """Make a new instance of the GaussianSmoothing class."""
        super().__init__()
        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="xy",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):  # noqa
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = functional.conv1d
        elif dim == 2:
            self.conv = functional.conv2d
        elif dim == 3:
            self.conv = functional.conv3d
        else:
            raise RuntimeError(
                f"Only 1, 2 and 3 dimensions are supported. Received {dim}."
            )

    def forward(self, input):
        """Forward pass of the gaussian smoothing."""
        return self.conv(
            input.unsqueeze(0),
            weight=self.weight,
            groups=self.groups,
            padding=(self.kernel_size - 1) // 2,
        ).squeeze(0)


def get_gaussian_weight(
    ps: Union[List[int], Tuple[int, int, int]], std: Union[float, int]
) -> torch.Tensor:
    """Returns a gaussian weight for the given patch size and standard deviation."""
    n_slices = ps[0]
    gaussian_vec = (
        1
        / torch.sqrt(torch.tensor(std) * torch.pi)
        * torch.exp(
            -0.5 * (torch.square(torch.arange(n_slices) + 0.5 - n_slices / 2.0) / std)
        )
    )
    gaussian_vec = gaussian_vec.unsqueeze(1).unsqueeze(1)
    weight = torch.ones(ps)
    weight *= gaussian_vec
    return weight.unsqueeze(0).unsqueeze(0)


def get_model(
    model_type: Literal["standard", "dummy", "determine"] = "determine"
) -> ZeroDose:
    """Returns the ZeroDose model and loads the parameters."""
    if model_type == "determine":
        if os.environ.get("ZERODOSE_USE_DUMMY_MODEL"):
            model_type = "dummy"
        else:
            model_type = "standard"

    if model_type == "dummy":
        return ZeroDose(model_type=model_type)
    elif model_type == "standard":
        model = ZeroDose(model_type=model_type)
        if not os.environ.get("GITHUB_ACTIONS"):
            _maybe_download_parameters()
            weights_path = _get_model_fname()
            model.generator.load_state_dict(torch.load(weights_path))
        return model
    else:
        raise ValueError(f"Unknown model type '{model_type!r}'.")


def _create_output_fname(mri_fname, suffix="_sb", file_type=".nii.gz"):
    """Create output filename from input filename."""
    out_fname = mri_fname
    if out_fname.endswith(".nii.gz"):
        out_fname = re.sub(".nii.gz$", "", out_fname)
    if out_fname.endswith(".nii"):
        out_fname = re.sub(".nii$", "", out_fname)
    out_fname += suffix + file_type
    return out_fname
