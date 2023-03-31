"""Utility functions."""
import math
import os
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union
from urllib.request import urlopen

import nibabel as nib
import torch
import torch.nn as nn
from torch.nn import functional

from zerodose.model import ZeroDose
from zerodose.paths import folder_with_parameter_files


def get_model_fname() -> str:
    """Returns the path to the parameter file for the given fold."""
    return os.path.join(folder_with_parameter_files, "model_1.pt")


def _download_file(url: str, filename: str) -> None:
    data = urlopen(url).read()  # noqa
    with open(filename, "wb") as f:
        f.write(data)


def maybe_download_parameters(
    force_overwrite: bool = False, verbose: bool = True
) -> None:
    """Downloads the parameters for some fold if it is not present yet.

    :param force_overwrite: if True the old parameter file will be\
          deleted (if present) prior to download
    :return:
    """
    if not os.path.isdir(folder_with_parameter_files):
        maybe_mkdir_p(folder_with_parameter_files)

    out_filename = get_model_fname()

    if force_overwrite and os.path.isfile(out_filename):
        os.remove(out_filename)

    if not os.path.isfile(out_filename):
        url = "http://sandbox.zenodo.org/record/1165160/files/gen2.pt?download=1"
        if verbose:
            print("Downloading", url, "...")
        _download_file(url, out_filename)


def maybe_mkdir_p(directory: str) -> None:
    """Creates a directory if it does not exist yet."""
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[: i + 1])):
            os.mkdir(os.path.join("/", *splits[: i + 1]))


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


def save_nifty(data: torch.Tensor, filename_out: str, affine_ref: str) -> None:
    """Saves a nifty file."""
    save_directory = os.path.dirname(filename_out)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    func = nib.load(affine_ref).affine
    data = data.squeeze().cpu().detach().numpy()
    ni_img = nib.Nifti1Image(data, func)
    nib.save(ni_img, filename_out)


def load_nifty(fname: str) -> torch.Tensor:
    """Loads a nifty file and returns it as a torch tensor."""
    return torch.from_numpy(nib.load(fname).get_fdata()).float()


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
        maybe_download_parameters()
        weights_path = get_model_fname()
        model.generator.load_state_dict(torch.load(weights_path))
        return model
    else:
        raise ValueError(f"Unknown model type '{model_type!r}'.")
