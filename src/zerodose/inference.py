"""Inference functions."""
from typing import Tuple
from typing import Union

import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridAggregator
from torchio.data import GridSampler

from zerodose import utils
from zerodose.models import ZeroDose


def _infer_single_subject(
    model: ZeroDose,
    subject: tio.Subject,
    ps: Tuple[int, int, int],
    po: Tuple[int, int, int],
    bs: int,
    std: Union[float, int],
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Infer a single subject."""
    grid_sampler = GridSampler(subject, ps, po)
    patch_loader = DataLoader(
        grid_sampler, batch_size=bs, num_workers=0  # type: ignore
    )
    aggregator = GridAggregator(grid_sampler, overlap_mode="average")
    aggregator_weight = GridAggregator(grid_sampler, overlap_mode="average")
    weight = utils.get_gaussian_weight(ps, std).to(device)
    with torch.no_grad():
        for patches_batch in patch_loader:
            patch_x = patches_batch["mr"][tio.DATA].to(device=device)
            locations = patches_batch[tio.LOCATION]
            patch_y = model(patch_x)
            aggregator.add_batch(patch_y * weight, locations)
            aggregator_weight.add_batch(weight, locations)

    return (
        aggregator.get_output_tensor().detach()
        / aggregator_weight.get_output_tensor().detach()
    )


def synthesize(mr, model, device="cuda:0"):
    """Synthesize baseline PET images from MRI images."""
    if len(mr.shape) == 3:
        mr = mr.unsqueeze(0)

    sub = tio.Subject({"mr": tio.ScalarImage(tensor=mr)})
    stride = 2
    patch_size = (32, 192, 192)
    patch_overlap = (32 - stride, 192 - stride, 192 - stride)
    sd_weight = (5,)
    batch_size = (1,)
    stride = (2,)

    sbpet_tensor = _infer_single_subject(
        model, sub, patch_size, patch_overlap, batch_size, sd_weight, device
    )

    return sbpet_tensor


def get_inference_params():
    """Returns the default parameters for inference stitching."""
    stride = 2
    params = {
        "patch_size": (32, 192, 192),
        "patch_overlap": (32 - stride, 192 - stride, 192 - stride),
        "batch_size": 1,
        "sd_weight": 5,
    }
    return params
