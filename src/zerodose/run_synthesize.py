"""Functions called by the CLI."""


from typing import Iterable
from typing import Tuple
from typing import Union

import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridAggregator
from torchio.data import GridSampler

from zerodose import processing
from zerodose import utils
from zerodose.dataset import SubjectDataset
from zerodose.model import ZeroDose

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


def synthesize_baselines(
    mri_fnames: Iterable[str],
    mask_fnames: Iterable[str],
    out_fnames: Iterable[str],
    device: Union[torch.device, str] = "cuda:0",
    sd_weight: Union[float, int] = 5,
    verbose: bool = False,
    batch_size: int = 1,
    stride: int = 2,
    save_output: bool = True,
) -> None:
    """Synthesize baseline PET images from MRI images."""
    if not isinstance(mri_fnames, list):
        mri_fnames = list(mri_fnames)
    if not isinstance(mask_fnames, list):
        mask_fnames = list(mask_fnames)
    if not isinstance(out_fnames, list):
        out_fnames = list(out_fnames)

    if not (len(mri_fnames) == len(mask_fnames) and len(mri_fnames) == len(out_fnames)):
        Exception(
            """The number of input files {} mask files {}
            and output files {} must be identical""".format(
                len(mri_fnames), len(mask_fnames), len(out_fnames)
            )
        )

    dataset = SubjectDataset(mri_fnames, mask_fnames, out_fnames)
    model = utils.get_model()

    model = model.to(device)
    model.eval()
    patch_size = (32, 192, 192)
    patch_overlap = (32 - stride, 192 - stride, 192 - stride)

    for sub in dataset:  # type: ignoregaussian_blure
        if verbose:
            print(f"Synthesizing sbPET for {sub.mr.path}")

        sbpet = _infer_single_subject(
            model, sub, patch_size, patch_overlap, batch_size, sd_weight, device
        )


        sbpet = sbpet.cpu().data * sub.mask.data
        sbpet = processing.postprocess(sbpet)

        if save_output:
            if verbose:
                print(f"Saving to {sub.out_fname}")

            utils.save_nifty(sbpet, sub.out_fname, affine_ref=sub.mr.path)

    return 0
