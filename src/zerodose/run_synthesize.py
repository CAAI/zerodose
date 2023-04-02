"""Functions called by the CLI."""


from typing import Sequence
from typing import Tuple
from typing import Union

import nibabel as nib
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridAggregator
from torchio.data import GridSampler

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


def _run_reverse_transform(sbpet, sub):
    temp_sub = tio.Subject({"sbpet": sbpet})
    inverse_transform = sub.get_inverse_transform(warn=False)
    sbpet_img = inverse_transform(temp_sub)["sbpet"]
    mask = nib.load(sub["mask"].path).get_fdata()
    sbpet_img.set_data(sbpet_img.tensor * mask)
    return sbpet_img


def synthesize_baselines(
    mri_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Sequence[str],
    device: Union[torch.device, str] = "cuda:0",
    sd_weight: Union[float, int] = 5,
    verbose: bool = False,
    batch_size: int = 1,
    stride: int = 2,
    save_output: bool = True,
    do_registration: bool = True,
) -> None:
    """Synthesize baseline PET images from MRI images."""
    if isinstance(mri_fnames, str):
        mri_fnames = [mri_fnames]
    if isinstance(mask_fnames, str):
        mask_fnames = [mask_fnames]
    if isinstance(out_fnames, str):
        out_fnames = [out_fnames]

    if not (len(mri_fnames) == len(mask_fnames) and len(mri_fnames) == len(out_fnames)):
        raise Exception(
            """The number of input files {} mask files {}
            and output files {} must be identical""".format(
                len(mri_fnames), len(mask_fnames), len(out_fnames)
            )
        )

    dataset = SubjectDataset(
        mri_fnames, mask_fnames, out_fnames, do_registration=do_registration
    )
    model = utils.get_model()

    model = model.to(device)
    model.eval()
    patch_size = (32, 192, 192)
    patch_overlap = (32 - stride, 192 - stride, 192 - stride)

    for sub in dataset:  # type: ignore
        if verbose:
            print(f"Synthesizing sbPET for {sub.mr.path}")

        sbpet_tensor = _infer_single_subject(
            model, sub, patch_size, patch_overlap, batch_size, sd_weight, device
        )

        sbpet_img = tio.ScalarImage(
            tensor=sbpet_tensor.cpu().data, affine=sub["mr"].affine
        )

        sbpet_img = _run_reverse_transform(sbpet_img, sub)

        if save_output:
            if verbose:
                print(f"Saving to {sub.out_fname}")
            sbpet_img.save(sub.out_fname)
