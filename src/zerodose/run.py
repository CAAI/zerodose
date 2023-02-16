"""Functions called by the CLI."""

import os

import nibabel as nib
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridAggregator
from torchio.data import GridSampler

from zerodose import processing
from zerodose import utils
from zerodose.dataset import SubjectDataset
from zerodose.model import ZeroDose


def _get_gaussian_weight(ps, std):
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


def _infer_single_subject(model, subject, ps, po, bs, std, device):
    """Infer a single subject."""
    grid_sampler = GridSampler(subject, ps, po)
    patch_loader = DataLoader(grid_sampler, batch_size=bs, num_workers=4)
    aggregator = GridAggregator(grid_sampler, overlap_mode="average")
    aggregator_weight = GridAggregator(grid_sampler, overlap_mode="average")
    weight = _get_gaussian_weight(ps, std).to(device)
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


def _save_nifty(data, filename_out, affine_ref):
    save_directory = os.path.dirname(filename_out)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    func = nib.load(affine_ref).affine
    data = data.squeeze().cpu().detach().numpy()
    ni_img = nib.Nifti1Image(data, func)
    nib.save(ni_img, filename_out)


def _get_model():
    model = ZeroDose()
    utils.maybe_download_parameters()
    weights_path = utils.get_model_fname()
    model.generator.load_state_dict(torch.load(weights_path))
    return model


def synthesize_baselines(
    mri_fnames,
    mask_fnames,
    out_fnames,
    device="cuda",
    overwrite=True,
    sd_weight=5,
    verbose=True,
    batch_size=1,
    stride=2,
    save_output=True,
):
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
    model = _get_model()
    model = model.to(device)
    model.eval()
    patch_size = (32, 192, 192)
    patch_overlap = tuple(_size - stride for _size in patch_size)

    for sub in dataset:
        if verbose:
            print(f"Synthesizing sbPET for {sub.mr.path}")

        sbpet = _infer_single_subject(
            model, sub, patch_size, patch_overlap, batch_size, sd_weight, device
        )

        sbpet = sbpet.cpu().data * sub.mask.data
        sbpet = processing._postprocess(sbpet)

        if save_output:
            if verbose:
                print(f"Saving to {sub.out_fname}")

            _save_nifty(sbpet, sub.out_fname, affine_ref=sub.mr.path)
