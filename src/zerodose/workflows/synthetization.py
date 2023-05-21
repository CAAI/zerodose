"""Functions called by the CLI."""


from typing import Sequence
from typing import Union

import torch
import torchio as tio

from zerodose import utils
from zerodose.dataset import SubjectDataset
from zerodose.inference import _infer_single_subject


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
    apply_mask: bool = True,
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

    dataset = SubjectDataset(mri_fnames, mask_fnames)
    model = utils.get_model()

    model = model.to(device)
    model.eval()
    patch_size = (32, 192, 192)
    patch_overlap = (32 - stride, 192 - stride, 192 - stride)

    for i, sub in enumerate(dataset):  # type: ignore
        if verbose:
            print(f"Synthesizing sbPET for {sub.mr.path}")

        sbpet_tensor = _infer_single_subject(
            model, sub, patch_size, patch_overlap, batch_size, sd_weight, device
        )

        sbpet_img = tio.ScalarImage(
            tensor=sbpet_tensor.cpu().data, affine=sub["mr"].affine
        )

        inverse_transform = sub.get_inverse_transform(warn=False)

        sub.add_image(sbpet_img, "sbpet")
        sub.remove_image("mr")
        sub = inverse_transform(sub)

        if save_output:
            if verbose:
                print(f"Saving to {out_fnames[i]}")
            sub["sbpet"].save(out_fnames[i])
