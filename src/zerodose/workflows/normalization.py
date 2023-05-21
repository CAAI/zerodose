"""Functions called by the CLI."""


from typing import Sequence

import torch

from zerodose import utils
from zerodose.models import QuantileNormalization


def normalize_to_pet(
    pet_fnames: Sequence[str],
    sbpet_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Sequence[str],
    verbose: bool = False,
    save_output: bool = True,
    device: str = "cuda:0",
) -> None:
    """Normalize sbPET to PET images."""
    if isinstance(pet_fnames, str):
        pet_fnames = [pet_fnames]
    if isinstance(sbpet_fnames, str):
        sbpet_fnames = [sbpet_fnames]
    if isinstance(mask_fnames, str):
        mask_fnames = [mask_fnames]
    if isinstance(out_fnames, str):
        out_fnames = [out_fnames]

    if not (
        len(pet_fnames) == len(mask_fnames)
        and len(pet_fnames) == len(sbpet_fnames)
        and len(pet_fnames) == len(out_fnames)
    ):
        raise ValueError(
            """The number of PET files {} sbPET files {},
            mask files {} and output files {} must be identical""".format(
                len(pet_fnames), len(sbpet_fnames), len(mask_fnames), len(out_fnames)
            )
        )

    normalization = QuantileNormalization(quantile=0.97, sigma_normalization=3).to(
        device
    )

    with torch.no_grad():
        for i in range(len(pet_fnames)):  # type: ignore
            if verbose:
                print(f"Normalizing to {pet_fnames[i]}")

            pet = utils.load_nifty(pet_fnames[i]).to(device)
            sbpet = utils.load_nifty(sbpet_fnames[i]).to(device)
            mask = utils.load_nifty(mask_fnames[i]).type(torch.bool).to(device)

            sbpet = normalization(pet, sbpet, mask)

            if save_output:
                if verbose:
                    print(f"Saving to {out_fnames[i]}")

                utils.save_nifty(sbpet, out_fnames[i], affine_ref=str(pet_fnames[i]))
