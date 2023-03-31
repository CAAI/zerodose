"""Functions called by the CLI."""


from typing import Iterable

import torch
from zerodose import utils

from zerodose.model import AbnormalityMap
from zerodose.processing import QuantileNormalization

def create_abnormality_maps(
        pet_fnames: Iterable[str],
        sbpet_fnames: Iterable[str],
        mask_fnames: Iterable[str],
        out_fnames: Iterable[str],
        verbose: bool = False,
        save_output: bool =True,
        device: str = "cuda:0",
        ):
    if not isinstance(pet_fnames, list):
        pet_fnames = list(pet_fnames)
    if not isinstance(sbpet_fnames, list):
        sbpet_fnames = list(sbpet_fnames)
    if not isinstance(mask_fnames, list):
        mask_fnames = list(mask_fnames)
    if not isinstance(out_fnames, list):
        out_fnames = list(out_fnames)

    if not (len(pet_fnames) == len(mask_fnames) and len(pet_fnames) == len(sbpet_fnames) and  len(pet_fnames) == len(out_fnames)):
        raise Exception(
            """The number of PET files {} sbPET files {}, 
            mask files {} and output files {} must be identical""".format(
                len(pet_fnames),len(sbpet_fnames), len(mask_fnames), len(out_fnames)
            )
        )
    
    normalization = QuantileNormalization(quantile=0.97,sigma_normalization=3).to(device)
    abnormality_map = AbnormalityMap(sigma_smooth=3).to(device)

    with torch.no_grad():
        for i in range(len(pet_fnames)):  # type: ignore
            if verbose:
                print(f"Creating abnormality map for {pet_fnames[i]}")

            pet = utils.load_nifty(pet_fnames[i]).to(device)
            sbpet = utils.load_nifty(sbpet_fnames[i]).to(device)
            mask = utils.load_nifty(mask_fnames[i]).type(torch.bool).to(device)

            sbpet = normalization(pet,sbpet,mask)
            abnormality_map = abnormality_map(pet,sbpet,mask)

            if save_output:
                if verbose:
                    print(f"Saving to {out_fnames[i]}")
            
                utils.save_nifty(abnormality_map, out_fnames[i], affine_ref=pet_fnames[i])
    
    return out_fnames