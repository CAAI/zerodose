"""Full pipeline."""

import torch
import torchio as tio

from zerodose import utils
from zerodose.dataset import SubjectDataset
from zerodose.inference import _infer_single_subject
from zerodose.inference import get_inference_params
from zerodose.models import AbnormalityMap
from zerodose.models import QuantileNormalization


def run(
    mri_fnames,
    mask_fnames,
    pet_fnames,
    out_sbpets,
    out_abns,
    verbose=False,
    device="cuda:0",
):
    """Run ZeroDose."""
    if isinstance(pet_fnames, str):
        pet_fnames = [pet_fnames]
    if isinstance(mri_fnames, str):
        mri_fnames = [mri_fnames]
    if isinstance(mask_fnames, str):
        mask_fnames = [mask_fnames]
    if isinstance(out_sbpets, str):
        out_sbpets = [out_sbpets]
    if isinstance(out_abns, str):
        out_abns = [out_abns]

    if not (
        len(pet_fnames) == len(out_sbpets)
        and len(pet_fnames) == len(mri_fnames)
        and len(pet_fnames) == len(mask_fnames)
        and len(pet_fnames) == len(out_abns)
    ):
        raise ValueError(
            """The number of MRI file {} PET files {},
            mask files {} and output sbPET files {} and
              output ABN files {} must be identical""".format(
                len(mri_fnames),
                len(pet_fnames),
                len(mask_fnames),
                len(out_sbpets),
                len(out_abns),
            )
        )

    dataset = SubjectDataset(
        mri_fnames,
        mask_fnames,
        pet_fnames,
    )

    model = utils.get_model()
    model = model.to(device)
    model.eval()

    abnormality_mapper = AbnormalityMap.get_default().to(device)
    normalization = QuantileNormalization.get_default().to(device)

    par = get_inference_params()
    for i, sub in enumerate(dataset):  # type: ignore
        if verbose:
            print(f"Synthesizing sbPET for {sub.mr.path}")

        sbpet_tensor = _infer_single_subject(
            model,
            sub,
            par["patch_size"],
            par["patch_overlap"],
            par["batch_size"],
            par["sd_weight"],
            device,
        )

        sbpetraw_img = tio.ScalarImage(
            tensor=sbpet_tensor.cpu().data, affine=sub["mr"].affine
        )

        inverse_transform = sub.get_inverse_transform(warn=False)
        sub.add_image(sbpetraw_img, "sbpetraw")
        sub = inverse_transform(sub)

        with torch.no_grad():
            pet = sub["pet"].tensor.to(device)
            sbpetraw = sub["sbpetraw"].tensor.to(device)
            mask = sub["mask"].tensor.type(torch.bool).to(device)

            sbpet = normalization(pet, sbpetraw, mask)
            abnormality_map = abnormality_mapper(pet, sbpet, mask)

            sbpet = sbpet.cpu().data
            abnormality_map = abnormality_map.cpu().data

        if verbose:
            print("Saving volumes...")

        utils.save_nifty(sbpet, out_sbpets[i], affine_ref=sub["mr"].path)
        utils.save_nifty(abnormality_map, out_abns[i], affine_ref=sub["mr"].path)
