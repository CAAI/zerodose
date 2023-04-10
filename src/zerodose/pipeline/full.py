"""Full pipeline."""

import os
import tempfile

from zerodose.pipeline.main import run
from zerodose.pipeline.niftyreg import from_mni
from zerodose.pipeline.niftyreg import to_mni
from zerodose.visualizations import create_article_figure


def run_with_registration(  # noqa: C901
    mri_fname,
    mask_fname,
    pet_fname,
    out_sbpet,
    out_abn,
    out_img,
    reg_pet_to_mri=True,
    verbose=False,
    device="cuda:0",
    outputspace="pet",
):
    """Run full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        if outputspace == "pet":
            output_target = pet_fname
        elif outputspace == "mr":
            output_target = mri_fname

        sbpet = os.path.join(tmpdir, "sbpet.nii.gz")
        abn = os.path.join(tmpdir, "abn.nii.gz")
        affine_fname = os.path.join(tmpdir, "affine.txt")
        mri = os.path.join(tmpdir, "mri_mni.nii.gz")
        pet = os.path.join(tmpdir, "pet_mni.nii.gz")
        mask = os.path.join(tmpdir, "mask_mni.nii.gz")

        if verbose:
            print("Registration to MNI space...")

        to_mni(
            in_mri_fname=mri_fname,
            in_pet_fname=pet_fname,
            in_mask_fname=mask_fname,
            out_mri_fname=mri,
            out_pet_fname=pet,
            out_mask_fname=mask,
            out_affine_fname=affine_fname,
            reg_pet_to_mri=reg_pet_to_mri,
        )

        run(
            mri,
            mask,
            pet,
            sbpet,
            abn,
            verbose=verbose,
            device=device,
        )

        if out_img is not None:
            if verbose:
                print(f"Creating figure {out_img}...")
            create_article_figure(mri, pet, sbpet, mask, abn, save_fname=out_img)

        if verbose:
            print(f"Registration of sbPET back to {outputspace} space...")

        from_mni(
            ref_fname=output_target,
            in_affine_fname=affine_fname,
            float_fname=sbpet,
            out_fname=out_sbpet,
        )

        if verbose:
            print(f"Registration of abnormality map back to {outputspace} space...")

        from_mni(
            ref_fname=output_target,
            in_affine_fname=affine_fname,
            float_fname=abn,
            out_fname=out_abn,
        )
