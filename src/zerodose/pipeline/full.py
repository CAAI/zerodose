"""Full pipeline."""

import os
import shutil
import tempfile

from zerodose.pipeline.abnormality import create_abnormality_maps
from zerodose.pipeline.niftyreg import from_mni
from zerodose.pipeline.niftyreg import to_mni
from zerodose.pipeline.normalization import normalize_to_pet
from zerodose.pipeline.synthetization import synthesize_baselines


def run_full(  # noqa: C901
    mri_fname,
    mask_fname,
    out_sbpet,
    pet_fname=None,
    out_abn=None,
    out_img=None,
    do_registration=True,
    do_abnormality=True,
    do_normalization=True,
    do_image=True,
    verbose=False,
    device="cuda:0",
    outputspace="pet",
):
    """Run full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        if pet_fname is None:
            print("No pet file provided:")
            print("\tsetting outputspace = 'mr'")
            print("\tskipping normalization")
            print("\tskipping abnormality map")
            print("\tSkipping image creation")
            outputspace = "mr"
            do_normalization = False
            do_abnormality = False
            do_image = False

        if outputspace == "pet":
            output_target = pet_fname
        elif outputspace == "mr":
            output_target = mri_fname

        sbpetraw = os.path.join(tmpdir, "sbpetraw.nii.gz")
        sbpet = os.path.join(tmpdir, "sbpet.nii.gz")

        if do_registration:
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
            )
        else:
            mri = mri_fname
            pet = pet_fname
            mask = mask_fname

        synthesize_baselines(
            mri_fnames=mri,
            mask_fnames=mask,
            out_fnames=sbpetraw,
            verbose=verbose,
            device=device,
        )

        if do_normalization:
            normalize_to_pet(
                pet_fnames=pet,
                sbpet_fnames=sbpetraw,
                mask_fnames=mask,
                out_fnames=sbpet,
                verbose=verbose,
                device=device,
            )
        else:
            sbpet = sbpetraw

        if do_registration:
            if verbose:
                print(f"Registration of sbPET back to {outputspace} space...")
            from_mni(
                ref_fname=output_target,
                in_affine_fname=affine_fname,
                float_fname=sbpet,
                out_fname=out_sbpet,
            )
        else:
            shutil.copyfile(sbpet, out_sbpet)

        if verbose:
            print("sbPET saved to", out_sbpet)

        if do_abnormality:
            abn = os.path.join(tmpdir, "abn.nii.gz")

            create_abnormality_maps(
                pet_fnames=pet,
                sbpet_fnames=sbpet,
                mask_fnames=mask,
                out_fnames=abn,
                verbose=verbose,
                device=device,
            )

            if do_image:
                create_abn_image(
                    mri=mri,
                    pet=pet,
                    sbpet=sbpet,
                    abn=abn,
                    mask=mask,
                    out_fname=out_img,
                )

            if do_registration:
                if verbose:
                    print(
                        f"Registration of abnormality map back\
                              to {outputspace} space..."
                    )
                from_mni(
                    ref_fname=output_target,
                    in_affine_fname=affine_fname,
                    float_fname=abn,
                    out_fname=out_abn,
                )
            else:
                shutil.copyfile(abn, out_abn)
            if verbose:
                print("Abnormality map saved to", out_abn)


def create_abn_image(mri, pet, sbpet, abn, mask, out_fname):
    """Create abnormality image."""
    pass
