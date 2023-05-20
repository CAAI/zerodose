"""Command-line interface."""
from typing import Sequence
from typing import Union

import click

from zerodose.pipeline import create_abnormality_maps
from zerodose.pipeline import normalize_to_pet
from zerodose.pipeline import run_with_registration
from zerodose.pipeline import synthesize_baselines
from zerodose.utils import _create_output_fname


@click.group()
@click.version_option()
def main() -> None:
    """Zerodose CLI."""
    pass


device_option = click.option(
    "-d",
    "--device",
    "device",
    type=click.Choice(
        [
            "cpu",
            "cuda:0",
            "cuda:1",
            "cuda:2",
            "cuda:3",
            "cuda:4",
            "cuda:5",
            "cuda:6",
            "cuda:7",
        ]
    ),
    default="cuda:0",
    help="Device to use for inference.",
)

mri_option = click.option(
    "-i",
    "--in",
    "mri_fnames",
    type=click.Path(exists=True),
    required=True,
    multiple=True,
    help="Help test",
)

mask_option = click.option(
    "-m",
    "--mask",
    "mask_fnames",
    type=click.Path(exists=True),
    required=True,
    multiple=True,
)

sbpet_output_option = click.option(
    "-o", "--out", "out_fnames", type=click.Path(), multiple=True
)

verbose_option = click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)

no_registration_option = click.option(
    "-n",
    "--no-registration",
    "no_registration",
    is_flag=True,
    default=False,
    help="""Skip registration to MNI space.
    Useful if the input images already are in MNI space""",
)

outputspace_option = click.option(
    "--space",
    "outputspace",
    type=click.Choice(
        [
            "mr",
            "pet",
        ]
    ),
    default="mr",
    help="Which space to ",
)


@main.command()
@mri_option
@mask_option
@sbpet_output_option
@verbose_option
@device_option
def syn(
    mri_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Union[Sequence[str], None] = None,
    verbose: bool = True,
    device: str = "cuda:0",
) -> None:
    """Synthesize baseline PET images."""
    if out_fnames is None or len(out_fnames) == 0:
        out_fnames = [
            _create_output_fname(mri_fname, suffix="_sbraw") for mri_fname in mri_fnames
        ]

    synthesize_baselines(
        mri_fnames,
        mask_fnames,
        out_fnames,
        verbose=verbose,
        device=device,
    )


pet_option = click.option(
    "-p",
    "--pet",
    "pet_fnames",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
)

sbpet_option = click.option(
    "-s",
    "--sbpet",
    "sbpet_fnames",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
)

abn_output_option = click.option(
    "-o",
    "--o",
    "out_fnames",
    type=click.Path(),
    multiple=True,
)

no_resample_mask_option = click.option(
    "--no-resample-mask",
    "no_resample_mask",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)


@pet_option
@sbpet_option
@mask_option
@abn_output_option
@verbose_option
@device_option
@main.command()
def abn(
    pet_fnames: Sequence[str],
    sbpet_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Union[Sequence[str], None] = None,
    verbose: bool = False,
    device: str = "cuda:0",
):
    """Create abnormality maps."""
    if out_fnames is None or len(out_fnames) == 0:
        out_fnames = [
            _create_output_fname(pet_fname, suffix="_abn") for pet_fname in pet_fnames
        ]

    create_abnormality_maps(
        pet_fnames,
        sbpet_fnames,
        mask_fnames,
        out_fnames,
        verbose=verbose,
        device=device,
    )


no_resample_sbpet_option = click.option(
    "--no-resample-sbpet",
    "no_resample_sbpet",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)


@pet_option
@sbpet_option
@mask_option
@abn_output_option
@verbose_option
@device_option
@main.command()
def norm(
    pet_fnames: Sequence[str],
    sbpet_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Union[Sequence[str], None] = None,
    verbose: bool = False,
    device: str = "cuda:0",
):
    """Normalize sbPET images to PET images."""
    if out_fnames is None or len(out_fnames) == 0:
        out_fnames = [
            _create_output_fname(pet_name, suffix="_sb") for pet_name in pet_fnames
        ]

    normalize_to_pet(
        pet_fnames=pet_fnames,
        sbpet_fnames=sbpet_fnames,
        mask_fnames=mask_fnames,
        out_fnames=out_fnames,
        verbose=verbose,
        device=device,
    )


no_image_option = click.option(
    "--no-image",
    "no_image",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)

mri_option_single = click.option(
    "-i",
    "--mri",
    "mri_fname",
    type=click.Path(exists=True),
    required=True,
)
mask_option_single = click.option(
    "-m",
    "--mask",
    "mask_fname",
    type=click.Path(exists=True),
    required=True,
)

pet_option_single = click.option(
    "-p",
    "--pet",
    "pet_fname",
    type=click.Path(exists=True),
    required=False,
)

sbpet_output_option_single = click.option(
    "-os",
    "--out-sbpet",
    "out_sbpet",
    type=click.Path(),
    required=False,
)

abn_output_option_single = click.option(
    "-oa",
    "--out-abn",
    "out_abn",
    type=click.Path(),
    required=False,
)

img_output_option_single = click.option(
    "-oi",
    "--out-img",
    "out_img",
    type=click.Path(),
    required=False,
)

no_abnormality_option = click.option(
    "--no-abn",
    "no_abnormality",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)

no_normalization_option = click.option(
    "--no-norm",
    "no_normalization",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)

no_image_option = click.option(
    "--no-img",
    "no_image",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)

no_reg_pet_to_mr_option = click.option(
    "--no-pet-rigid",
    "no_reg_pet_to_mr",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)


@mri_option_single
@mask_option_single
@pet_option_single
@sbpet_output_option_single
@abn_output_option_single
@img_output_option_single
@no_reg_pet_to_mr_option
@no_image_option
@verbose_option
@device_option
@outputspace_option
@main.command()
def pipeline(
    mri_fname,
    mask_fname,
    pet_fname,
    out_sbpet,
    out_abn,
    out_img,
    no_reg_pet_to_mr,
    no_image,
    verbose,
    device,
    outputspace,
):
    """Run full pipeline."""
    reg_pet_to_mr = not no_reg_pet_to_mr
    do_image = not no_image

    if out_sbpet is None:
        out_sbpet = _create_output_fname(pet_fname, suffix="_sb")
    if out_abn is None:
        out_abn = _create_output_fname(pet_fname, suffix="_abn")
    if out_img is None and do_image:
        out_img = _create_output_fname(
            pet_fname, suffix="_abn_figure", file_type=".png"
        )

    run_with_registration(
        mri_fname=mri_fname,
        mask_fname=mask_fname,
        out_sbpet=out_sbpet,
        pet_fname=pet_fname,
        out_abn=out_abn,
        out_img=out_img,
        reg_pet_to_mri=reg_pet_to_mr,
        verbose=verbose,
        device=device,
        outputspace=outputspace,
    )
