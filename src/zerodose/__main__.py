"""Command-line interface."""
import re
from typing import Sequence
from typing import Union

import click

from zerodose.run_abnormality import create_abnormality_maps
from zerodose.run_synthesize import synthesize_baselines


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


@main.command()
@mri_option
@mask_option
@sbpet_output_option
@verbose_option
@device_option
@no_registration_option
def syn(
    mri_fnames: Sequence[str],
    mask_fnames: Sequence[str],
    out_fnames: Union[Sequence[str], None] = None,
    verbose: bool = True,
    device: str = "cuda:0",
    no_registration: bool = False,
) -> None:
    """Synthesize baseline PET images."""
    if out_fnames is None or len(out_fnames) == 0:
        out_fnames = [
            _create_output_fname(mri_fname, suffix="_sb") for mri_fname in mri_fnames
        ]

    do_registration = not no_registration
    synthesize_baselines(
        mri_fnames,
        mask_fnames,
        out_fnames,
        verbose=verbose,
        device=device,
        do_registration=do_registration,
    )


def _create_output_fname(mri_fname, suffix="_sb", file_type=".nii.gz"):
    """Create output filename from input filename."""
    out_fname = mri_fname
    if out_fname.endswith(".nii.gz"):
        out_fname = re.sub(".nii.gz$", "", out_fname)
    if out_fname.endswith(".nii"):
        out_fname = re.sub(".nii$", "", out_fname)
    out_fname += suffix + file_type
    return out_fname


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
