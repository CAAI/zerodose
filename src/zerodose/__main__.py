"""Command-line interface."""
import re
from typing import Iterable
from typing import Union

import click

from zerodose.run_synthesize import synthesize_baselines
from zerodose.run_abnormality import create_abnormality_maps


@click.group()
@click.version_option()
def main() -> None:
    """Zerodose CLI."""
    pass


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
    "-o", 
    "--out", 
    "out_fnames", 
    type=click.Path(), 
    multiple=True
)

verbose_option = click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    default=False,
    help="Print verbose output.",
)


@main.command()
@mri_option
@mask_option
@sbpet_output_option
#@verbose_option
def syn(
    mri_fnames: Iterable[str],
    mask_fnames: Iterable[str],
    out_fnames: Union[Iterable[str], None] = None,
   # verbose: bool = False,
) -> None:
    """Synthesize baseline PET images."""
    if out_fnames is None:
        out_fnames = [_create_output_fname(mri_fname,suffix="_sb") for mri_fname in mri_fnames]
    synthesize_baselines(mri_fnames, mask_fnames, out_fnames)

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
@main.command()
def abn(
    pet_fnames: Iterable[str],
    sbpet_fnames: Iterable[str],
    mask_fnames: Iterable[str],
    out_fnames: Union[Iterable[str], None] = None,
    ):
    """Create abnormality maps."""
    if out_fnames is None or len(out_fnames) == 0:
        out_fnames = [_create_output_fname(pet_fname,suffix="_abn") for pet_fname in pet_fnames]

    create_abnormality_maps(pet_fnames, sbpet_fnames, mask_fnames, out_fnames)


if __name__ == "__main__":
    main(
        prog_name="zerodose",
    )
