"""Command-line interface."""
import re
from typing import Iterable
from typing import Union

import click

from zerodose.run import synthesize_baselines


@click.group()
@click.version_option()
def main() -> None:
    """Zerodose CLI."""
    pass


@main.command()
@click.option(
    "-i",
    "--in",
    "mri_fnames",
    type=click.Path(exists=True),
    required=True,
    multiple=True,
    help="Help test",
)
@click.option(
    "-m",
    "--mask",
    "mask_fnames",
    type=click.Path(exists=True),
    required=True,
    multiple=True,
)
@click.option("-o", "--out", "out_fnames", type=click.Path(), multiple=True)
def syn(
    mri_fnames: Iterable[str],
    mask_fnames: Iterable[str],
    out_fnames: Union[Iterable[str], None] = None,
) -> None:
    """Synthesize baseline PET images."""
    if out_fnames is None:
        out_fnames = [_create_output_fname(mri_fname) for mri_fname in mri_fnames]

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


@main.command()
def abn(pet_fnames, sbpet_fnames, mask_names):
    """Create abnormality maps."""
    pass


if __name__ == "__main__":
    main(
        prog_name="zerodose",
    )
