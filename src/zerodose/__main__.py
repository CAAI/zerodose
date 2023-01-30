"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Zerodose."""


if __name__ == "__main__":
    main(prog_name="zerodose")  # pragma: no cover
