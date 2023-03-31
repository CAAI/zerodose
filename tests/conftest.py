"""Configuration for pytest."""

import pytest
from click.testing import CliRunner


def pytest_addoption(parser):
    """Add --runslow option to pytest command-line interface."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests with the full model",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests if --runslow is not specified."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()
