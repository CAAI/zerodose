"""Configuration for pytest."""

import os

import pytest
from click.testing import CliRunner


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        """Debugging in vscode."""
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        """Debugging in vscode."""
        raise excinfo.value


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


@pytest.fixture()
def use_dummy_model():
    """Use the dummy model for testing."""
    os.environ["ZERODOSE_USE_DUMMY_MODEL"] = "1"
    yield
    del os.environ["ZERODOSE_USE_DUMMY_MODEL"]
